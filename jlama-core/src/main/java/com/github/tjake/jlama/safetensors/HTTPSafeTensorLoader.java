/*
 * Copyright 2024 T Jake Luciani
 *
 * The Jlama Project licenses this file to you under the Apache License,
 * version 2.0 (the "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */
package com.github.tjake.jlama.safetensors;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.github.tjake.jlama.model.DistributedContext;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.SegmentedTensor;
import com.github.tjake.jlama.tensor.TensorShape;
import com.github.tjake.jlama.util.HttpSupport;
import com.github.tjake.jlama.util.JsonSupport;
import com.github.tjake.jlama.util.Pair;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

public class HTTPSafeTensorLoader implements WeightLoader {
    private static final Logger logger = LoggerFactory.getLogger(HTTPSafeTensorLoader.class);

    private final Path modelRoot;
    private final String indexFile;
    private final String modelName;
    private final Optional<String> branch;
    private final Optional<String> authToken;
    private final SafeTensorIndex index;
    private final Map<String, Pair<RandomAccessFile, AbstractTensor>> layerFiles;
    private final Map<String, TensorInfo> dynamicTensorInfoMap;
    private final Map<String, Integer> tensorFileOffsets;
    private final DType modelDType;

    /**
     * Used for distributed inference
     *
     * Dynamically fetches weights from a remote server based on the distributed context
     *
     * @param modelRoot
     * @param owner
     * @param modelName
     * @param branch
     * @param authToken
     * @throws JsonProcessingException
     */
    public HTTPSafeTensorLoader(
        Path modelRoot,
        String owner,
        String modelName,
        DType modelDType,
        Optional<String> branch,
        Optional<String> authToken
    ) {
        this.modelRoot = modelRoot;
        this.modelName = owner + "/" + modelName;
        this.branch = branch;
        this.indexFile = String.format("%s/%s", modelRoot, SafeTensorIndex.MODEL_INDEX_JSON);
        this.authToken = authToken;

        // Check we have the index file
        if (!new File(indexFile).exists()) {
            this.index = new SafeTensorIndex(Collections.emptyMap(), Map.of("model-file", SafeTensorIndex.SINGLE_MODEL_NAME));
        } else {
            try {
                this.index = JsonSupport.om.readValue(new File(indexFile), SafeTensorIndex.class);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        this.layerFiles = new HashMap<>();
        this.dynamicTensorInfoMap = new HashMap<>();
        this.tensorFileOffsets = new HashMap<>();
        this.modelDType = modelDType;
    }

    @Override
    public Map<String, String> metadata() {
        return index.metadata();
    }

    @Override
    public Map<String, TensorInfo> tensorInfoMap() {
        return dynamicTensorInfoMap;
    }

    @Override
    public AbstractTensor load(String name, DistributedContext dctx, boolean sparseRows, boolean sparseColumns) {
        Preconditions.checkArgument(!sparseColumns || !sparseRows, "Cannot have both sparse rows and columns");
        Preconditions.checkArgument(index.weightFileMap.containsKey(name) || index.weightFileMap.size() == 1, "Unknown weight: " + name);

        // Check if we already have the layer loaded
        if (layerFiles.containsKey(name)) {
            return layerFiles.get(name).right();
        }

        try {
            TensorInfo info = maybeLoadTensorInfo(name);

            Pair<TensorShape, Pair<Long, Long>> offsets = Weights.getLoadOffsets(info, dctx, sparseRows);

            Integer headerOffset = tensorFileOffsets.get(name);

            assert headerOffset != null && headerOffset > 0 : "Failed to find header offset for: " + name;
            String weightFile = index.weightFileMap.getOrDefault(name, SafeTensorIndex.SINGLE_MODEL_NAME);

            TensorShape shape = offsets.left;
            long positionOffset = offsets.right.left + headerOffset;
            long positionLimit = offsets.right.right + headerOffset;
            long length = positionLimit - positionOffset;

            if (length > Integer.MAX_VALUE) {
                // Make a segmented tensor
                assert info.shape.length == 2 : "Only 2D tensors supported";

                List<AbstractTensor> tensors = new ArrayList<>();
                int bytesPerColumn = info.dType.size() * info.shape[1];
                long offset = positionOffset;
                // Chunk size needs to be a multiple of the column size
                long chunkSize = Integer.MAX_VALUE - (Integer.MAX_VALUE % bytesPerColumn);
                int chunkNum = 0;
                while (offset < positionLimit) {
                    long chunkEnd = Math.min(offset + chunkSize, positionLimit);
                    int numRowsInChunk = Ints.checkedCast((chunkEnd - offset) / bytesPerColumn);
                    TensorShape chunkShape = TensorShape.of(numRowsInChunk, info.shape[1]);
                    tensors.add(
                        downloadAndLoadTensor(
                            name + ".part." + chunkNum++,
                            weightFile,
                            info,
                            chunkShape,
                            offset,
                            chunkEnd,
                            dctx,
                            sparseRows,
                            sparseColumns
                        )
                    );
                    offset = chunkEnd;
                }

                AbstractTensor wrapped = SegmentedTensor.wrap(tensors);
                layerFiles.put(name, Pair.of(null, wrapped));

                return wrapped;
            } else {
                return downloadAndLoadTensor(name, weightFile, info, shape, positionOffset, positionLimit, dctx, sparseRows, sparseColumns);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private AbstractTensor downloadAndLoadTensor(
        String name,
        String weightFile,
        TensorInfo info,
        TensorShape shape,
        long positionOffset,
        long positionLimit,
        DistributedContext dctx,
        boolean sparseRows,
        boolean sparseColumns
    ) throws IOException {
        Path weightPath = modelRoot.resolve(weightFile + ".part." + positionOffset + "_" + positionLimit);

        if (!weightPath.toFile().exists()) {
            logger.info("Downloading file: {} for {} {}MB", weightPath, name, (positionLimit - positionOffset) / 1024 / 1024);
            HttpSupport.downloadFile(
                modelName,
                weightFile,
                branch,
                authToken,
                Optional.of(Pair.of(positionOffset, positionLimit)),
                weightPath,
                Optional.empty()
            );
        }

        int length = Ints.checkedCast(positionLimit - positionOffset);

        RandomAccessFile raf = new RandomAccessFile(weightPath.toFile(), "r");
        ByteBuffer buf = raf.getChannel()
            .map(FileChannel.MapMode.READ_ONLY, 0, raf.length())
            .duplicate()
            .order(ByteOrder.LITTLE_ENDIAN)
            .position(0)
            .limit(length);

        if (raf.length() < length) {
            throw new RuntimeException(
                "Failed to download the correct number of bytes: " + raf.length() + " != " + length + " for " + weightPath
            );
        }

        logger.debug("Loading tensor: {} from {} with offsets: {} {}", name, weightPath, positionOffset, positionLimit);

        AbstractTensor tensor = Weights.loadTensorFromBuffer(
            name,
            info.dType,
            modelDType,
            shape,
            buf,
            sparseRows,
            sparseColumns,
            dctx,
            this
        );

        layerFiles.put(name, Pair.of(raf, tensor));

        return tensor;
    }

    private TensorInfo maybeLoadTensorInfo(String name) throws IOException {
        if (dynamicTensorInfoMap.containsKey(name)) {
            return dynamicTensorInfoMap.get(name);
        }

        String weightFile = index.weightFileMap.getOrDefault(name, SafeTensorIndex.SINGLE_MODEL_NAME);

        Path headerFile = modelRoot.resolve(weightFile + ".header");

        if (!Files.exists(headerFile)) {
            // Download the first 1MB of the file to get the tensor info
            HttpSupport.downloadFile(
                modelName,
                weightFile,
                branch,
                authToken,
                Optional.of(Pair.of(0L, (long) 1 << 20)),
                headerFile,
                Optional.empty()
            );
        }

        try (RandomAccessFile raf = new RandomAccessFile(headerFile.toFile(), "r")) {
            ByteBuffer header = raf.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, Math.min(1 << 20, raf.length()));
            Map<String, TensorInfo> info = SafeTensorSupport.readTensorInfoMap(header, Optional.empty());
            int endOfHeaderPosition = header.position();
            for (Map.Entry<String, TensorInfo> e : info.entrySet()) {
                dynamicTensorInfoMap.put(e.getKey(), e.getValue());
                tensorFileOffsets.put(e.getKey(), endOfHeaderPosition);
            }
        }

        assert dynamicTensorInfoMap.containsKey(name) : "Failed to load tensor info for: " + name;
        return dynamicTensorInfoMap.get(name);
    }

    @Override
    public DType getModelDType() {
        return modelDType;
    }

    @Override
    public void close() {
        for (Pair<RandomAccessFile, AbstractTensor> pair : layerFiles.values()) {
            try {
                if (pair.left() != null) pair.left().close();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        layerFiles.clear();
        dynamicTensorInfoMap.clear();
    }

}
