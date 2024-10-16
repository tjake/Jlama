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

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.tjake.jlama.model.DistributedContext;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.SegmentedTensor;

import com.google.common.collect.ImmutableMap;
import com.google.common.primitives.Ints;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SafeTensorIndex implements WeightLoader, AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(SafeTensorIndex.class);
    private static final ObjectMapper om = new ObjectMapper();

    public static final String SINGLE_MODEL_NAME = "model.safetensors";
    public static final String MODEL_INDEX_JSON = "model.safetensors.index.json";

    private final Map<String, String> metadata;

    final Map<String, TensorInfo> allTensorInfoMap = new HashMap<>();

    // Map from weight name to file name (this is what's in the JSON file)
    final Map<String, String> weightFileMap;

    // Map from weight name to Weights data
    private final Map<String, Weights> weightMap = new HashMap<>();

    // Map from file name to RandomAccessFile
    private final Map<String, RandomAccessFile> fileMap = new HashMap<>();

    public static SafeTensorIndex loadWithWeights(Path modelRoot) {
        try {
            File indexFile = Paths.get(modelRoot.toString(), MODEL_INDEX_JSON).toFile();

            SafeTensorIndex index = om.readValue(indexFile, SafeTensorIndex.class);
            loadWeights(index, modelRoot);

            return index;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static SafeTensorIndex loadSingleFile(Path modelRoot, String modelFile) {
        try {
            SafeTensorIndex index = new SafeTensorIndex(Collections.emptyMap(), Map.of("model-file", modelFile));
            loadWeights(index, modelRoot);

            return index;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    static void loadWeights(SafeTensorIndex index, Path modelRoot) throws IOException {
        for (Map.Entry<String, String> e : index.weightFileMap.entrySet()) {
            // Only load the file if it's not already loaded
            if (!index.fileMap.containsKey(e.getValue())) {
                RandomAccessFile raf = new RandomAccessFile(Paths.get(modelRoot.toString(), e.getValue()).toFile(), "r");
                index.fileMap.put(e.getValue(), raf);

                // Read the first 1MB of the file to get the TensorInfo
                ByteBuffer header = raf.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, Math.min(1 << 20, raf.length()));

                Map<String, String> metadata = new HashMap<>();
                Map<String, TensorInfo> tensorInfoMap = SafeTensorSupport.readTensorInfoMap(header, Optional.of(metadata));
                index.allTensorInfoMap.putAll(tensorInfoMap);
                int endOfHeaderPosition = header.position();

                Map<List<Long>, List<String>> splits = index.computeMmapSplits(tensorInfoMap, raf.length());
                for (Map.Entry<List<Long>, List<String>> split : splits.entrySet()) {
                    long offset = split.getKey().get(0);
                    long length = split.getKey().get(1);
                    List<String> tensors = split.getValue();
                    int lengthInt = Ints.checkedCast(length - offset);

                    ByteBuffer buf = raf.getChannel().map(FileChannel.MapMode.READ_ONLY, endOfHeaderPosition + offset, lengthInt);

                    Map<String, TensorInfo> mmapTensorInfoMap = tensorInfoMap.entrySet()
                        .stream()
                        .filter(x -> tensors.contains(x.getKey()))
                        .collect(ImmutableMap.toImmutableMap(Map.Entry::getKey, Map.Entry::getValue));

                    Weights mmapWeights = new Weights(metadata, mmapTensorInfoMap, buf, Optional.of(index));
                    for (String tensor : tensors) {
                        index.weightMap.put(tensor, mmapWeights);
                    }
                }
            }
        }
    }

    /**
     * Group tensors into splits that can be mmaped together.
     * Since mmap limitation is integer max_value length.
     *
     * This also adjusts (inplace) the tensor offsets to be relative to the start of the split.
     *
     */
    private Map<List<Long>, List<String>> computeMmapSplits(Map<String, TensorInfo> tensorInfoMap, long fileLength) {
        Map<List<Long>, List<String>> splits = new HashMap<>();
        long lastSplitOffset = 0;
        int tensorsInFile = tensorInfoMap.size();
        int tensorsSplit = 0;
        List<String> tensors = new ArrayList<>();

        Iterator<Map.Entry<String, TensorInfo>> it = new ArrayList<>(tensorInfoMap.entrySet()).iterator();
        Map.Entry<String, TensorInfo> next = null;
        while (tensorsSplit < tensorsInFile && (it.hasNext() || next != null)) {
            tensors.clear();
            long limit = lastSplitOffset + Integer.MAX_VALUE;
            long startOffset = fileLength;
            long endOffset = 0;

            while (it.hasNext() || next != null) {
                next = next == null ? it.next() : next;
                TensorInfo info = next.getValue();
                logger.debug("Tensor {} {} {} limit {}", next.getKey(), info.dataOffsets[0], info.dataOffsets[1], limit);
                if (info.dataOffsets[1] < limit) {
                    tensors.add(next.getKey());
                    tensorsSplit++;

                    if (info.dataOffsets[1] > endOffset) endOffset = info.dataOffsets[1];
                    if (info.dataOffsets[0] < startOffset) startOffset = info.dataOffsets[0];

                    // Adjust the offset to be relative to the start of the split
                    info.dataOffsets[0] -= lastSplitOffset;
                    info.dataOffsets[1] -= lastSplitOffset;

                    logger.debug("Adding tensor {} to split {}-{}", next.getKey(), info.dataOffsets[0], info.dataOffsets[1]);

                    // Used so fetch the tensor from the mmap
                    next = null;
                } else {
                    // Split large tensors up (they will be reassembled in the Weights class)
                    if (tensors.size() == 0) {
                        int bytesPerColumn = info.dType.size() * info.shape[1];

                        // This tensor is too large to fit in a single split
                        // We'll split it up into smaller chunks
                        if (info.dataOffsets[1] > endOffset) endOffset = info.dataOffsets[1];
                        if (info.dataOffsets[0] < startOffset) startOffset = info.dataOffsets[0];

                        // Adjust the offset to be relative to the start of the split
                        info.dataOffsets[0] -= lastSplitOffset;
                        info.dataOffsets[1] -= lastSplitOffset;

                        long offset = info.dataOffsets[0];
                        long length = info.dataOffsets[1] - offset;

                        // Chunk size needs to be a multiple of the column size
                        long chunkSize = Integer.MAX_VALUE - (Integer.MAX_VALUE % bytesPerColumn);
                        long offsetAdded = 0;
                        int chunk = 0;
                        boolean added = false;
                        while (length > 0) {
                            long chunkEnd = Math.min(offset + chunkSize, endOffset);
                            String chunkName = next.getKey() + "-part-" + chunk++;
                            logger.debug(
                                "Adding chunk {} to split {}-{} {}",
                                chunkName,
                                offset,
                                chunkEnd,
                                Ints.checkedCast(chunkEnd - offset)
                            );
                            splits.put(List.of(offset, chunkEnd), List.of(chunkName));

                            // Add TensorInfo for the chunk
                            assert info.shape.length == 2 : "Only 2D tensors supported";
                            int numRowsInChunk = Ints.checkedCast((chunkEnd - offset) / bytesPerColumn);

                            // This tensorInfo is relative to the split which we know is at least the mmap limit
                            // We track the offsetAdded so we can make the offset relative to the current split
                            TensorInfo chunkInfo = new TensorInfo(
                                info.dType,
                                new long[] { numRowsInChunk, info.shape[1] },
                                new long[] { offset - offsetAdded, chunkEnd - offsetAdded }
                            );
                            tensorInfoMap.put(chunkName, chunkInfo);
                            added = true;
                            offsetAdded += chunkEnd - offset;

                            offset = chunkEnd;
                            length -= chunkSize;
                        }

                        if (added) {
                            tensorsSplit++;
                            next = null;
                        }
                    }

                    break;
                }
            }

            assert tensorsSplit > 0 : "No tensors in split";
            logger.debug("Adding split {}-{} with {} tensors of {}", startOffset, endOffset, tensors.size(), tensorsSplit);

            // Add any sections that were split
            if (!tensors.isEmpty()) splits.put(List.of(startOffset, endOffset), new ArrayList<>(tensors));

            if (endOffset > lastSplitOffset) lastSplitOffset = endOffset;
        }

        assert tensorsInFile == tensorsSplit : "Not all tensors were split: " + tensorsSplit + " != " + tensorsInFile;
        return splits;
    }

    @JsonCreator
    SafeTensorIndex(@JsonProperty("metadata") Map<String, String> metadata, @JsonProperty("weight_map") Map<String, String> weightFileMap) {
        this.metadata = ImmutableMap.copyOf(metadata);
        this.weightFileMap = ImmutableMap.copyOf(weightFileMap);
    }

    @Override
    public Map<String, String> metadata() {
        return metadata;
    }

    @Override
    public Map<String, TensorInfo> tensorInfoMap() {
        return allTensorInfoMap;
    }

    @Override
    public AbstractTensor load(String name, DistributedContext dctx, boolean sparseRows, boolean sparseColumns) {
        Weights w = weightMap.get(name);
        if (w == null) {
            // Maybe assemble the tensor from segments
            List<AbstractTensor> segments = new ArrayList<>();
            int idx = 0;
            while (true) {
                String segmentName = name + "-part-" + idx++;
                if (!weightMap.containsKey(segmentName)) break;
                segments.add(weightMap.get(segmentName).load(segmentName, dctx, sparseRows, sparseColumns));
            }

            if (segments.size() > 0) {
                return SegmentedTensor.wrap(segments);
            }

            throw new NoSuchElementException(name);
        }

        return w.load(name, dctx, sparseRows, sparseColumns);
    }

    @Override
    public DType getModelDType() {
        // FIXME: This assumes all weights have the same dtype
        return weightMap.values().iterator().next().getModelDType();
    }

    @Override
    public void close() throws Exception {
        weightMap.clear();
        fileMap.forEach((k, v) -> {
            try {
                v.close();
            } catch (IOException e) {
                // Close quietly
            }
        });
        fileMap.clear();
        allTensorInfoMap.clear();
    }
}
