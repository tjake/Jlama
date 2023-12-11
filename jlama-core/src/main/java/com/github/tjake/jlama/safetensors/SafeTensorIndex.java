package com.github.tjake.jlama.safetensors;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.util.Pair;
import com.google.common.collect.ImmutableMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class SafeTensorIndex implements WeightLoader, AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(SafeTensorIndex.class);
    private static final ObjectMapper om = new ObjectMapper();

    public static final String SINGLE_MODEL_NAME = "model.safetensors";
    public static final String MODEL_INDEX_JSON = "model.safetensors.index.json";

    private final Map<String, String> metadata;

    // Map from weight name to file name (this is what's in the JSON file)
    private final Map<String, String> weightFileMap;

    // Map from weight name to Weights data
    private final Map<String, Weights> weightMap = new HashMap<>();


    // Map from file name to RandomAccessFile
    private final Map<String, RandomAccessFile> fileMap = new HashMap<>();

    public static SafeTensorIndex loadWithWeights(Path modelRoot) throws IOException {
        File indexFile = Paths.get(modelRoot.toString(), MODEL_INDEX_JSON).toFile();

        SafeTensorIndex index = om.readValue(indexFile, SafeTensorIndex.class);
        loadWeights(index, modelRoot);

        return index;
    }

    public static SafeTensorIndex loadSingleFile(Path modelRoot, String modelFile) throws IOException {
        SafeTensorIndex index = new SafeTensorIndex(Collections.emptyMap(), Map.of("model-file", modelFile));
        loadWeights(index, modelRoot);

        return index;
    }

    static void loadWeights(SafeTensorIndex index, Path modelRoot) throws IOException {
        for (Map.Entry<String, String> e : index.weightFileMap.entrySet()) {
            // Only load the file if it's not already loaded
            if (!index.fileMap.containsKey(e.getValue())) {
                RandomAccessFile raf = new RandomAccessFile(Paths.get(modelRoot.toString(), e.getValue()).toFile(), "r");
                index.fileMap.put(e.getValue(), raf);

                //Read the first 1MB of the file to get the TensorInfo
                ByteBuffer header = raf.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, Math.min(1 << 20, raf.length()));

                Map<String, String> metadata = new HashMap<>();
                Map<String, TensorInfo> tensorInfoMap = SafeTensorSupport.readTensorInfoMap(header, Optional.of(metadata));
                int endOfHeaderPosition = header.position();

                Map<List<Long>, List<String>> splits = index.computeMmapSplits(tensorInfoMap, raf.length());
                for (Map.Entry<List<Long>, List<String>> split : splits.entrySet()) {
                    long offset = split.getKey().get(0);
                    long length = split.getKey().get(1);
                    List<String> tensors = split.getValue();

                    ByteBuffer buf = raf.getChannel().map(FileChannel.MapMode.READ_ONLY, endOfHeaderPosition + offset, (length - offset)).load();
                    Map<String, TensorInfo> mmapTensorInfoMap = tensorInfoMap.entrySet().stream()
                            .filter(x -> tensors.contains(x.getKey())).collect(ImmutableMap.toImmutableMap(Map.Entry::getKey, Map.Entry::getValue));

                    Weights mmapWeights = new Weights(metadata, mmapTensorInfoMap, buf);
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
        Set<String> added = new HashSet<>();
        Map<List<Long>, List<String>> splits = new HashMap<>();
        long lastSplitOffset = 0;
        while (added.size() < tensorInfoMap.size()) {
            List<String> tensors = new ArrayList<>();
            long limit = lastSplitOffset + Integer.MAX_VALUE;
            long startOffset = fileLength;
            long endOffset = 0;

            for (Map.Entry<String, TensorInfo> e : tensorInfoMap.entrySet()) {
                if (added.contains(e.getKey()))
                    continue;

                TensorInfo info = e.getValue();

                if (info.dataOffsets[1] < limit) {
                    tensors.add(e.getKey());
                    added.add(e.getKey());

                    if (info.dataOffsets[1] > endOffset)
                        endOffset = info.dataOffsets[1];

                    if (info.dataOffsets[0] < startOffset)
                        startOffset = info.dataOffsets[0];

                    // Adjust the offset to be relative to the start of the split
                    info.dataOffsets[0] -= lastSplitOffset;
                    info.dataOffsets[1] -= lastSplitOffset;

                    logger.debug("Adding tensor {} to split {}-{}", e.getKey(), info.dataOffsets[0], info.dataOffsets[1]);
                }
            }

            logger.debug("Adding split {}-{} with {} tensors", startOffset, endOffset, tensors.size());
            assert endOffset - startOffset < Integer.MAX_VALUE : "Mmap split too large " + (endOffset - startOffset) + " > " + Integer.MAX_VALUE + " " + lastSplitOffset;
            splits.put(List.of(startOffset, endOffset), tensors);
            lastSplitOffset = endOffset;
        }

        return splits;
    }

    @JsonCreator
    SafeTensorIndex(@JsonProperty("metadata") Map<String, String> metadata,
                           @JsonProperty("weight_map") Map<String, String> weightFileMap) {
        this.metadata = ImmutableMap.copyOf(metadata);
        this.weightFileMap = ImmutableMap.copyOf(weightFileMap);
    }

    @Override
    public Map<String, String> metadata() {
        return metadata;
    }

    @Override
    public Map<String, TensorInfo> tensorInfoMap() {
        Map<String, TensorInfo> tensorInfoMap = new HashMap<>();
        for (String name : weightMap.keySet()) {
            Weights w = weightMap.get(name);
            if (w == null)
                throw new NoSuchElementException(name);

            tensorInfoMap.put(name, w.tensorInfoMap().get(name));
        }

        return tensorInfoMap;
    }

    @Override
    public AbstractTensor load(String name, Optional<Pair<Integer, Integer>> offset) {
        Weights w = weightMap.get(name);
        if (w == null)
            throw new NoSuchElementException(name);

        AbstractTensor t = w.load(name);
        return offset.map(o -> {
            logger.info("Sparsifying tensor {} with shape {}", name, o);
            return t.sparsify(o.left, o.right);
        }).orElse(t);
    }

    @Override
    public DType getModelDType() {
        // FIXME: This assumes all weights have the same dtype
        return weightMap.values().iterator().next().getModelDType();
    }

    @Override
    public void close() throws Exception {
        weightMap.clear();
        fileMap.forEach((k,v) -> {
            try {
                v.close();
            } catch (IOException e) {
                // Close quietly
            }
        });
        fileMap.clear();
    }
}
