package com.github.tjake.llmj.safetensors;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.tjake.llmj.model.Tensor;
import com.google.common.collect.ImmutableMap;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.NoSuchElementException;

public class SafeTensorIndex implements AutoCloseable {
    private static final ObjectMapper om = new ObjectMapper();

    private final Map<String, Object> metadata;
    private final Map<String, String> weightFileMap;
    private final Map<String, Weights> weightMap = new HashMap<>();

    private final Map<String, RandomAccessFile> fileMap = new HashMap<>();

    public static SafeTensorIndex loadWithWeights(Path modelRoot) throws IOException {
        SafeTensorIndex index = om.readValue(Paths.get(modelRoot.toString(), "model.safetensors.index.json").toFile(), SafeTensorIndex.class);

        for (Map.Entry<String, String> e : index.weightFileMap.entrySet()) {
            if (!index.fileMap.containsKey(e.getValue())) {
                RandomAccessFile raf = new RandomAccessFile(Paths.get(modelRoot.toString(), e.getValue()).toFile(), "r");
                index.fileMap.put(e.getValue(), raf);
                long s = raf.length();
                long s2 = Integer.MAX_VALUE;
                if (s > s2)
                    throw new IllegalArgumentException("File too large: " + e.getValue());

                index.weightMap.put(e.getValue(), SafeTensors.readBytes(raf.getChannel().map(FileChannel.MapMode.READ_ONLY, 0, raf.length())));
            }
        }

        return index;
    }

    public Tensor load(String name) {
        String f = weightFileMap.get(name);
        if (f == null)
            throw new NoSuchElementException(name);

        return weightMap.get(f).load(name);
    }

    @JsonCreator
    SafeTensorIndex(@JsonProperty("metadata") Map<String, Object> metadata,
                           @JsonProperty("weight_map") Map<String, String> weightFileMap) {
        this.metadata = ImmutableMap.copyOf(metadata);
        this.weightFileMap = ImmutableMap.copyOf(weightFileMap);
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
