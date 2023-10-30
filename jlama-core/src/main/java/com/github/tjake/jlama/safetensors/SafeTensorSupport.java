package com.github.tjake.jlama.safetensors;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.type.MapType;
import com.github.tjake.jlama.model.ModelSupport.ModelType;
import com.github.tjake.jlama.safetensors.tokenizer.TokenizerModel;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class SafeTensorSupport {
    private static final ObjectMapper om = new ObjectMapper().configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
    private static final MapType metadataTypeReference = om.getTypeFactory().constructMapType(Map.class, String.class, String.class);

    public static Map<String, TensorInfo> readTensorInfoMap(ByteBuffer buf, Optional<Map<String, String>> saveMetadata) {
        long headerLength = buf.order() == ByteOrder.BIG_ENDIAN ? Long.reverseBytes(buf.getLong()) : buf.getLong();
        byte[] header = new byte[Ints.checkedCast(headerLength)];
        buf.get(header);

        try {
            JsonNode rootNode = om.readTree(header);
            Iterator<Map.Entry<String, JsonNode>> fields = rootNode.fields();
            Map<String, TensorInfo> tensorInfoMap = new HashMap<>();
            Map<String, String> metadata = Collections.emptyMap();

            while (fields.hasNext()) {
                Map.Entry<String, JsonNode> field = fields.next();
                if (field.getKey().equalsIgnoreCase("__metadata__")) {
                    metadata = om.treeToValue(field.getValue(), metadataTypeReference);
                } else {
                    TensorInfo tensorInfo = om.treeToValue(field.getValue(), TensorInfo.class);
                    tensorInfoMap.put(field.getKey(), tensorInfo);
                }
            }

            final Map<String, String> finalMetadata = metadata;
            saveMetadata.ifPresent(m -> m.putAll(finalMetadata));

            return tensorInfoMap;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static Weights readWeights(ByteBuffer safeBuf) {
        safeBuf = safeBuf.duplicate();
        Map<String, String> metadata = new HashMap<>();
        Map<String, TensorInfo> tensorInfoMap = readTensorInfoMap(safeBuf, Optional.of(metadata));

        return new Weights(metadata, tensorInfoMap, safeBuf.slice());
    }

    public static ModelType detectModel(File configFile) throws IOException {
        JsonNode rootNode = om.readTree(configFile);
        if (!rootNode.has("model_type"))
            throw new IllegalArgumentException("Config missing model_type field.");

        return ModelType.valueOf(rootNode.get("model_type").textValue().toUpperCase());
    }

    public static WeightLoader loadWeights(File baseDir) throws IOException {
        if (Files.exists(Paths.get(baseDir.getAbsolutePath(), SafeTensorIndex.MODEL_INDEX_JSON)))
            return SafeTensorIndex.loadWithWeights(baseDir.toPath());

        if (Files.exists(Paths.get(baseDir.getAbsolutePath(), SafeTensorIndex.SINGLE_MODEL_NAME)))
            return SafeTensorIndex.loadSingleFile(baseDir.toPath(), SafeTensorIndex.SINGLE_MODEL_NAME);

        throw new IllegalArgumentException("No safetensors model found in: " + baseDir);
    }

    public static TokenizerModel loadTokenizer(Path modelRoot) throws IOException {
        File tokenizerJson = modelRoot.resolve("tokenizer.json").toFile();
        Preconditions.checkArgument(tokenizerJson.exists(), "No tokenizer.jsom found in " + modelRoot);

        JsonNode rootNode = om.readTree(tokenizerJson);
        if (!rootNode.has("model"))
            throw new IllegalArgumentException("Json missing 'model' key");

        return om.treeToValue(rootNode.get("model"), TokenizerModel.class);
    }
}
