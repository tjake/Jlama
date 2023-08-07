package com.github.tjake.llmj.safetensors;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.type.MapType;
import com.google.common.primitives.Ints;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.*;

public class SafeTensors {
    private static final ObjectMapper om = new ObjectMapper();
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

    public static Weights readBytes(ByteBuffer safeBuf) {
        safeBuf = safeBuf.duplicate();
        Map<String, String> metadata = new HashMap<>();
        Map<String, TensorInfo> tensorInfoMap = readTensorInfoMap(safeBuf, Optional.of(metadata));

        return new Weights(metadata, tensorInfoMap, safeBuf.slice());
    }
}
