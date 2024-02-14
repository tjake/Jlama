package com.github.tjake.jlama.safetensors;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.type.MapType;
import com.github.tjake.jlama.model.ModelSupport.ModelType;
import com.github.tjake.jlama.safetensors.tokenizer.TokenizerModel;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.Q4ByteBufferTensor;
import com.github.tjake.jlama.tensor.Q5ByteBufferTensor;
import com.github.tjake.jlama.tensor.Q8ByteBufferTensor;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SafeTensorSupport {
    private static final Logger logger = LoggerFactory.getLogger(SafeTensorSupport.class);
    private static final ObjectMapper om = new ObjectMapper().configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
    private static final MapType metadataTypeReference = om.getTypeFactory().constructMapType(Map.class, String.class, String.class);

    public static Map<String, TensorInfo> readTensorInfoMap(ByteBuffer buf, Optional<Map<String, String>> saveMetadata) {
        buf = buf.order(ByteOrder.LITTLE_ENDIAN);
        long headerLength = buf.getLong();
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

        return new Weights(metadata, tensorInfoMap, safeBuf.slice(), Optional.empty());
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

        throw new IllegalArgumentException("No safetensor model found in: " + baseDir);
    }

    public static TokenizerModel loadTokenizer(Path modelRoot) throws IOException {
        File tokenizerJson = modelRoot.resolve("tokenizer.json").toFile();
        Preconditions.checkArgument(tokenizerJson.exists(), "No tokenizer.jsom found in " + modelRoot);

        JsonNode rootNode = om.readTree(tokenizerJson);
        if (!rootNode.has("model"))
            throw new IllegalArgumentException("Json missing 'model' key");

        return om.treeToValue(rootNode.get("model"), TokenizerModel.class);
    }

    public static Path quantizeModel(Path modelRoot, DType modelQuantization, String[] skipLayerPrefixes, String[] dropLayerPrefixes, Optional<Path> outputRoot) throws IOException {
        File tmp = File.createTempFile("safe", "tensor");
        tmp.deleteOnExit();
        WeightLoader wl = SafeTensorSupport.loadWeights(modelRoot.toFile());
        Map<String, Object> writtenInfo = new HashMap<>();

        try (RandomAccessFile raf = new RandomAccessFile(tmp, "rw")) {
            Map<String, TensorInfo> tensors = wl.tensorInfoMap();

            for (Map.Entry<String, TensorInfo> e : tensors.entrySet()) {
                boolean drop = false;
                if (dropLayerPrefixes != null) {
                    for (String dropLayerPrefix : dropLayerPrefixes) {
                        if (e.getKey().startsWith(dropLayerPrefix)) {
                            logger.info("Dropping layer: " + e.getKey());
                            drop = true;
                        }
                    }
                }

                if (drop)
                    continue;

                try (AbstractTensor tr = wl.load(e.getKey())) {

                    boolean skipQ = false;
                    if (skipLayerPrefixes != null) {
                        for (String skipLayerPrefix : skipLayerPrefixes) {
                            if (e.getKey().startsWith(skipLayerPrefix)) {
                                skipQ = true;
                                break;
                            }
                        }
                    }

                    AbstractTensor t = skipQ ? tr : tr.quantize(modelQuantization);

                    switch (t.dType()) {
                        case F32:
                        case BF16:
                        case F16:
                            writtenInfo.put(e.getKey(), t.save(raf.getChannel()));
                            break;
                        case Q4:
                            writtenInfo.put(e.getKey(), t.save(raf.getChannel()));
                            writtenInfo.put(e.getKey() + ".qb", ((Q4ByteBufferTensor) t).getBlockF().save(raf.getChannel()));
                            break;
                        case Q5:
                            writtenInfo.put(e.getKey(), t.save(raf.getChannel()));
                            writtenInfo.put(e.getKey() + ".qb", ((Q5ByteBufferTensor) t).getBlockF().save(raf.getChannel()));
                            //FIXME: Need to add b5 bits
                            throw new UnsupportedOperationException("TODO");
                            //break;
                        case I8:
                            writtenInfo.put(e.getKey(), t.save(raf.getChannel()));
                            writtenInfo.put(e.getKey() + ".qb", ((Q8ByteBufferTensor) t).getBlockF().save(raf.getChannel()));
                            break;
                        default:
                            throw new UnsupportedOperationException("" + t.dType() + " not implemented");
                    }
                }
            }
        }

        //Now create the output file
        String baseDirName = modelRoot.getName(modelRoot.getNameCount() - 1).toString();
        Path parentPath = modelRoot.getParent();

        Path qPath = outputRoot.orElseGet(() -> Paths.get(parentPath.toString(), baseDirName + "-jlama-" + modelQuantization.name()));
        File qDir = qPath.toFile();
        qDir.mkdirs();

        //Copy config.json and tokenizer.json
        Files.copy(modelRoot.resolve("config.json"), qPath.resolve("config.json"));
        Files.copy(modelRoot.resolve("tokenizer.json"), qPath.resolve("tokenizer.json"));

        try (RandomAccessFile raf = new RandomAccessFile(qPath.resolve("model.safetensors").toFile(), "rw")) {
            FileChannel chan = raf.getChannel();

            byte[] header = om.writeValueAsBytes(writtenInfo);
            logger.debug("pos = {}", chan.position());
            byte[] hsize = new byte[Long.BYTES];
            ByteBuffer.wrap(hsize).order(ByteOrder.LITTLE_ENDIAN).putLong(header.length);
            raf.write(hsize);
            logger.debug("pos = {}", chan.position());
            raf.write(header);
            logger.debug("pos = {}", chan.position());

            Files.copy(tmp.toPath(), new OutputStream() {
                @Override
                public void write(int b) throws IOException {
                    raf.write(b);
                }

                @Override
                public void write(byte[] b) throws IOException {
                    raf.write(b);
                }

                @Override
                public void write(byte[] b, int off, int len) throws IOException {
                    raf.write(b, off, len);
                }
            });
        }

        return qPath;
    }
}
