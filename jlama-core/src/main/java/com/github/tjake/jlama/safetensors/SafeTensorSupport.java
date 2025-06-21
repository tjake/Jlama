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

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.type.MapType;
import com.github.tjake.jlama.model.ModelSupport;
import com.github.tjake.jlama.safetensors.tokenizer.TokenizerModel;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.Q4ByteBufferTensor;
import com.github.tjake.jlama.tensor.Q5ByteBufferTensor;
import com.github.tjake.jlama.tensor.Q8ByteBufferTensor;
import com.github.tjake.jlama.util.HttpSupport;
import com.github.tjake.jlama.util.ProgressReporter;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

import static com.github.tjake.jlama.util.JsonSupport.om;

public class SafeTensorSupport {
    private static final Logger logger = LoggerFactory.getLogger(SafeTensorSupport.class);
    private static final MapType metadataTypeReference = om.getTypeFactory().constructMapType(Map.class, String.class, String.class);

    public static Map<String, TensorInfo> readTensorInfoMap(ByteBuffer buf, Optional<Map<String, String>> saveMetadata) {
        final long MAX_HEADER_LENGTH = 1024 * 1024 * 1024; // 1 GB
        buf = buf.order(ByteOrder.LITTLE_ENDIAN);
        long headerLength = buf.getLong();

        // headerLength is negative
        if (headerLength < 0) {
            throw new IllegalArgumentException("Header length cannot be negative: " + headerLength);
        }
        // headerLength exceeds the maximum allowed length MAX_HEADER_LENGTH
        if (headerLength > MAX_HEADER_LENGTH) {
            throw new IllegalArgumentException(String.format("Header length %d exceeds the maximum allowed length %d.", headerLength, MAX_HEADER_LENGTH));
        }

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

            // Sort by value using a lambda expression
            Map<String, TensorInfo> sortedMap = tensorInfoMap.entrySet()
                .stream()
                .sorted(Map.Entry.comparingByValue())
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue, (e1, e2) -> e1, LinkedHashMap::new));

            final Map<String, String> finalMetadata = metadata;
            saveMetadata.ifPresent(m -> m.putAll(finalMetadata));

            return sortedMap;
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

    public static ModelSupport.ModelType detectModel(File configFile) throws IOException {
        JsonNode rootNode = om.readTree(configFile);
        if (!rootNode.has("model_type")) throw new IllegalArgumentException("Config missing model_type field.");

        return ModelSupport.getModelType(rootNode.get("model_type").textValue().toUpperCase());
    }

    public static WeightLoader loadWeights(File baseDir) {
        if (Files.exists(Paths.get(baseDir.getAbsolutePath(), SafeTensorIndex.MODEL_INDEX_JSON))) return SafeTensorIndex.loadWithWeights(
            baseDir.toPath()
        );

        if (Files.exists(Paths.get(baseDir.getAbsolutePath(), SafeTensorIndex.SINGLE_MODEL_NAME))) return SafeTensorIndex.loadSingleFile(
            baseDir.toPath(),
            SafeTensorIndex.SINGLE_MODEL_NAME
        );

        throw new IllegalArgumentException("No safetensor model found in: " + baseDir);
    }

    public static boolean isModelLocal(Path modelRoot) {

        if (Files.exists(modelRoot.resolve(SafeTensorIndex.SINGLE_MODEL_NAME))) return true;
        try {
            if (Files.exists(modelRoot.resolve(SafeTensorIndex.MODEL_INDEX_JSON))) {
                SafeTensorIndex index = om.readValue(modelRoot.resolve(SafeTensorIndex.MODEL_INDEX_JSON).toFile(), SafeTensorIndex.class);

                for (String file : index.weightFileMap.values()) {
                    if (!Files.exists(modelRoot.resolve(file))) {
                        return false;
                    }
                }

                return true;
            }
        } catch (IOException e) {
            logger.error("Error reading model index", e);
            return false;
        }

        return false;
    }

    public static TokenizerModel loadTokenizer(Path modelRoot) throws IOException {
        File tokenizerJson = modelRoot.resolve("tokenizer.json").toFile();
        Preconditions.checkArgument(tokenizerJson.exists(), "No tokenizer.json found in " + modelRoot);

        JsonNode rootNode = om.readTree(tokenizerJson);
        if (!rootNode.has("model")) throw new IllegalArgumentException("Json missing 'model' key");

        TokenizerModel model = om.treeToValue(rootNode.get("model"), TokenizerModel.class);

        if (rootNode.has("added_tokens") && rootNode.get("added_tokens") != null) {
            List<Map<String, Object>> addedTokens = om.convertValue(rootNode.get("added_tokens"), List.class);
            model.setAddedTokens(addedTokens);
        }

        if (rootNode.has("pre_tokenizer") && rootNode.get("pre_tokenizer") != null) model.setPreTokenizer(
            om.treeToValue(rootNode.get("pre_tokenizer"), TokenizerModel.PreTokenizer.class)
        );

        if (rootNode.has("normalizer") && rootNode.get("normalizer") != null) model.setNormalizer(
            om.treeToValue(rootNode.get("normalizer"), TokenizerModel.Normalizer.class)
        );

        File tokenizerConfigJson = modelRoot.resolve("tokenizer_config.json").toFile();
        if (tokenizerConfigJson.exists()) {
            JsonNode configNode = om.readTree(tokenizerConfigJson);
            if (configNode.has("legacy")) model.setLegacy(configNode.get("legacy").asBoolean());

            if (configNode.has("chat_template")) {
                JsonNode chatTemplateNode = configNode.get("chat_template");
                Map<String, String> promptTemplates = new HashMap<>();
                if (chatTemplateNode.isTextual()) {
                    promptTemplates.put("default", chatTemplateNode.asText());
                } else if (chatTemplateNode.isArray()) {
                    List<Map<String, String>> chatTemplates = om.convertValue(chatTemplateNode, List.class);
                    for (Map<String, String> chatTemplate : chatTemplates) {
                        if (chatTemplate.containsKey("name") && chatTemplate.containsKey("template")) {
                            promptTemplates.put(chatTemplate.get("name"), chatTemplate.get("template"));
                        } else {
                            throw new IllegalArgumentException("Invalid chat_template format");
                        }
                    }
                } else {
                    throw new IllegalArgumentException("Invalid chat_template format");
                }

                model.setPromptTemplates(promptTemplates);
            }

            if (configNode.has("eos_token")) {
                model.setEosToken(configNode.get("eos_token").asText());
            }

            if (configNode.has("bos_token")) {
                model.setBosToken(configNode.get("bos_token").asText());
            }
        }

        return model;
    }

    public static Path quantizeModel(
        Path modelRoot,
        DType modelQuantization,
        String[] skipLayerPrefixes,
        String[] dropLayerPrefixes,
        Optional<Path> outputRoot
    ) throws IOException {
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

                if (drop) continue;

                try (AbstractTensor tr = wl.load(e.getKey())) {

                    boolean skipQ = false;
                    if (skipLayerPrefixes != null) {
                        for (String skipLayerPrefix : skipLayerPrefixes) {
                            if (e.getKey().contains(skipLayerPrefix)) {
                                logger.info("Skipping quantization of layer: " + e.getKey());
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
                            // FIXME: Need to add b5 bits
                            throw new UnsupportedOperationException("TODO");
                        // break;
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

        // Now create the output file
        String baseDirName = modelRoot.getName(modelRoot.getNameCount() - 1).toString();
        Path parentPath = modelRoot.getParent();

        Path qPath = outputRoot.orElseGet(() -> Paths.get(parentPath.toString(), baseDirName + "-J" + modelQuantization.name()));
        File qDir = qPath.toFile();
        qDir.mkdirs();

        // Copy config.json and tokenizer.json
        Files.copy(modelRoot.resolve("config.json"), qPath.resolve("config.json"));
        Files.copy(modelRoot.resolve("tokenizer.json"), qPath.resolve("tokenizer.json"));
        Files.copy(modelRoot.resolve("README.md"), qPath.resolve("README.md"));

        // Copy README.md and add jlama header
        addJlamaHeader(baseDirName, qPath.resolve("README.md"));

        if (Files.exists(modelRoot.resolve("tokenizer_config.json"))) Files.copy(
            modelRoot.resolve("tokenizer_config.json"),
            qPath.resolve("tokenizer_config.json")
        );

        try (RandomAccessFile raf = new RandomAccessFile(qPath.resolve("model.safetensors").toFile(), "rw")) {
            byte[] header = om.writeValueAsBytes(writtenInfo);
            byte[] hsize = new byte[Long.BYTES];
            ByteBuffer.wrap(hsize).order(ByteOrder.LITTLE_ENDIAN).putLong(header.length);
            raf.write(hsize);
            raf.write(header);

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

    private static void addJlamaHeader(String modelName, Path readmePath) throws IOException {
        String cleanName = modelName.replaceAll("_", "/");
        String header = String.format(
            "# Quantized Version of %s \n\n"
                + "This model is a quantized variant of the %s model, optimized for use with Jlama, a Java-based inference engine. "
                + "The quantization process reduces the model's size and improves inference speed, while maintaining high accuracy "
                + "for efficient deployment in production environments.\n\n"
                + "For more information on Jlama, visit the [Jlama GitHub repository](https://github.com/tjake/jlama).\n\n"
                + "---\n\n",
            cleanName,
            cleanName
        );
        String readme = new String(Files.readAllBytes(readmePath));
        boolean startMeta = false;
        boolean finishedMeta = false;
        int linenum = 0;
        StringBuilder finalReadme = new StringBuilder();
        for (String line : readme.split("\n")) {
            if (linenum++ == 0) {
                if (line.startsWith("---")) {
                    startMeta = true;
                } else {
                    finalReadme.append(header);
                }
            } else if (startMeta && !finishedMeta && line.startsWith("---")) {
                finishedMeta = true;
                line += "\n\n" + header;
            }
            finalReadme.append(line).append("\n");
        }

        Files.write(readmePath, finalReadme.toString().getBytes());
    }

    public static File maybeDownloadModel(String modelDir, String fullModelName, ProgressReporter progressReporter) throws IOException {
        String[] parts = fullModelName.split("/");
        if (parts.length == 0 || parts.length > 2) {
            throw new IllegalArgumentException("Model must be in the form owner/name");
        }

        String owner;
        String name;

        if (parts.length == 1) {
            owner = null;
            name = fullModelName;
        } else {
            owner = parts[0];
            name = parts[1];
        }

        return maybeDownloadModel(
            modelDir,
            Optional.ofNullable(owner),
            name,
            true,
            Optional.empty(),
            Optional.empty(),
            Optional.ofNullable(progressReporter)
        );
    }

    public static File maybeDownloadModel(String modelDir, String fullModelName) throws IOException {
        return maybeDownloadModel(modelDir, fullModelName, null);
    }

    public static Path constructLocalModelPath(String modelDir, String owner, String modelName) {
        return Paths.get(modelDir, owner + "_" + modelName);
    }

    static String FINISHED_MARKER = ".finished";

    /**
     * Download a model from HuggingFace and return the path to the model directory
     *
     * @param modelDir The directory to save the model to
     * @param modelOwner The owner of the HF model (if any)
     * @param modelName The name of the HF model
     * @param downloadWeights Include the weights or leave them out
     * @param optionalBranch The branch of the model to download
     * @param optionalAuthHeader The authorization header to use for the request
     * @param optionalProgressReporter A consumer to report download progress
     * @return The path to the downloaded model directory
     * @throws IOException
     */
    public static File maybeDownloadModel(
        String modelDir,
        Optional<String> modelOwner,
        String modelName,
        boolean downloadWeights,
        Optional<String> optionalBranch,
        Optional<String> optionalAuthHeader,
        Optional<ProgressReporter> optionalProgressReporter
    ) throws IOException {

        Path localModelDir = constructLocalModelPath(modelDir, modelOwner.orElse("na"), modelName);
        // Check if the model is already downloaded
        if (Files.exists(localModelDir.resolve(FINISHED_MARKER))) {
            return localModelDir.toFile();
        }

        String hfModel = modelOwner.map(mo -> mo + "/" + modelName).orElse(modelName);
        InputStream modelInfoStream = HttpSupport.getResponse(
            "https://huggingface.co/api/models/" + hfModel + "/tree/" + optionalBranch.orElse("main"),
            optionalAuthHeader,
            Optional.empty()
        ).left;
        String modelInfo = HttpSupport.readInputStream(modelInfoStream);

        if (modelInfo == null) {
            throw new IOException("No valid model found or trying to access a restricted model (please include correct access token)");
        }

        List<String> allFiles = parseFileList(modelInfo);
        if (allFiles.isEmpty()) {
            throw new IOException("No valid model found");
        }

        List<String> tensorFiles = new ArrayList<>();
        boolean hasSafetensor = false;
        for (String currFile : allFiles) {
            String f = currFile.toLowerCase();
            if ((f.contains("safetensor") && !f.contains("consolidated"))
                || f.contains("readme")
                || f.equals("config.json")
                || f.contains("tokenizer")) {

                if (f.contains("safetensor")) {
                    hasSafetensor = true;
                }

                if (!downloadWeights && f.contains("safetensor")) {
                    continue;
                }

                tensorFiles.add(currFile);
            }
        }

        if (!hasSafetensor) {
            throw new IOException("Model is not available in safetensor format");
        }

        Files.createDirectories(localModelDir);

        for (String currFile : tensorFiles) {
            HttpSupport.downloadFile(
                hfModel,
                currFile,
                optionalBranch,
                optionalAuthHeader,
                Optional.empty(),
                localModelDir.resolve(currFile),
                optionalProgressReporter
            );
        }

        // When fully downloaded, create a .finished file
        Files.createFile(localModelDir.resolve(FINISHED_MARKER));

        return localModelDir.toFile();
    }

    private static List<String> parseFileList(String modelInfo) throws IOException {
        List<String> fileList = new ArrayList<>();

        ObjectMapper objectMapper = new ObjectMapper();
        JsonNode siblingsNode = objectMapper.readTree(modelInfo);
        if (siblingsNode.isArray()) {
            for (JsonNode siblingNode : siblingsNode) {
                String rFilename = siblingNode.path("path").asText();
                fileList.add(rFilename);
            }
        }

        return fileList;
    }
}
