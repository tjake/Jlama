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
package com.github.tjake.jlama.model;

import com.github.tjake.jlama.model.bert.BertModelType;
import com.github.tjake.jlama.model.gemma.GemmaModelType;
import com.github.tjake.jlama.model.gemma2.Gemma2ModelType;
import com.github.tjake.jlama.model.gpt2.GPT2ModelType;
import com.github.tjake.jlama.model.granite.GraniteModelType;
import com.github.tjake.jlama.model.llama.LlamaModelType;
import com.github.tjake.jlama.model.mistral.MistralModelType;
import com.github.tjake.jlama.model.mixtral.MixtralModelType;
import com.github.tjake.jlama.model.qwen2.Qwen2ModelType;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.SafeTensorSupport;
import com.github.tjake.jlama.safetensors.WeightLoader;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;
import com.github.tjake.jlama.util.PhysicalCoreExecutor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Function;

import static com.github.tjake.jlama.util.JsonSupport.om;

public class ModelSupport {
    private static final Logger logger = LoggerFactory.getLogger(ModelSupport.class);
    private static final Map<String, ModelType> registry = new HashMap<>();

    // Initialize default model types
    static {
        register("BERT", new BertModelType());
        register("GEMMA", new GemmaModelType());
        register("GEMMA2", new Gemma2ModelType());
        register("GPT2", new GPT2ModelType());
        register("GRANITE", new GraniteModelType());
        register("LLAMA", new LlamaModelType());
        register("MISTRAL", new MistralModelType());
        register("MIXTRAL", new MixtralModelType());
        register("QWEN2", new Qwen2ModelType());
    }

    // Register a model type with a unique name
    public static void register(String name, ModelType modelType) {
        registry.putIfAbsent(name, modelType);
    }

    // Retrieve a model type by name
    public static ModelType getModelType(String name) {
        ModelType modelType = registry.get(name);
        if (modelType == null) {
            throw new IllegalArgumentException("Unknown model type: " + name);
        }
        return modelType;
    }

    /** Shortcut for loading a model for token generation*/
    public static AbstractModel loadModel(File model, DType workingMemoryType, DType workingQuantizationType) {
        return loadModel(model, null, workingMemoryType, workingQuantizationType, Optional.empty(), Optional.empty());
    }

    /** Shortcut for loading a model for embeddings */
    public static AbstractModel loadEmbeddingModel(File model, DType workingMemoryType, DType workingQuantizationType) {
        return loadModel(
            AbstractModel.InferenceType.FULL_EMBEDDING,
            model,
            null,
            workingMemoryType,
            workingQuantizationType,
            Optional.empty(),
            Optional.empty(),
            Optional.empty(),
            SafeTensorSupport::loadWeights
        );
    }

    /** Shortcut for loading a model for embeddings */
    public static AbstractModel loadClassifierModel(File model, DType workingMemoryType, DType workingQuantizationType) {
        return loadModel(
            AbstractModel.InferenceType.FULL_CLASSIFICATION,
            model,
            null,
            workingMemoryType,
            workingQuantizationType,
            Optional.empty(),
            Optional.empty(),
            Optional.empty(),
            SafeTensorSupport::loadWeights
        );
    }

    public static AbstractModel loadModel(
        File model,
        File workingDirectory,
        DType workingMemoryType,
        DType workingQuantizationType,
        Optional<DType> modelQuantization,
        Optional<Integer> threadCount
    ) {
        return loadModel(
            AbstractModel.InferenceType.FULL_GENERATION,
            model,
            workingDirectory,
            workingMemoryType,
            workingQuantizationType,
            modelQuantization,
            threadCount,
            Optional.empty(),
            SafeTensorSupport::loadWeights
        );
    }

    public static AbstractModel loadModel(
        AbstractModel.InferenceType inferenceType,
        File model,
        File workingDirectory,
        DType workingMemoryType,
        DType workingQuantizationType,
        Optional<DType> modelQuantization,
        Optional<Integer> threadCount,
        Optional<Function<Config, DistributedContext>> distributedContextLoader,
        Function<File, WeightLoader> weightLoaderSupplier
    ) {

        if (!model.exists()) {
            throw new IllegalArgumentException("Model location does not exist: " + model);
        }

        File baseDir = model.isFile() ? model.getParentFile() : model;

        // Find config
        if (!baseDir.isDirectory()) {
            throw new IllegalArgumentException("Model directory does not exist: " + baseDir);
        }

        File configFile = null;
        for (File f : Objects.requireNonNull(baseDir.listFiles())) {
            if (f.getName().equals("config.json")) {
                configFile = f;
                break;
            }
        }

        if (configFile == null) {
            throw new IllegalArgumentException("config.json in model directory does not exist: " + baseDir);
        }

        try {
            threadCount.ifPresent(PhysicalCoreExecutor::overrideThreadCount);

            ModelType modelType = SafeTensorSupport.detectModel(configFile);
            Config c = om.readValue(configFile, modelType.getConfigClass());
            distributedContextLoader.ifPresent(loader -> c.setDistributedContext(loader.apply(c)));

            c.setWorkingDirectory(workingDirectory);

            Tokenizer t = modelType.getTokenizerClass().getConstructor(Path.class).newInstance(baseDir.toPath());
            WeightLoader wl = weightLoaderSupplier.apply(baseDir);

            return modelType.getModelClass().getConstructor(
                AbstractModel.InferenceType.class,
                Config.class,
                WeightLoader.class,
                Tokenizer.class,
                DType.class,
                DType.class,
                Optional.class
            ).newInstance(inferenceType, c, wl, t, workingMemoryType, workingQuantizationType, modelQuantization);

        } catch (IOException | NoSuchMethodException | InvocationTargetException | InstantiationException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Interface to define the type of model, including its configuration, tokenizer, and model class.
     * This is used to ensure that the correct classes are used for different model types.
     */
    public interface ModelType {
        Class<? extends AbstractModel> getModelClass();
        Class<? extends Config> getConfigClass();
        Class<? extends Tokenizer> getTokenizerClass();
    }
}
