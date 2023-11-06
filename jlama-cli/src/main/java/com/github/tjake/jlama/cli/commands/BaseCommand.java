package com.github.tjake.jlama.cli.commands;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Path;
import java.util.Objects;
import java.util.Optional;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.tjake.jlama.cli.JlamaCli;
import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.model.ModelSupport;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.SafeTensorSupport;
import com.github.tjake.jlama.safetensors.WeightLoader;
import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;
import com.github.tjake.jlama.util.PhysicalCoreExecutor;
import picocli.CommandLine;

public class BaseCommand extends SimpleBaseCommand {

    private static final ObjectMapper om = new ObjectMapper()
            .configure(DeserializationFeature.FAIL_ON_IGNORED_PROPERTIES, false)
            .configure(DeserializationFeature.FAIL_ON_TRAILING_TOKENS, false)
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
            .configure(DeserializationFeature.FAIL_ON_MISSING_CREATOR_PROPERTIES, true);


    @CommandLine.Option(names={"-wm", "--working-dtype"}, description = "Working memory data type")
    protected DType workingMemoryType = DType.F32;

    @CommandLine.Option(names={"-wq", "--working-qtype"}, description = "Working memory quantization data type")
    protected DType workingQuantizationType = DType.I8;

    @CommandLine.Option(names={"-tc", "--threads"}, description = "Number of threads to use")
    protected int threadCount = Runtime.getRuntime().availableProcessors() / 2;

    @CommandLine.Option(names={"-q", "--quantization"}, description = "Model quantization type")
    protected DType modelQuantization;


    protected AbstractModel loadModel(File model) {

        if (!model.exists()) {
            System.err.println("Model location does not exist: " + model);
            System.exit(1);
        }

        File baseDir = model.isFile() ? model.getParentFile() : model;

        //Find config
        if (!baseDir.isDirectory()) {
            System.err.println("Model directory does not exist: " + baseDir);
            System.exit(1);
        }

        File configFile = null;
        for (File f : Objects.requireNonNull(baseDir.listFiles())) {
            if (f.getName().equals("config.json")) {
                configFile = f;
                break;
            }
        }

        if (configFile == null) {
            System.err.println("config.json in model directory does not exist: " + baseDir);
            System.exit(1);
        }

        try {
            PhysicalCoreExecutor.overrideThreadCount(threadCount);

            ModelSupport.ModelType modelType = SafeTensorSupport.detectModel(configFile);

            Config c = om.readValue(configFile, modelType.configClass);
            Tokenizer t = modelType.tokenizerClass.getConstructor(Path.class).newInstance(baseDir.toPath());
            WeightLoader wl = SafeTensorSupport.loadWeights(baseDir);

            return modelType.modelClass.getConstructor(Config.class, WeightLoader.class, Tokenizer.class, DType.class, DType.class, Optional.class)
                    .newInstance(c, wl, t, workingMemoryType, workingQuantizationType, Optional.ofNullable(modelQuantization));

        } catch (IOException | NoSuchMethodException | InvocationTargetException | InstantiationException |
                 IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }
}
