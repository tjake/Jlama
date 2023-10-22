package com.github.tjake.jlama.cli.commands;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Path;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.model.ModelSupport;
import com.github.tjake.jlama.safetensors.Config;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.SafeTensorSupport;
import com.github.tjake.jlama.safetensors.Tokenizer;
import com.github.tjake.jlama.safetensors.WeightLoader;
import com.github.tjake.jlama.util.PhysicalCoreExecutor;
import picocli.CommandLine.*;

public class ModelBaseCommand extends BaseCommand {

    private static final ObjectMapper om = new ObjectMapper()
            .configure(DeserializationFeature.FAIL_ON_IGNORED_PROPERTIES, false)
            .configure(DeserializationFeature.FAIL_ON_TRAILING_TOKENS, false)
            .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
            .configure(DeserializationFeature.FAIL_ON_MISSING_CREATOR_PROPERTIES, true);

    @Option(names = {"-p", "--prompt"}, description = "Text to complete", required = true)
    protected String prompt;

    @Option(names={"-t", "--temperature"}, description = "Temperature of response [0,1]", defaultValue = "0.6")
    protected Float temperature;

    @Option(names={"--top-p"}, description = "Controls how many different words the model considers per token [0,1]", defaultValue = ".9")
    protected Float topp;

    @Option(names={"-n", "--tokens"}, description = "Number of tokens to generate", defaultValue = "256")
    protected Integer tokens;

    @Option(names={"-q", "--quantization"}, description = "Model quantization type")
    protected DType modelQuantization;

    @Option(names={"-wm", "--working-dtype"}, description = "Working memory data type")
    protected DType workingMemoryType = DType.F32;

    @Option(names={"-wq", "--working-qtype"}, description = "Working memory quantization data type")
    protected DType workingQuantizationType = DType.I8;

    @Option(names={"-tc", "--threads"}, description = "Number of threads to use")
    protected int threadCount = Runtime.getRuntime().availableProcessors() / 2;


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

    protected BiConsumer<String, Float> makeOutHandler() {
        PrintWriter out;
        BiConsumer<String, Float> outCallback;
        if (System.console() == null) {
            AtomicInteger i = new AtomicInteger(0);
            StringBuilder b = new StringBuilder();
            out = new PrintWriter(System.out);
            outCallback = (w,t) ->  {
                b.append(w);
                out.println(String.format("%d: %s [took %.2fms])", i.getAndIncrement(), b, t));
                out.flush();
            };
        } else {
            out = System.console().writer();
            outCallback = (w,t) -> {
                out.print(w);
                out.flush();
            };
        }

        return outCallback;
    }
}
