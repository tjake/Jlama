package com.github.tjake.jlama.cli.commands;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;

import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.safetensors.DType;
import picocli.CommandLine.*;

public class ModelBaseCommand extends BaseCommand {

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
        try {
            return AbstractModel.load(model, threadCount, workingMemoryType, workingQuantizationType);
        } catch (FileNotFoundException e) {
            System.err.println(e.getMessage());
            System.exit(1);
        }
        throw new IllegalStateException("Should have exited");
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
