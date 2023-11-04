package com.github.tjake.jlama.cli.commands;


import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;

import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.SafeTensorSupport;
import picocli.CommandLine;

@CommandLine.Command(name = "quantize", description = "Quantize the specified model")

public class QuantizeCommand extends BaseCommand {

    @CommandLine.Parameters(index = "1", arity = "0..1", description = "The output location")
    protected Path output;

    @CommandLine.Option(names = { "-q", "--quantization"}, description = "Model quantization type", arity = "1")
    protected DType modelQuantization;

    @CommandLine.Option(names = {"-s", "--skip-layer"}, description = "Layer name prefix to not quantize")
    protected String[] skipLayerPrefixes;

    @Override
    public void run() {

        if (!model.exists()) {
            System.err.println("Model location does not exist: " + model);
            System.exit(1);
        }

        File baseDir = model.isFile() ? model.getParentFile() : model;

        if (!baseDir.isDirectory()) {
            System.err.println("Model directory does not exist: " + baseDir);
            System.exit(1);
        }

        try {
            Path out = SafeTensorSupport.quantizeModel(baseDir.toPath(), modelQuantization, skipLayerPrefixes, Optional.ofNullable(output));

            System.out.println("Quantized model written to: " + out);
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(2);
        }
    }
}
