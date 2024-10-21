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
package com.github.tjake.jlama.cli.commands;

import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.SafeTensorSupport;
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Optional;
import picocli.CommandLine;

@CommandLine.Command(name = "quantize", description = "Quantize the specified model", abbreviateSynopsis = true)
public class QuantizeCommand extends SimpleBaseCommand {

    @CommandLine.Parameters(index = "1", arity = "0..1", description = "The output location")
    protected Path output;

    @CommandLine.Option(names = {
        "--quantization" }, paramLabel = "ARG", description = "Model quantization type (default: ${DEFAULT-VALUE})", arity = "1", defaultValue = "Q4")
    protected DType modelQuantization = DType.Q4;

    @CommandLine.Option(names = {
        "--skip-layer" }, paramLabel = "ARG", description = "Layer name prefix to not quantize (default: ${DEFAULT-VALUE})", defaultValue = "norm")
    protected String[] skipLayerPrefixes;

    @CommandLine.Option(names = { "--drop-layer" }, paramLabel = "ARG", description = "Layer name prefix to drop")
    protected String[] dropLayerPrefixes;

    @Override
    public void run() {

        Path modelPath = SimpleBaseCommand.getModel(
            modelName,
            modelDirectory,
            downloadSection.autoDownload,
            downloadSection.branch,
            downloadSection.authToken
        );
        File model = modelPath.toFile();

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
            Path out = SafeTensorSupport.quantizeModel(
                baseDir.toPath(),
                modelQuantization,
                skipLayerPrefixes,
                dropLayerPrefixes,
                Optional.ofNullable(output)
            );

            System.out.println("Quantized model written to: " + out);
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(2);
        }
    }
}
