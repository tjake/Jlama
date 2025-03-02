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

import com.github.tjake.jlama.cli.JlamaCli;
import com.github.tjake.jlama.model.ModelSupport;
import com.github.tjake.jlama.safetensors.SafeTensorSupport;
import picocli.CommandLine;

import java.io.File;
import java.io.IOException;
import java.util.Objects;

@CommandLine.Command(name = "list", description = "Lists local models", abbreviateSynopsis = true)
public class ListCommand extends JlamaCli {
    @CommandLine.Option(names = {
        "--model-cache" }, paramLabel = "ARG", description = "The local directory for all downloaded models (default: ${DEFAULT-VALUE})")
    protected File modelDirectory = new File(JlamaCli.DEFAULT_MODEL_DIRECTORY);

    @Override
    public void run() {
        if (!modelDirectory.exists()) {
            System.out.println("No models found in " + modelDirectory.getAbsolutePath());
            System.exit(0);
        }

        File[] files = modelDirectory.listFiles();
        if (files == null || files.length == 0) {
            System.out.println("No models found in " + modelDirectory.getAbsolutePath());
            System.exit(0);
        }
        int idx = 1;
        for (File file : files) {
            if (file.isDirectory()) {
                String[] parts = file.getName().split("_");
                if (parts.length == 2) {

                    File baseDir = file;
                    File configFile = null;
                    for (File f : Objects.requireNonNull(baseDir.listFiles())) {
                        if (f.getName().equals("config.json")) {
                            configFile = f;
                            break;
                        }
                    }

                    if (configFile != null) {
                        ModelSupport.ModelType modelType = null;
                        try {
                            modelType = SafeTensorSupport.detectModel(configFile);
                        } catch (IOException | IllegalArgumentException e) {
                            // ignore Unknown model type
                        }
                        if (modelType != null) {
                            System.out.println(idx++ + ": " + parts[0] + "/" + parts[1]);
                        }
                    }
                }
            }
        }
    }
}
