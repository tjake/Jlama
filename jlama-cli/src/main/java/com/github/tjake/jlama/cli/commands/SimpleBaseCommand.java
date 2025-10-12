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
import java.io.File;
import java.io.IOException;
import java.net.URLEncoder;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Stream;

import com.github.tjake.jlama.model.ModelSupport;
import com.github.tjake.jlama.safetensors.SafeTensorSupport;
import com.github.tjake.jlama.util.ProgressReporter;
import com.google.common.util.concurrent.Uninterruptibles;
import me.tongfei.progressbar.ProgressBar;
import me.tongfei.progressbar.ProgressBarBuilder;
import me.tongfei.progressbar.ProgressBarStyle;
import picocli.CommandLine;

public class SimpleBaseCommand extends JlamaCli {
    static AtomicReference<ProgressBar> progressRef = new AtomicReference<>();

    @CommandLine.ArgGroup(exclusive = false, heading = "Download Options:%n")
    protected DownloadSection downloadSection = new DownloadSection();

    @CommandLine.Option(names = {
        "--model-cache" }, paramLabel = "ARG", description = "The local directory for downloaded models (default: ${DEFAULT-VALUE})")
    protected File modelDirectory = new File(JlamaCli.DEFAULT_MODEL_DIRECTORY);

    @CommandLine.Parameters(index = "0", arity = "0", paramLabel = "<model name>", description = "The huggingface model owner/name pair")
    protected String modelName;

    static class DownloadSection {
        @CommandLine.Option(names = {
            "--auto-download" }, paramLabel = "ARG", description = "Download the model if missing (default: ${DEFAULT-VALUE})", defaultValue = "false")
        Boolean autoDownload = false;

        @CommandLine.Option(names = {
            "--branch" }, paramLabel = "ARG", description = "The model branch to download from (default: ${DEFAULT-VALUE})", defaultValue = "main")
        String branch = "main";

        @CommandLine.Option(names = { "--auth-token" }, paramLabel = "ARG", description = "HuggingFace auth token (for restricted models)")
        String authToken = null;

        @CommandLine.Option(names = {
            "--sequential-download" }, description = "Use sequential download instead of parallel when auto-downloading (default: parallel)")
        Boolean useSequential = false;
    }

    protected record ModelId(String owner, String name) {
        protected ModelId {
            if (owner == null || owner.isEmpty() || name == null || name.isEmpty()) {
                throw new IllegalArgumentException("ModelId owner and name cannot be null or empty");
            }
        }

        protected String fullName() {
            return owner + "/" + name;
        }
    }

    ModelId requireModelId() throws IllegalArgumentException {
        return tryResolveModelId().orElseGet(() -> {
            StringBuilder sb = new StringBuilder();
            sb.append("Invalid model name: ").append(modelName).append("\n");
            sb.append("Must be in the format owner/name or the index of a listed model\n");
            sb.append("Available models:\n");
            AtomicReference<Integer> idx = new AtomicReference<>(1);
            getExistingModels()
                .map(m -> idx.getAndSet(idx.get() + 1) + ": " + m.owner() + "/" + m.name())
                .forEach(line -> sb.append("  ").append(line).append("\n"));
            System.out.println(sb);
            System.exit(1);
            throw new IllegalArgumentException(sb.toString());
        });
    }

    Optional<ModelId> tryResolveModelId() {
        try {
            int modelIndex = Integer.parseInt(modelName);
            List<ModelId> models = getExistingModels().toList();
            if (modelIndex >= 1 && modelIndex <= models.size()) {
                return Optional.of(models.get(modelIndex - 1));
            } else {
                return Optional.empty();
            }
        } catch (IllegalArgumentException e) {
            // Not an integer, continue
        }
        String[] parts = modelName.split("/");
        if (parts.length != 2 || parts[0].isEmpty() || parts[1].isEmpty()) {
            return Optional.empty();
        }
        return Optional.of(new ModelId(parts[0], parts[1]));
    }

    static Optional<ProgressReporter> getProgressConsumer() {
        if (System.console() == null) return Optional.empty();

        return Optional.of((ProgressReporter) (filename, sizeDownloaded, totalSize) -> {
            if (progressRef.get() == null || !progressRef.get().getTaskName().equals(filename)) {
                ProgressBarBuilder builder = new ProgressBarBuilder().setTaskName(filename)
                    .setInitialMax(totalSize)
                    .setStyle(ProgressBarStyle.ASCII);

                if (totalSize > 1000000) {
                    builder.setUnit("MB", 1000000);
                } else if (totalSize > 1000) {
                    builder.setUnit("KB", 1000);
                } else {
                    builder.setUnit("B", 1);
                }

                progressRef.set(builder.build());
            }

            progressRef.get().stepTo(sizeDownloaded);
            Uninterruptibles.sleepUninterruptibly(150, TimeUnit.MILLISECONDS);
        });
    }

    static void downloadModel(
        String owner,
        String name,
        File modelDirectory,
        String branch,
        String authToken,
        boolean downloadWeights,
        boolean useSequential
    ) {
        try {
            SafeTensorSupport.maybeDownloadModel(
                modelDirectory.getAbsolutePath(),
                Optional.ofNullable(owner),
                name,
                downloadWeights,
                Optional.ofNullable(URLEncoder.encode(branch, "UTF-8")),
                Optional.ofNullable(authToken),
                getProgressConsumer(),
                useSequential
            );
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }
    }

    protected Path getModel(ModelId modelId, File modelDirectory, boolean autoDownload, String branch, String authToken) {
        return getModel(modelId, modelDirectory, autoDownload, branch, authToken, true, false);
    }

    protected Path getModel(
        ModelId modelId,
        File modelDirectory,
        boolean autoDownload,
        String branch,
        String authToken,
        boolean downloadWeights,
        boolean useSequential
    ) {

        Path modelPath = SafeTensorSupport.constructLocalModelPath(modelDirectory.getAbsolutePath(), modelId.owner,
            modelId.name);

        if (autoDownload) {
            downloadModel(modelId.owner, modelId.name, modelDirectory, branch, authToken, downloadWeights, useSequential);
        } else if (!modelPath.toFile().exists()) {
            System.err.println("Model not found: " + modelPath);
            System.err.println("Use --auto-download to download the model");
            System.exit(1);
        }

        return modelPath;
    }

    protected Stream<ModelId> getExistingModels() {
        if (!modelDirectory.exists()) {
            System.out.println("No models found in " + modelDirectory.getAbsolutePath());
            System.exit(0);
        }
        File[] files = modelDirectory.listFiles();
        if (files == null || files.length == 0) {
            return Stream.empty();
        }
        return Arrays.stream(files)
            .filter(File::isDirectory)
            .map(file -> file.getName().split("_"))
            .filter(parts -> parts.length == 2)
            .map(parts -> {
                File baseDir = new File(modelDirectory, parts[0] + "_" + parts[1]);
                File configFile = null;
                for (File f : Objects.requireNonNull(baseDir.listFiles())) {
                    if (f.getName().equals("config.json")) {
                        configFile = f;
                        break;
                    }
                }
                if (configFile != null) {
                    try {
                        ModelSupport.ModelType modelType = SafeTensorSupport.detectModel(configFile);
                        if (modelType != null) {
                            return new ModelId(parts[0], parts[1]);
                        }
                    } catch (IOException | IllegalArgumentException e) {
                        // ignore Unknown model type
                    }
                }
                return null;
            })
            .filter(Objects::nonNull)
            .distinct()
            .sorted(Comparator.comparing(ModelId::fullName));

    }
}
