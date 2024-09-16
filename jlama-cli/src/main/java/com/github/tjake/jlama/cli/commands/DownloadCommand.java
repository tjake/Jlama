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
import com.github.tjake.jlama.safetensors.SafeTensorSupport;
import com.google.common.util.concurrent.Uninterruptibles;
import java.io.File;
import java.io.IOException;
import java.net.URLEncoder;
import java.util.Optional;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import me.tongfei.progressbar.ProgressBar;
import me.tongfei.progressbar.ProgressBarBuilder;
import me.tongfei.progressbar.ProgressBarStyle;
import picocli.CommandLine;

@CommandLine.Command(name = "download", description = "Downloads a HuggingFace model - use owner/name format")
public class DownloadCommand extends JlamaCli {
    @CommandLine.Option(names = { "-d",
        "--model-directory" }, description = "The directory to download the model to (default: ${DEFAULT-VALUE})", defaultValue = "models")
    protected File modelDirectory = new File("models");

    @CommandLine.Option(names = { "-b",
        "--branch" }, description = "The branch to download from (default: ${DEFAULT-VALUE})", defaultValue = "main")
    protected String branch = "main";

    @CommandLine.Option(names = { "-t", "--auth-token" }, description = "The auth token to use for downloading the model (if required)")
    protected String authToken = null;

    @CommandLine.Parameters(index = "0", arity = "1", description = "The model owner/name pair to download")
    protected String model;

    @Override
    public void run() {
        AtomicReference<ProgressBar> progressRef = new AtomicReference<>();

        String[] parts = model.split("/");
        if (parts.length == 0 || parts.length > 2) {
            System.err.println("Model must be in the form owner/name");
            System.exit(1);
        }

        String owner;
        String name;

        if (parts.length == 1) {
            owner = null;
            name = model;
        } else {
            owner = parts[0];
            name = parts[1];
        }

        try {
            SafeTensorSupport.maybeDownloadModel(
                modelDirectory.getAbsolutePath(),
                Optional.ofNullable(owner),
                name,
                false,
                Optional.ofNullable(URLEncoder.encode(branch)),
                Optional.ofNullable(authToken),
                Optional.of((n, c, t) -> {
                    if (progressRef.get() == null || !progressRef.get().getTaskName().equals(n)) {
                        ProgressBarBuilder builder = new ProgressBarBuilder().setTaskName(n)
                            .setInitialMax(t)
                            .setStyle(ProgressBarStyle.ASCII);

                        if (t > 1000000) {
                            builder.setUnit("MB", 1000000);
                        } else if (t > 1000) {
                            builder.setUnit("KB", 1000);
                        } else {
                            builder.setUnit("B", 1);
                        }

                        progressRef.set(builder.build());
                    }

                    progressRef.get().stepTo(c);
                    Uninterruptibles.sleepUninterruptibly(150, TimeUnit.MILLISECONDS);
                })
            );
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }
    }
}
