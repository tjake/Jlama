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

import static com.github.tjake.jlama.cli.commands.SimpleBaseCommand.getName;
import static com.github.tjake.jlama.cli.commands.SimpleBaseCommand.getOwner;

@CommandLine.Command(name = "download", description = "Downloads a HuggingFace model - use owner/name format")
public class DownloadCommand extends JlamaCli {
    @CommandLine.Option(names = { "-d", "--model-directory" }, description = "The local model directory for all models (default: ${DEFAULT-VALUE})", defaultValue = "models")
    protected File modelDirectory = new File("models");

    @CommandLine.Option(names = { "-b", "--branch" }, description = "The branch to download from (default: ${DEFAULT-VALUE})", defaultValue = "main")
    protected String branch = "main";

    @CommandLine.Option(names = { "-t", "--auth-token" }, description = "The auth token to use for downloading the model (if required)")
    protected String authToken = null;

    @CommandLine.Parameters(index = "0", arity = "1", description = "The huggingface model owner/name pair")
    protected String modelName;

    @Override
    public void run() {
        String owner = getOwner(modelName);
        String name = getName(modelName);

        SimpleBaseCommand.downloadModel(owner, name, modelDirectory, branch, authToken, true);
    }
}
