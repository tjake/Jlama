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
import picocli.CommandLine;

import static com.github.tjake.jlama.cli.commands.SimpleBaseCommand.getName;
import static com.github.tjake.jlama.cli.commands.SimpleBaseCommand.getOwner;

@CommandLine.Command(name = "download", description = "Downloads a HuggingFace model - use owner/name format", abbreviateSynopsis = true)
public class DownloadCommand extends JlamaCli {
    @CommandLine.Option(names = {
        "--model-cache" }, paramLabel = "ARG", description = "The local directory for all downloaded models (default: ${DEFAULT-VALUE})")
    protected File modelDirectory = new File(JlamaCli.DEFAULT_MODEL_DIRECTORY);

    @CommandLine.Option(names = {
        "--branch" }, paramLabel = "ARG", description = "The branch to download from (default: ${DEFAULT-VALUE})", defaultValue = "main")
    protected String branch = "main";

    @CommandLine.Option(names = {
        "--auth-token" }, paramLabel = "ARG", description = "The auth token to use for downloading the model (if required)")
    protected String authToken = null;

    @CommandLine.Parameters(index = "0", arity = "1", paramLabel = "<model name>", description = "The huggingface model owner/name pair")
    protected String modelName;

    @Override
    public void run() {
        String owner = getOwner(modelName);
        String name = getName(modelName);

        SimpleBaseCommand.downloadModel(owner, name, modelDirectory, branch, authToken, true);
    }
}
