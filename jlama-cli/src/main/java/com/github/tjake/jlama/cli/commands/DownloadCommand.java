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

import picocli.CommandLine;

@CommandLine.Command(name = "download", description = "Downloads a HuggingFace model - use owner/name format", abbreviateSynopsis = true)
public class DownloadCommand extends SimpleBaseCommand {

    @CommandLine.Option(names = { "--sequential" }, description = "Use sequential download instead of parallel (default: parallel)")
    protected boolean useSequential;

    @Override
    public void run() {
        ModelId modelId = requireModelId();
        downloadModel(modelId.owner(), modelId.name(), modelDirectory, downloadSection.branch, downloadSection.authToken, true, useSequential);
    }
}
