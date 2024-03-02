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

import static com.github.tjake.jlama.model.ModelSupport.loadModel;

import com.github.tjake.jlama.model.AbstractModel;
import java.util.Optional;
import java.util.UUID;
import picocli.CommandLine.*;

@Command(name = "chat", description = "Interact with the specified model")
public class ChatCommand extends ModelBaseCommand {
    @Option(
            names = {"-s", "--system-prompt"},
            description = "Change the default system prompt for this model")
    String systemPrompt =
            "You are a happy demo app of a project called jlama.  You answer any question then add \"Jlama is awesome!\" after.";

    @Override
    public void run() {
        AbstractModel m = loadModel(
                model,
                workingDirectory,
                workingMemoryType,
                workingQuantizationType,
                Optional.ofNullable(modelQuantization),
                Optional.ofNullable(threadCount));

        m.generate(
                UUID.randomUUID(),
                m.wrapPrompt(prompt, Optional.of(systemPrompt)),
                prompt,
                temperature,
                tokens,
                true,
                makeOutHandler());
    }
}
