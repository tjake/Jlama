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

import com.diogonunes.jcolor.AnsiFormat;
import com.diogonunes.jcolor.Attribute;
import com.github.tjake.jlama.model.AbstractModel;
import com.github.tjake.jlama.model.functions.Generator;
import com.github.tjake.jlama.safetensors.prompt.PromptContext;
import com.github.tjake.jlama.safetensors.prompt.PromptSupport;

import java.io.PrintWriter;
import java.nio.file.Path;
import java.util.Optional;
import java.util.Scanner;
import java.util.UUID;
import java.util.function.BiConsumer;

import picocli.CommandLine.*;

@Command(name = "chat", description = "Interact with the specified model", abbreviateSynopsis = true)
public class ChatCommand extends ModelBaseCommand {
    private static final AnsiFormat chatText = new AnsiFormat(Attribute.CYAN_TEXT());
    private static final AnsiFormat statsColor = new AnsiFormat(Attribute.BLUE_TEXT());

    @Option(names = { "--system-prompt" }, paramLabel = "ARG", description = "Change the default system prompt for this model")
    String systemPrompt = null;

    @Override
    public void run() {
        Path modelPath = SimpleBaseCommand.getModel(
            modelName,
            modelDirectory,
            downloadSection.autoDownload,
            downloadSection.branch,
            downloadSection.authToken
        );
        AbstractModel m = loadModel(
            modelPath.toFile(),
            workingDirectory,
            advancedSection.workingMemoryType,
            advancedSection.workingQuantizationType,
            Optional.ofNullable(advancedSection.modelQuantization),
            Optional.ofNullable(advancedSection.threadCount)
        );

        if (m.promptSupport().isEmpty()) {
            System.err.println("This model does not support chat prompting");
            System.exit(1);
        }

        UUID session = UUID.randomUUID();
        PromptSupport promptSupport = m.promptSupport().get();
        PromptSupport.Builder builder = promptSupport.builder();
        PrintWriter out = System.console().writer();

        out.println("\nChatting with " + modelName + "...\n");
        out.flush();
        Scanner sc = new Scanner(System.in);
        boolean first = true;
        while (true) {
            out.print("\nYou: ");
            out.flush();
            String prompt = sc.nextLine();
            out.println();
            out.flush();
            if (prompt.isEmpty()) {
                break;
            }

            if (first && systemPrompt != null) {
                builder.addSystemMessage(systemPrompt);
            }
            builder.addUserMessage(prompt);
            PromptContext builtPrompt = builder.build();

            Generator.Response r = m.generate(
                session,
                builtPrompt,
                temperature,
                tokens == null ? m.getConfig().contextLength : tokens,
                makeOutHandler()
            );

            // New prompt builder and strip out the preamble since we're continuing the conversation
            builder = promptSupport.builder().stripPreamble();

            out.println(
                "\n\n"
                    + statsColor.format(
                        Math.round(r.promptTokens / (double) (r.promptTimeMs / 1000f))
                            + " tokens/s (prompt), "
                            + Math.round(r.generatedTokens / (double) (r.generateTimeMs / 1000f))
                            + " tokens/s (gen)"
                    )
            );

            first = false;
        }
    }

    protected BiConsumer<String, Float> makeOutHandler() {
        PrintWriter out;
        BiConsumer<String, Float> outCallback;

        out = System.console().writer();
        out.print(chatText.format("Jlama: "));
        out.flush();
        outCallback = (w, t) -> {
            out.print(chatText.format(w));
            out.flush();
        };

        return outCallback;
    }
}
