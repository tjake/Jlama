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
import com.github.tjake.jlama.safetensors.tokenizer.PromptSupport;
import java.io.PrintWriter;
import java.util.Optional;
import java.util.Scanner;
import java.util.UUID;
import java.util.function.BiConsumer;
import picocli.CommandLine.*;

@Command(name = "chat", description = "Interact with the specified model")
public class ChatCommand extends BaseCommand {
    private static final AnsiFormat chatText = new AnsiFormat(Attribute.CYAN_TEXT());
    private static final AnsiFormat statsColor = new AnsiFormat(Attribute.BLUE_TEXT());

    @Option(
            names = {"-s", "--system-prompt"},
            description = "Change the default system prompt for this model")
    String systemPrompt = null;

    @Option(
            names = {"-t", "--temperature"},
            description = "Temperature of response [0,1] (default: ${DEFAULT-VALUE})",
            defaultValue = "0.6")
    protected Float temperature;

    @Option(
            names = {"--top-p"},
            description =
                    "Controls how many different words the model considers per token [0,1] (default: ${DEFAULT-VALUE})",
            defaultValue = ".9")
    protected Float topp;

    @Override
    public void run() {
        AbstractModel m = loadModel(
                model,
                workingDirectory,
                workingMemoryType,
                workingQuantizationType,
                Optional.ofNullable(modelQuantization),
                Optional.ofNullable(threadCount));

        if (m.promptSupport().isEmpty()) {
            System.err.println("This model does not support chat prompting");
            System.exit(1);
        }

        UUID session = UUID.randomUUID();
        PromptSupport promptSupport = m.promptSupport().get();
        PrintWriter out = System.console().writer();

        out.println("Chatting with " + model + "...\n");
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

            PromptSupport.Builder builder = promptSupport.newBuilder();
            if (first && systemPrompt != null) {
                builder.addSystemMessage(systemPrompt);
            }
            builder.addUserMessage(prompt);
            String builtPrompt = builder.build();

            m.generate(session, builtPrompt, temperature, Integer.MAX_VALUE, false, makeOutHandler());

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
