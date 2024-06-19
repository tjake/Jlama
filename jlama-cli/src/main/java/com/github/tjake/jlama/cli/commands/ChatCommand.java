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

import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.util.Optional;
import java.util.Scanner;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;
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

    @Option(
            names = {"--override-prompt-start"},
            description = "Override of prompt instruction format before the prompt (example: [INST] )")
    protected String promptStart;

    @Option(
            names = {"--override-prompt-end"},
            description = "Override of prompt instruction format after the prompt (example: [/INST] )")
    protected String promptEnd;

    @Override
    public void run() {
        AbstractModel m = loadModel(
                model,
                workingDirectory,
                workingMemoryType,
                workingQuantizationType,
                Optional.ofNullable(modelQuantization),
                Optional.ofNullable(threadCount));

        UUID session = UUID.randomUUID();
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

            String wrappedPrompt = wrap(m, prompt, first ? Optional.ofNullable(systemPrompt) : Optional.empty());
            m.generate(
                    session,
                    wrappedPrompt,
                    prompt,
                    temperature,
                    Integer.MAX_VALUE,
                    true,
                    makeOutHandler());

            first = false;
        }
    }

    protected BiConsumer<String, Float> makeOutHandler()
    {
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

    protected String wrap(AbstractModel model, String prompt, Optional<String> systemPrompt) {
        if (promptStart == null && promptEnd == null) {
            return model.wrapPrompt(prompt, systemPrompt);
        }

        return promptStart + prompt + promptEnd;
    }

}
