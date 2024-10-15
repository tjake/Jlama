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

import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;
import picocli.CommandLine.*;

public class ModelBaseCommand extends BaseCommand {

    @Option(names = {
        "--temperature" }, paramLabel = "ARG", description = "Temperature of response [0,1] (default: ${DEFAULT-VALUE})", defaultValue = "0.6")
    protected Float temperature;

    @Option(names = {
        "--tokens" }, paramLabel = "ARG", description = "Number of tokens to generate (default: ${DEFAULT-VALUE})", defaultValue = "128000")
    protected Integer tokens;

    @Option(names = {
        "--top-p" }, paramLabel = "ARG", description = "Controls how many different words the model considers per token [0,1] (default: ${DEFAULT-VALUE})", defaultValue = ".9")
    protected Float topp;

    protected BiConsumer<String, Float> makeOutHandler() {
        PrintWriter out;
        Charset utf8 = Charset.forName("UTF-8");
        BiConsumer<String, Float> outCallback;
        if (System.console() == null) {
            AtomicInteger i = new AtomicInteger(0);
            StringBuilder b = new StringBuilder();
            out = new PrintWriter(System.out, true, utf8);
            outCallback = (w, t) -> {
                b.append(w);
                out.println(String.format("%d: %s [took %.2fms])", i.getAndIncrement(), b, t));
                out.flush();
            };
        } else {
            out = System.console().writer();
            outCallback = (w, t) -> {
                out.print(w);
                out.flush();
            };
        }

        return outCallback;
    }
}
