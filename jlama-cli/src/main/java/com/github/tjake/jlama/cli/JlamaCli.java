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
package com.github.tjake.jlama.cli;

import com.github.tjake.jlama.cli.commands.*;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;
import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;
import picocli.CommandLine.*;

@Command(name = "jlama", mixinStandardHelpOptions = true, requiredOptionMarker = '*',
        usageHelpAutoWidth = true, sortOptions = true,
        description = "Jlama is a modern LLM inference engine for Java!\n\n" +
                "Quantized models are maintained at https://hf.co/tjake\n")
public class JlamaCli implements Runnable {
    static {
        System.setProperty("jdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK", "0");
        TensorOperationsProvider.get();
    }

    @Option(names = { "-h", "--help" }, usageHelp = true, hidden = true)
    boolean helpRequested = false;

    public static void main(String[] args) {
        Logger root = (Logger) LoggerFactory.getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME);
        root.setLevel(Level.INFO);

        CommandLine cli = new CommandLine(new JlamaCli());
        cli.addSubcommand("download", new DownloadCommand());
        cli.addSubcommand("quantize", new QuantizeCommand());
        cli.addSubcommand("chat", new ChatCommand());
        cli.addSubcommand("complete", new CompleteCommand());
        cli.addSubcommand("restapi", new ApiServiceCommand());
        cli.addSubcommand("cluster-coordinator", new ClusterCoordinatorCommand());
        cli.addSubcommand("cluster-worker", new ClusterWorkerCommand());

        cli.setUsageHelpLongOptionsMaxWidth(256);

        String[] pargs = args.length == 0 ? new String[] { "-h" } : args;
        cli.parseWithHandler(new RunLast(), pargs);
    }

    @Override
    public void run() {}
}
