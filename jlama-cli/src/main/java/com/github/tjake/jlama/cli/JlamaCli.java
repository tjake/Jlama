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

import ch.qos.logback.classic.LoggerContext;
import ch.qos.logback.classic.encoder.PatternLayoutEncoder;
import ch.qos.logback.core.ConsoleAppender;
import com.github.tjake.jlama.cli.commands.*;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;
import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;
import picocli.CommandLine.*;

@Command(name = "jlama", mixinStandardHelpOptions = true, requiredOptionMarker = '*', usageHelpAutoWidth = true, sortOptions = true,
        description = "\nJlama is a modern LLM inference engine for Java!\n\nQuantized models are maintained at https://hf.co/tjake\n",
        defaultValueProvider = PropertiesDefaultProvider.class)
public class JlamaCli implements Runnable {
    static {
        setupLogging();
    }

    @Option(names = { "-h", "--help" }, usageHelp = true, hidden = true)
    boolean helpRequested = false;

    public static void main(String[] args) {
        CommandLine cli = new CommandLine(new JlamaCli());
        cli.addSubcommand("download", new DownloadCommand());
        cli.addSubcommand("quantize", new QuantizeCommand());
        cli.addSubcommand("chat", new ChatCommand());
        cli.addSubcommand("complete", new CompleteCommand());
        cli.addSubcommand("restapi", new ApiServiceCommand());
        cli.addSubcommand("cluster-coordinator", new ClusterCoordinatorCommand());
        cli.addSubcommand("cluster-worker", new ClusterWorkerCommand());

        cli.setUsageHelpLongOptionsMaxWidth(256);
        cli.setUsageHelpAutoWidth(true);

        String[] pargs = args.length == 0 ? new String[] { "-h" } : args;
        cli.parseWithHandler(new RunLast(), pargs);
    }

    @Override
    public void run() {}

    private static void setupLogging() {
        Logger root = (Logger) LoggerFactory.getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME);
        LoggerContext logCtx = root.getLoggerContext();

        logCtx.reset();

        PatternLayoutEncoder logEncoder = new PatternLayoutEncoder();
        logEncoder.setContext(logCtx);
        logEncoder.setPattern("%msg%n");
        logEncoder.start();

        ConsoleAppender logConsoleAppender = new ConsoleAppender();
        logConsoleAppender.setContext(logCtx);
        logConsoleAppender.setName("console");
        logConsoleAppender.setEncoder(logEncoder);
        logConsoleAppender.start();

        root.addAppender(logConsoleAppender);
        root.setAdditive(false);
        root.setLevel(Level.INFO);
    }
}
