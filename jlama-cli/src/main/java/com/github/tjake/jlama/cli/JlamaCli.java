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
import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;
import picocli.CommandLine.*;

import java.util.*;
import java.io.File;

import static java.util.Arrays.asList;
import static picocli.CommandLine.Help.Column.Overflow.*;
import static picocli.CommandLine.Model.UsageMessageSpec.*;

@Command(name = "jlama", sortOptions = false, headerHeading = "Usage:%n", synopsisHeading = "%n", descriptionHeading = "%nDescription:%n%n", parameterListHeading = "%nParameters:%n", optionListHeading = "%nCommand Options:%n", mixinStandardHelpOptions = true, usageHelpAutoWidth = true, requiredOptionMarker = '*', description = "Jlama is a modern LLM inference engine for Java!\nQuantized models are maintained at https://hf.co/tjake\n\nChoose from the available commands:", defaultValueProvider = PropertiesDefaultProvider.class)
public class JlamaCli implements Runnable {
    public static final String DEFAULT_MODEL_DIRECTORY = setupDefaultModelDirectory();

    static {
        setupLogging();
    }

    @Option(names = { "-h", "--help" }, usageHelp = true, hidden = true)
    boolean helpRequested = false;

    public static void main(String[] args) {
        CommandLine cli = new CommandLine(new JlamaCli());
        cli.addSubcommand("chat", new ChatCommand());
        cli.addSubcommand("restapi", new ApiServiceCommand());
        cli.addSubcommand("complete", new CompleteCommand());
        cli.addSubcommand("list", new ListCommand());
        cli.addSubcommand("download", new DownloadCommand());
        cli.addSubcommand("quantize", new QuantizeCommand());
        cli.addSubcommand("cluster-coordinator", new ClusterCoordinatorCommand());
        cli.addSubcommand("cluster-worker", new ClusterWorkerCommand());
        cli.addSubcommand("rm", new RemoveCommand());
        cli.addSubcommand("version", new VersionCommand());

        cli.getHelpSectionMap().remove(SECTION_KEY_COMMAND_LIST_HEADING);
        cli.getHelpSectionMap().put(SECTION_KEY_COMMAND_LIST, getCommandRenderer());

        String[] pargs = args.length == 0 ? new String[] { "-h" } : args;
        cli.execute(pargs);
    }

    @Override
    public void run() {}

    /** Shamelessly stolen from jbang */
    public static CommandGroupRenderer getCommandRenderer() {
        Map<String, List<String>> sections = new LinkedHashMap<>();
        sections.put("Inference", asList("chat", "restapi", "complete"));
        sections.put("Distributed Inference", asList("cluster-coordinator", "cluster-worker"));
        sections.put("Other", asList("download", "list", "quantize", "rm", "version"));
        CommandGroupRenderer renderer = new CommandGroupRenderer(sections);
        return renderer;
    }

    public static class CommandGroupRenderer implements CommandLine.IHelpSectionRenderer {
        private final Map<String, List<String>> sections;

        public CommandGroupRenderer(Map<String, List<String>> sections) {
            this.sections = sections;
        }

        /**
         * validate all commands in Help is covered by section and each section command
         * exist in help.
         *
         * @param help
         */
        public void validate(CommandLine.Help help) {
            Set<String> cmds = new HashSet<>();
            sections.forEach((key, value) -> cmds.addAll(value));

            Set<String> actualcmds = new HashSet<>(help.subcommands().keySet());

            actualcmds.removeAll(cmds);

            cmds.removeAll(help.subcommands().keySet());

            if (cmds.size() > 0) {
                throw new IllegalStateException("Section help defined for non existent commands" + cmds);
            }

            if (actualcmds.size() > 0) {
                throw new IllegalStateException(("Commands found with no assigned section" + actualcmds));
            }

            sections.forEach((key, value) -> cmds.addAll(value));

        }

        // @Override
        public String render(CommandLine.Help help) {
            if (help.commandSpec().subcommands().isEmpty()) {
                return "";
            }

            StringBuilder result = new StringBuilder();

            sections.forEach((key, value) -> result.append(renderSection(key, value, help)));
            return result.toString();
        }

        private String renderSection(String sectionHeading, List<String> cmdNames, CommandLine.Help help) {
            Help.TextTable textTable = createTextTable(help);

            for (String name : cmdNames) {
                Model.CommandSpec sub = help.commandSpec().subcommands().get(name).getCommandSpec();

                // create comma-separated list of command name and aliases
                String names = sub.names().toString();
                names = names.substring(1, names.length() - 1); // remove leading '[' and trailing ']'

                // description may contain line separators; use Text::splitLines to handle this
                String description = description(sub.usageMessage());
                CommandLine.Help.Ansi.Text[] lines = help.colorScheme().text(description).splitLines();

                for (int i = 0; i < lines.length; i++) {
                    CommandLine.Help.Ansi.Text cmdNamesText = help.colorScheme().commandText(i == 0 ? names : "");
                    textTable.addRowValues(cmdNamesText, lines[i]);
                }
            }
            return help.createHeading("%n" + sectionHeading + ":%n") + textTable.toString();
        }

        private static Help.TextTable createTextTable(CommandLine.Help help) {
            Model.CommandSpec spec = help.commandSpec();
            // prepare layout: two columns
            // the left column overflows, the right column wraps if text is too long
            int commandLength = maxLength(spec.subcommands(), 37);
            Help.TextTable textTable = Help.TextTable.forColumns(
                help.colorScheme(),
                new CommandLine.Help.Column(commandLength + 2, 2, SPAN),
                new CommandLine.Help.Column(spec.usageMessage().width() - (commandLength + 2), 2, WRAP)
            );
            textTable.setAdjustLineBreaksForWideCJKCharacters(spec.usageMessage().adjustLineBreaksForWideCJKCharacters());
            return textTable;
        }

        private static int maxLength(Map<String, CommandLine> subcommands, int max) {
            int result = subcommands.values()
                .stream()
                .map(cmd -> cmd.getCommandSpec().names().toString().length() - 2)
                .max(Integer::compareTo)
                .get();
            return Math.min(max, result);
        }

        private String description(Model.UsageMessageSpec usageMessage) {
            if (usageMessage.header().length > 0) {
                return usageMessage.header()[0];
            }
            if (usageMessage.description().length > 0) {
                return usageMessage.description()[0];
            }
            return "";
        }
    }

    private static void setupLogging() {
        Logger root = (Logger) LoggerFactory.getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME);
        LoggerContext logCtx = root.getLoggerContext();

        logCtx.reset();

        PatternLayoutEncoder logEncoder = new PatternLayoutEncoder();
        logEncoder.setContext(logCtx);
        logEncoder.setPattern("%msg%n");
        logEncoder.start();

        ConsoleAppender<ILoggingEvent> logConsoleAppender = new ConsoleAppender<>();
        logConsoleAppender.setContext(logCtx);
        logConsoleAppender.setName("console");
        logConsoleAppender.setEncoder(logEncoder);
        logConsoleAppender.start();

        root.addAppender(logConsoleAppender);
        root.setAdditive(false);
        root.setLevel(Boolean.getBoolean("jlama.debug") ? Level.DEBUG : Level.INFO);
    }

    private static String setupDefaultModelDirectory() {
        String envHome = System.getenv("JLAMA_MODEL_HOME");
        String propHome = System.getProperty("jlama.model.home");
        String defaultHome = System.getProperty("user.home", "") + File.separator + ".jlama" + File.separator + "models";

        if (envHome != null && !envHome.isEmpty()) {
            return envHome;
        } else if (propHome != null && !propHome.isEmpty()) {
            return propHome;
        } else {
            return defaultHome;
        }
    }
}
