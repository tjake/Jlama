package com.github.tjake.jlama.cli;

import com.github.tjake.jlama.cli.commands.*;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;
import picocli.CommandLine;
import picocli.CommandLine.*;


@Command(name="jlama", mixinStandardHelpOptions = true, requiredOptionMarker = '*', usageHelpAutoWidth = true, sortOptions = true)
public class JlamaCli implements Runnable {

    static {
        System.setProperty("jdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK", "0");
        TensorOperationsProvider.get();
    }

    @Option(names = { "-h", "--help" }, usageHelp = true, hidden = true)
    boolean helpRequested = false;

    public static void main(String[] args) {
        CommandLine cli = new CommandLine(new JlamaCli());
        cli.addSubcommand("download", new DownloadCommand());
        cli.addSubcommand("quantize", new QuantizeCommand());
        cli.addSubcommand("chat", new ChatCommand());
        cli.addSubcommand("complete", new CompleteCommand());
        cli.addSubcommand("serve", new ServeCommand());
        cli.addSubcommand("cluster-coordinator", new ClusterCoordinatorCommand());
        cli.addSubcommand("cluster-worker", new ClusterWorkerCommand());

        cli.setUsageHelpLongOptionsMaxWidth(256);

        String[] pargs = args.length == 0 ? new String[]{"-h"} : args;
        cli.parseWithHandler(new RunLast(), pargs);
    }

    @Override
    public void run() {

    }
}
