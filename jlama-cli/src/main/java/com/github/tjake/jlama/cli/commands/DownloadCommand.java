package com.github.tjake.jlama.cli.commands;

import com.github.tjake.jlama.cli.JlamaCli;
import com.github.tjake.jlama.safetensors.SafeTensorSupport;
import com.google.common.util.concurrent.Uninterruptibles;
import me.tongfei.progressbar.ProgressBar;
import me.tongfei.progressbar.ProgressBarBuilder;
import me.tongfei.progressbar.ProgressBarStyle;
import picocli.CommandLine;

import java.io.File;
import java.io.IOException;
import java.util.Optional;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

@CommandLine.Command(name = "download", description = "Downloads the specified model")
public class DownloadCommand extends JlamaCli {
    @CommandLine.Option(names={"-d", "--model-directory"}, description = "The directory to download the model to (default: ${DEFAULT-VALUE})", defaultValue = "models")
    protected File modelDirectory = new File("models");

    @CommandLine.Option(names={"-t", "--auth-token"}, description = "The auth token to use for downloading the model (if required)")
    protected String authToken = null;


    @CommandLine.Parameters(index = "0", arity = "1", description = "The model owner/name pair to download")
    protected String model;

    @Override
    public void run() {
        AtomicReference<ProgressBar> progressRef = new AtomicReference<>();

        String[] parts = model.split("/");
        if (parts.length == 0 || parts.length > 2) {
            System.err.println("Model must be in the form owner/name");
            System.exit(1);
        }

        String owner;
        String name;

        if (parts.length == 1) {
            owner = null;
            name = model;
        } else {
            owner = parts[0];
            name = parts[1];
        }


        try {
            SafeTensorSupport.maybeDownloadModel(modelDirectory.getAbsolutePath(), Optional.ofNullable(owner), name, Optional.ofNullable(authToken), Optional.of((n,c,t) -> {
                if (progressRef.get() == null || !progressRef.get().getTaskName().equals(n)) {
                    ProgressBarBuilder builder = new ProgressBarBuilder()
                            .setTaskName(n)
                            .setInitialMax(t)
                            .setStyle(ProgressBarStyle.ASCII);

                    if (t > 1000000) {
                        builder.setUnit("MB", 1000000);
                    } else if (t > 1000) {
                        builder.setUnit("KB", 1000);
                    } else {
                        builder.setUnit("B", 1);
                    }

                    progressRef.set(builder.build());
                }

                progressRef.get().stepTo(c);
                Uninterruptibles.sleepUninterruptibly(150, TimeUnit.MILLISECONDS);
            }));
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
        }
    }
}
