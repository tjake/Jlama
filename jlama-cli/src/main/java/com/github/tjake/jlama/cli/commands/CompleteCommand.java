package com.github.tjake.jlama.cli.commands;

import com.github.tjake.jlama.model.AbstractModel;
import picocli.CommandLine.*;

import java.util.Optional;
import java.util.UUID;

import static com.github.tjake.jlama.model.ModelSupport.loadModel;

@Command(name = "complete", description = "Completes a prompt using the specified model", mixinStandardHelpOptions = true)
public class CompleteCommand extends ModelBaseCommand {

    @Override
    public void run() {
        AbstractModel m = loadModel(model, workingDirectory, workingMemoryType, workingQuantizationType, Optional.ofNullable(modelQuantization), Optional.ofNullable(threadCount));
        m.generate(UUID.randomUUID(), prompt, temperature, tokens, false, makeOutHandler());
    }
}
