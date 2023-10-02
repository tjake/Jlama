package com.github.tjake.jlama.cli.commands;

import com.github.tjake.jlama.model.AbstractModel;
import picocli.CommandLine.*;

@Command(name = "complete", description = "Completes a prompt using the specified model", mixinStandardHelpOptions = true)
public class CompleteCommand extends ModelBaseCommand {

    @Override
    public void run() {
        AbstractModel m = loadModel(model);
        m.generate(prompt, temperature, tokens, false, makeOutHandler());
    }
}
