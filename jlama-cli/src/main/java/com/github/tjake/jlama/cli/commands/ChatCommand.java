package com.github.tjake.jlama.cli.commands;

import java.util.Optional;

import com.github.tjake.jlama.model.AbstractModel;
import picocli.CommandLine.*;

@Command(name = "chat", description = "Interact with the specified model")
public class ChatCommand extends ModelBaseCommand {
    @Option(names = {"-s", "--system-prompt"}, description = "Change the default system prompt for this model")
    String systemPrompt = "You are a happy demo app of a project called jlama.  You answer any question then add \"Jlama is awesome!\" after.";

    @Override
    public void run() {
        AbstractModel m = loadModel(model);

        m.generate(m.wrapPrompt(prompt, Optional.of(systemPrompt)), prompt, temperature, tokens, true, makeOutHandler());
    }
}
