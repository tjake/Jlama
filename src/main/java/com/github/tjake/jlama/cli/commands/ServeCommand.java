package com.github.tjake.jlama.cli.commands;

import com.github.tjake.jlama.cli.JlamaCli;
import picocli.CommandLine;

@CommandLine.Command(name = "serve", description = "Starts a rest api for interacting with this model")
public class ServeCommand extends JlamaCli {
}
