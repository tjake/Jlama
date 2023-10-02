package com.github.tjake.jlama.cli.commands;

import java.io.File;

import com.github.tjake.jlama.cli.JlamaCli;
import picocli.CommandLine;

public class BaseCommand extends JlamaCli {
    @CommandLine.Parameters(index = "0", arity = "1", description = "The model location")
    protected File model;
}
