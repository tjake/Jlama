package com.github.tjake.jlama.cli.commands;

import com.github.tjake.jlama.cli.JlamaCli;
import picocli.CommandLine;

import java.io.File;

public class SimpleBaseCommand extends JlamaCli {
    @CommandLine.Parameters(index = "0", arity = "1", description = "The model location")
    protected File model;

}
