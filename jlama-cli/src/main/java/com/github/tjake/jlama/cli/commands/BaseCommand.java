package com.github.tjake.jlama.cli.commands;

import java.io.File;

import com.github.tjake.jlama.safetensors.DType;
import picocli.CommandLine;

public class BaseCommand extends SimpleBaseCommand {
    @CommandLine.Option(names={"-d", "--working-directory"}, description = "Working directory for attention cache")
    protected File workingDirectory = null;

    @CommandLine.Option(names={"-wm", "--working-dtype"}, description = "Working memory data type (default: ${DEFAULT-VALUE})", defaultValue = "F32")
    protected DType workingMemoryType = DType.F32;

    @CommandLine.Option(names={"-wq", "--working-qtype"}, description = "Working memory quantization data type (default: ${DEFAULT-VALUE})", defaultValue = "I8")
    protected DType workingQuantizationType = DType.I8;

    @CommandLine.Option(names={"-tc", "--threads"}, description = "Number of threads to use (default: number of cores)")
    protected Integer threadCount = null;

    @CommandLine.Option(names={"-q", "--quantization"}, description = "Model quantization type")
    protected DType modelQuantization;
}
