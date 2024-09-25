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
package com.github.tjake.jlama.cli.commands;

import com.github.tjake.jlama.safetensors.DType;
import java.io.File;
import picocli.CommandLine;

public class BaseCommand extends SimpleBaseCommand {
    @CommandLine.Option(names = { "--work-directory" }, paramLabel = "ARG", description = "Working directory for attention cache")
    protected File workingDirectory = null;

    @CommandLine.ArgGroup(exclusive = false, heading = "Advanced Options:%n")
    protected AdvancedSection advancedSection = new AdvancedSection();

    static class AdvancedSection {
        @CommandLine.Option(names = {
            "--working-dtype" }, paramLabel = "ARG", description = "Working memory data type (default: ${DEFAULT-VALUE})", defaultValue = "F32")
        protected DType workingMemoryType = DType.F32;

        @CommandLine.Option(names = {
            "--working-qtype" }, paramLabel = "ARG", description = "Working memory quantization data type (default: ${DEFAULT-VALUE})", defaultValue = "I8")
        protected DType workingQuantizationType = DType.I8;

        @CommandLine.Option(names = {
            "--threads" }, paramLabel = "ARG", description = "Number of threads to use (default: number of physical cores)")
        protected Integer threadCount = null;

        @CommandLine.Option(names = { "--quantize-to" }, paramLabel = "ARG", description = "Runtime Model quantization type")
        protected DType modelQuantization;
    }
}
