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

import com.github.tjake.jlama.safetensors.SafeTensorSupport;
import picocli.CommandLine;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Files;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.FileVisitResult;
import java.nio.file.attribute.BasicFileAttributes;

@CommandLine.Command(name = "rm", description = "Removes local model", abbreviateSynopsis = true)
public class RemoveCommand extends SimpleBaseCommand {

    @Override
    public void run() {
        ModelId modelId = requireModelId();

        Path modelPath = SafeTensorSupport.constructLocalModelPath(modelDirectory.getAbsolutePath(),
            modelId.owner(), modelId.name());

        if (!modelPath.toFile().exists()) {
            System.err.println("Model not found: " + modelPath);
            System.exit(1);
        }

        try {
            Files.walkFileTree(modelPath, new SimpleFileVisitor<>() {
                @Override
                public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                    Files.delete(file);
                    return FileVisitResult.CONTINUE;
                }

                @Override
                public FileVisitResult postVisitDirectory(Path modelPath, IOException exc) throws IOException {
                    Files.delete(modelPath);
                    return FileVisitResult.CONTINUE;
                }
            });

            System.out.println("Model successfully removed: " + modelPath);
        } catch (IOException e) {
            System.err.println("Failed to delete model: " + e.getMessage());
            System.exit(1);
        }
    }
}
