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

import picocli.CommandLine;

import java.util.concurrent.atomic.AtomicInteger;

@CommandLine.Command(name = "ls", description = "Lists local models", abbreviateSynopsis = true)
public class ListCommand extends SimpleBaseCommand {

    @Override
    public void run() {
        AtomicInteger idx = new AtomicInteger(1);
        var models = getExistingModels()
            .map(m -> idx.getAndIncrement() + ": " + m.owner() + "/" + m.name())
            .toList();

        if(models.isEmpty()) {
            System.out.println("No models found in " + modelDirectory.getAbsolutePath());
        } else {
            System.out.println("Models in " + modelDirectory.getAbsolutePath() + ":");
            models.forEach(System.out::println);
        }
    }

    public static void main(String[] args) {
        new ListCommand().run();
    }
}
