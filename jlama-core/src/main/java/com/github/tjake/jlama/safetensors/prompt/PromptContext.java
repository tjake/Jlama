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
package com.github.tjake.jlama.safetensors.prompt;

import java.util.List;
import java.util.Optional;

public class PromptContext {
    private final String prompt;
    private final Optional<List<Tool>> optionalTools;

    public static PromptContext of(String prompt) {
        return new PromptContext(prompt);
    }

    PromptContext(String prompt, Optional<List<Tool>> optionalTools) {
        this.prompt = prompt;
        this.optionalTools = optionalTools;
    }

    PromptContext(String prompt) {
        this.prompt = prompt;
        this.optionalTools = Optional.empty();
    }

    public boolean hasTools() {
        return optionalTools.isPresent();
    }

    public Optional<List<Tool>> getTools() {
        return optionalTools;
    }

    public String getPrompt() {
        return prompt;
    }

    @Override
    public String toString() {
        return "PromptContext{" + "prompt='" + prompt + '\'' + ", optionalTools=" + optionalTools + '}';
    }
}
