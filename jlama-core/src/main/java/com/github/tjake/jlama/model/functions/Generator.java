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
package com.github.tjake.jlama.model.functions;

import com.github.tjake.jlama.safetensors.tokenizer.Tokenizer;
import java.util.UUID;
import java.util.function.BiConsumer;

/**
 * Used to define a function that generates tokens from a prompt
 */
public interface Generator {
    void generate(
            UUID session,
            String prompt,
            float temperature,
            int ntokens,
            boolean useEOS,
            BiConsumer<String, Float> onTokenWithTimings);

    Tokenizer getTokenizer();
}
