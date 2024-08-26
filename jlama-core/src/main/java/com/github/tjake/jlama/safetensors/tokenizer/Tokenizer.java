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
package com.github.tjake.jlama.safetensors.tokenizer;

import com.github.tjake.jlama.safetensors.prompt.PromptSupport;
import java.util.List;
import java.util.Optional;

/**
 * Tokenizer interface
 */
public interface Tokenizer {

    /**
     * Tokenize a sentence
     * @param sentence
     * @return list of token strings
     */
    List<String> tokenize(String sentence);

    /**
     * Encode a sentence into a list of token ids
     * @param sentence
     * @return list of token ids
     */
    long[] encode(String sentence);

    /**
     * Decode a token id into its string representation
     * @param id
     * @return token string
     */
    String decode(long id);

    /**
     * Decode a list of token ids into their string representation
     * @param ids list of token ids
     * @return list of token strings
     */
    String decode(long[] ids);

    /**
     * Get the prompt support for this tokenizer model if it exists
     * @return prompt support
     */
    Optional<PromptSupport> promptSupport();

    /**
     * Get the model for this tokenizer (expert mode)
     * @return tokenizer model
     */
    TokenizerModel getModel();
}
