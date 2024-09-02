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
package com.github.tjake.jlama.model.gpt2;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.github.tjake.jlama.math.ActivationFunction;
import com.github.tjake.jlama.safetensors.Config;
import java.util.List;

public class GPT2Config extends Config {

    @JsonCreator
    public GPT2Config(
        @JsonProperty("n_ctx") int contextLength,
        @JsonProperty("n_embd") int embeddingLength,
        @JsonProperty("n_head") int numberOfHeads,
        @JsonProperty("n_layer") int numberOfLayers,
        @JsonProperty("layer_norm_epsilon") float layerNormEps,
        @JsonProperty("vocab_size") int vocabularySize,
        @JsonProperty("bos_token_id") int bosToken,
        @JsonProperty("eos_token_id") int eosToken
    ) {
        super(
            contextLength,
            embeddingLength,
            embeddingLength * 4,
            numberOfHeads,
            numberOfHeads,
            numberOfLayers,
            layerNormEps,
            vocabularySize,
            bosToken,
            List.of(eosToken),
            ActivationFunction.Type.GELU,
            null,
            null
        );
    }
}
