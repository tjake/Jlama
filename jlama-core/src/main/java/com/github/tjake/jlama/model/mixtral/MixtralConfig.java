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
package com.github.tjake.jlama.model.mixtral;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.github.tjake.jlama.math.ActivationFunction;
import com.github.tjake.jlama.safetensors.Config;
import java.util.List;

public class MixtralConfig extends Config {

    public final int numberOfExperts;

    public final int numberOfExpertsPerToken;

    @JsonCreator
    public MixtralConfig(
        @JsonProperty("max_position_embeddings") int contextLength,
        @JsonProperty("hidden_size") int embeddingLength,
        @JsonProperty("intermediate_size") int hiddenLength,
        @JsonProperty("num_attention_heads") int numberOfHeads,
        @JsonProperty("num_key_value_heads") int numberOfKeyValueHeads,
        @JsonProperty("num_hidden_layers") int numberOfLayers,
        @JsonProperty("rms_norm_eps") float layerNormEps,
        @JsonProperty("vocab_size") int vocabularySize,
        @JsonProperty("bos_token_id") int bosToken,
        @JsonProperty("eos_token_id") int eosToken,
        @JsonProperty("rope_theta") Double ropeTheta,
        @JsonProperty("hidden_act") ActivationFunction.Type activationFunction,
        @JsonProperty("num_local_experts") int numberOfExperts,
        @JsonProperty("num_experts_per_tok") int numberOfExpertsPerToken
    ) {
        super(
            contextLength,
            embeddingLength,
            hiddenLength,
            numberOfHeads,
            numberOfKeyValueHeads,
            numberOfLayers,
            layerNormEps,
            vocabularySize,
            bosToken,
            List.of(eosToken),
            activationFunction,
            ropeTheta,
            1.0
        );

        this.numberOfExperts = numberOfExperts;
        this.numberOfExpertsPerToken = numberOfExpertsPerToken;
    }
}
