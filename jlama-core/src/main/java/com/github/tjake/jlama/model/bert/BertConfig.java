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
package com.github.tjake.jlama.model.bert;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.github.tjake.jlama.math.ActivationFunction;
import com.github.tjake.jlama.safetensors.Config;
import java.util.List;
import java.util.Map;

public class BertConfig extends Config {
    @JsonCreator
    public BertConfig(
        @JsonProperty("max_position_embeddings") int contextLength,
        @JsonProperty("hidden_size") int embeddingLength,
        @JsonProperty("intermediate_size") int hiddenLength,
        @JsonProperty("num_attention_heads") int numberOfHeads,
        @JsonProperty("num_hidden_layers") int numberOfLayers,
        @JsonProperty("layer_norm_eps") float layerNormEps,
        @JsonProperty("hidden_act") ActivationFunction.Type activationFunction,
        @JsonProperty("vocab_size") int vocabularySize,
        @JsonProperty("label2id") Map<String, Integer> classificationLabels,
        @JsonProperty("sep_token") Integer sepToken,
        @JsonProperty("cls_token") Integer clsToken
    ) {
        super(
            contextLength,
            embeddingLength,
            hiddenLength,
            numberOfHeads,
            numberOfHeads,
            numberOfLayers,
            layerNormEps,
            vocabularySize,
            sepToken == null ? 0 : sepToken,
            clsToken == null ? List.of(0) : List.of(clsToken),
            activationFunction,
            null,
            null,
            classificationLabels
        );
    }
}
