package com.github.tjake.jlama.model.bert;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.github.tjake.jlama.safetensors.Config;

public class BertConfig extends Config {
    @JsonCreator
    public BertConfig(  @JsonProperty("max_position_embeddings") int contextLength,
                        @JsonProperty("hidden_size") int embeddingLength,
                        @JsonProperty("intermediate_size") int hiddenLength,
                        @JsonProperty("num_attention_heads") int numberOfHeads,
                        @JsonProperty("num_hidden_layers") int numberOfLayers,
                        @JsonProperty("layer_norm_eps") float layerNormEps,
                        @JsonProperty("vocab_size") int vocabularySize) {
        super(contextLength, embeddingLength, hiddenLength, numberOfHeads, numberOfHeads, numberOfLayers, layerNormEps, vocabularySize, 0, 0, null, null);
    }
}
