package com.github.tjake.llmj.model.llama;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.github.tjake.llmj.safetensors.Config;

public class LlamaConfig extends Config {

    @JsonCreator
    public LlamaConfig( @JsonProperty("max_position_embeddings") int embeddingLength,
                        @JsonProperty("intermediate_size") int hiddenLength,
                        @JsonProperty("num_attention_heads") int numberOfHeads,
                        @JsonProperty("num_hidden_layers") int numberOfLayers,
                        @JsonProperty("rms_norm_eps") float layerNormEps,
                        @JsonProperty("vocab_size") int vocabularySize) {
        super(2048, embeddingLength, hiddenLength, numberOfHeads, numberOfLayers, layerNormEps, vocabularySize);
    }
}
