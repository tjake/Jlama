package com.github.tjake.jlama.model.llama;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.github.tjake.jlama.safetensors.Config;

public class LlamaConfig extends Config {

    @JsonCreator
    public LlamaConfig( @JsonProperty("hidden_size") int embeddingLength,
                        @JsonProperty("intermediate_size") int hiddenLength,
                        @JsonProperty("num_attention_heads") int numberOfHeads,
                        @JsonProperty("num_key_value_heads") int numberOfKeyValueHeads,
                        @JsonProperty("num_hidden_layers") int numberOfLayers,
                        @JsonProperty("rms_norm_eps") float layerNormEps,
                        @JsonProperty("vocab_size") int vocabularySize,
                        @JsonProperty("bos_token_id") int bosToken,
                        @JsonProperty("eos_token_id") int eosToken) {
        super(2048, embeddingLength, hiddenLength, numberOfHeads, numberOfKeyValueHeads, numberOfLayers, layerNormEps, vocabularySize, bosToken, eosToken, 10000.0);
    }
}
