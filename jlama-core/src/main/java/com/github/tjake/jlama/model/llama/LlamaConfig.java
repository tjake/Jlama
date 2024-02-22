package com.github.tjake.jlama.model.llama;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.github.tjake.jlama.math.ActivationFunction;
import com.github.tjake.jlama.safetensors.Config;

import java.util.Map;

public class LlamaConfig extends Config {

    @JsonCreator
    public LlamaConfig( @JsonProperty("max_position_embeddings") int contextLength,
                        @JsonProperty("hidden_size") int embeddingLength,
                        @JsonProperty("intermediate_size") int hiddenLength,
                        @JsonProperty("num_attention_heads") int numberOfHeads,
                        @JsonProperty("num_key_value_heads") int numberOfKeyValueHeads,
                        @JsonProperty("num_hidden_layers") int numberOfLayers,
                        @JsonProperty("rms_norm_eps") float layerNormEps,
                        @JsonProperty("vocab_size") int vocabularySize,
                        @JsonProperty("bos_token_id") int bosToken,
                        @JsonProperty("eos_token_id") int eosToken,
                        @JsonProperty("hidden_act") ActivationFunction.Type activationFunction,
                        @JsonProperty("rope_theta") Double ropeFreqsTheta,
                        @JsonProperty("rope_scaling") Map<String,String> ropeScaling) {
        super(contextLength, embeddingLength, hiddenLength, numberOfHeads, numberOfKeyValueHeads, numberOfLayers, layerNormEps, vocabularySize, bosToken, eosToken, activationFunction,
                ropeFreqsTheta == null ? 10000.0 : ropeFreqsTheta,
                ropeScaling == null ? 1.0 : Double.parseDouble(ropeScaling.get("factor")));
    }
}
