package com.github.tjake.jlama.safetensors.tokenizer;

import java.util.Map;

import com.google.common.base.Preconditions;
import com.google.common.collect.BiMap;
import com.google.common.collect.ImmutableBiMap;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

public class TokenizerModel
{
    @JsonProperty("type")
    public final String type;
    @JsonProperty("unk_token")
    public final String unkToken;
    @JsonProperty("fuse_unk")
    public final boolean fuseUnk;
    @JsonProperty("byte_fallback")
    public final boolean byteFallback;
    @JsonProperty("vocab")
    public final BiMap<String, Long> vocabLookup;

    @JsonCreator
    public TokenizerModel(
            @JsonProperty("type") String type,
            @JsonProperty("unk_token") String unkToken,
            @JsonProperty("fuse_unk") boolean fuseUnk,
            @JsonProperty("byte_fallback") boolean byteFallback,
            @JsonProperty("vocab") Map<String, Long> vocabLookup) {
        this.type = type;
        this.unkToken = unkToken;
        this.fuseUnk = fuseUnk;
        this.byteFallback = byteFallback;
        this.vocabLookup = ImmutableBiMap.copyOf(vocabLookup);
   }
}
