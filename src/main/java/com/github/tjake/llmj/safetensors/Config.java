package com.github.tjake.llmj.safetensors;

import com.github.tjake.llmj.model.FloatBufferTensor;

public class Config {
    public final int contextLength;
    public final int embeddingLength;
    public final int hiddenLength;
    public final int numberOfHeads;
    public final int numberOfLayers;
    public final float layerNormEps;
    public final int vocabularySize;
    public final int bosToken;
    public final int eosToken;

    public final FloatBufferTensor.BufferCache bufferCache;

    public Config(int contextLength,
                  int embeddingLength,
                  int hiddenLength,
                  int numberOfHeads,
                  int numberOfLayers,
                  float layerNormEps,
                  int vocabularySize,
                  int bosToken,
                  int eosToken) {
        this.contextLength = contextLength;
        this.embeddingLength = embeddingLength;
        this.hiddenLength = hiddenLength;
        this.numberOfHeads = numberOfHeads;
        this.numberOfLayers = numberOfLayers;
        this.layerNormEps = layerNormEps;
        this.vocabularySize = vocabularySize;
        this.bosToken = bosToken;
        this.eosToken = eosToken;
        this.bufferCache = new FloatBufferTensor.BufferCache(100 * 1024 * 1024);
    }
}
