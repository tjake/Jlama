package com.github.tjake.jlama.safetensors;

import com.github.tjake.jlama.tensor.TensorCache;
import com.github.tjake.jlama.util.Pair;

import java.util.Optional;

public class Config {
    public final int contextLength;
    public final int embeddingLength;
    public final int hiddenLength;
    public final int numberOfHeads;
    public final int headSize;
    protected final int numberOfLayers;
    public final float layerNormEps;
    public final int vocabularySize;
    public final int bosToken;
    public final int eosToken;

    public volatile Optional<Pair<Integer, Integer>> offset;

    public final TensorCache tensorCache;

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
        this.tensorCache = TensorCache.instance;
        this.headSize = embeddingLength / numberOfHeads;
        this.offset = Optional.empty();
    }

    public void setOffset(Pair<Integer, Integer> offset) {
        this.offset = Optional.ofNullable(offset);
    }

    public int layerStart() {
        return 0;
    }

    public int layerEnd() {
        return numberOfLayers;
    }

    public int getNumberOfLayers() {
        return numberOfLayers;
    }

    public int embeddingSegmentStart() {
        return offset.map(Pair::left).orElse(0);
    }

    public int embeddingSegmentEnd() {
        return embeddingSegmentStart() + embeddingSegmentLength();
    }

    public int embeddingSegmentLength() {
        return offset.map(Pair::right).orElse(embeddingLength);
    }

    public int headStart() {
        return embeddingSegmentStart() / headSize;
    }

    public int headEnd() {
        return embeddingSegmentEnd() / headSize;
    }
}
