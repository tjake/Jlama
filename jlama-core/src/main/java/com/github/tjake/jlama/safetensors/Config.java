package com.github.tjake.jlama.safetensors;

import com.github.tjake.jlama.tensor.TensorCache;
import com.github.tjake.jlama.util.Pair;

import java.io.File;
import java.util.Optional;

import com.google.common.base.Preconditions;
import com.google.common.io.Files;

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

    private volatile Optional<Pair<Integer, Integer>> offset;
    private volatile File workingDirectory;

    //Suppliers to store values that chance when offset is adjusted
    private volatile int embeddingSegmentStart;
    private volatile int embeddingSegmentLength;
    private volatile int embeddingSegmentEnd;
    private volatile int headStart;
    private volatile int headEnd;

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
        setOffset(null);
    }

    public void setOffset(Pair<Integer, Integer> offset) {
        this.offset = Optional.ofNullable(offset);

        this.embeddingSegmentStart = this.offset.map(Pair::left).orElse(0);
        this.embeddingSegmentLength = this.offset.map(Pair::right).orElse(embeddingLength);
        this.embeddingSegmentEnd = embeddingSegmentStart + embeddingSegmentLength;
        this.headStart = embeddingSegmentStart / headSize;
        this.headEnd = embeddingSegmentEnd / headSize;
    }

    public void setWorkingDirectory(File workingDirectory) {
        if (workingDirectory == null) {
            this.workingDirectory = Files.createTempDir();
            this.workingDirectory.deleteOnExit();
        } else {
            Preconditions.checkArgument(workingDirectory.isDirectory());
            this.workingDirectory = workingDirectory;
        }
    }

    public Optional<File> workingDirectory() {
        return Optional.ofNullable(this.workingDirectory);
    }

    public Optional<Pair<Integer, Integer>> offset() {
        return offset;
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
        return embeddingSegmentStart;
    }

    public int embeddingSegmentLength() {
        return embeddingSegmentLength;
    }

    public int headStart() {
        return headStart;
    }

    public int headEnd() {
        return headEnd;
    }
}
