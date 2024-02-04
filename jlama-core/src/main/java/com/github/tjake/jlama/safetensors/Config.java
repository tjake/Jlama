package com.github.tjake.jlama.safetensors;

import com.github.tjake.jlama.math.VectorMath;
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
    public final int numberOfKeyValueHeads;
    public final int headSize;
    public final int headGroupSize;
    public final int kvLength;
    public final boolean isGQA;
    protected final int numberOfLayers;
    public final float layerNormEps;
    public final int vocabularySize;
    public final int bosToken;
    public final int eosToken;

    public final Optional<float[][]> ropeFreqs;
    public final Optional<float[][]> groupRopeFreqs;

    private volatile Optional<Pair<Integer, Integer>> offset;
    private volatile File workingDirectory;

    //Suppliers to store values that chance when offset is adjusted
    private volatile int embeddingSegmentStart;
    private volatile int embeddingSegmentLength;
    private volatile int embeddingSegmentEnd;
    private volatile int kvSegmentStart;
    private volatile int kvSegmentLength;
    private volatile int kvSegmentEnd;
    private volatile int headStart;
    private volatile int headEnd;
    private volatile int groupHeadStart;
    private volatile int groupHeadEnd;

    public final TensorCache tensorCache;

    public Config(int contextLength,
                  int embeddingLength,
                  int hiddenLength,
                  int numberOfHeads,
                  int numberOfKeyValueHeads,
                  int numberOfLayers,
                  float layerNormEps,
                  int vocabularySize,
                  int bosToken,
                  int eosToken,
                  Double ropeFreqsTheta) {
        this.contextLength = contextLength;
        this.embeddingLength = embeddingLength;
        this.hiddenLength = hiddenLength;
        this.numberOfHeads = numberOfHeads;
        this.numberOfKeyValueHeads = numberOfKeyValueHeads;
        this.numberOfLayers = numberOfLayers;
        this.layerNormEps = layerNormEps;
        this.vocabularySize = vocabularySize;
        this.bosToken = bosToken;
        this.eosToken = eosToken;
        this.tensorCache = TensorCache.instance;
        this.headSize = embeddingLength / numberOfHeads;
        this.headGroupSize = numberOfHeads / numberOfKeyValueHeads;
        this.kvLength = numberOfKeyValueHeads * headSize;
        this.isGQA = numberOfKeyValueHeads < numberOfHeads;
        if (ropeFreqsTheta != null) {
            this.ropeFreqs = Optional.of(VectorMath.precomputeFreqsCis(embeddingLength / numberOfHeads, contextLength, ropeFreqsTheta));
            this.groupRopeFreqs = Optional.of(VectorMath.precomputeFreqsCis(embeddingLength / numberOfKeyValueHeads, contextLength, ropeFreqsTheta));
        } else {
            this.ropeFreqs = Optional.empty();
            this.groupRopeFreqs = Optional.empty();
        }
        setOffset(null);
    }

    public void setOffset(Pair<Integer, Integer> offset) {
        this.offset = Optional.ofNullable(offset);

        this.embeddingSegmentStart = this.offset.map(Pair::left).orElse(0);
        this.embeddingSegmentLength = this.offset.map(Pair::right).orElse(embeddingLength);
        this.embeddingSegmentEnd = embeddingSegmentStart + embeddingSegmentLength;
        this.kvSegmentStart = embeddingSegmentStart / headGroupSize;
        this.kvSegmentEnd = embeddingSegmentEnd / headGroupSize;
        this.kvSegmentLength = embeddingSegmentLength / headGroupSize;
        this.headStart = embeddingSegmentStart / headSize;
        this.headEnd = embeddingSegmentEnd / headSize;
        this.groupHeadStart = kvSegmentStart / headSize;
        this.groupHeadEnd = kvSegmentEnd / headSize;
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

    public int kvSegmentStart() {
        return kvSegmentStart;
    }

    public int kvSegmentLength() {
        return kvSegmentLength;
    }

    public int headStart() {
        return headStart;
    }

    public int headEnd() {
        return headEnd;
    }

    public int maybeMapToGroupHead(int head) {
        int i = (int) Math.floor((double) head / headGroupSize);
        //System.out.println("i: " + i + " head: " + head);
        return i;
    }

    public int groupHeadStart() {
        return groupHeadStart;
    }

    public int groupHeadEnd() {
        return groupHeadEnd;
    }
}
