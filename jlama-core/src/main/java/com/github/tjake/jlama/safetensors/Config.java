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
package com.github.tjake.jlama.safetensors;

import com.github.tjake.jlama.math.ActivationFunction;
import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.tensor.TensorCache;
import com.github.tjake.jlama.util.Pair;
import com.google.common.base.Preconditions;
import com.google.common.io.Files;
import java.io.File;
import java.util.List;
import java.util.Optional;

public class Config {
    public final int contextLength;
    public final int embeddingLength;
    public final int hiddenLength;
    public final int numberOfHeads;
    public final int numberOfKeyValueHeads;
    public final int headSize;
    public final ActivationFunction.Type activationFunction;
    public final int headGroupSize;
    public final int kvLength;
    public final boolean isGQA;
    protected final int numberOfLayers;
    public final float layerNormEps;
    public final int vocabularySize;
    public final int bosToken;
    public final List<Integer> eosTokens;
    public final Optional<float[][]> ropeFreqs;
    private volatile Optional<Pair<Integer, Integer>> offset;
    private volatile File workingDirectory;

    // Suppliers to store values that chance when offset is adjusted
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


    public Config(
            int contextLength,
            int embeddingLength,
            int hiddenLength,
            int numberOfHeads,
            int numberOfKeyValueHeads,
            int numberOfLayers,
            float layerNormEps,
            int vocabularySize,
            int bosToken,
            List<Integer> eosToken,
            ActivationFunction.Type activationFunction,
            Double ropeFreqsTheta,
            Double ropeScalingFactor) {
        this(
                contextLength,
                embeddingLength,
                hiddenLength,
                numberOfHeads,
                numberOfKeyValueHeads,
                numberOfLayers,
                layerNormEps,
                vocabularySize,
                bosToken,
                eosToken,
                activationFunction,
                ropeFreqsTheta,
                ropeScalingFactor,
                embeddingLength / numberOfHeads);
    }

    public Config(
            int contextLength,
            int embeddingLength,
            int hiddenLength,
            int numberOfHeads,
            int numberOfKeyValueHeads,
            int numberOfLayers,
            float layerNormEps,
            int vocabularySize,
            int bosToken,
            List<Integer> eosTokens,
            ActivationFunction.Type activationFunction,
            Double ropeFreqsTheta,
            Double ropeScalingFactor,
            Integer headSize) {
        this.contextLength = contextLength;
        this.embeddingLength = embeddingLength;
        this.hiddenLength = hiddenLength;
        this.numberOfHeads = numberOfHeads;
        this.numberOfKeyValueHeads = numberOfKeyValueHeads;
        this.numberOfLayers = numberOfLayers;
        this.layerNormEps = layerNormEps;
        this.vocabularySize = vocabularySize;
        this.bosToken = bosToken;
        this.eosTokens = eosTokens;
        this.tensorCache = TensorCache.instance;
        this.headSize = headSize;
        this.headGroupSize = numberOfHeads / numberOfKeyValueHeads;
        this.kvLength = numberOfKeyValueHeads * headSize;
        this.isGQA = numberOfKeyValueHeads < numberOfHeads;
        this.activationFunction = activationFunction;
        this.ropeFreqs = ropeFreqsTheta == null
                ? Optional.empty()
                : Optional.of(VectorMath.precomputeFreqsCis(
                        headSize,
                        contextLength,
                        ropeFreqsTheta,
                        ropeScalingFactor == null ? 1.0 : ropeScalingFactor));

        // Set default values
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
        if (!isGQA) return head;
        return Math.floorDiv(head, headGroupSize);
    }

    public int groupHeadStart() {
        return groupHeadStart;
    }

    public int groupHeadEnd() {
        return groupHeadEnd;
    }
}
