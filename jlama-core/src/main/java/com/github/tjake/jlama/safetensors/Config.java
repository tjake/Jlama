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
import com.github.tjake.jlama.model.DistributedContext;
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
    public final int attentionLength;
    public final int hiddenLength;
    public final int numberOfHeads;
    public final int numberOfKeyValueHeads;
    public final int headSize;
    public final ActivationFunction.Type activationFunction;
    public final int headGroupSize;
    public final int kvLength;
    public final boolean isGQA;
    public final int numberOfLayers;
    public final float layerNormEps;
    public final int vocabularySize;
    public final int bosToken;
    public final List<Integer> eosTokens;
    public final Optional<float[][]> ropeFreqs;
    private volatile DistributedContext dctx;
    private volatile File workingDirectory;


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
        Double ropeScalingFactor
    ) {
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
            embeddingLength / numberOfHeads
        );
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
        Integer headSize
    ) {
        this.contextLength = contextLength;
        this.attentionLength = numberOfHeads * headSize;
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
            : Optional.of(
                VectorMath.precomputeFreqsCis(headSize, contextLength, ropeFreqsTheta, ropeScalingFactor == null ? 1.0 : ropeScalingFactor)
            );

        // Set default values
        this.dctx = DistributedContext.builder(this).build();
    }

    public void setDistributedContext(DistributedContext dctx) {
        this.dctx = dctx;
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

    public DistributedContext dctx() {
        return dctx;
    }

    public int maybeMapToGroupHead(int head) {
        if (!isGQA) return head;
        return Math.floorDiv(head, headGroupSize);
    }
}
