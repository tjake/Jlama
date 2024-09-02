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
package com.github.tjake.jlama.model;

import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.util.Pair;
import com.google.common.base.Preconditions;
import java.util.Optional;
import java.util.function.BiFunction;

public class LayerNorm {

    protected final AbstractModel m;
    private final AbstractTensor bias;
    protected final AbstractTensor weights;

    public LayerNorm(AbstractModel m, AbstractTensor bias, AbstractTensor weights) {
        this.m = m;
        this.bias = bias;
        this.weights = weights;
    }

    public AbstractTensor forward(AbstractTensor input) {
        return forward(input, Optional.empty());
    }

    public AbstractTensor forward(AbstractTensor input, Optional<BiFunction<Float, Float, Pair<Float, Float>>> reducer) {
        Preconditions.checkArgument(input.shape().dims() == 2);
        int size = input.shape().last();
        Preconditions.checkArgument(size == m.c.embeddingLength);
        return forward(input, m.c.embeddingSegmentStart(), m.c.embeddingSegmentLength(), reducer);
    }

    public AbstractTensor forward(
        AbstractTensor input,
        int offset,
        int length,
        Optional<BiFunction<Float, Float, Pair<Float, Float>>> reducer
    ) {

        int batchSize = input.shape().first();
        AbstractTensor output = input.copyShape();

        for (int b = 0; b < batchSize; b++) {
            float sum = 0;
            float sumSq = 0;
            int limit = offset + length;
            for (int i = offset; i < limit; i++) {
                float v = input.get(b, i);
                sum += v;
                sumSq += v * v;
            }

            if (reducer.isPresent()) {
                Pair<Float, Float> p = reducer.get().apply(sumSq, sum);
                sumSq = p.left;
                sum = p.right;
            }

            float mean = sum / m.c.embeddingLength;
            float variance = sumSq / m.c.embeddingLength - mean * mean;
            float invStddev = 1.0f / (float) Math.sqrt(variance + m.c.layerNormEps);

            for (int i = offset; i < limit; i++) {
                float v = (input.get(b, i) - mean) * invStddev * weights.get(0, i) + bias.get(0, i);
                output.set(v, b, i);
            }
        }

        return output;
    }
}
