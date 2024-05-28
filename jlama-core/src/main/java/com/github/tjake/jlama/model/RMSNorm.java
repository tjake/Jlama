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

public class RMSNorm extends LayerNorm {
    private final float weightAdjustment;

    public RMSNorm(AbstractModel m, AbstractTensor weights) {
        this(m, weights, 0.0f);
    }

    public RMSNorm(AbstractModel m, AbstractTensor weights, float weightAdjustment) {
        super(m, null, weights);
        this.weightAdjustment = weightAdjustment;
    }

    @Override
    public AbstractTensor forward(
            AbstractTensor input,
            int offset,
            int length,
            Optional<BiFunction<Float, Float, Pair<Float, Float>>> reducer) {

        int batchSize = input.shape().first();
        AbstractTensor output = input.copyShape();

        int limit = offset + length;
        for (int b = 0; b < batchSize; b++) {
            float ss = 0.0f;
            for (int j = offset; j < limit; j++) {
                float v = input.get(b, j);
                ss += v * v;
            }

            if (reducer.isPresent())
            {
                Pair<Float, Float> p = reducer.get().apply(ss, 0f);
                ss = p.left;
            }

            ss /= m.c.embeddingLength;
            ss += m.c.layerNormEps;
            ss = (float) (1.0 / StrictMath.sqrt(ss));
            // normalize and scale
            for (int j = offset; j < limit; j++) {
                output.set((weightAdjustment + weights.get(0, j)) * (ss * input.get(b, j)), b, j);
            }
        }
        return output;
    }
}
