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

import com.github.tjake.jlama.math.VectorMath;import com.github.tjake.jlama.safetensors.DType;import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.FloatBufferTensor;import com.github.tjake.jlama.tensor.TensorCache;import com.github.tjake.jlama.tensor.TensorShape;import com.github.tjake.jlama.util.Pair;
import com.google.common.base.Preconditions;import jdk.incubator.vector.FloatVector;import jdk.incubator.vector.VectorOperators;
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

    public AbstractTensor forward(
            AbstractTensor input, Optional<BiFunction<Float, Float, Pair<Float, Float>>> reducer) {
        Preconditions.checkArgument(input.shape().dims() == 2);
        int size = input.shape().last();
        Preconditions.checkArgument(size == m.c.embeddingLength);
        return forward(input, m.c.embeddingSegmentStart(), m.c.embeddingSegmentLength(), reducer);
    }

    public AbstractTensor forward(
            AbstractTensor input,
            int offset,
            int length,
            Optional<BiFunction<Float, Float, Pair<Float, Float>>> reducer) {

        int batchSize = input.shape().first();

        try (AbstractTensor sum = TensorCache.instance.get(DType.F32, TensorShape.of(batchSize));
                AbstractTensor sumSq = TensorCache.instance.get(DType.F32, TensorShape.of(batchSize))) {

            int limit = offset + length;
            int vlimit = offset + FloatVector.SPECIES_PREFERRED.loopBound(length);
            boolean useVector = vlimit > offset;

            for (int b = 0; b < batchSize; b++) {
                int i = offset;
                if (useVector) {
                    FloatVector vsum = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                    FloatVector vsumSq = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

                    for (; i < vlimit; i += FloatVector.SPECIES_PREFERRED.length()) {
                        FloatVector v = input.getVector(FloatVector.SPECIES_PREFERRED, b, i)
                                .reinterpretAsFloats();
                        vsum = vsum.add(v);
                        vsumSq = v.fma(v, vsumSq);
                    }

                    sum.set(vsum.reduceLanes(VectorOperators.ADD), 0, b);
                    sumSq.set(vsumSq.reduceLanes(VectorOperators.ADD), 0, b);
                }

                for (; i < limit; i++) {
                    float v = input.get(b, i);
                    sum.set(sum.get(0, b) + v, 0, b);
                    sumSq.set(sumSq.get(0, b) + v * v, 0, b);
                }
            }

            /*if (reducer.isPresent()) {
                Pair<Float, Float> p = reducer.get().apply(sumSq, sum);
                sumSq = p.left;
                sum = p.right;
            }*/

            AbstractTensor output = input.copyShape();

            for (int b = 0; b < batchSize; b++) {
                float mean = sum.get(0, b) / m.c.embeddingLength;
                float variance = sumSq.get(0, b) / m.c.embeddingLength - mean * mean;
                float invStddev = 1.0f / (float) Math.sqrt(variance + m.c.layerNormEps);

                int i = offset;

                if (useVector) {
                    FloatVector vmean = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, mean);
                    FloatVector vinvStddev = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, invStddev);

                    for (; i < vlimit; i += FloatVector.SPECIES_PREFERRED.length()) {
                        FloatVector v = input.getVector(FloatVector.SPECIES_PREFERRED, b, i).reinterpretAsFloats();
                        v = v.sub(vmean).mul(vinvStddev).mul(weights.getVector(FloatVector.SPECIES_PREFERRED, 0, i)).add(bias.getVector(FloatVector.SPECIES_PREFERRED, 0, i));
                        output.intoTensor(v, b, i);
                    }
                }

                for (; i < limit; i++) {
                    float v = (input.get(b, i) - mean) * invStddev * weights.get(i) + bias.get(i);
                    input.set(v, b, i);
                }
            }

            return output;
        }
    }

    public AbstractTensor batchForward(AbstractTensor input) {
        AbstractTensor output = input.copyShape();

        int batchSize = input.shape().first();
        VectorMath.pfor(0, batchSize, i -> {
            try(AbstractTensor o = forward(input.slice(i), Optional.empty())) {
                output.copyFrom(o, 0, i * m.c.embeddingLength, m.c.embeddingLength);
            }
        });

        return output;
    }
}
