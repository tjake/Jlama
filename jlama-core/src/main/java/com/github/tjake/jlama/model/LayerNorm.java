package com.github.tjake.jlama.model;

import com.github.tjake.jlama.util.Pair;
import com.google.common.base.Function;
import com.google.common.base.Preconditions;

import com.github.tjake.jlama.tensor.AbstractTensor;
import com.google.common.base.Supplier;

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

    public AbstractTensor forward(AbstractTensor input, Optional<BiFunction<Float, Float,Pair<Float, Float>>> reducer) {
        Preconditions.checkArgument(input.shape().dims() == 1);
        int size = input.shape().first();
        Preconditions.checkArgument(size == m.c.embeddingLength);
        return forward(input, m.c.embeddingSegmentStart(), m.c.embeddingSegmentLength(), reducer);
    }

    public AbstractTensor forward(AbstractTensor input, int offset, int length, Optional<BiFunction<Float, Float,Pair<Float, Float>>> reducer) {
        float sum = 0;
        float sumSq = 0;
        int limit = offset + length;
        for (int i = offset; i < limit; i++) {
            float v = input.get(i);
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

        AbstractTensor output = input.copyShape();
        for (int i = offset; i < limit; i++) {
            float v = (input.get(i) - mean) * invStddev * weights.get(i) + bias.get(i);
            output.set(v, i);
        }

        return output;
    }
}
