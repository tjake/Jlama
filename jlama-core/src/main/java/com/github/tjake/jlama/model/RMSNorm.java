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
    public AbstractTensor forward(AbstractTensor input, int offset, int length, Optional<BiFunction<Float, Float, Pair<Float, Float>>> reducer) {
        Preconditions.checkArgument(input.shape().dims() == 1);
        int limit = offset + length;
        float ss = 0.0f;
        for (int j = offset; j < limit; j++) {
            float v = input.get(j);
            ss += v * v;
        }

        if (reducer.isPresent()) {
            Pair<Float, Float> p = reducer.get().apply(ss, 0f);
            ss = p.left;
        }

        ss /= m.c.embeddingLength;
        ss += m.c.layerNormEps;
        ss = (float)(1.0 / StrictMath.sqrt(ss));
        // normalize and scale
        AbstractTensor output = input.copyShape();
        for (int j = offset; j < limit; j++) {
            output.set((weightAdjustment + weights.get(j)) * (ss * input.get(j)), j);
        }
        return output;
    }
}
