package com.github.tjake.jlama.model;

import com.github.tjake.jlama.tensor.AbstractTensor;

import com.google.common.base.Preconditions;

public class RMSNorm extends LayerNorm {
    public RMSNorm(AbstractModel m, AbstractTensor weights) {
        super(m, null, weights);
    }

    @Override
    public AbstractTensor forward(AbstractTensor input, int offset, int length) {
        Preconditions.checkArgument(input.shape().length == 1);
        int limit = offset + length;
        float ss = 0.0f;
        for (int j = offset; j < limit; j++) {
            float v = input.get(j);
            ss += v * v;
        }
        ss /= length;
        ss += m.c.layerNormEps;
        ss = (float)(1.0 / StrictMath.sqrt(ss));
        // normalize and scale
        AbstractTensor out = m.makeTensor(input.shape());
        for (int j = offset; j < limit; j++) {
             out.set(weights.get(j) * (ss * input.get(j)), j);
        }
        return out;
    }
}
