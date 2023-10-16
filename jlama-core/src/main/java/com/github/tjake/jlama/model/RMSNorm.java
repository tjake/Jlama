package com.github.tjake.jlama.model;

import com.github.tjake.jlama.tensor.AbstractTensor;

import com.google.common.base.Preconditions;

public class RMSNorm extends LayerNorm {
    public RMSNorm(AbstractModel m, AbstractTensor weights) {
        super(m, null, weights);
    }

    @Override
    public AbstractTensor forward(AbstractTensor input) {
        Preconditions.checkArgument(input.shape().length == 1);
        int size = input.shape()[0];
        float ss = 0.0f;
        for (int j = 0; j < size; j++) {
            float v = input.get(j);
            ss += v * v;
        }
        ss /= size;
        ss += m.c.layerNormEps;
        ss = (float)(1.0 / StrictMath.sqrt(ss));
        // normalize and scale
        AbstractTensor out = m.makeTensor(size);
        for (int j = 0; j < size; j++) {
             out.set(weights.get(j) * (ss * input.get(j)), j);
        }
        return out;
    }
}
