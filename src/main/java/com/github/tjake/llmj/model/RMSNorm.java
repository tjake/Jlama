package com.github.tjake.llmj.model;

import com.github.tjake.llmj.safetensors.Config;
import com.google.common.base.Preconditions;

public class RMSNorm extends LayerNorm {
    public RMSNorm(Config c, Tensor bias, Tensor weights) {
        super(c, bias, weights);
    }

    @Override
    public Tensor forward(Tensor input) {
        Preconditions.checkArgument(input.shape().length == 1);
        int size = input.shape()[0];
        float ss = 0.0f;
        for (int j = 0; j < size; j++) {
            float v = input.get(j);
            ss += v * v;
        }
        ss /= size;
        ss += c.layerNormEps;
        ss = (float)(1.0 / StrictMath.sqrt(ss));
        // normalize and scale
        FloatBufferTensor out = c.bufferCache.get(size);
        for (int j = 0; j < size; j++) {
             out.set(weights.get(j) * (ss * input.get(j)), j);
        }
        return out;
    }
}
