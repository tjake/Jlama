package com.github.tjake.jlama.model;

import com.github.tjake.jlama.safetensors.Config;
import com.google.common.base.Preconditions;

public class LayerNorm {

    protected final Config c;
    private final Tensor bias;
    protected final Tensor weights;

    public LayerNorm(Config c, Tensor bias, Tensor weights)
    {
        this.c = c;
        this.bias = bias;
        this.weights = weights;
    }

    public Tensor forward(Tensor input)
    {
        Preconditions.checkArgument(input.shape().length == 1);
        float sum = 0;
        float sumSq = 0;
        int size = input.shape()[0];
        for (int i = 0; i < size; i++) {
            float v = input.get(i);
            sum += v;
            sumSq += v * v;
        }

        float mean = sum / size;
        float variance = sumSq / size - mean * mean;
        float invStddev = 1.0f / (float) Math.sqrt(variance + c.layerNormEps);

        FloatBufferTensor out = c.bufferCache.get(input.shape());
        for (int i = 0; i < size; i++) {
            float v = (input.get(i) - mean) * invStddev * weights.get(i) + bias.get(i);
            out.set(v, i);
        }

        return out;
    }
}
