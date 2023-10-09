package com.github.tjake.jlama.tensor.operations;

import com.github.tjake.jlama.tensor.TensorCache;
import com.google.common.base.Preconditions;

import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.tensor.AbstractTensor;

public interface TensorOperations
{
    String name();

    boolean requiresOffHeapTensor();

    default float dotProduct(AbstractTensor a, AbstractTensor b, int limit) {
        return dotProduct(a, b, 0, 0, limit);
    }

    float dotProduct(AbstractTensor a, AbstractTensor b, int aoffset, int boffset, int limit);

    /**
     * For each position in the tensor, add a to b.  Must be same size.
     */
    void accumulate(AbstractTensor a, AbstractTensor b);

    /**
     * The value computed is (alpha * X[i]) + Y[i]
     */
    void saxpy(float alpha, AbstractTensor x, AbstractTensor y, int xoffset, int yoffset, int limit);

    /**
     * The value computed is Y[i] = X[i] + (beta * Y[i])
     */
    void sxpby(float beta, AbstractTensor x, AbstractTensor y, int xoffset, int yoffset, int limit);

    /**
     * For each position multiply value by the scale factor
     */
    void scale(float factor, AbstractTensor x, int offset, int length);

    /**
     * Quantizes the model to the specified type (if supported)
     */
    default AbstractTensor quantize(AbstractTensor t, DType qtype) {
        AbstractTensor t2 = TensorCache.instance.get(t.dType(), t.shape());
        t2.copyFrom(t, 0, 0, t.size());
        return t2;
    }

    /**
     * Collects the total sum of each position in the tensor.  (For testing purposes)
     */
    default float sum(AbstractTensor a) {
        Preconditions.checkArgument( a.dims() == 1);
        float sum = 0f;
        for (int i = 0; i < a.size(); i++)
            sum += a.get(i);
        return sum;
    }
}
