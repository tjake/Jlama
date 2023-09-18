package com.github.tjake.jlama.tensor.operations;

import com.google.common.base.Preconditions;

import com.github.tjake.jlama.tensor.AbstractTensor;

public interface TensorOperations
{
    default float dotProduct(AbstractTensor a, AbstractTensor b, int limit) {
        return dotProduct(a, b, 0, 0, limit);
    }

    float dotProduct(AbstractTensor a, AbstractTensor b, int aoffset, int boffset, int limit);

    /**
     * For each position in the tensor, add a to b.  Must be same size.
     */
    void accumulate(AbstractTensor a, AbstractTensor b);

    default float sum(AbstractTensor a) {
        Preconditions.checkArgument( a.dims() == 1);
        float sum = 0f;
        for (int i = 0; i < a.size(); i++)
            sum += a.get(i);
        return sum;
    }


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
}
