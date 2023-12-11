package com.github.tjake.jlama.tensor.operations;

import com.google.common.base.Preconditions;

import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.tensor.AbstractTensor;

public class NaiveTensorOperations implements TensorOperations
{
    @Override
    public String name() {
        return "Naive Java Operations";
    }

    @Override
    public boolean requiresOffHeapTensor() {
        return false;
    }

    // a[0..n] += b[0..n]
    @Override
    public void accumulate(AbstractTensor a, AbstractTensor b, int offset, int length) {
        Preconditions.checkArgument(a.size() == b.size() && a.dims() == b.dims() && a.dims() == 1);

        for (int i = offset; i < offset + length; ++i) {
            a.set(a.get(i) + b.get(i), i);
        }
    }

    // a[0..n] *= b[0..n]
    @Override
    public void maccumulate(AbstractTensor a, AbstractTensor b, int offset, int length) {
        Preconditions.checkArgument(a.size() == b.size() && a.dims() == b.dims() && a.dims() == 1);

        for (int i = offset; i < offset + length; ++i) {
            a.set(a.get(i) * b.get(i), i);
        }
    }

    @Override
    public float dotProduct(AbstractTensor a, AbstractTensor b, int aoffset, int boffset, int limit) {
        Preconditions.checkArgument(a.dims() == b.dims());

        int alen = aoffset + limit;
        int blen = boffset + limit;

        float s = 0;
        for (; aoffset < alen && boffset < blen; aoffset++, boffset++) {
            s += a.get(aoffset) * b.get(boffset);
        }

        return s;
    }

    // Computes a constant times a vector plus a vector (single-precision).
    // On return, the contents of vector Y are replaced with the result. The value computed is (alpha * X[i]) + Y[i].
    @Override
    public void saxpy(float alpha, AbstractTensor x, AbstractTensor y, int xoffset, int yoffset, int limit) {
        Preconditions.checkArgument(x.dims() == y.dims());
        for (int xo = xoffset, yo = yoffset; xo < (xoffset + limit) && yo < (yoffset + limit) ; xo++, yo++) {
            float v = (alpha * x.get(xo)) + y.get(yo);
            y.set(v, yo);
        }
    }

    // y = x + b*y variant
    @Override
    public void sxpby(float beta, AbstractTensor x, AbstractTensor y, int xoffset, int yoffset, int limit) {
        Preconditions.checkArgument(x.dims() == y.dims());

        for (int xo = xoffset, yo = yoffset; xo < (xoffset + limit) && yo < (yoffset + limit) ; xo++, yo++) {
            float v = x.get(xo) + (beta * y.get(yo));
            y.set(v, yo);
        }
    }

    @Override
    public void scale(float factor, AbstractTensor x, int offset, int length)
    {
        int limit = offset + length;
        for (; offset < limit; ++offset)
            x.set(x.get(offset) * factor, offset);
    }
}
