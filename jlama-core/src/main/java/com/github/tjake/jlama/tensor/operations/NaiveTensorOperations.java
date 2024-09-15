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
package com.github.tjake.jlama.tensor.operations;

import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.google.common.base.Preconditions;

public class NaiveTensorOperations implements TensorOperations {
    @Override
    public String name() {
        return "Naive Java Operations";
    }

    // a[0..n] += b[0..n]
    @Override
    public void accumulate(AbstractTensor a, AbstractTensor b, int offset, int length) {
        Preconditions.checkArgument(a.dims() == b.dims());

        boolean isBatch = b.shape().first() > 1;
        for (int ai = 0; ai < a.shape().first(); ai++) {
            AbstractTensor as = a.slice(ai);
            AbstractTensor bs = isBatch ? b.slice(ai) : b;
            for (int i = offset; i < offset + length; ++i) {
                as.set(as.get(0, i) + bs.get(0, i), 0, i);
            }
        }
    }

    // a[0..n] *= b[0..n]
    @Override
    public void maccumulate(AbstractTensor a, AbstractTensor b, int offset, int length) {
        Preconditions.checkArgument(a.size() == b.size() && a.dims() == b.dims());

        boolean isBatch = b.shape().first() > 1;
        for (int ai = 0; ai < a.shape().first(); ai++) {
            AbstractTensor as = a.slice(ai);
            AbstractTensor bs = isBatch ? b.slice(ai) : b;
            for (int i = offset; i < offset + length; ++i) {
                as.set(as.get(0, i) * bs.get(0, i), 0, i);
            }
        }
    }

    @Override
    public float dotProduct(AbstractTensor a, AbstractTensor b, int aoffset, int boffset, int limit) {
        Preconditions.checkArgument(a.dims() == b.dims() && a.shape().first() == 1);

        int alen = aoffset + limit;
        int blen = boffset + limit;

        float s = 0;
        for (; aoffset < alen && boffset < blen; aoffset++, boffset++) {
            s += a.get(0, aoffset) * b.get(0, boffset);
        }

        return s;
    }

    @Override
    public void batchDotProduct(
        AbstractTensor result,
        AbstractTensor a,
        AbstractTensor b,
        int aColumnOffset,
        int bColumnOffset,
        int columnLength,
        int rRowOffset,
        int bRowOffset,
        int rowChunkSize
    ) {
        Preconditions.checkArgument(a.dims() == 2 && b.dims() == 2 && result.dims() == 2);

        int bRowLimit = bRowOffset + rowChunkSize;

        for (int i = 0; i < a.shape().first(); i++) {
            for (int j = bRowOffset, r = rRowOffset; j < bRowLimit; j++, r++) {
                float d = dotProduct(a.slice(i), b.slice(j), aColumnOffset, bColumnOffset, columnLength);
                result.set(d, i, r);
            }
        }
    }

    // Computes a constant times a vector plus a vector (single-precision).
    // On return, the contents of vector Y are replaced with the result. The value computed is (alpha * X[i]) + Y[i].
    @Override
    public void saxpy(float alpha, AbstractTensor x, AbstractTensor y, int xoffset, int yoffset, int limit) {
        Preconditions.checkArgument(x.shape().first() == 1 && y.shape().first() == 1);
        for (int xo = xoffset, yo = yoffset; xo < (xoffset + limit) && yo < (yoffset + limit); xo++, yo++) {
            float v = (alpha * x.get(0, xo)) + y.get(0, yo);
            y.set(v, 0, yo);
        }
    }

    @Override
    public void scale(float factor, AbstractTensor x, int offset, int length) {
        int limit = offset + length;

        for (int b = 0; b < x.shape().first(); b++)
            for (int i = offset; i < limit; ++i)
                x.set(x.get(b, i) * factor, b, i);
    }

    @Override
    public AbstractTensor quantize(AbstractTensor t, DType qtype, int offset, int length) {
        return t.quantize(qtype, true);
    }
}
