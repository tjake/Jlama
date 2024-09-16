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
import com.github.tjake.jlama.tensor.FloatBufferTensor;
import com.github.tjake.jlama.tensor.TensorCache;
import com.github.tjake.jlama.tensor.TensorShape;
import com.google.common.base.Preconditions;

public interface TensorOperations {
    ThreadLocal<FloatBufferTensor> scratch = ThreadLocal.withInitial(() -> new FloatBufferTensor(TensorShape.one));

    String name();

    default int parallelSplitSize() {
        return 1;
    }

    default float dotProduct(AbstractTensor a, AbstractTensor b, int limit) {
        return dotProduct(a, b, 0, 0, limit);
    }

    default float dotProduct(AbstractTensor a, AbstractTensor b, int aoffset, int boffset, int limit) {
        FloatBufferTensor r = scratch.get();
        batchDotProduct(r, a, b, aoffset, boffset, limit);
        return r.get(0, 0);
    }

    default void batchDotProduct(
        AbstractTensor result,
        AbstractTensor a,
        AbstractTensor b,
        int aColumnOffset,
        int bColumnOffset,
        int columnLimit
    ) {
        batchDotProduct(result, a, b, aColumnOffset, bColumnOffset, columnLimit, 0, 0, b.shape().first());
    }

    void batchDotProduct(
        AbstractTensor result,
        AbstractTensor a,
        AbstractTensor b,
        int aColumnOffset,
        int bColumnOffset,
        int columnLimit,
        int rRowOffset,
        int bRowOffset,
        int rowChunkSize
    );

    default void dotProductChunk(
        AbstractTensor result,
        AbstractTensor a,
        AbstractTensor b,
        int columnOffset,
        int columnLimit,
        int rowOffset,
        int rowChunkSize
    ) {
        batchDotProduct(result, a, b, columnOffset, columnOffset, columnLimit, 0, rowOffset, rowChunkSize);
    }

    default void dotProductBatchChunk(
        AbstractTensor[] result,
        AbstractTensor a,
        AbstractTensor[] b,
        int offset,
        int limit,
        int chunkStart,
        int chunkSize
    ) {
        Preconditions.checkArgument(b[0].dims() == 2 && result.length == b.length);
        for (int j = 0; j < result.length; j++) {
            dotProductChunk(result[j], a, b[j], offset, limit, chunkStart, chunkSize);
        }
    }

    /**
     * For each position in the tensor, add b into a.  Must be same size.
     */
    void accumulate(AbstractTensor a, AbstractTensor b, int offset, int length);

    /**
     * For each position in the tensor, multiply b into a.  Must be same size.
     */
    void maccumulate(AbstractTensor a, AbstractTensor b, int offset, int length);

    /**
     * The value computed is Y[i] = (alpha * X[i]) + Y[i]
     */
    void saxpy(float alpha, AbstractTensor x, AbstractTensor y, int xoffset, int yoffset, int limit);

    /**
     * The value computed is Y[i] = (alpha[j] * X[j, i]) + Y[i]
     */
    default void saxpy(
        AbstractTensor alpha,
        AbstractTensor x,
        AbstractTensor y,
        int xoffset,
        int yoffset,
        int limit,
        int aOffset,
        int xRowOffset,
        int batchSize
    ) {
        Preconditions.checkArgument(alpha.shape().last() == x.shape().first() && y.shape().first() == 1);
        int batchLimit = xRowOffset + batchSize;
        for (int xi = xRowOffset; xi < batchLimit; xi++) {
            saxpy(alpha.get(0, aOffset++), x.slice(xi), y, xoffset, yoffset, limit);
        }
    }

    /**
     * For each position multiply value by the scale factor
     */
    void scale(float factor, AbstractTensor x, int offset, int length);

    /**
     * Quantizes the tensor to the specified type (if supported)
     */
    default AbstractTensor quantize(AbstractTensor t, DType qtype, int offset, int length) {
        AbstractTensor t2 = TensorCache.instance.get(t.dType(), t.shape());
        t2.copyFrom(t, offset, offset, length);
        return t2;
    }

    /**
     * Collects the total sum of each position in the tensor.  (For testing purposes)
     */
    default float sum(AbstractTensor a) {
        float sum = 0f;
        int[] cursor = new int[a.dims()];
        while (a.iterate(cursor))
            sum += a.get(cursor);
        return sum;
    }
}
