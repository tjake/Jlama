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
package com.github.tjake.jlama.tensor;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.ShortBuffer;
import java.nio.channels.FileChannel;
import java.util.List;

import com.google.common.base.Preconditions;

import com.github.tjake.jlama.safetensors.TensorInfo;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorSpecies;

public class SegmentedTensor extends BFloat16BufferTensor {
    public static SegmentedTensor wrap(List<AbstractTensor> ft) {
        Preconditions.checkArgument(ft.size() > 1, "Must have at least two tensor to segment");
        Preconditions.checkArgument(ft.get(0).shape().dims() == 2, "First tensor must be 2D");

        AbstractTensor ft0 = ft.get(0);

        int firstDim = ft0.shape().first();
        int secondDim = ft0.shape().last();
        int[] splitPoints = new int[ft.size()];
        splitPoints[0] = firstDim;
        for (int i = 1; i < ft.size(); i++) {
            AbstractTensor t = ft.get(i);
            Preconditions.checkArgument(t.shape().last() == secondDim, "All tensors must have the same second dimension");
            firstDim += t.shape().first();
            splitPoints[i] = firstDim;
        }

        SegmentedTensor st = new SegmentedTensor(TensorShape.of(firstDim, secondDim), splitPoints, ft.toArray(new AbstractTensor[0]));

        return st;
    }

    private final AbstractTensor[] tensors;
    private final int[] splitPoints;

    protected SegmentedTensor(TensorShape shape, int[] splitPoints, AbstractTensor... tensors) {
        super("segmented-tensor", ShortBuffer.allocate(0), shape, false);
        this.splitPoints = splitPoints;
        this.tensors = tensors;
    }

    @Override
    public TensorInfo save(FileChannel out) throws IOException {

        long startOffset = out.position();
        for (AbstractTensor t : tensors) {
            t.save(out);
        }

        long[] lshape = new long[shape.dims()];
        for (int i = 0; i < shape.dims(); i++)
            lshape[i] = shape.dim(i);

        return new TensorInfo(dType, lshape, new long[] { startOffset, out.position() });
    }

    @Override
    public AbstractTensor slice(int... dims) {
        Preconditions.checkArgument(dims.length == 1, "Must slice on first dimension");
        int index = dims[0];
        for (int i = 0; i < splitPoints.length; i++) {
            if (index < splitPoints[i]) {
                return tensors[i].slice(index - (i == 0 ? 0 : splitPoints[i - 1]));
            }
        }

        throw new IllegalArgumentException("Index out of range");
    }

    ////////////////////// Everything below this line is not supported //////////////////////
    @Override
    public AbstractTensor slice(boolean cacheInnerSlice, int... dims) {
        return super.slice(dims);
    }

    @Override
    protected AbstractTensor make(TensorShape shape) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    protected AbstractTensor make(int heapOffset, int heapLength, TensorShape shape, boolean cacheSlices) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public float get(int... dims) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void set(float v, int... dims) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public ShortVector getVector(VectorSpecies<Short> species, int... voffset) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void intoTensor(ShortVector vector, int... aoffset) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public String toString() {
        return "SegmentedBF16Tensor{" + "shape=" + shape + ", tensors=" + tensors.length + '}';
    }

    @Override
    public MemorySegment getMemorySegment() {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public int getMemorySegmentOffset(int offset) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void copyFrom(AbstractTensor src, int srcOffset, int destOffset, int length) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void clear() {
        throw new UnsupportedOperationException("Not supported");
    }
}
