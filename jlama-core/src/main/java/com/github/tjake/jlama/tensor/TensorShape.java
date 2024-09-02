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

import com.github.tjake.jlama.util.Pair;
import com.google.common.base.Preconditions;
import java.util.Arrays;
import java.util.Objects;
import java.util.Optional;

/**
 *
 */
public class TensorShape {
    public static TensorShape one = of(1, 1);

    public static TensorShape of(int... shape) {
        // Special case for vectors
        if (shape.length == 1) shape = new int[] { 1, shape[0] };

        return new TensorShape(shape, Optional.empty());
    }

    public static TensorShape sparse(int[] shape, Pair<Integer, Integer> sparseOffset) {
        return new TensorShape(shape, Optional.of(sparseOffset));
    }

    private final int[] tshape;
    private final long capacity;
    private final Optional<Pair<Integer, Integer>> sparseRange;
    private final boolean isSparse;
    private final int sparseOffset;
    private final int sparseLength;

    private TensorShape(int[] shape, Optional<Pair<Integer, Integer>> sparseRange) {
        Preconditions.checkArgument(
            shape.length > 1,
            "Shape must have at least two dimensions, even if first is 1 (to represent a vector)"
        );

        this.tshape = shape;
        this.sparseRange = sparseRange;
        this.isSparse = sparseRange.isPresent();
        this.sparseOffset = sparseRange.map(Pair::left).orElse(0);
        this.sparseLength = sparseRange.map(Pair::right).orElse(shape[shape.length - 1]);

        long c = 1;
        for (int i = 0; i < shape.length - 1; i++)
            c *= shape[i];

        c *= sparseLength;
        this.capacity = c;
    }

    public final boolean isSparse() {
        return isSparse;
    }

    public int dims() {
        return tshape.length;
    }

    public int dim(int i) {
        Preconditions.checkArgument(i < tshape.length);
        return tshape[i];
    }

    public final int getOffset(int... pdims) {
        // Preconditions.checkArgument(pdims.length == dims(), "Method requires all dimensions specified");
        switch (pdims.length) {
            case 1:
                return sparseLength * pdims[0] - sparseOffset;
            case 2:
                return sparseLength * pdims[0] + pdims[1] - sparseOffset; // Most common case
            case 3:
                return (sparseLength * tshape[1] * pdims[0]) + (sparseLength * pdims[1]) + pdims[2] - sparseOffset;
            default:
                int totalOffset = 0;
                for (int d = 0; d < pdims.length - 1; ++d) { // Stop before last dimension
                    int offset = sparseLength;
                    for (int i = tshape.length - 2; i > d; --i) { // factor scaling of each dim shape
                        offset *= tshape[i];
                    }

                    totalOffset += pdims[d] * offset;
                }

                return totalOffset + pdims[pdims.length - 1] - sparseOffset;
        }
    }

    public int sparseLength() {
        return sparseLength;
    }

    public int sparseOffset() {
        return sparseOffset;
    }

    public int sparseAdjustment(int offset) {
        // Preconditions.checkArgument(sparseOffset <= offset, "Offset is outside of sparse range");
        return offset - sparseOffset;
    }

    public TensorShape scaleLastDim(float scale) {
        int[] copy = Arrays.copyOf(tshape, tshape.length);
        copy[copy.length - 1] *= scale;
        return isSparse ? sparse(copy, Pair.create((int) (sparseOffset * scale), (int) (sparseLength * scale))) : of(copy);
    }

    public TensorShape setDimValue(int dim, int value) {
        Preconditions.checkArgument(dim < tshape.length);
        int[] copy = Arrays.copyOf(tshape, tshape.length);
        copy[dim] = value;
        int newSparseLength = copy[copy.length - 1];
        return isSparse ? sparse(copy, Pair.create(sparseOffset, newSparseLength)) : of(copy);
    }

    public int first() {
        return tshape[0];
    }

    public int last() {
        return tshape[tshape.length - 1];
    }

    public long size() {
        return capacity;
    }

    public TensorShape sparsify(int offset, int length) {
        Preconditions.checkArgument(!isSparse, "Cannot sparsify a sparse tensor");
        return new TensorShape(tshape, Optional.of(Pair.create(offset, length)));
    }

    public TensorShape slice(int numDims) {
        Preconditions.checkArgument(numDims < tshape.length, "Too many dimensions specified for tensor");
        int newLength = tshape.length - numDims;
        if (newLength == 1) return new TensorShape(new int[] { 1, tshape[tshape.length - 1] }, sparseRange);

        return new TensorShape(Arrays.copyOfRange(tshape, numDims, tshape.length), sparseRange);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        TensorShape that = (TensorShape) o;
        return Arrays.equals(tshape, that.tshape) && Objects.equals(sparseRange, that.sparseRange);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(sparseRange);
        result = 31 * result + Arrays.hashCode(tshape);
        return result;
    }

    @Override
    public String toString() {
        return "TensorShape{" + "tshape=" + Arrays.toString(tshape) + ", capacity=" + capacity + ", sparseRange=" + sparseRange + '}';
    }
}
