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

import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.safetensors.TensorInfo;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import jdk.incubator.vector.Vector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorSpecies;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A Tensor is a multi-dimensional array of data.
 * See {@link FloatBufferTensor} for primary impl.
 *
 * All implementations must be heap based. Meaning the data is stored in a
 * contiguous array in memory.  This is required for vector math operations.
 *
 * This class is abstract because there are multiple implementations
 * for different types of data.
 **/
public abstract class AbstractTensor<V extends Vector<?>, T extends Number> implements AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(AbstractTensor.class);

    protected final TensorShape shape;
    protected final DType dType;
    protected final AbstractTensor[] sliceCache;
    private final int stride;
    private volatile TensorCache originCache = null;

    protected AbstractTensor(DType dType, TensorShape shape, boolean cacheSlices) {
        Preconditions.checkArgument(shape != null && shape.dims() > 0);
        this.dType = dType;
        this.shape = shape;
        this.sliceCache = cacheSlices ? new AbstractTensor[shape.first()] : null;
        this.stride = shape.first() > 1 && dims() == 2 ? getOffset(shape.sparseRowOffset() + 1, shape.sparseColumnOffset()) : 0;
    }

    public static AbstractTensor make(DType dType, TensorShape shape) {
        return switch (dType) {
            case F32 -> new FloatBufferTensor(shape);
            case BF16 -> new BFloat16BufferTensor(shape);
            case I8 -> new Q8ByteBufferTensor(shape);
            default -> throw new RuntimeException("Unsupported tensor type: " + dType);
        };
    }

    /** Create a new tensor with the given shape of the same Tensor implementation */
    protected abstract AbstractTensor make(TensorShape shape);

    /** Create a new tensor with the given shape of the same Tensor implementation, including offsets to underlying heap */
    protected abstract AbstractTensor make(int heapOffset, int heapLength, TensorShape shape, boolean cacheSlices);

    /** Create a new tensor with the same shape and the same Tensor implementation */
    public AbstractTensor copyShape() {
        return TensorCache.instance.get(dType, shape);
    }

    /** Number of dimensions */
    public final int dims() {
        return shape.dims();
    }

    /** Represents the dimensions of the tensor */
    public final TensorShape shape() {
        return shape;
    }

    /** Total capacity of the tensor */
    public final long size() {
        return shape.size();
    }

    /** Get a value at the given coordinates */
    public abstract float get(int... dims);

    /** Set a value at the given coordinates */
    public abstract void set(float v, int... dims);

    public AbstractTensor slice(int... dims) {
        return slice(false, dims);
    }

    /** Get a slice of the tensor along the given dimension */
    public AbstractTensor slice(boolean cacheInnerSlice, int... dims) {
        Preconditions.checkArgument(dims.length < shape.dims(), "Too many dimensions specified for tensor");
        try {
            if (dims.length == 1 && sliceCache != null && sliceCache[dims[0]] != null) return sliceCache[dims[0]];
        } catch (Throwable t) {
            logger.warn("Dims = {}", Arrays.toString(dims), t);
            throw t;
        }

        TensorShape slicedShape = shape.slice(dims.length);
        int totalOffset = 0;
        if (dims.length == 1 && this.shape.dims() == 2) {
            totalOffset = shape.sparseColumnLength() * dims[0];
        } else {
            for (int d = 0; d <= dims.length - 1; d++) {
                int offset = shape.sparseColumnLength();
                for (int i = shape.dims() - 2; i > d; i--) { // factor scaling of each dim shape
                    offset *= shape.dim(i);
                }

                totalOffset += dims[d] * offset;
            }
        }

        AbstractTensor r = this.make(totalOffset, (int) slicedShape.size(), slicedShape, cacheInnerSlice);
        if (dims.length == 1 && sliceCache != null) sliceCache[dims[0]] = r;
        return r;
    }

    /**
     * Creates a sparse tensor that acts like a dense one but is missing the data outside
     * the range of in last dimension.
     */
    public AbstractTensor<V, T> sparsify(int offset, int length) {
        if (shape.isSparse()) return this;

        if (length == shape.last()) return this;

        AbstractTensor<V, T> sparseT = this.make(shape.sparsifyColumns(offset, length));
        int originalLength = shape.last();

        int[] cursor = new int[shape.dims()];

        try {
            do {
                cursor[cursor.length - 1] = offset;
                sparseT.copyFrom(this, getOffset(cursor), sparseT.getOffset(cursor), length);
                cursor[cursor.length - 1] = originalLength - 1; // Reset last dimension, so it iterates in the next lower dimension
            } while (iterate(cursor));
        } catch (Throwable t) {
            logger.warn("Cursor = {}", Arrays.toString(cursor), t);
            throw t;
        }
        return sparseT;
    }

    /** Split the tensor into numChunks along the given dimension */
    public AbstractTensor[] split(int numChunks, int dim) {
        AbstractTensor[] chunks = new AbstractTensor[numChunks];
        int innerLength = this.shape.dim(dim) / numChunks;

        if (innerLength * numChunks != this.shape.dim(dim)) {
            throw new IllegalStateException("Chunks must be of equal size");
        }

        TensorShape newShape = shape.setDimValue(dim, innerLength);
        for (int i = 0; i < numChunks; i++) {
            chunks[i] = this.make(Ints.checkedCast(i * newShape.size()), Ints.checkedCast(newShape.size()), newShape, true);
        }

        return chunks;
    }

    /**
     * Does inplace iteration based on the current values from innermost to outermost offset
     * Meaning if I pass [0,0,99] for a tensor of shape (100,100,100).  It will alter values to be
     * [0,1,0] and so on
     * @param cursor
     * @return false if cursor has hit its limit, otherwise true
     */
    public final boolean iterate(int[] cursor) {
        Preconditions.checkArgument(cursor.length == shape.dims());

        for (int i = cursor.length - 1; i >= 0; i--) {
            Preconditions.checkArgument(cursor[i] >= 0 && cursor[i] < shape.dim(i));
            if (cursor[i] + 1 < shape.dim(i)) {
                cursor[i]++;
                break;
            } else {
                cursor[i] = 0;
                if (i == 0) return false;
            }
        }

        return true;
    }

    public final int getStride() {
        return stride;
    }

    public final int getOffset(int... dims) {
        return shape.getOffset(dims);
    }

    /** Transpose the tensor across all dimensions*/
    public final AbstractTensor transpose() {
        Preconditions.checkArgument(!shape.isSparse(), "Cannot transpose a sparse tensor");

        // Reverse the dimensions
        int[] tshape = new int[dims()];
        for (int i = 0; i < tshape.length; i++)
            tshape[i] = shape.dim(shape.dims() - i - 1);

        AbstractTensor tt = this.make(TensorShape.of(tshape));
        int[] cursor = new int[dims()];
        int[] tcursor = new int[dims()];
        do {
            float v = this.get(cursor);

            // 1000(0), 100(0), 10(1) -> 1
            // 10, 100, 1000 -> 10001
            for (int i = 0; i < tcursor.length; i++)
                tcursor[i] = cursor[cursor.length - i - 1];

            tt.set(v, tcursor);
        } while (iterate(cursor));

        return tt;
    }

    public final DType dType() {
        return dType;
    }

    public abstract V getVector(VectorSpecies<T> species, int... offset);

    public abstract void intoTensor(V vector, int... offset);

    public void intoTensor(V vector, VectorMask<T> mask, int... offset) {
        throw new UnsupportedOperationException();
    }

    public abstract MemorySegment getMemorySegment();

    public abstract int getMemorySegmentOffset(int offset);

    public abstract void copyFrom(AbstractTensor src, int srcOffset, int destOffset, int length);

    /** Zero out the tensor */
    public abstract void clear();

    public void close() {
        if (originCache != null) originCache.release(this);
    }

    void setOwnerCache(TensorCache cache) {
        this.originCache = cache;
    }

    public AbstractTensor quantize(DType dType) {
        return quantize(dType, false);
    }

    public AbstractTensor quantize(DType dType, boolean force) {

        if (!force && (this.shape().first() == 1 || this.dType == dType || this.dType.size() < dType.size())) return this;

        if (shape.isSparse()) {
            logger.info("Quantizing sparse tensor is not supported");
            return this;
        }

        return switch (dType) {
            case Q4 -> new Q4ByteBufferTensor(this);
            case I8 -> new Q8ByteBufferTensor(this);
            case F32 -> new FloatBufferTensor(this);
            case BF16 -> new BFloat16BufferTensor(this);
            default -> this;
        };
    }

    public TensorInfo save(FileChannel out) throws IOException {
        Preconditions.checkArgument(!shape.isSparse(), "Cannot save a sparse tensor");
        ByteBuffer bb = getMemorySegment().asByteBuffer().order(ByteOrder.LITTLE_ENDIAN);

        long startOffset = out.position();
        out.write(bb);

        long[] lshape = new long[shape.dims()];
        for (int i = 0; i < shape.dims(); i++)
            lshape[i] = shape.dim(i);

        return new TensorInfo(dType, lshape, new long[] { startOffset, out.position() });
    }

    public void debug(String id) {
        if (true) {
            double tmp = 0.0;
            for (int i = 0; i < size(); i++) {
                tmp += get(0, i);
            }
            System.out.println(String.format("%s = %.5f", id, tmp));
        }
    }
}
