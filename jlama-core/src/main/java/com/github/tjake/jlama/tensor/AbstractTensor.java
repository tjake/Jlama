package com.github.tjake.jlama.tensor;

import com.github.tjake.jlama.safetensors.DType;
import com.google.common.base.Preconditions;

import com.github.tjake.jlama.safetensors.TensorInfo;
import jdk.incubator.vector.Vector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorSpecies;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.Arrays;

/** A Tensor is a multi-dimensional array of data.
 * See {@link FloatBufferTensor} for primary impl.
 *
 * All implementations must be heap based. Meaning the data is stored in a
 * contiguous array in memory.  This is required for vector math operations.
 *
 * This class is abstract because there are multiple implementations
 * for different types of data.
 **/
public abstract class AbstractTensor<V extends Vector<?>, T extends Number, A> implements AutoCloseable {
    private static final Logger logger = LoggerFactory.getLogger(AbstractTensor.class);

    protected final TensorShape shape;
    protected final DType dType;
    protected final AbstractTensor[] sliceCache;
    private volatile TensorCache originCache = null;

    protected AbstractTensor(DType dType, TensorShape shape, boolean cacheSlices) {
        Preconditions.checkArgument(shape != null && shape.dims() > 0);
        this.dType = dType;
        this.shape = shape;
        this.sliceCache = cacheSlices ? new AbstractTensor[shape.first()] : null;
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
    final public int dims() {
        return shape.dims();
    }

    /** Represents the dimensions of the tensor */
    final public TensorShape shape() {
        return shape;
    }

    /** Total capacity of the tensor */
    final public int size() {
        return shape.size();
    }

    /** Get a value at the given coordinates */
    public abstract float get(int ...dims);

    /** Set a value at the given coordinates */
    public abstract void set(float v, int ...dims);

    public AbstractTensor slice(int ...dims) {
        return slice(false, dims);
    }

    /** Get a slice of the tensor along the given dimension */
    public AbstractTensor slice(boolean cacheInnerSlice, int ...dims) {
        Preconditions.checkArgument(dims.length < shape.dims(), "Too many dimensions specified for tensor");

        if (dims.length == 1 && sliceCache != null && sliceCache[dims[0]] != null)
            return sliceCache[dims[0]];

        TensorShape slicedShape = shape.slice(dims.length);

        int totalOffset = 0;
        for (int d = 0; d <= dims.length - 1; d++) {
            int offset = shape.sparseLength();
            for (int i = shape.dims() - 2; i > d; i--) { // factor scaling of each dim shape
                offset *= shape.dim(i);
            }

            totalOffset += dims[d] * offset;
        }

        int length = slicedShape.sparseLength();
        for (int i = 0; i < slicedShape.dims() - 1; i++)
            length *= slicedShape.dim(i);

        AbstractTensor r = this.make(totalOffset, length, slicedShape, cacheInnerSlice);
        if (dims.length == 1 && sliceCache != null)
            sliceCache[dims[0]] = r;

        return r;
    }

    /**
     * Creates a sparse tensor that acts like a dense one but is missing the data outside
     * the range of in last dimension.
     */
    public AbstractTensor<V, T, A> sparsify(int offset, int length) {
        if (shape.isSparse())
            return this;

        //if(length == shape.last())
        //    return this;

        AbstractTensor<V,T,A> sparseT = this.make(shape.sparsify(offset, length));
        int originalLength = shape.last();

        int[] cursor = new int[shape.dims()];
        do {
            cursor[cursor.length - 1] = offset;
            sparseT.copyFrom(this, getOffset(cursor), sparseT.getOffset(cursor), length);
            cursor[cursor.length - 1] = originalLength - 1; // Reset last dimension, so it iterates in the next lower dimension
        } while (iterate(cursor));

        return sparseT;
    }

    /** Split the tensor into numChunks along the given dimension */
    public AbstractTensor[] split(int numChunks, int dim) {
        AbstractTensor[] chunks = new AbstractTensor[numChunks];
        int innerLength = this.shape.dim(dim) / numChunks;

        if (Integer.bitCount(innerLength) != 1) {
            throw new IllegalStateException("Chunks must be power of 2");
        }

        TensorShape newShape = shape.setDimValue(dim, innerLength);
        for (int i = 0; i < numChunks; i++) {
            chunks[i] = this.make(i * newShape.size(), newShape.size(), newShape, true);
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
    final public boolean iterate(int[] cursor) {
        Preconditions.checkArgument(cursor.length == shape.dims());

        for (int i = cursor.length - 1; i >= 0; i--) {
            Preconditions.checkArgument(cursor[i] >= 0 && cursor[i] < shape.dim(i));
            if (cursor[i] + 1 < shape.dim(i)) {
                cursor[i]++;
                break;
            } else {
                cursor[i] = 0;
                if (i == 0)
                    return false;
            }
        }

        return true;
    }

    final public int getOffset(int... dims) {
        Preconditions.checkArgument(dims.length == shape.dims(), "Method requires all dimensions specified");
        int totalOffset = 0;

        for (int d = 0; d < dims.length - 1; d++) { // Stop before last dimension
            int offset = shape.sparseLength();
            for (int i = shape.dims() - 2; i > d; i--) { // factor scaling of each dim shape
                offset *= shape.dim(i);
            }

            totalOffset += dims[d] * offset;
        }

        return totalOffset + shape.sparseAdjustment(dims[dims.length - 1]);
    }

    /** Transpose the tensor across all dimensions*/
    final public AbstractTensor transpose() {
        Preconditions.checkArgument(!shape.isSparse(), "Cannot transpose a sparse tensor");

        //Reverse the dimensions
        int[] tshape = new int[dims()];
        for (int i = 0; i < tshape.length; i++)
            tshape[i] = shape.dim(shape.dims() - i - 1);

        AbstractTensor tt = this.make(TensorShape.of(tshape));
        int[] cursor = new int[dims()];
        int[] tcursor = new int[dims()];
        do {
            float v = this.get(cursor);

            //1000(0), 100(0), 10(1) -> 1
            //10, 100, 1000 -> 10001
            for (int i = 0; i < tcursor.length; i++)
                tcursor[i] = cursor[cursor.length - i - 1];

            tt.set(v, tcursor);
        } while(iterate(cursor));

        return tt;
    }

    final public DType dType() {
        return dType;
    }

    public abstract A getArray();
    public abstract int getArrayOffset(int offset);

    public abstract V getVector(VectorSpecies<T> species, int offset);

    public abstract void intoTensor(V vector, int offset);

    public void intoTensor(V vector, int offset, VectorMask<T> mask) {
        throw new UnsupportedOperationException();
    }

    public abstract MemorySegment getMemorySegment();

    public abstract int getMemorySegmentOffset(int offset);

    public abstract void copyFrom(AbstractTensor src, int srcOffset, int destOffset, int length);

    /** Zero out the tensor */
    public abstract void clear();

    public void close() {
        if (originCache != null)
            originCache.release(this);
    }

    void setOwnerCache(TensorCache cache) {
        this.originCache = cache;
    }

    public AbstractTensor quantize(DType dType) {

        if (this.dims() != 2 || this.dType == dType)
            return this;

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

        return new TensorInfo(dType, lshape, new long[]{startOffset, out.position()});
    }

    public void debug(String id) {
        if (false) {
            double tmp = 0.0;
            for (int i = 0; i < size(); i++) {
                tmp += get(i);
            }
            System.out.println(String.format("%s = %.5f", id, tmp));
        }
    }
}
