package com.github.tjake.jlama.tensor;

import com.github.tjake.jlama.safetensors.DType;
import com.google.common.base.Preconditions;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.Vector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.MemorySegment;
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
    protected final int[] shape;
    protected final DType dType;
    protected final AbstractTensor[] sliceCache;
    protected final int capacity;
    private volatile TensorCache originCache = null;

    protected AbstractTensor(DType dType, int[] shape, boolean cacheSlices) {
        Preconditions.checkArgument(shape != null && shape.length > 0);
        this.dType = dType;
        this.shape = shape;

        int c = 1;
        for (int i = 0; i < shape.length; i++)
            c *= shape[i];

        this.capacity = c;

        this.sliceCache = cacheSlices ? new AbstractTensor[shape[0]] : null;
    }

    /** Create a new tensor with the given shape of the same Tensor implementation */
    protected abstract AbstractTensor make(int ...shape);

    /** Create a new tensor with the given shape of the same Tensor implementation, including offsets to underlying heap */
    protected abstract AbstractTensor make(int heapOffset, int heapLength, int[] shape, boolean cacheSlices);

    /** Number of dimensions */
    final public int dims() {
        return shape.length;
    }

    /** Represents the dimensions of the tensor */
    final public int[] shape() {
        return shape;
    }

    /** Total capacity of the tensor */
    final public int size() {
        return capacity;
    }

    /** Get a value at the given coordinates */
    public abstract float get(int ...dims);

    /** Set a value at the given coordinates */
    public abstract void set(float v, int ...dims);

    /** Get a slice of the tensor along the given dimension */
    public AbstractTensor slice(int ...dims) {
        Preconditions.checkArgument(dims.length < shape.length, "Too many dimensions specified for tensor");

        if (dims.length == 1 && sliceCache != null && sliceCache[dims[0]] != null)
            return sliceCache[dims[0]];

        int[] slicedShape = Arrays.copyOfRange(shape, dims.length, shape.length);

        int totalOffset = 0;
        for (int d = 0; d <= dims.length - 1; d++) {
            int offset = 1;
            for (int i = shape.length - 1; i > d; i--) { // factor scaling of each dim shape
                offset *= shape[i];
            }

            totalOffset += dims[d] * offset;
        }

        int length = 1;
        for (int i = 0; i < slicedShape.length; i++)
            length *= slicedShape[i];

        AbstractTensor r = this.make(totalOffset, length, slicedShape, false);
        if (dims.length == 1 && sliceCache != null)
            sliceCache[dims[0]] = r;

        return r;
    }

    /** Split the tensor into numChunks along the given dimension */
    public AbstractTensor[] split(int numChunks, int dim) {
        AbstractTensor[] chunks = new AbstractTensor[numChunks];
        int innerLength = this.shape[dim] / numChunks;

        if (Integer.bitCount(innerLength) != 1) {
            throw new IllegalStateException("Chunks must be power of 2");
        }

        int[] newShape = Arrays.copyOf(this.shape, shape.length);
        newShape[dim] = innerLength;
        int newCapacity = 1;
        for (int i = 0; i < newShape.length; i++) {
            newCapacity *= newShape[i];
        }

        for (int i = 0; i < numChunks; i++) {
            chunks[i] = this.make(i * newCapacity, newCapacity, newShape, true);
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
        int[] shape = shape();
        Preconditions.checkArgument(cursor.length == shape.length);

        for (int i = cursor.length - 1; i >= 0; i--) {
            Preconditions.checkArgument(cursor[i] >= 0 && cursor[i] < shape[i]);
            if (cursor[i] + 1 < shape[i]) {
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

    final public int getOffset(int[] dims) {
        int[] shape = shape();
        Preconditions.checkArgument(dims.length == shape.length, "Method requires all dimensions specified");
        int totalOffset = 0;

        for (int d = 0; d < dims.length - 1; d++) { // Stop before last dimension
            int offset = 1;
            for (int i = shape.length - 1; i > d; i--) { // factor scaling of each dim shape
                offset *= shape[i];
            }

            totalOffset += dims[d] * offset;
        }

        return totalOffset + dims[shape.length - 1];
    }

    /** Transpose the tensor across all dimensions*/
    final public AbstractTensor transpose() {
        int[] shape = shape();
        int[] tshape = new int[dims()];

        for (int i = 0; i < tshape.length; i++)
            tshape[i] = shape[shape.length - i - 1];

        AbstractTensor tt = this.make(tshape);
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

    public abstract boolean hasMemorySegment();

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

        if (this.dims() != 2)
            return this;

        return switch (dType) {
            case Q4 -> new Q4ByteBufferTensor(this);
            case I8 -> new Q8ByteBufferTensor(this);
            case F32 -> new FloatBufferTensor(this);
            case BF16 -> new BFloat16BufferTensor(this);
            default -> this;
        };
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
