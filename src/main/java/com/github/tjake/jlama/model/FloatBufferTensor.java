package com.github.tjake.jlama.model;

import com.github.tjake.jlama.math.VectorMath;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;
import jdk.incubator.vector.FloatVector;
import org.jctools.queues.MpmcUnboundedXaddArrayQueue;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.Buffer;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * A Tensor is a multi-dimensional array of floats.  It is backed by a FloatBuffer
 * and can be used to perform vector math operations on the data.
 *
 * The Tensor can be read only, or read/write.  If it is read only, it can be shared (model weights)
 *
 * The Tensor is backed by a FloatBuffer, but can be converted to a float[] for use with
 * other libraries.
 *
 * The Tensor is thread safe for read operations, but not for write operations.
 */
public class FloatBufferTensor implements Tensor {
    private static final Logger logger = LoggerFactory.getLogger(FloatBufferTensor.class);
    private final int[] shape;
    private final FloatBufferTensor[] sliceCache;
    private final FloatBuffer b;
    private final int capacity;
    private volatile float[] cachedRoBuffer = null;

    private BufferCache originCache = null;
    private boolean isReadOnly;

    public FloatBufferTensor(int ...shape) {
        Preconditions.checkArgument(shape != null && shape.length > 0);
        this.shape = shape;

        int c = 1;
        for (int i = 0; i < shape.length; i++)
            c *= shape[i];

        this.capacity = c;
        this.sliceCache = new FloatBufferTensor[shape[0]];
        this.b = FloatBuffer.allocate(capacity);
        this.isReadOnly = false;
    }

    public FloatBufferTensor(FloatBuffer b, int[] shape, boolean readOnly, boolean cacheSlices)
    {
        this.b = b;
        this.shape = shape;
        this.capacity = b.remaining();
        this.isReadOnly = readOnly;
        this.sliceCache = cacheSlices ? new FloatBufferTensor[shape[0]] : null;
    }

    @Override
    public int dims() {
        return shape.length;
    }
    @Override
    public int[] shape() {
        return shape;
    }
    @Override
    public int size() {
        return capacity;
    }

    @VisibleForTesting
    public int getOffset(int[] dims) {
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

    /**
     * Does inplace iteration based on the current values from innermost to outermost offset
     * Meaning if I pass [0,0,99] for a tensor of shape (100,100,100).  It will alter values to be
     * [0,1,0] and so on
     * @param cursor
     * @return false if cursor has hit its limit, otherwise true
     */
    public boolean iterate(int[] cursor) {
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

    static boolean isPowerOfTwo(int n)
    {
        double v = Math.log(n) / Math.log(2);
        return (int)(Math.ceil(v)) == (int)(Math.floor(v));
    }

    public FloatBufferTensor[] split(int numChunks, int dim) {
        FloatBufferTensor[] chunks = new FloatBufferTensor[numChunks];
        int innerLength = this.shape[dim] / numChunks;

        if (!isPowerOfTwo(innerLength)) {
            throw new IllegalStateException("Chunks must be power of 2");
        }

        int[] newShape = Arrays.copyOf(this.shape, shape.length);
        newShape[dim] = innerLength;
        int newCapacity = 1;
        for (int i = 0; i < newShape.length; i++) {
            newCapacity *= newShape[i];
        }

        for (int i = 0; i < numChunks; i++) {
            chunks[i] = new FloatBufferTensor(b.slice(i * newCapacity, newCapacity), newShape, isReadOnly, true);
        }

        return chunks;
    }

    public FloatBufferTensor transpose() {
        int[] tshape = new int[dims()];

        for (int i = 0; i < tshape.length; i++)
            tshape[i] = shape[shape.length - i - 1];

        FloatBufferTensor tt = new FloatBufferTensor(tshape);
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

    @Override
    public float get(int... dims) {
        Preconditions.checkArgument(dims.length <= shape.length, "Too many dimensions specified");
        Preconditions.checkArgument(dims.length == shape.length, "Must specify all dimensions");

        return b.get(getOffset(dims));
    }

    @Override
    public FloatBufferTensor slice(int... dims) {
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

        FloatBufferTensor r =  new FloatBufferTensor(b.slice(totalOffset, length), slicedShape, isReadOnly, false);
        if (dims.length == 1 && sliceCache != null)
            sliceCache[dims[0]] = r;

        return r;
    }

    public void set(float v, int ...dims) {
        Preconditions.checkArgument(dims.length <= shape.length, "Too many dimensions specified for tensor");
        Preconditions.checkArgument(dims.length == shape.length, "Must specify all dimensions");
        Preconditions.checkArgument(!b.isReadOnly(), "Can't modify a read only buffer");
        b.put(getOffset(dims), v);
    }

    @Override
    public float[] getFloatArray() {
        if (dims() > 1)
            throw new UnsupportedOperationException("dims must be 1");

        if (b.hasArray())
            return b.array();

        if (b.isReadOnly() || isReadOnly) {
            if (cachedRoBuffer == null) {
                cachedRoBuffer = new float[b.remaining()];
                b.duplicate().get(cachedRoBuffer);
            }
            return cachedRoBuffer;
        }

        float[] buf = new float[b.remaining()];
        b.duplicate().get(buf);
        return buf;
    }

    @Override
    public FloatVector getVector(int offset) {
        if (VectorMath.hasVectorAPI) {
            return FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, getFloatArray(), getArrayOffset() + offset);
        }

        throw new UnsupportedOperationException("No vector API available");
    }

    public int getArrayOffset()
    {
        return b.hasArray() ? b.arrayOffset() : 0;
    }

    public boolean hasArray() {
        return b.hasArray();
    }

    public Buffer getBuffer() {
        return this.b;
    }

    private void setOwnerCache(BufferCache cache) {
        this.originCache = cache;
    }

    @Override
    public void close() {
        if (originCache != null)
            originCache.release(this);
    }

    @Override
    public String toString() {
        float[] sample = new float[Math.min(10, b.remaining())];
        b.duplicate().get(sample);
        return "FloatBufferTensor{" +
                "shape=" + Arrays.toString(shape) +
                ", b=" + Arrays.toString(sample) +
                "...}";
    }

    /**
     * In LLMs a lot of buffers are used for inference.  Rather than allocating each one or using a fixed pool
     * this BufferCache allow a limited number of different shaped buffers to be shared across threads
     */
    public static class BufferCache {
        private class ShapeKey {
            final int[] shape;
            ShapeKey(int[] shape){
                this.shape = shape;
            }

            @Override
            public boolean equals(Object o) {
                if (this == o) return true;
                if (o == null || getClass() != o.getClass()) return false;
                ShapeKey shapeKey = (ShapeKey) o;
                return Arrays.equals(shape, shapeKey.shape);
            }

            @Override
            public int hashCode() {
                return Arrays.hashCode(shape);
            }
        }

        private final long bytesCapacity;
        private final AtomicLong currentBytes;

        private final ConcurrentMap<ShapeKey, MpmcUnboundedXaddArrayQueue<FloatBufferTensor>> availibleByShape;
        public BufferCache(long bytesCapacity) {
            this.bytesCapacity = bytesCapacity;
            this.currentBytes = new AtomicLong(0);
            this.availibleByShape = Maps.newConcurrentMap();
        }

        public FloatBufferTensor get(int ...shape) {
            MpmcUnboundedXaddArrayQueue<FloatBufferTensor> availableQueue = availibleByShape.computeIfAbsent(new ShapeKey(shape), s -> new MpmcUnboundedXaddArrayQueue<>(128));
            FloatBufferTensor t = availableQueue.poll();

            if (t != null)
                return t;

            t = new FloatBufferTensor(shape);

            //Assign to this cache or just over allocate
            if (currentBytes.addAndGet(t.capacity) < bytesCapacity) {
                t.setOwnerCache(this);
            } else {
                logger.debug("Full!");
                currentBytes.addAndGet(-t.capacity);
            }

            return t;
        }

        private void release(FloatBufferTensor b) {
            Arrays.fill(b.getFloatArray(), b.getArrayOffset(), b.getArrayOffset() + b.size(), 0);
            MpmcUnboundedXaddArrayQueue<FloatBufferTensor> availableQueue = availibleByShape.computeIfAbsent(new ShapeKey(b.shape), s -> new MpmcUnboundedXaddArrayQueue<>(128));
            availableQueue.offer(b);
        }

        public int numShapes()
        {
            return availibleByShape.size();
        }

    }
}
