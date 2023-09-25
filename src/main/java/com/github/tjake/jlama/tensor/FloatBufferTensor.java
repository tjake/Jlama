package com.github.tjake.jlama.tensor;

import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.tensor.operations.PanamaTensorOperations;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;
import com.google.common.base.Preconditions;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sun.nio.ch.DirectBuffer;

import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Arrays;

/**
 * A Tensor is a multidimensional array of floats.  It is backed by a FloatBuffer
 * and can be used to perform vector math operations on the data.
 *
 * The Tensor can be read only, or read/write.  If it is read only, it can be shared (model weights)
 *
 * The Tensor is backed by a FloatBuffer, but can be converted to a float[] for use with
 * other libraries.
 *
 * The Tensor is thread safe for read operations, but not for write operations.
 */
public final class FloatBufferTensor extends AbstractTensor<FloatVector, Float, float[]>
{
    private static final Logger logger = LoggerFactory.getLogger(FloatBufferTensor.class);
    private final FloatBuffer b;

    private final String name;
    private final MemorySegment segment;

    public FloatBufferTensor(AbstractTensor ft) {
        this(ft.shape);
        Preconditions.checkArgument(ft.dType != DType.I32, "This should never happen, likely a bug");

        int[] cursor = new int[ft.shape.length];
        do {
            set(ft.get(cursor), cursor);
        } while (ft.iterate(cursor));
    }

    public FloatBufferTensor(int ...shape) {
        super(DType.F32, shape, true);
        this.name = "tmp";
        if (TensorOperationsProvider.get().requiresOffHeapTensor()) {
            this.segment = Arena.global().allocate(MemoryLayout.sequenceLayout(capacity, ValueLayout.JAVA_FLOAT));
            this.b = segment.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        } else {
            this.b = FloatBuffer.allocate(capacity);
            this.segment = MemorySegment.ofBuffer(b);
        }
    }

    public FloatBufferTensor(FloatBuffer b, int[] shape, boolean cacheSlices) {
        this("none", b, shape, cacheSlices);
    }

    private FloatBufferTensor(String name, FloatBuffer b, int[] shape, boolean cacheSlices) {
        super(DType.F32, shape, cacheSlices);
        this.name = name;
        this.b = b;
        this.segment = MemorySegment.ofBuffer(b);
    }

    @Override
    protected AbstractTensor make(int... shape) {
        return new FloatBufferTensor(shape);
    }

    @Override
    protected AbstractTensor make(int offset, int length, int[] shape, boolean cacheSlices) {
        return new FloatBufferTensor(name, b.slice(offset, length), shape, cacheSlices);
    }

    @Override
    public float get(int... dims) {
        Preconditions.checkArgument(dims.length <= shape.length, "Too many dimensions specified");
        Preconditions.checkArgument(dims.length == shape.length, "Must specify all dimensions");
        return b.hasArray() ? b.array()[b.arrayOffset() + getOffset(dims)] : b.get(getOffset(dims));
    }

    @Override
    public void set(float v, int ...dims) {
        Preconditions.checkArgument(dims.length <= shape.length, "Too many dimensions specified for tensor");
        Preconditions.checkArgument(dims.length == shape.length, "Must specify all dimensions");
        Preconditions.checkArgument(!b.isReadOnly(), "Can't modify a read only buffer");
        b.put(getOffset(dims), v);
    }

    @Override
    public float[] getArray() {
        Preconditions.checkArgument(b.hasArray());
        return b.array();
    }

    public int getArrayOffset(int offset)
    {
        return (b.hasArray() ? b.arrayOffset() : 0) + offset;
    }


    @Override
    public MemorySegment getMemorySegment() {
        Preconditions.checkArgument(hasMemorySegment());
        return segment;
    }

    @Override
    public boolean hasMemorySegment() {
        return segment != null;
    }

    @Override
    public void copyFrom(AbstractTensor src, int srcOffset, int destOffset, int length) {
        Preconditions.checkArgument(this.dType == src.dType, "Different types");
        Preconditions.checkArgument(!b.isReadOnly());
        if (this.hasMemorySegment() && src.hasMemorySegment()) {
            segment.asSlice(getMemorySegmentOffset(destOffset), length * dType.size())
                    .copyFrom(src.getMemorySegment().asSlice(src.getMemorySegmentOffset(srcOffset), length * dType.size()));
        } else {
            System.arraycopy(src.getArray(), src.getArrayOffset(srcOffset), getArray(), this.getArrayOffset(destOffset), length);
        }
    }

    @Override
    public int getMemorySegmentOffset(int offset) {
        Preconditions.checkArgument(hasMemorySegment());
        return offset * Float.BYTES;
    }

    @Override
    public FloatVector getVector(VectorSpecies<Float> species, int offset) {
        if (!TensorOperationsProvider.get().requiresOffHeapTensor())
            return FloatVector.fromArray(species, getArray(), getArrayOffset(offset));
        else
            return FloatVector.fromMemorySegment(species, segment, getMemorySegmentOffset(offset), ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    public void intoTensor(FloatVector vector, int offset) {
        Preconditions.checkArgument(!b.isReadOnly());
        if (!TensorOperationsProvider.get().requiresOffHeapTensor())
            vector.intoArray(getArray(), getArrayOffset(offset));
        else
            vector.intoMemorySegment(segment, getMemorySegmentOffset(offset), ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    public void clear() {
        if (b.hasArray()) {
            Arrays.fill(b.array(), getArrayOffset(0), getArrayOffset(size()), 0);
        } else {
            segment.fill((byte) 0);
        }
    }

    @Override
    public String toString() {
        float[] sample = new float[Math.min(10, b.remaining())];
        b.duplicate().get(sample);
        return "FloatBufferTensor{" +
                "name='" + name + '\'' +
                " shape=" + Arrays.toString(shape) +
                ", b=" + Arrays.toString(sample) +
                "...}";
    }
}
