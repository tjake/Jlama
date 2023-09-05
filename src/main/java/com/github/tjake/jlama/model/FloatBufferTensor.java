package com.github.tjake.jlama.model;

import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.safetensors.DType;
import com.google.common.base.Preconditions;
import jdk.incubator.vector.FloatVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sun.nio.ch.DirectBuffer;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
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
public class FloatBufferTensor extends AbstractTensor {
    private static final Logger logger = LoggerFactory.getLogger(FloatBufferTensor.class);
    private final FloatBuffer b;

    private final String name;

    private final boolean mmapped;
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
        this.b = ByteBuffer.allocateDirect(capacity * dType().size()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        this.mmapped = false;
        this.segment = MemorySegment.ofAddress(((DirectBuffer)b).address() + b.position(), (long) size() * dType().size());;
    }

    public FloatBufferTensor(FloatBuffer b, int[] shape, boolean cacheSlices, boolean mmapped) {
        this("none", b, shape, cacheSlices, mmapped);
    }

    public FloatBufferTensor(String name, FloatBuffer b, int[] shape, boolean cacheSlices, boolean mmapped) {
        super(DType.F32, shape, cacheSlices);
        this.name = name;
        this.b = b;
        this.mmapped = mmapped;
        this.segment = b.isDirect() ? MemorySegment.ofAddress(((DirectBuffer)b).address() + b.position(), (long) size() * dType().size()) : null;
    }

    @Override
    protected AbstractTensor make(int... shape) {
        return new FloatBufferTensor(shape);
    }

    @Override
    protected AbstractTensor make(int offset, int length, int[] shape, boolean cacheSlices) {
        return new FloatBufferTensor(name, b.slice(offset, length), shape, cacheSlices, mmapped);
    }

    @Override
    public float get(int... dims) {
        Preconditions.checkArgument(dims.length <= shape.length, "Too many dimensions specified");
        Preconditions.checkArgument(dims.length == shape.length, "Must specify all dimensions");
        return b.get(getOffset(dims));
    }

    @Override
    public void set(float v, int ...dims) {
        Preconditions.checkArgument(dims.length <= shape.length, "Too many dimensions specified for tensor");
        Preconditions.checkArgument(dims.length == shape.length, "Must specify all dimensions");
        Preconditions.checkArgument(!b.isReadOnly() && !mmapped, "Can't modify a read only buffer");
        b.put(getOffset(dims), v);
    }

    @Override
    public float[] getFloatArray() {
        if (shape.length > 1)
            throw new UnsupportedOperationException("dims must be 1");

        if (!mmapped && b.hasArray())
            return b.array();

        float[] buf = new float[b.remaining()];
        b.duplicate().get(buf);
        return buf;
    }

    public int getArrayOffset()
    {
        return b.hasArray() ? b.arrayOffset() : 0;
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
        if (this.hasMemorySegment() && src.hasMemorySegment())
            segment.asSlice(getMemorySegmentOffset(destOffset), length * dType.size())
                    .copyFrom(src.getMemorySegment().asSlice(src.getMemorySegmentOffset(srcOffset), length * dType.size()));
        else
            System.arraycopy(src.getFloatArray(), src.getArrayOffset() + srcOffset, getFloatArray(), this.getArrayOffset() + destOffset, length);
    }

    @Override
    public int getMemorySegmentOffset(int offset) {
        Preconditions.checkArgument(hasMemorySegment());
        return offset * Float.BYTES;
    }

    @Override
    public FloatVector getFloatVector(int offset) {
        if (VectorMath.hasVectorAPI) {
            if (segment == null)
                return FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, getFloatArray(), getArrayOffset() + offset);
            else
                return FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, segment, (long) offset * dType.size(), ByteOrder.LITTLE_ENDIAN);
        }

        throw new UnsupportedOperationException("No vector API available");
    }


    /** Since getFloatArray() returns the actual backing array, we can just ignore this call */
    @Override
    public void update(float[] data, int... offset) {
        Preconditions.checkArgument(!b.isReadOnly() && b.hasArray() && !mmapped, "Can't modify a read only buffer");
    }

    @Override
    public void clear() {
        if (b.hasArray()) {
            Arrays.fill(b.array(), getArrayOffset(), getArrayOffset() + size(), 0);
        } else {
            segment.fill((byte) 0);
        }
    }

    @Override
    public void scale(float factor, int offset, int length) {
        int limit = offset + length;
        for (int i = offset; i < limit; i++)
            this.set(this.get(i) * factor, i);
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
