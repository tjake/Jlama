package com.github.tjake.jlama.tensor;

import com.github.tjake.jlama.safetensors.DType;
import com.google.common.base.Preconditions;
import jdk.incubator.vector.FloatVector;
import sun.nio.ch.DirectBuffer;

import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.ShortBuffer;
import java.util.Arrays;

public class Float16BufferTensor extends AbstractTensor {

    private final ShortBuffer b;

    private final String name;

    private final boolean mmapped;
    private final MemorySegment segment;

    public Float16BufferTensor(AbstractTensor ft) {
        this(ft.shape);
        Preconditions.checkArgument(ft.dType != DType.F16, "This should never happen, likely a bug");

        int[] cursor = new int[ft.shape.length];
        do {
            set(ft.get(cursor), cursor);
        } while (ft.iterate(cursor));
    }

    public Float16BufferTensor(int ...shape) {
        super(DType.F16, shape, true);
        this.name = "tmp";
        this.segment = Arena.global().allocate(MemoryLayout.sequenceLayout(capacity, ValueLayout.JAVA_SHORT));
        this.mmapped = false;
        this.b = this.segment.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asShortBuffer();
    }

    public Float16BufferTensor(ShortBuffer b, int[] shape, boolean cacheSlices, boolean mmapped) {
        this("none", b, shape, cacheSlices, mmapped);
    }

    private Float16BufferTensor(String name, ShortBuffer b, int[] shape, boolean cacheSlices, boolean mmapped) {
        super(DType.F16, shape, cacheSlices);
        Preconditions.checkArgument(b.isDirect(), "Must use direct buffers");
        this.name = name;
        this.b = b;
        this.mmapped = mmapped;
        this.segment = MemorySegment.ofBuffer(b);
    }

    @Override
    protected AbstractTensor make(int... shape) {
        return new Float16BufferTensor(shape);
    }

    @Override
    protected AbstractTensor make(int offset, int length, int[] shape, boolean cacheSlices) {
        return new Float16BufferTensor(name, b.slice(offset, length), shape, cacheSlices, mmapped);
    }

    @Override
    public float get(int... dims) {
        Preconditions.checkArgument(dims.length <= shape.length, "Too many dimensions specified");
        Preconditions.checkArgument(dims.length == shape.length, "Must specify all dimensions");
        return Float.float16ToFloat(b.get(getOffset(dims)));
    }

    @Override
    public void set(float v, int ...dims) {
        Preconditions.checkArgument(dims.length <= shape.length, "Too many dimensions specified for tensor");
        Preconditions.checkArgument(dims.length == shape.length, "Must specify all dimensions");
        Preconditions.checkArgument(!b.isReadOnly() && !mmapped, "Can't modify a read only buffer");
        b.put(getOffset(dims), Float.floatToFloat16(v));
    }

    @Override
    public float[] getFloatArray() {
        throw new UnsupportedOperationException("Not implemented");
    }

    public int getArrayOffset() {
        return b.arrayOffset();
    }

    @Override
    public FloatVector getFloatVector(int offset) {
        throw new UnsupportedOperationException();
    }

    @Override
    public MemorySegment getMemorySegment() {
        return segment;
    }

    @Override
    public int getMemorySegmentOffset(int offset) {
        return offset * Short.BYTES;
    }

    @Override
    public boolean hasMemorySegment() {
        return true;
    }

    @Override
    public void copyFrom(AbstractTensor src, int srcOffset, int destOffset, int length) {
        Preconditions.checkArgument(this.dType == src.dType, "different types");
        Preconditions.checkArgument(!b.isReadOnly(), "Read-only");
        segment.asSlice(getMemorySegmentOffset(destOffset), length)
                .copyFrom(src.getMemorySegment().asSlice(src.getMemorySegmentOffset(srcOffset), length));
    }

    /** Since getFloatArray() returns the actual backing array, we can just ignore this call */
    @Override
    public void update(float[] data, int... offset) {
        Preconditions.checkArgument(!b.isReadOnly(), "Can't modify a read only buffer");

        int noff = getOffset(offset);
        for (int i = 0; i < data.length; i++) {
            b.put(noff + i, Float.floatToFloat16(data[i]));
        }
    }

    @Override
    public void clear() {
        Preconditions.checkArgument(!mmapped, "Can't clear a read-only buffer");
        segment.fill((byte)0);
    }

    @Override
    public void scale(float factor, int offset, int length) {
        Preconditions.checkArgument(length % 8 == 0);
        for (int i = offset; i < length; i++)
            this.set(this.get(i) * factor, i);
    }

    @Override
    public String toString() {
        short[] sample = new short[Math.min(10, b.remaining())];
        b.duplicate().get(sample);
        return "Float16BufferTensor{" +
                "name='" + name + '\'' +
                "shape=" + Arrays.toString(shape) +
                ", b=" + Arrays.toString(sample) +
                "...}";
    }
}
