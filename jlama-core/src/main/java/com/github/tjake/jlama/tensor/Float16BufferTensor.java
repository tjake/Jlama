package com.github.tjake.jlama.tensor;

import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;
import com.google.common.base.Preconditions;

import com.github.tjake.jlama.util.UnsafeDirectByteBuffer;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorSpecies;

import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.ShortBuffer;
import java.util.Arrays;

public class Float16BufferTensor extends AbstractTensor<ShortVector, Short, short[]> {
    private final ShortBuffer b;
    private final String name;
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
        if (TensorOperationsProvider.get().requiresOffHeapTensor()) {
            this.b = UnsafeDirectByteBuffer.allocateAlignedByteBuffer(capacity * dType().size(), UnsafeDirectByteBuffer.CACHE_LINE_SIZE).asShortBuffer();
        } else {
            this.b = ShortBuffer.allocate(capacity);
        }
        this.segment = MemorySegment.ofBuffer(b);
    }

    public Float16BufferTensor(ShortBuffer b, int[] shape, boolean cacheSlices) {
        this("none", b, shape, cacheSlices);
    }

    public Float16BufferTensor(String name, ShortBuffer b, int[] shape, boolean cacheSlices) {
        super(DType.F16, shape, cacheSlices);
        Preconditions.checkArgument(b.isDirect(), "Must use direct buffers");
        this.name = name;
        this.b = b;
        this.segment = MemorySegment.ofBuffer(b);
    }

    @Override
    protected AbstractTensor make(int... shape) {
        return new Float16BufferTensor(shape);
    }

    @Override
    protected AbstractTensor make(int offset, int length, int[] shape, boolean cacheSlices) {
        return new Float16BufferTensor(name, b.slice(offset, length), shape, cacheSlices);
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
        Preconditions.checkArgument(!b.isReadOnly(), "Can't modify a read only buffer");
        b.put(getOffset(dims), Float.floatToFloat16(v));
    }

    @Override
    public short[] getArray() {
        if (b.hasArray())
            return b.array();
        else
            throw new UnsupportedOperationException("Can't get array from direct buffer");
    }

    @Override
    public int getArrayOffset(int offset) {
        return b.arrayOffset() + offset;
    }

    @Override
    public ShortVector getVector(VectorSpecies<Short> species, int offset) {
        if (!TensorOperationsProvider.get().requiresOffHeapTensor())
            return ShortVector.fromArray(species, getArray(), getArrayOffset(offset));
        else
            return ShortVector.fromMemorySegment(species, segment, getMemorySegmentOffset(offset), ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    public void intoTensor(ShortVector vector, int offset) {
        Preconditions.checkArgument(!b.isReadOnly());
        if (!TensorOperationsProvider.get().requiresOffHeapTensor())
            vector.intoArray(getArray(), getArrayOffset(offset));
        else
            vector.intoMemorySegment(segment, getMemorySegmentOffset(offset), ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    public MemorySegment getMemorySegment() {
        return segment;
    }

    @Override
    public int getMemorySegmentOffset(int offset) {
        return offset * dType.size();
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

    @Override
    public void clear() {
        Preconditions.checkArgument(!b.isReadOnly(), "Can't clear a read-only buffer");
        segment.fill((byte)0);
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
