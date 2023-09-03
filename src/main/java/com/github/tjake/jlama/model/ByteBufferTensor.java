package com.github.tjake.jlama.model;

import com.github.tjake.jlama.safetensors.DType;
import com.google.common.base.Preconditions;
import jdk.incubator.vector.FloatVector;
import sun.nio.ch.DirectBuffer;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;

public class ByteBufferTensor extends AbstractTensor {

    public static final int BLOCK_SIZE = 32;
    private static final float I_BLOCK_SIZE = 1.0f / 32;

    final ByteBuffer b;
    final FloatBufferTensor blockF;
    private final String name;

    private final boolean mmapped;
    private final MemorySegment segment;

    public ByteBufferTensor(AbstractTensor ft) {
        this(ft.shape);
        Preconditions.checkArgument(ft.dType != DType.I8, "This should never happen, likely a bug");

        for (int i = 0; i < this.blockF.size(); i++) {
            float max = Float.MIN_VALUE;
            for (int j = 0; j < BLOCK_SIZE; j++) {
                float v = ft.get(i * BLOCK_SIZE + j);
                float absv = v > 0 ? v : -v;
                if (absv > max) max = absv;
            }

            float d = max / Byte.MAX_VALUE;
            float id = d != 0.0f ? 1.0f / d : d;
            this.blockF.set(d, i);

            for (int j = 0; j < BLOCK_SIZE; j++) {
                float f0 = ft.get(i * BLOCK_SIZE + j) * id; //scale
                this.b.put(i * BLOCK_SIZE + j, (byte) f0);
            }
        }
    }

    protected ByteBufferTensor(int[] shape) {
        super(DType.I8, shape, true);
        Preconditions.checkArgument(this.size() % BLOCK_SIZE == 0, "Tensor must be ");
        this.b = ByteBuffer.allocateDirect(this.size()).order(ByteOrder.LITTLE_ENDIAN);
        this.blockF = new FloatBufferTensor(this.size() / BLOCK_SIZE);
        this.name = "tmp";
        this.mmapped = false;
        this.segment = MemorySegment.ofAddress(((DirectBuffer)b).address() + b.position(), (long) size() * dType().size());
    }

    public ByteBufferTensor(String name, ByteBuffer b, int[] shape, boolean cacheSlices, boolean mmapped) {
        super(DType.I8, shape, cacheSlices);
        Preconditions.checkArgument(b.isDirect(), "Must use direct buffers");
        this.name = name;
        this.b = b;
        this.blockF = new FloatBufferTensor(this.size() / BLOCK_SIZE);
        this.mmapped = mmapped;
        this.segment = MemorySegment.ofAddress(((DirectBuffer)b).address() + b.position(), (long) size() * Short.BYTES);
    }

    public ByteBufferTensor(ByteBuffer b, int[] shape, boolean cacheSlices, boolean mmapped) {
        this("none", b, shape, cacheSlices, mmapped);
    }

    @Override
    protected AbstractTensor make(int... shape) {
        return new ByteBufferTensor(shape);
    }

    @Override
    protected AbstractTensor make(int offset, int length, int[] shape, boolean cacheSlices) {
        return new ByteBufferTensor(name, b.slice(offset, length), shape, cacheSlices, mmapped);
    }

    @Override
    public float get(int... dims) {
        Preconditions.checkArgument(dims.length <= shape.length, "Too many dimensions specified");
        Preconditions.checkArgument(dims.length == shape.length, "Must specify all dimensions");

        int i = getOffset(dims);
        float d = getFactorForIndex(i);
        return b.get(i) * d;
    }

    public final float getFactorForIndex(int i) {
        float f = (i * I_BLOCK_SIZE);
        int ix = (int)(i * I_BLOCK_SIZE);
        if (ix >= blockF.size())
            throw new RuntimeException();
        return blockF.get(ix);
    }

    @Override
    public void set(float v, int... dims) {
        Preconditions.checkArgument(dims.length <= shape.length, "Too many dimensions specified for tensor");
        Preconditions.checkArgument(dims.length == shape.length, "Must specify all dimensions");
        Preconditions.checkArgument(!b.isReadOnly() && !mmapped, "Can't modify a read only buffer");
        int i = getOffset(dims);
        float d = blockF.get((int)(i * I_BLOCK_SIZE));
        float max = d * Byte.MAX_VALUE;
        if (v <= max) {
            float id = d != 0.0f ? 1.0f / d : d;
            b.put(i, (byte)(v * id));
        }
    }

    @Override
    public float[] getFloatArray() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int getArrayOffset() {
        return 0;
    }

    @Override
    public FloatVector getFloatVector(int offset) {
        return null;
    }

    @Override
    public MemorySegment getMemorySegment() {
        return segment;
    }

    @Override
    public int getMemorySegmentOffset(int offset) {
        return offset;
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
    public void update(float[] data, int... offset) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void clear() {
        Preconditions.checkArgument(!mmapped, "Can't clear a read-only buffer");
        segment.fill((byte)0);
    }

    @Override
    public void scale(float factor, int offset, int length) {
        throw new UnsupportedOperationException();
    }

    @Override
    public String toString() {
        byte[] sample = new byte[Math.min(10, b.remaining())];
        b.duplicate().get(sample);
        return "ByteBufferTensor{" +
                "name='" + name + '\'' +
                "shape=" + Arrays.toString(shape) +
                ", b=" + Arrays.toString(sample) +
                "...}";
    }
}
