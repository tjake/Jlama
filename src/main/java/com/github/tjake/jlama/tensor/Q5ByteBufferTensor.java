package com.github.tjake.jlama.tensor;

import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;
import com.google.common.base.Preconditions;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sun.nio.ch.DirectBuffer;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Q5ByteBufferTensor extends AbstractTensor<ByteVector, Byte, byte[]> {
    private static final Logger logger = LoggerFactory.getLogger(Q5ByteBufferTensor.class);
    public static final int BLOCK_SIZE = 32;
    private static final float I_BLOCK_SIZE = 1.0f / BLOCK_SIZE;

    final ByteBuffer b;
    final FloatBufferTensor blockF; //Deltas
    final int[] b5; //Fith bit of quants
    private final String name;
    private final MemorySegment segment;

    public Q5ByteBufferTensor(AbstractTensor ft) {
        this(ft.shape);
        Preconditions.checkArgument(ft.dType != DType.Q5, "This should never happen, likely a bug");
        Preconditions.checkArgument(ft.size() % BLOCK_SIZE == 0, "I8 buffer must be a multiple of BLOCK_SIZE");

        List<int[]> startBlockCursors = new ArrayList<>();
        int[] cursor = new int[ft.shape.length];
        int c = 0;
        do {
            if (c++ % BLOCK_SIZE == 0) {
                startBlockCursors.add(Arrays.copyOf(cursor, cursor.length));
            }
        } while (ft.iterate(cursor));

        //Process each block in parallel
        VectorMath.pfor(0, startBlockCursors.size(), (i) -> {
            int[] blockStartCursor = startBlockCursors.get(i);
            processBlock(ft, blockStartCursor);
        });
    }

    void processBlock(AbstractTensor ft, int[] blockStartCursor) {
        int[] cursor = Arrays.copyOf(blockStartCursor, blockStartCursor.length);
        float max = Float.MIN_VALUE;
        float amax = Float.MIN_VALUE;

        //Accumulate the max value for this block
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float v = ft.get(cursor);
            float absv = v < 0 ? -v : v;
            if (absv > amax) {
                max = v;
                amax = absv;
            }
            ft.iterate(cursor);
        }

        // Process the block and save it
        float scale = max / -16f;
        float iscale = scale != 0.0f ? 1.0f / scale : 0.0f;
        this.blockF.set(scale, makeBlockShape(blockStartCursor));
        int i = ft.getOffset(blockStartCursor);
        int q = 0;

        // Reset the cursor
        cursor = Arrays.copyOf(blockStartCursor, blockStartCursor.length);
        for (int j = 0;  j < BLOCK_SIZE / 2; j++, i+=2) {
            float f0 = ft.get(cursor) * iscale;
            ft.iterate(cursor);
            float f1 = ft.get(cursor) * iscale;
            ft.iterate(cursor);

            short fb0 =  (byte) Math.min(31, (byte)(f0 + 16.5f));
            short fb1 =  (byte) Math.min(31, (byte)(f1 + 16.5f));

            this.b.put(i/2, (byte) ((fb0 & 0x0F) | ((fb1 & 0x0F) << 4)));

            // 5th bit of each quant placed deterministically
            q |= ((fb0 & 0x10) >>> 4) << (j + 0);
            q |= ((fb1 & 0x10) >>> 4) << (j + BLOCK_SIZE/2);
        }

        this.b5[getOffset(makeBlockShape(blockStartCursor))] = q;
    }


    private static int[] makeBlockShape(int[] shape) {
        int[] blockShape = new int[shape.length];
        for (int i = 0; i < shape.length; i++) {
            if (i == shape.length - 1)
                blockShape[i] = shape[i] / BLOCK_SIZE;
            else
                blockShape[i] = shape[i];
        }

        return blockShape;
    }


    protected Q5ByteBufferTensor(int[] shape) {
        super(DType.Q5, shape, true);
        Preconditions.checkArgument(this.size() % BLOCK_SIZE == 0, "Tensor must be a multiple of BLOCK_SIZE");
        this.blockF = new FloatBufferTensor(makeBlockShape(shape));
        this.b5 = new int[Arrays.stream(makeBlockShape(shape)).sum()];
        this.name = "tmp";

        if (TensorOperationsProvider.get().requiresOffHeapTensor()) {
            this.b = ByteBuffer.allocateDirect(this.size() / 2).order(ByteOrder.LITTLE_ENDIAN);
            this.segment = MemorySegment.ofBuffer(b);
        } else {
            this.b = ByteBuffer.allocate(this.size() / 2).order(ByteOrder.LITTLE_ENDIAN);
            this.segment = MemorySegment.ofBuffer(b);
        }
    }

    public Q5ByteBufferTensor(String name, ByteBuffer b, FloatBufferTensor blockF, int[] b5, int[] shape, boolean cacheSlices) {
        super(DType.Q5, shape, cacheSlices);
        Preconditions.checkArgument(b.isDirect(), "Must use direct buffers");
        this.name = name;
        this.b = b;
        this.blockF = blockF;
        this.b5 = b5;
        this.segment = MemorySegment.ofBuffer(b);
    }

    @Override
    protected AbstractTensor make(int... shape) {
        return new Q5ByteBufferTensor(shape);
    }

    @Override
    protected AbstractTensor make(int offset, int length, int[] shape, boolean cacheSlices) {
        FloatBufferTensor newBlockF = (FloatBufferTensor) this.blockF.make((int)(offset * I_BLOCK_SIZE), (int)(length * I_BLOCK_SIZE), makeBlockShape(shape), cacheSlices);
        return new Q5ByteBufferTensor(name, b.slice(offset, length), newBlockF, b5, shape, cacheSlices);
    }

    @Override
    public float get(int... dims) {
        Preconditions.checkArgument(dims.length <= shape.length, "Too many dimensions specified");
        Preconditions.checkArgument(dims.length == shape.length, "Must specify all dimensions");
        int i = getOffset(dims);
        float scale = blockF.get(makeBlockShape(dims));
        int q = b5[getOffset(makeBlockShape(dims))];
        byte b0 = this.b.get(i/2);
        int j = i % BLOCK_SIZE / 2;

        byte xh;
        int x;
        if (i % 2 == 0) {
            xh = (byte) (((q >> (j + 0)) << 4) & 0x10);
            x = ((b0 & 0x0F) | xh) - 16;
        } else {
            xh = (byte) (((q >> (j + 12))     ) & 0x10); // j + 12 == j + BLOCK_SIZE/2 << 4
            x = ((b0 >> 4 & 0x0F) | xh) - 16;
        }

        return x * scale;
    }

    public final float getFactorForIndex(int i) {
        int ix = (int)(i * I_BLOCK_SIZE);
        if (ix >= blockF.size())
            throw new RuntimeException();
        return blockF.get(ix);
    }

    @Override
    public void set(float v, int... dims) {
        throw new UnsupportedOperationException();
    }

    @Override
    public byte[] getArray() {
        if (b.hasArray())
            return b.array();
        else
            throw new UnsupportedOperationException();
    }

    @Override
    public int getArrayOffset(int i) {
        return b.arrayOffset() + i;
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
    public ByteVector getVector(VectorSpecies<Byte> species, int offset) {
        if (!TensorOperationsProvider.get().requiresOffHeapTensor())
            return ByteVector.fromArray(species, getArray(), getArrayOffset(offset));
        else
            return ByteVector.fromMemorySegment(species, segment, getMemorySegmentOffset(offset), ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    public void intoTensor(ByteVector vector, int offset) {
        Preconditions.checkArgument(!b.isReadOnly());
        if (!TensorOperationsProvider.get().requiresOffHeapTensor())
            vector.intoArray(getArray(), getArrayOffset(offset));
        else
            vector.intoMemorySegment(segment, getMemorySegmentOffset(offset), ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    public void clear() {
        Preconditions.checkArgument(!b.isReadOnly(), "Can't clear a read-only buffer");
        segment.fill((byte)0);
    }

    @Override
    public String toString() {
        byte[] sample = new byte[Math.min(10, b.remaining())];
        b.duplicate().get(sample);
        return "Q5BufferTensor{" +
                "name='" + name + '\'' +
                "shape=" + Arrays.toString(shape) +
                ", b=" + Arrays.toString(sample) +
                "...}";
    }
}
