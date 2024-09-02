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
import com.github.tjake.jlama.util.UnsafeDirectByteBuffer;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.VectorSpecies;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class Q4ByteBufferTensor extends AbstractTensor<ByteVector, Byte> {
    private static final Logger logger = LoggerFactory.getLogger(Q4ByteBufferTensor.class);;
    public static final int BLOCK_SIZE = 32;
    public static final int HALF_BLOCK = (BLOCK_SIZE / 2);
    private static final float I_BLOCK_SIZE = 1.0f / BLOCK_SIZE;

    final ByteBuffer b;
    final FloatBufferTensor blockF; // Deltas
    private final String name;
    private final MemorySegment segment;

    public Q4ByteBufferTensor(AbstractTensor ft) {
        this(ft.shape);
        Preconditions.checkArgument(ft.dType != DType.Q4, "This should never happen, likely a bug");
        Preconditions.checkArgument(ft.size() % BLOCK_SIZE == 0, "I8 buffer must be a multiple of BLOCK_SIZE");

        List<int[]> startBlockCursors = new ArrayList<>();
        int[] cursor = new int[ft.shape.dims()];
        int c = 0;
        do {
            if (c++ % BLOCK_SIZE == 0) {
                startBlockCursors.add(Arrays.copyOf(cursor, cursor.length));
            }
        } while (ft.iterate(cursor));

        // Process each block in parallel
        IntStream.range(0, startBlockCursors.size()).parallel().forEach((i) -> {
            int[] blockStartCursor = startBlockCursors.get(i);
            processBlock(ft, blockStartCursor);
        });
    }

    void processBlock(AbstractTensor ft, int[] blockStartCursor) {
        int[] cursor = Arrays.copyOf(blockStartCursor, blockStartCursor.length);
        float max = Float.MIN_VALUE;
        float amax = Float.MIN_VALUE;

        // Accumulate the max value for this block
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
        float scale = max / -8f;
        float iscale = scale != 0.0f ? 1.0f / scale : 0.0f;
        this.blockF.set(scale, makeBlockShape(blockStartCursor));
        int i = ft.getOffset(blockStartCursor);

        int ibyte = i / 2;
        for (int j = 0; j < BLOCK_SIZE / 2; j++, i++, ibyte++) {
            float f0 = ft.get(blockStartCursor) * iscale;

            // Since the iterator doesn't work for Q4, we need to manually increment the cursor
            // to get the next value since the byte is packed with [0,15] position.
            // we do this because on the simd side we need to load 2 values at a time
            // and want to keep the order of the values in the byte the same as the original layout
            blockStartCursor[blockStartCursor.length - 1] += HALF_BLOCK;
            float f1 = ft.get(blockStartCursor) * iscale;

            // Reset the cursor back to previous position
            blockStartCursor[blockStartCursor.length - 1] -= HALF_BLOCK;
            ft.iterate(blockStartCursor);

            byte fb0 = (byte) Math.min(15, (byte) (f0 + 8.5f));
            byte fb1 = (byte) Math.min(15, (byte) (f1 + 8.5f));

            this.b.put(ibyte, (byte) ((fb0) | ((fb1) << 4)));

            /*
            //DEBUG
            byte b0 = this.b.get(ibyte);
            int x0 = (b0 & 0x0F) - 8;
            int x1 = (b0 >> 4 & 0x0F) - 8;
            
            float f11 = x0 * scale;
            float f2 = x1 * scale;
            logger.info("i:{} ibyte:{} f1={} | i:{} f2={} | c={}",i,ibyte, x0, i+HALF_BLOCK, x1, b0);
            */

        }
    }

    static int[] makeBlockShape(int... shape) {
        int[] blockShape = new int[shape.length];
        for (int i = 0; i < shape.length - 1; i++) {
            blockShape[i] = shape[i];
        }

        blockShape[shape.length - 1] = (int) (shape[shape.length - 1] * I_BLOCK_SIZE);

        return blockShape;
    }

    static TensorShape makeBlockShape(TensorShape shape) {
        return shape.scaleLastDim(I_BLOCK_SIZE);
    }

    protected Q4ByteBufferTensor(TensorShape shape) {
        super(DType.Q4, shape, true);
        Preconditions.checkArgument(this.size() % BLOCK_SIZE == 0, "Tensor must be a multiple of BLOCK_SIZE");
        this.blockF = new FloatBufferTensor(makeBlockShape(shape));
        this.name = "tmp";
        this.b = UnsafeDirectByteBuffer.allocateAlignedByteBuffer(Ints.checkedCast(this.size() / 2), UnsafeDirectByteBuffer.CACHE_LINE_SIZE)
            .order(ByteOrder.LITTLE_ENDIAN);

        this.segment = MemorySegment.ofBuffer(b);
    }

    public Q4ByteBufferTensor(String name, ByteBuffer b, FloatBufferTensor blockF, TensorShape shape, boolean cacheSlices) {
        super(DType.Q4, shape, cacheSlices);
        this.blockF = blockF;
        this.name = name;
        if (b.isDirect()) {
            this.b = b;
        } else {
            this.b = ByteBuffer.allocateDirect(b.remaining()).order(ByteOrder.LITTLE_ENDIAN);
            this.b.duplicate().put(b);
        }

        this.segment = MemorySegment.ofBuffer(this.b);
    }

    @Override
    protected AbstractTensor make(TensorShape shape) {
        return new Q4ByteBufferTensor(shape);
    }

    @Override
    protected AbstractTensor make(int offset, int length, TensorShape shape, boolean cacheSlices) {
        FloatBufferTensor newBlockF = (FloatBufferTensor) this.blockF.make(
            (int) (offset * I_BLOCK_SIZE),
            (int) (length * I_BLOCK_SIZE),
            makeBlockShape(shape),
            cacheSlices
        );
        return new Q4ByteBufferTensor(name, b.slice(offset / 2, length / 2), newBlockF, shape, cacheSlices);
    }

    @Override
    public float get(int... dims) {
        Preconditions.checkArgument(dims.length <= shape.dims(), "Too many dimensions specified");
        Preconditions.checkArgument(dims.length == shape.dims(), "Must specify all dimensions");
        int i = getOffset(dims);
        float scale = blockF.get(makeBlockShape(dims));

        // Represents the offset in the q4 byte array
        int ibyte = ((int) (i * I_BLOCK_SIZE)) * HALF_BLOCK + (i % BLOCK_SIZE);

        int x;
        if (i % BLOCK_SIZE < HALF_BLOCK) {
            byte b0 = this.b.get(ibyte);
            x = (b0 & 0x0F) - 8;
        } else {
            byte b0 = this.b.get(ibyte - HALF_BLOCK);
            x = (b0 >> 4 & 0x0F) - 8;
        }
        return x * scale;
    }

    public float getFactorForIndex(int d, int i) {
        int ix = (int) (i * I_BLOCK_SIZE);
        return blockF.get(d, ix);
    }

    public FloatBufferTensor getBlockF() {
        return blockF;
    }

    @Override
    public void set(float v, int... dims) {
        throw new UnsupportedOperationException();
    }

    @Override
    public ByteVector getVector(VectorSpecies<Byte> species, int... voffset) {
        int offset = getOffset(voffset);
        return ByteVector.fromMemorySegment(species, segment, getMemorySegmentOffset(offset), ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    public void intoTensor(ByteVector vector, int... aoffset) {
        Preconditions.checkArgument(!b.isReadOnly());
        int offset = getOffset(aoffset);
        vector.intoMemorySegment(segment, getMemorySegmentOffset(offset), ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    public MemorySegment getMemorySegment() {
        return segment;
    }

    @Override
    public int getMemorySegmentOffset(int offset) {
        return offset / 2;
    }

    @Override
    public void copyFrom(AbstractTensor src, int srcOffset, int destOffset, int length) {
        Preconditions.checkArgument(this.dType == src.dType, "different types");
        Preconditions.checkArgument(!b.isReadOnly(), "Read-only");
        segment.asSlice(getMemorySegmentOffset(destOffset), length / 2)
            .copyFrom(src.getMemorySegment().asSlice(src.getMemorySegmentOffset(srcOffset), length / 2));

        Q4ByteBufferTensor srcQ4 = (Q4ByteBufferTensor) src;
        blockF.copyFrom(srcQ4.blockF, srcOffset / BLOCK_SIZE, destOffset / BLOCK_SIZE, length / BLOCK_SIZE);
    }

    @Override
    public void clear() {
        Preconditions.checkArgument(!b.isReadOnly(), "Can't clear a read-only buffer");
        segment.fill((byte) 0);
    }

    @Override
    public String toString() {
        byte[] sample = new byte[Math.min(BLOCK_SIZE, b.remaining())];
        b.duplicate().get(sample);
        return "Q4BufferTensor{" + "name='" + name + '\'' + "shape=" + shape + ", b=" + Arrays.toString(sample) + "...}";
    }
}
