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

import static com.github.tjake.jlama.tensor.Q4ByteBufferTensor.makeBlockShape;

import com.github.tjake.jlama.math.VectorMath;
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
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorSpecies;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Q8ByteBufferTensor extends AbstractTensor<ByteVector, Byte> {
    private static final Logger logger = LoggerFactory.getLogger(Q8ByteBufferTensor.class);;
    public static final int BLOCK_SIZE = 32;
    public static final float I_BLOCK_SIZE = 1.0f / BLOCK_SIZE;

    final ByteBuffer b;
    final FloatBufferTensor blockF;
    private final String name;
    private final MemorySegment segment;

    public Q8ByteBufferTensor(AbstractTensor ft) {
        this(ft.shape);
        Preconditions.checkArgument(ft.dType != DType.I8, "This should never happen, likely a bug");
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
        VectorMath.pfor(0, startBlockCursors.size(), (i) -> {
            int[] blockStartCursor = startBlockCursors.get(i);
            processBlock(ft, blockStartCursor);
        });
    }

    void processBlock(AbstractTensor ft, int[] blockStartCursor) {
        int[] cursor = Arrays.copyOf(blockStartCursor, blockStartCursor.length);
        float max = Float.MIN_VALUE;

        // Accumulate the max value for this block
        for (int i = 0; i < BLOCK_SIZE; i++) {
            float v = ft.get(cursor);
            float absv = v < 0 ? -v : v;
            if (absv > max) max = absv;
            ft.iterate(cursor);
        }

        // Process the block and save it
        float iscale = 127f / max;
        float scale = iscale != 0.0f ? 1.0f / iscale : 0.0f;
        this.blockF.set(scale, makeBlockShape(blockStartCursor));
        int i = ft.getOffset(blockStartCursor);
        for (int j = 0; j < BLOCK_SIZE; j++, i++) {
            float f0 = ft.get(blockStartCursor) * iscale;
            this.b.put(i, (byte) Math.round(f0));
            ft.iterate(blockStartCursor);
        }
    }

    public Q8ByteBufferTensor(int... shape) {
        this(TensorShape.of(shape));
    }

    public Q8ByteBufferTensor(TensorShape shape) {
        super(DType.I8, shape, true);
        Preconditions.checkArgument(this.size() % BLOCK_SIZE == 0, "Tensor must be a multiple of BLOCK_SIZE");
        this.blockF = new FloatBufferTensor(makeBlockShape(shape));
        this.name = "tmp";

        this.b = UnsafeDirectByteBuffer.allocateAlignedByteBuffer(Ints.checkedCast(size()), UnsafeDirectByteBuffer.CACHE_LINE_SIZE)
            .order(ByteOrder.LITTLE_ENDIAN);

        this.segment = MemorySegment.ofBuffer(b);
    }

    public Q8ByteBufferTensor(String name, ByteBuffer b, FloatBufferTensor blockF, TensorShape shape, boolean cacheSlices) {
        super(DType.I8, shape, cacheSlices);
        this.name = name;
        this.blockF = blockF;
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
        return new Q8ByteBufferTensor(shape);
    }

    @Override
    protected AbstractTensor make(int offset, int length, TensorShape shape, boolean cacheSlices) {
        FloatBufferTensor newBlockF = (FloatBufferTensor) this.blockF.make(
            (int) (offset * I_BLOCK_SIZE),
            (int) (length * I_BLOCK_SIZE),
            makeBlockShape(shape),
            cacheSlices
        );
        return new Q8ByteBufferTensor(name, b.slice(offset, length), newBlockF, shape, cacheSlices);
    }

    @Override
    public float get(int... dims) {
        Preconditions.checkArgument(dims.length <= shape.dims(), "Too many dimensions specified");
        Preconditions.checkArgument(dims.length == shape.dims(), "Must specify all dimensions");
        int i = getOffset(dims);
        float d = blockF.get(makeBlockShape(dims));
        return b.get(i) * d;
    }

    public final FloatBufferTensor getBlockF() {
        return blockF;
    }

    public final float getFactorForIndex(int d, int i) {
        int ix = (int) (i * I_BLOCK_SIZE);
        if (ix >= blockF.size()) throw new RuntimeException();
        return blockF.get(d, ix);
    }

    @Override
    public void set(float v, int... dims) {
        Preconditions.checkArgument(dims.length <= shape.dims(), "Too many dimensions specified for tensor");
        Preconditions.checkArgument(dims.length == shape.dims(), "Must specify all dimensions");
        Preconditions.checkArgument(!b.isReadOnly(), "Can't modify a read only buffer");
        int i = getOffset(dims);
        float d = blockF.get(makeBlockShape(dims));
        float max = d * Byte.MAX_VALUE;
        if (v <= max) {
            float id = d != 0.0f ? 1.0f / d : d;
            b.put(i, (byte) (v * id));
        } else {
            throw new UnsupportedOperationException();
        }
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
    public void intoTensor(ByteVector vector, VectorMask<Byte> msk, int... aoffset) {
        Preconditions.checkArgument(!b.isReadOnly());
        int offset = getOffset(aoffset);
        vector.intoMemorySegment(segment, getMemorySegmentOffset(offset), ByteOrder.LITTLE_ENDIAN, msk);
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
    public void copyFrom(AbstractTensor src, int srcOffset, int destOffset, int length) {
        Preconditions.checkArgument(this.dType == src.dType, "different types");
        Preconditions.checkArgument(!b.isReadOnly(), "Read-only");
        segment.asSlice(getMemorySegmentOffset(destOffset), length)
            .copyFrom(src.getMemorySegment().asSlice(src.getMemorySegmentOffset(srcOffset), length));
    }

    @Override
    public void clear() {
        Preconditions.checkArgument(!b.isReadOnly(), "Can't clear a read-only buffer");
        segment.fill((byte) 0);
    }

    @Override
    public String toString() {
        byte[] sample = new byte[Math.min(10, b.remaining())];
        b.duplicate().get(sample);
        return "ByteBufferTensor{" + "name='" + name + '\'' + "shape=" + shape + ", b=" + Arrays.toString(sample) + "...}";
    }
}
