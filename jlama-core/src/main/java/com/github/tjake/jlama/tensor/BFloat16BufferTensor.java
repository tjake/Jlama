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

import com.github.tjake.jlama.math.FloatConversions;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.util.UnsafeDirectByteBuffer;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.nio.ShortBuffer;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorSpecies;

public class BFloat16BufferTensor extends AbstractTensor<ShortVector, Short> {

    private final ShortBuffer b;
    private final String name;
    private final MemorySegment segment;

    public BFloat16BufferTensor(AbstractTensor ft) {
        this(ft.shape);
        Preconditions.checkArgument(ft.dType != DType.BF16, "This should never happen, likely a bug");

        int[] cursor = new int[ft.shape.dims()];
        do {
            set(ft.get(cursor), cursor);
        } while (ft.iterate(cursor));
    }

    public BFloat16BufferTensor(int... shape) {
        this(TensorShape.of(shape));
    }

    public BFloat16BufferTensor(TensorShape shape) {
        super(DType.BF16, shape, true);
        this.name = "tmp";
        this.b = UnsafeDirectByteBuffer.allocateAlignedByteBuffer(
            Ints.checkedCast(size() * dType().size()),
            UnsafeDirectByteBuffer.CACHE_LINE_SIZE
        ).asShortBuffer();

        this.segment = MemorySegment.ofBuffer(b);
    }

    public BFloat16BufferTensor(String name, ShortBuffer b, TensorShape shape, boolean cacheSlices) {
        super(DType.BF16, shape, cacheSlices);
        this.name = name;
        this.b = b;
        this.segment = MemorySegment.ofBuffer(b);
    }

    @Override
    protected AbstractTensor make(TensorShape shape) {
        return new BFloat16BufferTensor(shape);
    }

    @Override
    protected AbstractTensor make(int offset, int length, TensorShape shape, boolean cacheSlices) {
        return new BFloat16BufferTensor(name, b.slice(offset, length), shape, cacheSlices);
    }

    @Override
    public float get(int... dims) {
        Preconditions.checkArgument(dims.length <= shape.dims(), "Too many dimensions specified");
        Preconditions.checkArgument(dims.length == shape.dims(), "Must specify all dimensions");
        return FloatConversions.bFloat16ToFloat32(b.get(getOffset(dims)));
    }

    @Override
    public void set(float v, int... dims) {
        Preconditions.checkArgument(dims.length <= shape.dims(), "Too many dimensions specified for tensor");
        Preconditions.checkArgument(dims.length == shape.dims(), "Must specify all dimensions");
        Preconditions.checkArgument(!b.isReadOnly(), "Can't modify a read only buffer");
        b.put(getOffset(dims), FloatConversions.float32ToBFloat16(v));
    }

    @Override
    public ShortVector getVector(VectorSpecies<Short> species, int... voffset) {
        int offset = getOffset(voffset);
        return ShortVector.fromMemorySegment(species, segment, getMemorySegmentOffset(offset), ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    public void intoTensor(ShortVector vector, int... aoffset) {
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
        return offset * dType.size();
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
        float[] sample = new float[Math.min(10, b.remaining())];
        for (int i = 0; i < sample.length; i++) {
            sample[i] = FloatConversions.bFloat16ToFloat32(b.get(i));
        }

        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < sample.length; i++) {
            sb.append(String.format("%8.4f", sample[i]));
            if (i < sample.length - 1) {
                sb.append(", ");
            }
        }

        for (int i = 0; i < sample.length; i++) {
            sample[i] = FloatConversions.bFloat16ToFloat32(b.get(i + (shape.first() / 2)));
        }

        StringBuffer sb2 = new StringBuffer();
        for (int i = 0; i < sample.length; i++) {
            sb2.append(String.format("%8.4f", sample[i]));
            if (i < sample.length - 1) {
                sb2.append(", ");
            }
        }

        return "BFloat16BufferTensor{" + "name='" + name + '\'' + ", shape=" + shape + ",\n b=" + sb + "..." + sb2 + "}";
    }
}
