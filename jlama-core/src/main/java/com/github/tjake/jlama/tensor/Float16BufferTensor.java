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
import java.nio.ByteOrder;
import java.nio.ShortBuffer;
import java.util.Arrays;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorSpecies;

public class Float16BufferTensor extends AbstractTensor<ShortVector, Short> {
    private final ShortBuffer b;
    private final String name;
    private final MemorySegment segment;

    public Float16BufferTensor(AbstractTensor ft) {
        this(ft.shape);
        Preconditions.checkArgument(ft.dType != DType.F16, "This should never happen, likely a bug");

        int[] cursor = new int[ft.shape.dims()];
        do {
            set(ft.get(cursor), cursor);
        } while (ft.iterate(cursor));
    }

    public Float16BufferTensor(int... shape) {
        this(TensorShape.of(shape));
    }

    public Float16BufferTensor(TensorShape shape) {
        super(DType.F16, shape, true);
        this.name = "tmp";
        this.b = UnsafeDirectByteBuffer.allocateAlignedByteBuffer(
            Ints.checkedCast(size() * dType().size()),
            UnsafeDirectByteBuffer.CACHE_LINE_SIZE
        ).asShortBuffer();

        this.segment = MemorySegment.ofBuffer(b);
    }

    public Float16BufferTensor(ShortBuffer b, TensorShape shape, boolean cacheSlices) {
        this("none", b, shape, cacheSlices);
    }

    public Float16BufferTensor(String name, ShortBuffer b, TensorShape shape, boolean cacheSlices) {
        super(DType.F16, shape, cacheSlices);
        Preconditions.checkArgument(b.isDirect(), "Must use direct buffers");
        this.name = name;
        this.b = b;
        this.segment = MemorySegment.ofBuffer(b);
    }

    @Override
    protected AbstractTensor make(TensorShape shape) {
        return new Float16BufferTensor(shape);
    }

    @Override
    protected AbstractTensor make(int offset, int length, TensorShape shape, boolean cacheSlices) {
        return new Float16BufferTensor(name, b.slice(offset, length), shape, cacheSlices);
    }

    @Override
    public float get(int... dims) {
        Preconditions.checkArgument(dims.length <= shape.dims(), "Too many dimensions specified");
        Preconditions.checkArgument(dims.length == shape.dims(), "Must specify all dimensions");
        return Float.float16ToFloat(b.get(getOffset(dims)));
    }

    @Override
    public void set(float v, int... dims) {
        Preconditions.checkArgument(dims.length <= shape.dims(), "Too many dimensions specified for tensor");
        Preconditions.checkArgument(dims.length == shape.dims(), "Must specify all dimensions");
        Preconditions.checkArgument(!b.isReadOnly(), "Can't modify a read only buffer");
        b.put(getOffset(dims), Float.floatToFloat16(v));
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
        short[] sample = new short[Math.min(10, b.remaining())];
        b.duplicate().get(sample);
        return "Float16BufferTensor{" + "name='" + name + '\'' + "shape=" + shape + ", b=" + Arrays.toString(sample) + "...}";
    }
}
