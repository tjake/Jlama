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
import com.github.tjake.jlama.util.DebugSupport;
import com.github.tjake.jlama.util.UnsafeDirectByteBuffer;
import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
public final class FloatBufferTensor extends AbstractTensor<FloatVector, Float> {
    private static final Logger logger = LoggerFactory.getLogger(FloatBufferTensor.class);
    private final FloatBuffer b;
    private final String name;
    private final MemorySegment segment;

    public FloatBufferTensor(AbstractTensor ft) {
        this(ft.shape);
        Preconditions.checkArgument(ft.dType != DType.I32, "This should never happen, likely a bug");

        int[] cursor = new int[ft.shape.dims()];
        do {
            set(ft.get(cursor), cursor);
        } while (ft.iterate(cursor));
    }

    public FloatBufferTensor(int... shape) {
        this(TensorShape.of(shape));
    }

    public FloatBufferTensor(TensorShape shape) {
        super(DType.F32, shape, true);
        this.name = "tmp";
        this.b = UnsafeDirectByteBuffer.allocateAlignedByteBuffer(
            Ints.checkedCast(shape.size() * dType().size()),
            UnsafeDirectByteBuffer.CACHE_LINE_SIZE
        ).asFloatBuffer();

        this.segment = MemorySegment.ofBuffer(b);
    }

    public FloatBufferTensor(FloatBuffer b, TensorShape shape, boolean cacheSlices) {
        this("none", b, shape, cacheSlices);
    }

    public FloatBufferTensor(String name, FloatBuffer b, TensorShape shape, boolean cacheSlices) {
        super(DType.F32, shape, cacheSlices);
        this.name = name;
        if (b.isDirect()) {
            this.b = b;
        } else {
            this.b = UnsafeDirectByteBuffer.allocateAlignedByteBuffer(
                Ints.checkedCast(size() * dType().size()),
                UnsafeDirectByteBuffer.CACHE_LINE_SIZE
            ).asFloatBuffer();
            this.b.duplicate().put(b);
        }

        this.segment = MemorySegment.ofBuffer(this.b);
    }

    @Override
    protected AbstractTensor make(TensorShape shape) {
        return new FloatBufferTensor(shape);
    }

    @Override
    protected AbstractTensor make(int offset, int length, TensorShape shape, boolean cacheSlices) {
        return new FloatBufferTensor(name, b.slice(offset, length), shape, cacheSlices);
    }

    @Override
    public float get(int... dims) {
        Preconditions.checkArgument(dims.length <= shape.dims(), "Too many dimensions specified");
        Preconditions.checkArgument(dims.length == shape.dims(), "Must specify all dimensions");
        return b.get(getOffset(dims));
    }

    @Override
    public void set(float v, int... dims) {
        Preconditions.checkArgument(dims.length <= shape.dims(), "Too many dimensions specified for tensor");
        Preconditions.checkArgument(dims.length == shape.dims(), "Must specify all dimensions");
        Preconditions.checkArgument(!b.isReadOnly(), "Can't modify a read only buffer");
        b.put(getOffset(dims), v);
    }

    @Override
    public MemorySegment getMemorySegment() {
        return segment;
    }

    @Override
    public void copyFrom(AbstractTensor src, int srcOffset, int destOffset, int length) {
        Preconditions.checkArgument(this.dType == src.dType, "Different types");
        // Preconditions.checkArgument(!b.isReadOnly());
        segment.asSlice(getMemorySegmentOffset(destOffset), length * dType.size())
            .copyFrom(src.getMemorySegment().asSlice(src.getMemorySegmentOffset(srcOffset), length * dType.size()));
    }

    @Override
    public int getMemorySegmentOffset(int offset) {
        return offset * Float.BYTES;
    }

    @Override
    public FloatVector getVector(VectorSpecies<Float> species, int... voffset) {
        int offset = getOffset(voffset);
        return FloatVector.fromMemorySegment(species, segment, getMemorySegmentOffset(offset), ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    public void intoTensor(FloatVector vector, int... aoffset) {
        // Preconditions.checkArgument(!b.isReadOnly());
        int offset = getOffset(aoffset);
        vector.intoMemorySegment(segment, getMemorySegmentOffset(offset), ByteOrder.LITTLE_ENDIAN);
    }

    @Override
    public void clear() {
        segment.fill((byte) 0);
    }

    @Override
    public String toString() {
        float[] sample = new float[DebugSupport.isDebug() ? b.remaining() : Math.min(10, b.remaining())];
        b.duplicate().get(sample);
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < sample.length; i++) {
            sb.append(String.format("%8.4f", sample[i]));
            if (i < sample.length - 1) {
                sb.append(", ");
            }
        }

        return "FloatBufferTensor{" + "name='" + name + '\'' + " shape=" + shape + ",\nb={" + sb + "...}";
    }
}
