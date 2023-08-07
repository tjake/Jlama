package com.github.tjake.llmj.safetensors;

import com.github.tjake.llmj.model.FloatBufferTensor;
import com.google.common.primitives.Ints;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;

public class Weights implements WeightLoader {

    private final Map<String, String> metadata;
    private final Map<String, TensorInfo> tensorInfoMap;
    private final ByteBuffer bytes;

    Weights(Map<String, String> metadata, Map<String, TensorInfo> tensorInfoMap, ByteBuffer bytes)
    {
        this.metadata = metadata;
        this.tensorInfoMap = tensorInfoMap;
        this.bytes = bytes.duplicate();
    }

    @Override
    public FloatBufferTensor load(String name) throws NoSuchElementException {
        TensorInfo info = tensorInfoMap.get(name);
        if (info == null)
            throw new NoSuchElementException();

        if (info.shape.length < 1)
            throw new RuntimeException("Invalid shape dimensions " + info.shape.length + " encountered for " + name);

        ByteBuffer b = bytes.duplicate().order(ByteOrder.LITTLE_ENDIAN)
                .position(Ints.checkedCast(info.dataOffsets[0]))
                .limit(Ints.checkedCast(info.dataOffsets[1]));

        FloatBuffer fb;
        switch (info.dType) {
            case F32:
                int len = b.remaining() / DType.F32.size();
                fb = FloatBuffer.allocate(len);
                for (int i = 0; i < len; i++) {
                    float v = b.getFloat();
                    fb.put(i, v);
                }
                //fb = b.asFloatBuffer().slice();
                break;
            case F16:
                 len = b.remaining() / DType.F16.size();
                fb = FloatBuffer.allocate(len);
                for (int i = 0; i < len; i++) {
                    short s = b.getShort();
                    float v = float16ToFloat32(s);
                    fb.put(i, v);
                }
                break;
            case BF16:
                len = b.remaining() / DType.F16.size();
                fb = FloatBuffer.allocate(len);
                for (int i = 0; i < len; i++) {
                    short s = b.getShort();
                    float v = bFloat16ToFloat32(s);
                    fb.put(i, v);
                }
                break;
            default:
                throw new IllegalArgumentException("Unsupported Tensor type: " + info.dType.name() + " for " + name);
        }

        return new FloatBufferTensor(fb, info.shape, true);
    }

    @Override
    public String toString() {
        return "SafeTensor{" +
                "metadata=" + metadata +
                ", tensorInfoMap=" + tensorInfoMap +
                ", bytes=" + bytes +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Weights weights = (Weights) o;
        return Objects.equals(metadata, weights.metadata) && Objects.equals(tensorInfoMap, weights.tensorInfoMap);
    }

    @Override
    public int hashCode() {
        return Objects.hash(metadata, tensorInfoMap);
    }

    /**
     * Convert this BFloat16 value to the nearest Float.
     *
     * Unlike Float16, since BFloat16 has the same size exponents as
     * Float32 it means that all we have to do is add some extra zeros
     * to the mantissa.
     *
     * From https://github.com/stripe-archive/agate/blob/master/core/src/main/scala/com/stripe/agate/tensor/BFloat16.scala
     */
    public static float bFloat16ToFloat32(short raw) {
        return Float.intBitsToFloat((raw & 0xffff) << 16);
    }

    /**
     * Convert a 16-bit floating-point number in ARM alternative half-precision format to a 32-bit floating-point number.
     *
     * Ported from https://github.com/Maratyszcza/FP16/blob/0a92994d729ff76a58f692d3028ca1b64b145d91/include/fp16/fp16.h#L255
     */
    public static float float16ToFloat32(short raw) {
        long  w = Integer.toUnsignedLong(raw << 16);
        long  sign =  w & 0x80000000L;
        long  nonsign = w & 0x7FFFFFFF;

        int renorm_shift = Long.numberOfLeadingZeros(nonsign);

        renorm_shift = renorm_shift > (32+5) ? renorm_shift - (32+5) : 0;

        long zero_mask = (nonsign - 1) >> (32+31);

        return Float.intBitsToFloat((int)(sign | (((nonsign << renorm_shift >> 3) + ((0x70 - renorm_shift) << 23)) & ~zero_mask)));
    }

}
