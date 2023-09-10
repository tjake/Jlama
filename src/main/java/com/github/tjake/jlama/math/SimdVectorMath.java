package com.github.tjake.jlama.math;

import com.github.tjake.jlama.model.*;
import com.github.tjake.jlama.safetensors.DType;
import com.google.common.base.Preconditions;
import jdk.incubator.vector.*;

import java.nio.ByteOrder;

public class SimdVectorMath {
    static final boolean hasAVX2 = FloatVector.SPECIES_PREFERRED == FloatVector.SPECIES_512;
    static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    static final VectorShuffle<Float> ZIP_EVEN = VectorShuffle.makeZip(FloatVector.SPECIES_256,0);
    static final VectorShuffle<Float> ZIP_ODD = VectorShuffle.makeZip(FloatVector.SPECIES_256,1);

    static final ByteVector Q4_BYTE_SUB = ByteVector.broadcast(ByteVector.SPECIES_64, 8);
    static final ByteVector Q4_BYTE_MASK = ByteVector.broadcast(ByteVector.SPECIES_64, 0xF);

    static final ByteVector Q4_BYTE_SHIFT = ByteVector.broadcast(ByteVector.SPECIES_64, 4);



    public static float dotProduct(AbstractTensor a, AbstractTensor b, int aoffset, int boffset, int limit) {
        Preconditions.checkArgument(limit % 8 == 0);

        return switch (a.dType()) {
            case F32 -> switch (b.dType()) {
                case F32 -> dotProductF32(a, b, aoffset, boffset, limit);
                case I8 -> hasAVX2 ? dotProductF32I8_512((FloatBufferTensor) a, (Q8ByteBufferTensor) b, aoffset, boffset, limit) : dotProductF32I8_256((FloatBufferTensor) a, (Q8ByteBufferTensor) b, aoffset, boffset, limit);
                //case Q5 -> dotProductF32Q5((FloatBufferTensor) a, (Q5ByteBufferTensor) b, aoffset, boffset, limit);
                case Q4 -> dotProductF32Q4((FloatBufferTensor) a, (Q4ByteBufferTensor) b, aoffset, boffset, limit);
                default -> throw new UnsupportedOperationException(b.dType().name());
            };
            case I8 -> dotProductI8((Q8ByteBufferTensor) a, (Q8ByteBufferTensor) b, aoffset, boffset, limit);

            default -> throw new UnsupportedOperationException();
        };
    }

    private static float dotProductI8(Q8ByteBufferTensor a, Q8ByteBufferTensor b, int aoffset, int boffset, int limit) {
        Preconditions.checkArgument(
                aoffset % Q8ByteBufferTensor.BLOCK_SIZE == 0 &&
                        boffset % Q8ByteBufferTensor.BLOCK_SIZE == 0 &&
                        limit % Q8ByteBufferTensor.BLOCK_SIZE == 0
        );

        int alim = aoffset + limit;
        int blim = boffset + limit;
        int slen = ByteVector.SPECIES_64.length();

        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_256);

        for (; aoffset < alim && boffset < blim; aoffset += slen, boffset += slen) {
            FloatVector scale = FloatVector.broadcast(FloatVector.SPECIES_256,a.getFactorForIndex(aoffset) * b.getFactorForIndex(boffset));
            var af = ByteVector.fromMemorySegment(ByteVector.SPECIES_64, a.getMemorySegment(), aoffset, ByteOrder.LITTLE_ENDIAN).convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0);
            var bf = ByteVector.fromMemorySegment(ByteVector.SPECIES_64, b.getMemorySegment(), boffset, ByteOrder.LITTLE_ENDIAN).convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0);

            acc = acc.add(af.mul(bf).mul(scale));
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    private static float dotProductF32I8_256(FloatBufferTensor a, Q8ByteBufferTensor b, int aoffset, int boffset, int limit) {
        Preconditions.checkArgument(a.dType() == DType.F32 && b.dType() == DType.I8);
        Preconditions.checkArgument(
                boffset % Q8ByteBufferTensor.BLOCK_SIZE == 0 &&
                        limit % Q8ByteBufferTensor.BLOCK_SIZE == 0
        );

        int alim = aoffset + limit;
        int blim = boffset + limit;
        int slen = ByteVector.SPECIES_64.length();

        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_256);

        //Unroll 4x
        for (; aoffset < alim && boffset < blim; aoffset += slen*4, boffset += slen*4) {
            FloatVector scale = FloatVector.broadcast(FloatVector.SPECIES_256, b.getFactorForIndex(boffset));
            var af = FloatVector.fromMemorySegment(FloatVector.SPECIES_256, a.getMemorySegment(), a.getMemorySegmentOffset(aoffset), ByteOrder.LITTLE_ENDIAN);
            var bf = ByteVector.fromMemorySegment(ByteVector.SPECIES_64, b.getMemorySegment(), boffset, ByteOrder.LITTLE_ENDIAN).convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0);
            acc = acc.add(af.mul(bf.mul(scale)));

            af = FloatVector.fromMemorySegment(FloatVector.SPECIES_256, a.getMemorySegment(), a.getMemorySegmentOffset(aoffset + slen), ByteOrder.LITTLE_ENDIAN);
            bf = ByteVector.fromMemorySegment(ByteVector.SPECIES_64, b.getMemorySegment(), boffset + slen, ByteOrder.LITTLE_ENDIAN).convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0);
            acc = acc.add(af.mul(bf.mul(scale)));

            af = FloatVector.fromMemorySegment(FloatVector.SPECIES_256, a.getMemorySegment(), a.getMemorySegmentOffset(aoffset + slen + slen), ByteOrder.LITTLE_ENDIAN);
            bf = ByteVector.fromMemorySegment(ByteVector.SPECIES_64, b.getMemorySegment(), boffset + slen + slen, ByteOrder.LITTLE_ENDIAN).convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0);
            acc = acc.add(af.mul(bf.mul(scale)));

            af = FloatVector.fromMemorySegment(FloatVector.SPECIES_256, a.getMemorySegment(), a.getMemorySegmentOffset(aoffset + slen + slen + slen ), ByteOrder.LITTLE_ENDIAN);
            bf = ByteVector.fromMemorySegment(ByteVector.SPECIES_64, b.getMemorySegment(), boffset + slen + slen + slen, ByteOrder.LITTLE_ENDIAN).convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0);
            acc = acc.add(af.mul(bf.mul(scale)));
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }


    private static float dotProductF32I8_512(FloatBufferTensor a, Q8ByteBufferTensor b, int aoffset, int boffset, int limit) {
        Preconditions.checkArgument(a.dType() == DType.F32 && b.dType() == DType.I8);
        Preconditions.checkArgument(
                        boffset % Q8ByteBufferTensor.BLOCK_SIZE == 0 &&
                        limit % Q8ByteBufferTensor.BLOCK_SIZE == 0
        );

        int alim = aoffset + limit;
        int blim = boffset + limit;
        int slen = ByteVector.SPECIES_128.length();

        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_512);

        //Unroll 4x
        for (; aoffset < alim && boffset < blim; aoffset += slen*4, boffset += slen*4) {
            FloatVector scale = FloatVector.broadcast(FloatVector.SPECIES_512, b.getFactorForIndex(boffset));
            var af = FloatVector.fromMemorySegment(FloatVector.SPECIES_512, a.getMemorySegment(), a.getMemorySegmentOffset(aoffset), ByteOrder.LITTLE_ENDIAN);
            var bf = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, b.getMemorySegment(), boffset, ByteOrder.LITTLE_ENDIAN).convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0);
            acc = acc.add(af.mul(bf.mul(scale)));

            af = FloatVector.fromMemorySegment(FloatVector.SPECIES_512, a.getMemorySegment(), a.getMemorySegmentOffset(aoffset + slen), ByteOrder.LITTLE_ENDIAN);
            bf = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, b.getMemorySegment(), boffset + slen, ByteOrder.LITTLE_ENDIAN).convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0);
            acc = acc.add(af.mul(bf.mul(scale)));

            af = FloatVector.fromMemorySegment(FloatVector.SPECIES_512, a.getMemorySegment(), a.getMemorySegmentOffset(aoffset + slen + slen), ByteOrder.LITTLE_ENDIAN);
            bf = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, b.getMemorySegment(), boffset + slen + slen, ByteOrder.LITTLE_ENDIAN).convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0);
            acc = acc.add(af.mul(bf.mul(scale)));

            af = FloatVector.fromMemorySegment(FloatVector.SPECIES_512, a.getMemorySegment(), a.getMemorySegmentOffset(aoffset + slen + slen + slen ), ByteOrder.LITTLE_ENDIAN);
            bf = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, b.getMemorySegment(), boffset + slen + slen + slen, ByteOrder.LITTLE_ENDIAN).convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0);
            acc = acc.add(af.mul(bf.mul(scale)));
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    private static float dotProductF32Q4(FloatBufferTensor a, Q4ByteBufferTensor b, int aoffset, int boffset, int limit) {
        Preconditions.checkArgument(a.dType() == DType.F32 && b.dType() == DType.Q4);
        Preconditions.checkArgument(
                boffset % Q4ByteBufferTensor.BLOCK_SIZE == 0 &&
                        limit % Q4ByteBufferTensor.BLOCK_SIZE == 0
        );

        int alim = aoffset + limit;
        int blim = boffset + limit;
        int slen = Q4ByteBufferTensor.BLOCK_SIZE;

        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_256);

        //Unroll 4x
        for (; aoffset < alim && boffset < blim; aoffset += slen, boffset += slen) {
            FloatVector scale = FloatVector.broadcast(FloatVector.SPECIES_256, b.getFactorForIndex(boffset));
            // BLOCK_SIZE Floats
            var af0 = FloatVector.fromMemorySegment(FloatVector.SPECIES_256, a.getMemorySegment(), a.getMemorySegmentOffset(aoffset), ByteOrder.LITTLE_ENDIAN);
            var af1 = FloatVector.fromMemorySegment(FloatVector.SPECIES_256, a.getMemorySegment(), a.getMemorySegmentOffset(aoffset + 8), ByteOrder.LITTLE_ENDIAN);
            var af2 = FloatVector.fromMemorySegment(FloatVector.SPECIES_256, a.getMemorySegment(), a.getMemorySegmentOffset(aoffset + 8 + 8), ByteOrder.LITTLE_ENDIAN);
            var af3 = FloatVector.fromMemorySegment(FloatVector.SPECIES_256, a.getMemorySegment(), a.getMemorySegmentOffset(aoffset + 8 + 8 + 8), ByteOrder.LITTLE_ENDIAN);

            //Make 8 bytes -> 16 4bit -> 16 bytes -> 16 32F
            var bf0 = ByteVector.fromMemorySegment(ByteVector.SPECIES_64, b.getMemorySegment(), b.getMemorySegmentOffset(boffset), ByteOrder.LITTLE_ENDIAN);
            var bf16 = ByteVector.fromMemorySegment(ByteVector.SPECIES_64, b.getMemorySegment(), b.getMemorySegmentOffset(boffset + Q4ByteBufferTensor.HALF_BLOCK), ByteOrder.LITTLE_ENDIAN);

            // Convert the first 4 bits into bytes
            var low0 = bf0.lanewise(VectorOperators.AND, Q4_BYTE_MASK)
                    .sub(Q4_BYTE_SUB)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0);


            // Convert the second 4 bits into bytes
            var high0 = bf0.lanewise(VectorOperators.ASHR, Q4_BYTE_SHIFT)
                    .lanewise(VectorOperators.AND, Q4_BYTE_MASK)
                    .sub(Q4_BYTE_SUB)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0);

            // Convert the first 4 bits into bytes
            var low1 = bf16.lanewise(VectorOperators.AND, Q4_BYTE_MASK)
                    .sub(Q4_BYTE_SUB)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0);


            // Convert the second 4 bits into bytes
            var high1 = bf16.lanewise(VectorOperators.ASHR, Q4_BYTE_SHIFT)
                    .lanewise(VectorOperators.AND, Q4_BYTE_MASK)
                    .sub(Q4_BYTE_SUB)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0);


            acc = acc.add(af0.mul(low0.mul(scale)));
            acc = acc.add(af1.mul(low1.mul(scale)));
            acc = acc.add(af2.mul(high0.mul(scale)));
            acc = acc.add(af3.mul(high1.mul(scale)));

        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    private static float dotProductF32(AbstractTensor a, AbstractTensor b, int aoffset, int boffset, int limit) {
         FloatVector acc = FloatVector.zero(SPECIES);
         int upperBound = SPECIES.loopBound(limit);
         int ao = aoffset;
         int bo = boffset;
         for (; ao < (aoffset + upperBound) && bo < (boffset + upperBound); ao += SPECIES.length(), bo += SPECIES.length()) {
             FloatVector va = a.getFloatVector(ao);
             FloatVector vb = b.getFloatVector(bo);
             acc = acc.add(va.mul(vb));
         }
         // reduce
         float res = acc.reduceLanes(VectorOperators.ADD);
         // tail
         for (; ao < (aoffset + limit) && bo < (boffset + limit); ao++, bo++) {
             res += a.get(ao) * b.get(bo);
         }
         return res;
     }

    public static void accumulate(AbstractTensor a, AbstractTensor b) {
        Preconditions.checkArgument(a.dType() == b.dType());
        Preconditions.checkArgument(a.size() % 8 == 0);

        switch (a.dType()) {
            case F32 -> accumulateF32(a, b);
            default -> throw new UnsupportedOperationException();
        }
    }

    static void accumulateF32(AbstractTensor a, AbstractTensor b) {
         int upperBound = SPECIES.loopBound(a.size());
         int i = 0;

         for (; i < upperBound; i += SPECIES.length()) {
             FloatVector va = a.getFloatVector(i);
             FloatVector vb = b.getFloatVector(i);
             va.add(vb).intoMemorySegment(a.getMemorySegment(), a.getMemorySegmentOffset(i), ByteOrder.LITTLE_ENDIAN);
         }

         // tail
         for (; i < a.size(); i++) {
             a.set(a.get(i) + b.get(i));
         }
     }

    public static void saxpy(float alpha, AbstractTensor x, AbstractTensor y, int xoffset, int yoffset, int limit) {
        Preconditions.checkArgument(x.dType() == y.dType());
        Preconditions.checkArgument(limit % 8 == 0);

        switch (x.dType()) {
            case F32 -> saxpyF32(alpha, x, y, xoffset, yoffset, limit);
            default -> throw new UnsupportedOperationException();
        }
    }

    public static void saxpyF32(float alpha, AbstractTensor x, AbstractTensor y, int xoffset, int yoffset, int limit) {
        int upperBound = SPECIES.loopBound(limit);
        int xo = xoffset;
        int yo = yoffset;
        for (; xo < (xoffset + upperBound) && yo < (yoffset + upperBound); xo += SPECIES.length(), yo += SPECIES.length()) {
            FloatVector vx = x.getFloatVector(xo);
            FloatVector vy = y.getFloatVector(yo);
            vy.add(vx.mul(alpha)).intoMemorySegment(y.getMemorySegment(), y.getMemorySegmentOffset(yo), ByteOrder.LITTLE_ENDIAN);
        }

        // tail
        for (; xo < (xoffset + limit) && yo < (yoffset + limit); xo++, yo++) {
            float v = y.get(yo) + (alpha * x.get(xo));
            y.set(v, yo);
        }
    }
}
