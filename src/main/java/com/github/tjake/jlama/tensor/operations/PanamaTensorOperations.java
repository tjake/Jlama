package com.github.tjake.jlama.tensor.operations;

import java.nio.ByteOrder;

import com.google.common.base.Preconditions;

import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.BFloat16BufferTensor;
import com.github.tjake.jlama.tensor.FloatBufferTensor;
import com.github.tjake.jlama.tensor.Q4ByteBufferTensor;
import com.github.tjake.jlama.tensor.Q8ByteBufferTensor;
import com.github.tjake.jlama.tensor.TensorCache;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class PanamaTensorOperations implements TensorOperations
{
    public static final boolean hasAVX2 = FloatVector.SPECIES_PREFERRED == FloatVector.SPECIES_512;
    static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    static final ByteVector Q4_BYTE_SUB;
    static final ByteVector Q4_BYTE_MASK;
    static final ByteVector Q4_BYTE_SHIFT;
    static {
        if (hasAVX2) {
            Q4_BYTE_SUB = ByteVector.broadcast(ByteVector.SPECIES_128, 8);
            Q4_BYTE_MASK = ByteVector.broadcast(ByteVector.SPECIES_128, 0xF);
            Q4_BYTE_SHIFT = ByteVector.broadcast(ByteVector.SPECIES_128, 4);
        } else {
            Q4_BYTE_SUB = ByteVector.broadcast(ByteVector.SPECIES_64, 8);
            Q4_BYTE_MASK = ByteVector.broadcast(ByteVector.SPECIES_64, 0xF);
            Q4_BYTE_SHIFT = ByteVector.broadcast(ByteVector.SPECIES_64, 4);
        }
    }

    public static final IntVector BF16_BYTE_SHIFT;
    static {
        if (hasAVX2) {
            BF16_BYTE_SHIFT = IntVector.broadcast(IntVector.SPECIES_512, 16);
        } else {
            BF16_BYTE_SHIFT = IntVector.broadcast(IntVector.SPECIES_256, 16);
        }
    }


    @Override
    public  float dotProduct(AbstractTensor a, AbstractTensor b, int aoffset, int boffset, int limit) {
        Preconditions.checkArgument(limit % 32 == 0);

        return switch (a.dType()) {
            case F32 -> switch (b.dType()) {
                case F32 -> dotProductF32(a, b, aoffset, boffset, limit);
                case I8 -> hasAVX2 ? dotProductF32I8_512((FloatBufferTensor) a, (Q8ByteBufferTensor) b, aoffset, boffset, limit) : dotProductF32I8_256((FloatBufferTensor) a, (Q8ByteBufferTensor) b, aoffset, boffset, limit);
                //case Q5 -> dotProductF32Q5((FloatBufferTensor) a, (Q5ByteBufferTensor) b, aoffset, boffset, limit);
                case Q4 -> hasAVX2 ? dotProductF32Q4_512((FloatBufferTensor) a, (Q4ByteBufferTensor) b, aoffset, boffset, limit) : dotProductF32Q4_256((FloatBufferTensor) a, (Q4ByteBufferTensor) b, aoffset, boffset, limit);
                default -> throw new UnsupportedOperationException(b.dType().name());
            };
            case I8 -> dotProductI8((Q8ByteBufferTensor) a, (Q8ByteBufferTensor) b, aoffset, boffset, limit);
            case BF16 -> switch (b.dType()) {
                case F32 -> dotProductBF16F32_512((BFloat16BufferTensor) a, (FloatBufferTensor) b, aoffset, boffset, limit);
                case I8 -> dotProductBF16I8_512((BFloat16BufferTensor) a, (Q8ByteBufferTensor) b, aoffset, boffset, limit);
                case BF16 -> dotProductBF16_512((BFloat16BufferTensor)a, (BFloat16BufferTensor)b, aoffset, boffset, limit);
                default -> throw new UnsupportedOperationException(b.dType().name());
            };
            default -> throw new UnsupportedOperationException();
        };
    }

    private float dotProductI8(Q8ByteBufferTensor a, Q8ByteBufferTensor b, int aoffset, int boffset, int limit) {
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

    private float dotProductF32I8_256(FloatBufferTensor a, Q8ByteBufferTensor b, int aoffset, int boffset, int limit) {
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

    public float dotProductF32I8_512(FloatBufferTensor a, Q8ByteBufferTensor b, int aoffset, int boffset, int limit) {
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

    private float dotProductF32Q4_256(FloatBufferTensor a, Q4ByteBufferTensor b, int aoffset, int boffset, int limit) {
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

    private float dotProductF32Q4_512(FloatBufferTensor a, Q4ByteBufferTensor b, int aoffset, int boffset, int limit) {
        Preconditions.checkArgument(a.dType() == DType.F32 && b.dType() == DType.Q4);
        Preconditions.checkArgument(
                boffset % Q4ByteBufferTensor.BLOCK_SIZE == 0 &&
                        limit % Q4ByteBufferTensor.BLOCK_SIZE == 0
        );

        int alim = aoffset + limit;
        int blim = boffset + limit;
        int slen = Q4ByteBufferTensor.BLOCK_SIZE;

        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_512);

        //Unroll 4x
        for (; aoffset < alim && boffset < blim; aoffset += slen, boffset += slen) {
            FloatVector scale = FloatVector.broadcast(FloatVector.SPECIES_512, b.getFactorForIndex(boffset));
            // BLOCK_SIZE Floats
            var af0 = FloatVector.fromMemorySegment(FloatVector.SPECIES_512, a.getMemorySegment(), a.getMemorySegmentOffset(aoffset), ByteOrder.LITTLE_ENDIAN);
            var af1 = FloatVector.fromMemorySegment(FloatVector.SPECIES_512, a.getMemorySegment(), a.getMemorySegmentOffset(aoffset + 16), ByteOrder.LITTLE_ENDIAN);

            //Make 16 bytes -> 32 4bit -> 32 bytes -> 32 32F
            var bf0 = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, b.getMemorySegment(), b.getMemorySegmentOffset(boffset), ByteOrder.LITTLE_ENDIAN);

            // Convert the first 4 bits into bytes
            var low0 = bf0.lanewise(VectorOperators.AND, Q4_BYTE_MASK)
                    .sub(Q4_BYTE_SUB)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0);

            // Convert the second 4 bits into bytes
            var high0 = bf0.lanewise(VectorOperators.ASHR, Q4_BYTE_SHIFT)
                    .lanewise(VectorOperators.AND, Q4_BYTE_MASK)
                    .sub(Q4_BYTE_SUB)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0);

            acc = acc.add(af0.mul(low0.mul(scale)));
            acc = acc.add(af1.mul(high0.mul(scale)));
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    public float dotProductBF16I8_512(BFloat16BufferTensor a, Q8ByteBufferTensor b, final int aoffset, final int boffset, int limit)
    {
        Preconditions.checkArgument(a.dType() == DType.BF16 && b.dType() == DType.I8);

        int ao = aoffset;
        int bo = boffset;
        final int alim = aoffset + limit;
        final int blim = boffset + limit;
        final int slen = ByteVector.SPECIES_128.length();

        FloatVector acc = FloatVector.SPECIES_512.zero().reinterpretAsFloats();

        //Unroll 4x
        for (; ao < alim && bo < blim; ao += slen*4, bo += slen*4)
        {
            FloatVector scale = FloatVector.broadcast(FloatVector.SPECIES_512, b.getFactorForIndex(bo));

            //Convert BF16 to F32
            var af = ShortVector.fromMemorySegment(ShortVector.SPECIES_256, a.getMemorySegment(), a.getMemorySegmentOffset(ao), ByteOrder.LITTLE_ENDIAN)
                    .convertShape(VectorOperators.ZERO_EXTEND_S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            var bf = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, b.getMemorySegment(), bo, ByteOrder.LITTLE_ENDIAN)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0);

            acc = acc.add(af.mul(bf.mul(scale)));

            //Convert BF16 to F32
            af = ShortVector.fromMemorySegment(ShortVector.SPECIES_256, a.getMemorySegment(), a.getMemorySegmentOffset(ao + slen), ByteOrder.LITTLE_ENDIAN)
                    .convertShape(VectorOperators.ZERO_EXTEND_S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            bf = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, b.getMemorySegment(), bo + slen, ByteOrder.LITTLE_ENDIAN)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0);

            acc = acc.add(af.mul(bf.mul(scale)));


            //Convert BF16 to F32
            af = ShortVector.fromMemorySegment(ShortVector.SPECIES_256, a.getMemorySegment(), a.getMemorySegmentOffset(ao + slen + slen), ByteOrder.LITTLE_ENDIAN)
                    .convertShape(VectorOperators.ZERO_EXTEND_S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            bf = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, b.getMemorySegment(), bo + slen + slen, ByteOrder.LITTLE_ENDIAN)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0);

            acc = acc.add(af.mul(bf.mul(scale)));


            //Convert BF16 to F32
            af = ShortVector.fromMemorySegment(ShortVector.SPECIES_256, a.getMemorySegment(), a.getMemorySegmentOffset(ao + slen + slen + slen), ByteOrder.LITTLE_ENDIAN)
                    .convertShape(VectorOperators.ZERO_EXTEND_S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            bf = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, b.getMemorySegment(), bo + slen + slen + slen, ByteOrder.LITTLE_ENDIAN)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0);

            acc = acc.add(af.mul(bf.mul(scale)));
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    private float dotProductBF16F32_512(BFloat16BufferTensor a, FloatBufferTensor b, int aoffset, int boffset, int limit) {
        Preconditions.checkArgument(a.dType() == DType.BF16 && b.dType() == DType.F32);

        int alim = aoffset + limit;
        int blim = boffset + limit;
        int slen = ByteVector.SPECIES_128.length();

        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_512);

        //Unroll 4x
        for (; aoffset < alim && boffset < blim; aoffset += slen, boffset += slen) {

            //Convert BF16 to F32
            var af = ShortVector.fromMemorySegment(ShortVector.SPECIES_256, a.getMemorySegment(), a.getMemorySegmentOffset(aoffset), ByteOrder.LITTLE_ENDIAN)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            FloatVector bf = b.getFloatVector(boffset);

            acc = acc.add(af.mul(bf));
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }


    private float dotProductBF16_512(BFloat16BufferTensor a, BFloat16BufferTensor b, int aoffset, int boffset, int limit) {
        Preconditions.checkArgument(a.dType() == DType.BF16 && b.dType() == DType.BF16);

        int alim = aoffset + limit;
        int blim = boffset + limit;
        int slen = ByteVector.SPECIES_128.length();

        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_512);

        //Unroll 4x
        for (; aoffset < alim && boffset < blim; aoffset += slen, boffset += slen) {

            //Convert BF16 to F32
            var af = ShortVector.fromMemorySegment(ShortVector.SPECIES_256, a.getMemorySegment(), a.getMemorySegmentOffset(aoffset), ByteOrder.LITTLE_ENDIAN)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            //Convert BF16 to F32
            var bf = ShortVector.fromMemorySegment(ShortVector.SPECIES_256, b.getMemorySegment(), b.getMemorySegmentOffset(boffset), ByteOrder.LITTLE_ENDIAN)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            acc = acc.add(af.mul(bf));
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    private float dotProductF32(AbstractTensor a, AbstractTensor b, int aoffset, int boffset, int limit) {
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

    @Override
    public void accumulate(AbstractTensor a, AbstractTensor b) {
        Preconditions.checkArgument(a.dType() == b.dType());
        Preconditions.checkArgument(a.size() % 8 == 0);

        switch (a.dType()) {
            case F32 -> accumulateF32(a, b);
            case BF16 -> accumulateBF16(a, b);
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

    static void accumulateBF16(AbstractTensor a, AbstractTensor b) {
        Preconditions.checkArgument(a.dType() == DType.BF16 && b.dType() == DType.BF16);
        int upperBound = FloatVector.SPECIES_512.loopBound(a.size());

        try(AbstractTensor buf = TensorCache.instance.get(DType.F32, a.shape()))
        {
            int i = 0;
            for (; i < upperBound; i += FloatVector.SPECIES_512.length())
            {

                //Convert BF16 to F32
                var af = ShortVector.fromMemorySegment(ShortVector.SPECIES_256, a.getMemorySegment(), a.getMemorySegmentOffset(i), ByteOrder.LITTLE_ENDIAN)
                        .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                        .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                        .reinterpretAsFloats();

                //Convert BF16 to F32
                var bf = ShortVector.fromMemorySegment(ShortVector.SPECIES_256, b.getMemorySegment(), b.getMemorySegmentOffset(i), ByteOrder.LITTLE_ENDIAN)
                        .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                        .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                        .reinterpretAsFloats();

                af.add(bf).intoMemorySegment(buf.getMemorySegment(), buf.getMemorySegmentOffset(i), ByteOrder.LITTLE_ENDIAN);
            }

            for (int j = 0; j < buf.size(); j++)
                a.set(buf.get(j), j);
        }
    }



    @Override
    public void scale(float factor, AbstractTensor a, int offset, int length)
    {
        int upperBound = FloatVector.SPECIES_PREFERRED.loopBound(offset + length);
        int i = offset;

        FloatVector sf = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, factor);
        for (; i < upperBound; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector va = a.getFloatVector(i);
            va.mul(sf).intoMemorySegment(a.getMemorySegment(), a.getMemorySegmentOffset(i), ByteOrder.LITTLE_ENDIAN);
        }

        // tail
        for (; i < (offset + length); i++) {
            a.set(a.get(i) + factor, i);
        }
    }

    @Override
    public void saxpy(float alpha, AbstractTensor x, AbstractTensor y, int xoffset, int yoffset, int limit) {
        Preconditions.checkArgument(x.dType() == y.dType());
        Preconditions.checkArgument(limit % 8 == 0);

        switch (x.dType()) {
            case F32 -> saxpyF32(alpha, x, y, xoffset, yoffset, limit);
            case BF16 -> saxpyBF16(alpha, x, y, xoffset, yoffset, limit);
            default -> throw new UnsupportedOperationException();
        }
    }

    static void saxpyF32(float alpha, AbstractTensor x, AbstractTensor y, int xoffset, int yoffset, int limit) {
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

    static void saxpyBF16(float alpha, AbstractTensor a, AbstractTensor b, int aoffset, int boffset, int limit)
    {
        Preconditions.checkArgument(a.dType() == DType.BF16 && b.dType() == DType.BF16);
        int upperBound = FloatVector.SPECIES_512.loopBound(limit);
        Preconditions.checkArgument(upperBound == limit);

        int ao = aoffset;
        int bo = boffset;
        int len = FloatVector.SPECIES_512.length();

        for (; ao < (aoffset + upperBound) && bo < (boffset + upperBound); ao += len, bo += len) {
            //Convert BF16 to F32
            var af = ShortVector.fromMemorySegment(ShortVector.SPECIES_256, a.getMemorySegment(), a.getMemorySegmentOffset(ao), ByteOrder.LITTLE_ENDIAN)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            //Convert BF16 to F32
            var bf = ShortVector.fromMemorySegment(ShortVector.SPECIES_256, b.getMemorySegment(), b.getMemorySegmentOffset(bo), ByteOrder.LITTLE_ENDIAN)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            var r = bf.add(af.mul(alpha));

            r.reinterpretAsInts()
                    .lanewise(VectorOperators.ASHR, BF16_BYTE_SHIFT)
                    .convertShape(VectorOperators.I2S, ShortVector.SPECIES_256, 0)
                    .intoMemorySegment(b.getMemorySegment(), b.getMemorySegmentOffset(bo), ByteOrder.LITTLE_ENDIAN);
        }

        // tail
        for (; ao < (aoffset + limit) && bo < (boffset + limit); ao++, bo++) {
            float v = a.get(ao) + alpha * b.get(bo);
            b.set(v, bo);
        }
    }

    @Override
    public void sxpby(float beta, AbstractTensor x, AbstractTensor y, int xoffset, int yoffset, int limit) {
        Preconditions.checkArgument(x.dType() == y.dType());
        Preconditions.checkArgument(limit % 8 == 0);

        switch (x.dType()) {
            case F32 -> sxpbyF32(beta, x, y, xoffset, yoffset, limit);
            case BF16 -> sxpbyBF16(beta, x, y, xoffset, yoffset, limit);
            default -> throw new UnsupportedOperationException();
        }
    }

    static void sxpbyF32(float beta, AbstractTensor x, AbstractTensor y, int xoffset, int yoffset, int limit) {
        int upperBound = SPECIES.loopBound(limit);
        int xo = xoffset;
        int yo = yoffset;
        for (; xo < (xoffset + upperBound) && yo < (yoffset + upperBound); xo += SPECIES.length(), yo += SPECIES.length()) {
            FloatVector vx = x.getFloatVector(xo);
            FloatVector vy = y.getFloatVector(yo);
            vx.add(vy.mul(beta)).intoMemorySegment(y.getMemorySegment(), y.getMemorySegmentOffset(yo), ByteOrder.LITTLE_ENDIAN);
        }

        // tail
        for (; xo < (xoffset + limit) && yo < (yoffset + limit); xo++, yo++) {
            float v = x.get(xo) + beta * y.get(yo);
            y.set(v, yo);
        }
    }

    static void sxpbyBF16(float beta, AbstractTensor x, AbstractTensor y, int xoffset, int yoffset, int limit) {
        Preconditions.checkArgument(x.dType() == DType.BF16 && y.dType() == DType.BF16);
        int upperBound = FloatVector.SPECIES_512.loopBound(limit);
        Preconditions.checkArgument(upperBound == limit);

        int xo = xoffset;
        int yo = yoffset;

        int len = FloatVector.SPECIES_512.length();

        for (; xo < (xoffset + upperBound) && yo < (yoffset + upperBound); xo += len, yo += len) {
            //Convert BF16 to F32
            var xv = ShortVector.fromMemorySegment(ShortVector.SPECIES_256, x.getMemorySegment(), x.getMemorySegmentOffset(xo), ByteOrder.LITTLE_ENDIAN)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            //Convert BF16 to F32
            var yv = ShortVector.fromMemorySegment(ShortVector.SPECIES_256, y.getMemorySegment(), y.getMemorySegmentOffset(yo), ByteOrder.LITTLE_ENDIAN)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            var res = xv.add(yv.mul(beta));

            //Turn back into BF16 and save
            res.reinterpretAsInts()
                    .lanewise(VectorOperators.ASHR, BF16_BYTE_SHIFT)
                    .convertShape(VectorOperators.I2S, ShortVector.SPECIES_256, 0)
                    .intoMemorySegment(y.getMemorySegment(), y.getMemorySegmentOffset(yo), ByteOrder.LITTLE_ENDIAN);
        }

        for (; xo < (xoffset + limit) && yo < (yoffset + limit); xo++, yo++) {
            float v = x.get(xo) + beta * y.get(yo);
            y.set(v, yo);
        }
    }
}
