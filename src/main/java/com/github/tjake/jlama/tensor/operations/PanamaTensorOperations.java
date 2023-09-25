package com.github.tjake.jlama.tensor.operations;

import com.google.common.base.Preconditions;

import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.BFloat16BufferTensor;
import com.github.tjake.jlama.tensor.FloatBufferTensor;
import com.github.tjake.jlama.tensor.Q4ByteBufferTensor;
import com.github.tjake.jlama.tensor.Q8ByteBufferTensor;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;

final public class PanamaTensorOperations implements TensorOperations
{
    public static final boolean hasAVX512 = FloatVector.SPECIES_PREFERRED == FloatVector.SPECIES_512;

    static final ByteVector Q4_BYTE_SUB;
    static final ByteVector Q4_BYTE_MASK;
    static final ByteVector Q4_BYTE_SHIFT;
    static {
        if (hasAVX512) {
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
        if (hasAVX512) {
            BF16_BYTE_SHIFT = IntVector.broadcast(IntVector.SPECIES_512, 16);
        } else {
            BF16_BYTE_SHIFT = IntVector.broadcast(IntVector.SPECIES_256, 16);
        }
    }

    @Override
    public boolean requiresOffHeapTensor() {
        return false;
    }

    @Override
    public float dotProduct(AbstractTensor a, AbstractTensor b, int aoffset, int boffset, int limit) {
        Preconditions.checkArgument(limit % 32 == 0);

        return switch (a.dType()) {
            case F32 -> switch (b.dType()) {
                case F32 -> dotProductF32((FloatBufferTensor) a, (FloatBufferTensor) b, aoffset, boffset, limit);
                case I8 -> hasAVX512 ? dotProductF32I8_512((FloatBufferTensor) a, (Q8ByteBufferTensor) b, aoffset, boffset, limit) : dotProductF32I8_256((FloatBufferTensor) a, (Q8ByteBufferTensor) b, aoffset, boffset, limit);
                //case Q5 -> dotProductF32Q5((FloatBufferTensor) a, (Q5ByteBufferTensor) b, aoffset, boffset, limit);
                case Q4 -> hasAVX512 ? dotProductF32Q4_512((FloatBufferTensor) a, (Q4ByteBufferTensor) b, aoffset, boffset, limit) : dotProductF32Q4_256((FloatBufferTensor) a, (Q4ByteBufferTensor) b, aoffset, boffset, limit);
                default -> throw new UnsupportedOperationException(b.dType().name());
            };
            case I8 -> dotProductI8((Q8ByteBufferTensor) a, (Q8ByteBufferTensor) b, aoffset, boffset, limit);
            case BF16 -> switch (b.dType()) {
                case F32 -> hasAVX512 ? dotProductBF16F32_512((BFloat16BufferTensor) a, (FloatBufferTensor) b, aoffset, boffset, limit) : dotProductBF16F32_256((BFloat16BufferTensor) a, (FloatBufferTensor) b, aoffset, boffset, limit);
                case I8 -> hasAVX512 ? dotProductBF16I8_512((BFloat16BufferTensor) a, (Q8ByteBufferTensor) b, aoffset, boffset, limit) : dotProductBF16I8_256((BFloat16BufferTensor) a, (Q8ByteBufferTensor) b, aoffset, boffset, limit);
                case Q4 -> hasAVX512 ? dotProductBF16Q4_512((BFloat16BufferTensor) a, (Q4ByteBufferTensor) b, aoffset, boffset, limit) : dotProductBF16Q4_256((BFloat16BufferTensor) a, (Q4ByteBufferTensor) b, aoffset, boffset, limit);
                case BF16 -> hasAVX512 ? dotProductBF16_512((BFloat16BufferTensor)a, (BFloat16BufferTensor)b, aoffset, boffset, limit) : dotProductBF16_256((BFloat16BufferTensor)a, (BFloat16BufferTensor)b, aoffset, boffset, limit);
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
            var af = a.getVector(ByteVector.SPECIES_64, aoffset).convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0);
            var bf = b.getVector(ByteVector.SPECIES_64, boffset).convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0);

            acc = acc.add(af.mul(bf).mul(scale));
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    private float dotProductF32I8_256(FloatBufferTensor a, Q8ByteBufferTensor b, int aoffset, int boffset, int limit) {
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
            var af = a.getVector(FloatVector.SPECIES_256, aoffset);
            var bf = b.getVector(ByteVector.SPECIES_64, boffset).convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0);
            acc = acc.add(af.mul(bf.mul(scale)));

            af = a.getVector(FloatVector.SPECIES_256, aoffset + slen);
            bf = b.getVector(ByteVector.SPECIES_64,  boffset + slen).convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0);
            acc = acc.add(af.mul(bf.mul(scale)));

            af = a.getVector(FloatVector.SPECIES_256,aoffset + slen + slen);
            bf = b.getVector(ByteVector.SPECIES_64, boffset + slen + slen).convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0);
            acc = acc.add(af.mul(bf.mul(scale)));

            af = a.getVector(FloatVector.SPECIES_256, aoffset + slen + slen + slen);
            bf = b.getVector(ByteVector.SPECIES_64, boffset + slen + slen + slen).convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0);
            acc = acc.add(af.mul(bf.mul(scale)));
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    public float dotProductF32I8_512(FloatBufferTensor a, Q8ByteBufferTensor b, int aoffset, int boffset, int limit) {
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
            var af = a.getVector(FloatVector.SPECIES_512, aoffset);
            var bf = b.getVector(ByteVector.SPECIES_128, boffset).convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0);
            acc = acc.add(af.mul(bf.mul(scale)));

            af = a.getVector(FloatVector.SPECIES_512, aoffset + slen);
            bf = b.getVector(ByteVector.SPECIES_128, boffset + slen).convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0);
            acc = acc.add(af.mul(bf.mul(scale)));

            af = a.getVector(FloatVector.SPECIES_512, aoffset + slen + slen);
            bf = b.getVector(ByteVector.SPECIES_128, boffset + slen + slen).convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0);
            acc = acc.add(af.mul(bf.mul(scale)));

            af = a.getVector(FloatVector.SPECIES_512, aoffset + slen + slen + slen);
            bf = b.getVector(ByteVector.SPECIES_128, boffset + slen + slen + slen).convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0);
            acc = acc.add(af.mul(bf.mul(scale)));
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    private float dotProductF32Q4_256(FloatBufferTensor a, Q4ByteBufferTensor b, int aoffset, int boffset, int limit)
    {
        Preconditions.checkArgument(
                boffset % Q4ByteBufferTensor.BLOCK_SIZE == 0 &&
                        limit % Q4ByteBufferTensor.BLOCK_SIZE == 0
        );

        int alim = aoffset + limit;
        int blim = boffset + limit;
        int slen = Q4ByteBufferTensor.BLOCK_SIZE;

        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_256);


        for (; aoffset < alim && boffset < blim; aoffset += slen, boffset += slen)
        {
            FloatVector scale = FloatVector.broadcast(FloatVector.SPECIES_256, b.getFactorForIndex(boffset));
            for (int i = 0; i < 2; i++)
            {
                // BLOCK_SIZE Floats
                var af0 = a.getVector(FloatVector.SPECIES_256, aoffset + (8 * i));
                var af1 = a.getVector(FloatVector.SPECIES_256, aoffset + Q4ByteBufferTensor.HALF_BLOCK + (8*i));

                //Make 8 bytes -> 16 4bit -> 16 bytes -> 16 32F
                var bf0 = b.getVector(ByteVector.SPECIES_64, boffset + (8 * i));

                // Convert the first 4 bits into bytes
                var low0 = bf0.lanewise(VectorOperators.AND, Q4_BYTE_MASK)
                        .sub(Q4_BYTE_SUB)
                        .convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0)
                        .mul(scale);

                var high0 = bf0.lanewise(VectorOperators.ASHR, Q4_BYTE_SHIFT)
                        .lanewise(VectorOperators.AND, Q4_BYTE_MASK)
                        .sub(Q4_BYTE_SUB)
                        .convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0)
                        .mul(scale);

                acc = af0.fma(low0, acc);
                acc = af1.fma(high0, acc);
            }
        }


        return acc.reduceLanes(VectorOperators.ADD);
    }

    private float dotProductF32Q4_512(FloatBufferTensor a, Q4ByteBufferTensor b, int aoffset, int boffset, int limit) {
        Preconditions.checkArgument(
                boffset % Q4ByteBufferTensor.BLOCK_SIZE == 0 &&
                        limit % Q4ByteBufferTensor.BLOCK_SIZE == 0
        );

        int alim = aoffset + limit;
        int blim = boffset + limit;
        int slen = Q4ByteBufferTensor.BLOCK_SIZE;

        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_512);

        for (; aoffset < alim && boffset < blim; aoffset += slen, boffset += slen) {
            FloatVector scale = FloatVector.broadcast(FloatVector.SPECIES_512, b.getFactorForIndex(boffset));
            // BLOCK_SIZE Floats
            var af0 = a.getVector(FloatVector.SPECIES_512, aoffset);
            var af1 = a.getVector(FloatVector.SPECIES_512, aoffset + Q4ByteBufferTensor.HALF_BLOCK);

            //Make 16 bytes -> 32 4bit -> 32 bytes -> 32 32F
            var bf0 = b.getVector(ByteVector.SPECIES_128, boffset);

            // Convert the first 4 bits into bytes
            var low0 = bf0.lanewise(VectorOperators.AND, Q4_BYTE_MASK)
                    .sub(Q4_BYTE_SUB)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0)
                    .mul(scale);

            var high0 = bf0.lanewise(VectorOperators.ASHR, Q4_BYTE_SHIFT)
                    .lanewise(VectorOperators.AND, Q4_BYTE_MASK)
                    .sub(Q4_BYTE_SUB)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0)
                    .mul(scale);

            acc = af0.fma(low0, acc);
            acc = af1.fma(high0, acc);
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    private float dotProductBF16Q4_512(BFloat16BufferTensor a, Q4ByteBufferTensor b, int aoffset, int boffset, int limit) {
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
            var af0 = a.getVector(ShortVector.SPECIES_256, aoffset)
                    .convertShape(VectorOperators.ZERO_EXTEND_S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            var af1 = a.getVector(ShortVector.SPECIES_256, aoffset + Q4ByteBufferTensor.HALF_BLOCK)
                    .convertShape(VectorOperators.ZERO_EXTEND_S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            //Make 16 bytes -> 32 4bit -> 32 bytes -> 32 32F
            var bf0 = b.getVector(ByteVector.SPECIES_128, boffset);

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

    private float dotProductBF16Q4_256(BFloat16BufferTensor a, Q4ByteBufferTensor b, int aoffset, int boffset, int limit) {
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
            var af0 = a.getVector(ShortVector.SPECIES_128, aoffset)
                    .convertShape(VectorOperators.ZERO_EXTEND_S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            var af1 = a.getVector(ShortVector.SPECIES_128, aoffset + 8)
                    .convertShape(VectorOperators.ZERO_EXTEND_S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            var af2 = a.getVector(ShortVector.SPECIES_128, aoffset + 8 + 8)
                    .convertShape(VectorOperators.ZERO_EXTEND_S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            var af3 = a.getVector(ShortVector.SPECIES_128, aoffset + 8 + 8 + 8)
                    .convertShape(VectorOperators.ZERO_EXTEND_S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            //Make 8 bytes -> 16 4bit -> 16 bytes -> 16 32F
            var bf0 = b.getVector(ByteVector.SPECIES_64, boffset);
            var bf16 = b.getVector(ByteVector.SPECIES_64, boffset + Q4ByteBufferTensor.HALF_BLOCK);

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


    public float dotProductBF16I8_256(BFloat16BufferTensor a, Q8ByteBufferTensor b, final int aoffset, final int boffset, int limit)
    {
        int ao = aoffset;
        int bo = boffset;
        final int alim = aoffset + limit;
        final int blim = boffset + limit;
        final int slen = ByteVector.SPECIES_64.length();

        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_256);

        //Unroll 4x
        for (; ao < alim && bo < blim; ao += slen, bo += slen)
        {
            FloatVector scale = FloatVector.broadcast(FloatVector.SPECIES_256, b.getFactorForIndex(bo));

            //Convert BF16 to F32
            var af = a.getVector(ShortVector.SPECIES_128, ao)
                    .convertShape(VectorOperators.ZERO_EXTEND_S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            var bf = b.getVector(ByteVector.SPECIES_64, bo)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0);

            acc = acc.add(af.mul(bf.mul(scale)));

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
        for (; ao < alim && bo < blim; ao += slen, bo += slen)
        {
            FloatVector scale = FloatVector.broadcast(FloatVector.SPECIES_512, b.getFactorForIndex(bo));

            //Convert BF16 to F32
            var af = a.getVector(ShortVector.SPECIES_256, ao)
                    .convertShape(VectorOperators.ZERO_EXTEND_S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            var bf = b.getVector(ByteVector.SPECIES_128, bo)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0);

            acc = acc.add(af.mul(bf.mul(scale)));
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    private float dotProductBF16F32_512(BFloat16BufferTensor a, FloatBufferTensor b, int aoffset, int boffset, int limit) {
        int alim = aoffset + limit;
        int blim = boffset + limit;
        int slen = ByteVector.SPECIES_128.length();

        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_512);

        //Unroll 4x
        for (; aoffset < alim && boffset < blim; aoffset += slen, boffset += slen) {

            //Convert BF16 to F32
            var af = a.getVector(ShortVector.SPECIES_256, aoffset)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            FloatVector bf = b.getVector(FloatVector.SPECIES_512, boffset);

            acc = acc.add(af.mul(bf));
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    private float dotProductBF16F32_256(BFloat16BufferTensor a, FloatBufferTensor b, int aoffset, int boffset, int limit) {
        int alim = aoffset + limit;
        int blim = boffset + limit;
        int slen = ByteVector.SPECIES_64.length();

        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_256);

        //Unroll 4x
        for (; aoffset < alim && boffset < blim; aoffset += slen, boffset += slen) {

            //Convert BF16 to F32
            var af = a.getVector(ShortVector.SPECIES_128, aoffset)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            FloatVector bf = b.getVector(FloatVector.SPECIES_256, boffset);

            acc = acc.add(af.mul(bf));
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    private float dotProductBF16_256(BFloat16BufferTensor a, BFloat16BufferTensor b, int aoffset, int boffset, int limit)
    {
        int alim = aoffset + limit;
        int blim = boffset + limit;
        int slen = ByteVector.SPECIES_64.length();

        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_256);

        //Unroll 4x
        for (; aoffset < alim && boffset < blim; aoffset += slen, boffset += slen)
        {

            //Convert BF16 to F32
            var af = a.getVector(ShortVector.SPECIES_128, aoffset)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            //Convert BF16 to F32
            var bf = b.getVector(ShortVector.SPECIES_128, boffset)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            acc = acc.add(af.mul(bf));
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    private float dotProductBF16_512(BFloat16BufferTensor a, BFloat16BufferTensor b, int aoffset, int boffset, int limit) {
        int alim = aoffset + limit;
        int blim = boffset + limit;
        int slen = ByteVector.SPECIES_128.length();

        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_512);

        //Unroll 4x
        for (; aoffset < alim && boffset < blim; aoffset += slen, boffset += slen) {

            //Convert BF16 to F32
            var af = a.getVector(ShortVector.SPECIES_256, aoffset)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            //Convert BF16 to F32
            var bf = b.getVector(ShortVector.SPECIES_256, boffset)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            acc = acc.add(af.mul(bf));
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    private float dotProductF32(FloatBufferTensor a, FloatBufferTensor b, int aoffset, int boffset, int limit) {
        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        int upperBound = FloatVector.SPECIES_PREFERRED.loopBound(limit);
        int ao = aoffset;
        int bo = boffset;
        int alim = aoffset + upperBound;
        int blim = boffset + upperBound;
        int slen = FloatVector.SPECIES_PREFERRED.length();
        for (; ao < alim && bo < blim; ao += slen*4, bo += slen*4) {
            FloatVector va = a.getVector(FloatVector.SPECIES_PREFERRED, ao);
            FloatVector vb = b.getVector(FloatVector.SPECIES_PREFERRED, bo);
            acc = va.fma(vb, acc);

            va = a.getVector(FloatVector.SPECIES_PREFERRED, ao + slen);
            vb = b.getVector(FloatVector.SPECIES_PREFERRED, bo + slen);
            acc = va.fma(vb, acc);

            va = a.getVector(FloatVector.SPECIES_PREFERRED, ao + slen + slen);
            vb = b.getVector(FloatVector.SPECIES_PREFERRED, bo + slen + slen);
            acc = va.fma(vb, acc);

            va = a.getVector(FloatVector.SPECIES_PREFERRED, ao + slen + slen + slen);
            vb = b.getVector(FloatVector.SPECIES_PREFERRED, bo + slen + slen + slen);
            acc = va.fma(vb, acc);
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
            case F32: accumulateF32((FloatBufferTensor)a, (FloatBufferTensor)b); break;
            case BF16:
                if (hasAVX512)
                    accumulateBF16_512((BFloat16BufferTensor) a, (BFloat16BufferTensor) b);
                else
                    accumulateBF16_256((BFloat16BufferTensor) a, (BFloat16BufferTensor) b);
                break;
            default: throw new UnsupportedOperationException();
        }
    }

    void accumulateF32(FloatBufferTensor a, FloatBufferTensor b) {
        int upperBound = FloatVector.SPECIES_PREFERRED.loopBound(a.size());
        int i = 0;

        for (; i < upperBound; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector va = a.getVector(FloatVector.SPECIES_PREFERRED, i);
            FloatVector vb = b.getVector(FloatVector.SPECIES_PREFERRED, i);
            a.intoTensor(va.add(vb), i);
        }

        // tail
        for (; i < a.size(); i++) {
            a.set(a.get(i) + b.get(i));
        }
    }

    void accumulateBF16_256(BFloat16BufferTensor a, BFloat16BufferTensor b) {
        int upperBound = FloatVector.SPECIES_256.loopBound(a.size());

        int i = 0;
        for (; i < upperBound; i += FloatVector.SPECIES_256.length()) {

            //Convert BF16 to F32
            var af = a.getVector(ShortVector.SPECIES_128, i)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            //Convert BF16 to F32
            var bf = b.getVector(ShortVector.SPECIES_128, i)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();


            var res = af.add(bf).reinterpretAsInts()
                    .lanewise(VectorOperators.ASHR, BF16_BYTE_SHIFT)
                    .convertShape(VectorOperators.I2S, ShortVector.SPECIES_128, 0);

            a.intoTensor((ShortVector) res , i);
        }

        // tail
        for (; i < a.size(); i++) {
            a.set(a.get(i) + b.get(i));
        }
    }

    void accumulateBF16_512(BFloat16BufferTensor a, BFloat16BufferTensor b) {
        int upperBound = FloatVector.SPECIES_512.loopBound(a.size());

        int i = 0;
        for (; i < upperBound; i += FloatVector.SPECIES_512.length()) {

            //Convert BF16 to F32
            var af = a.getVector(ShortVector.SPECIES_256, i)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            //Convert BF16 to F32
            var bf = b.getVector(ShortVector.SPECIES_256, i)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();


            var res = af.add(bf).reinterpretAsInts()
                    .lanewise(VectorOperators.ASHR, BF16_BYTE_SHIFT)
                    .convertShape(VectorOperators.I2S, ShortVector.SPECIES_256, 0);

            a.intoTensor((ShortVector) res , i);
        }

        // tail
        for (; i < a.size(); i++) {
            a.set(a.get(i) + b.get(i));
        }
    }

    @Override
    public void scale(float factor, AbstractTensor a, int offset, int length)
    {
        switch (a.dType()) {
            case F32: scaleF32(factor, (FloatBufferTensor) a, offset, length); break;
            case BF16:
                if (hasAVX512)
                    scaleBF16_512(factor, (BFloat16BufferTensor) a, offset, length);
                else
                    scaleBF16_256(factor, (BFloat16BufferTensor) a, offset, length);
                break;
            default: throw new UnsupportedOperationException();
        }
    }

    public void scaleF32(float factor, FloatBufferTensor a, int offset, int length)
    {
        int upperBound = FloatVector.SPECIES_PREFERRED.loopBound(offset + length);
        int i = offset;

        FloatVector sf = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, factor);
        for (; i < upperBound; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector va = a.getVector(FloatVector.SPECIES_PREFERRED, i);
            a.intoTensor(va.mul(sf), i);
        }

        // tail
        for (; i < (offset + length); i++) {
            a.set(a.get(i) + factor, i);
        }
    }

    public void scaleBF16_512(float factor, BFloat16BufferTensor a, int offset, int length)
    {
        int upperBound = FloatVector.SPECIES_512.loopBound(offset + length);
        int i = offset;

        FloatVector sf = FloatVector.broadcast(FloatVector.SPECIES_512, factor);
        for (; i < upperBound; i += FloatVector.SPECIES_512.length()) {
            var va = a.getVector(ShortVector.SPECIES_256, i)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            var res = va.mul(sf).reinterpretAsInts()
                    .lanewise(VectorOperators.ASHR, BF16_BYTE_SHIFT)
                    .convertShape(VectorOperators.I2S, ShortVector.SPECIES_256, 0);

            a.intoTensor((ShortVector) res, i);
        }

        // tail
        for (; i < (offset + length); i++) {
            a.set(a.get(i) + factor, i);
        }
    }

    public void scaleBF16_256(float factor, BFloat16BufferTensor a, int offset, int length)
    {
        int upperBound = FloatVector.SPECIES_256.loopBound(offset + length);
        int i = offset;

        FloatVector sf = FloatVector.broadcast(FloatVector.SPECIES_256, factor);
        for (; i < upperBound; i += FloatVector.SPECIES_256.length()) {
            var va = a.getVector(ShortVector.SPECIES_128, i)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            var res = va.mul(sf).reinterpretAsInts()
                    .lanewise(VectorOperators.ASHR, BF16_BYTE_SHIFT)
                    .convertShape(VectorOperators.I2S, ShortVector.SPECIES_128, 0);

            a.intoTensor((ShortVector) res, i);
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
            case F32: saxpyF32(alpha, (FloatBufferTensor) x, (FloatBufferTensor) y, xoffset, yoffset, limit); break;
            case BF16:
                if (hasAVX512)
                    saxpyBF16_512(alpha, (BFloat16BufferTensor) x, (BFloat16BufferTensor) y, xoffset, yoffset, limit);
                else
                    saxpyBF16_256(alpha, (BFloat16BufferTensor) x, (BFloat16BufferTensor) y, xoffset, yoffset, limit);
                break;
            default: throw new UnsupportedOperationException();
        }
    }

    void saxpyF32(float alpha, FloatBufferTensor x, FloatBufferTensor y, int xoffset, int yoffset, int limit) {
        int upperBound = FloatVector.SPECIES_PREFERRED.loopBound(limit);
        int xo = xoffset;
        int yo = yoffset;
        for (; xo < (xoffset + upperBound) && yo < (yoffset + upperBound); xo += FloatVector.SPECIES_PREFERRED.length(), yo += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector vx = x.getVector(FloatVector.SPECIES_PREFERRED, xo);
            FloatVector vy = y.getVector(FloatVector.SPECIES_PREFERRED, yo);
            y.intoTensor(vy.add(vx.mul(alpha)), yo);
        }

        // tail
        for (; xo < (xoffset + limit) && yo < (yoffset + limit); xo++, yo++) {
            float v = y.get(yo) + (alpha * x.get(xo));
            y.set(v, yo);
        }
    }

    void saxpyBF16_256(float alpha, BFloat16BufferTensor a, BFloat16BufferTensor b, int aoffset, int boffset, int limit) {
        int upperBound = FloatVector.SPECIES_256.loopBound(limit);
        Preconditions.checkArgument(upperBound == limit);

        int ao = aoffset;
        int bo = boffset;
        int len = FloatVector.SPECIES_256.length();

        for (; ao < (aoffset + upperBound) && bo < (boffset + upperBound); ao += len, bo += len) {
            //Convert BF16 to F32
            var af = a.getVector(ShortVector.SPECIES_128, ao)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            //Convert BF16 to F32
            var bf = b.getVector(ShortVector.SPECIES_128, bo)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            var r = bf.add(af.mul(alpha)).reinterpretAsInts()
                    .lanewise(VectorOperators.ASHR, BF16_BYTE_SHIFT)
                    .convertShape(VectorOperators.I2S, ShortVector.SPECIES_128, 0);

            b.intoTensor((ShortVector) r, bo);
        }

        // tail
        for (; ao < (aoffset + limit) && bo < (boffset + limit); ao++, bo++) {
            float v = a.get(ao) + alpha * b.get(bo);
            b.set(v, bo);
        }
    }

    void saxpyBF16_512(float alpha, BFloat16BufferTensor a, BFloat16BufferTensor b, int aoffset, int boffset, int limit) {
        int upperBound = FloatVector.SPECIES_512.loopBound(limit);
        Preconditions.checkArgument(upperBound == limit);

        int ao = aoffset;
        int bo = boffset;
        int len = FloatVector.SPECIES_512.length();

        for (; ao < (aoffset + upperBound) && bo < (boffset + upperBound); ao += len, bo += len) {
            //Convert BF16 to F32
            var af = a.getVector(ShortVector.SPECIES_256, ao)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            //Convert BF16 to F32
            var bf = b.getVector(ShortVector.SPECIES_256, bo)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            var r = bf.add(af.mul(alpha)).reinterpretAsInts()
                    .lanewise(VectorOperators.ASHR, BF16_BYTE_SHIFT)
                    .convertShape(VectorOperators.I2S, ShortVector.SPECIES_256, 0);

            b.intoTensor((ShortVector) r, bo);
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
            case F32: sxpbyF32(beta, (FloatBufferTensor) x, (FloatBufferTensor) y, xoffset, yoffset, limit); break;
            case BF16:
                if (hasAVX512)
                    sxpbyBF16_512(beta, (BFloat16BufferTensor)x, (BFloat16BufferTensor)y, xoffset, yoffset, limit);
                else
                    sxpbyBF16_256(beta, (BFloat16BufferTensor)x, (BFloat16BufferTensor)y, xoffset, yoffset, limit);
                break;
            default: throw new UnsupportedOperationException();
        }
    }

    void sxpbyF32(float beta, FloatBufferTensor x, FloatBufferTensor y, int xoffset, int yoffset, int limit) {
        int upperBound = FloatVector.SPECIES_PREFERRED.loopBound(limit);
        int xo = xoffset;
        int yo = yoffset;
        for (; xo < (xoffset + upperBound) && yo < (yoffset + upperBound); xo += FloatVector.SPECIES_PREFERRED.length(), yo += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector vx = x.getVector(FloatVector.SPECIES_PREFERRED, xo);
            FloatVector vy = y.getVector(FloatVector.SPECIES_PREFERRED, yo);
            y.intoTensor(vx.add(vy.mul(beta)), yo);
        }

        // tail
        for (; xo < (xoffset + limit) && yo < (yoffset + limit); xo++, yo++) {
            float v = x.get(xo) + beta * y.get(yo);
            y.set(v, yo);
        }
    }

    void sxpbyBF16_256(float beta, BFloat16BufferTensor x, BFloat16BufferTensor y, int xoffset, int yoffset, int limit) {
        int upperBound = FloatVector.SPECIES_256.loopBound(limit);
        Preconditions.checkArgument(upperBound == limit);

        int xo = xoffset;
        int yo = yoffset;

        int len = FloatVector.SPECIES_256.length();

        for (; xo < (xoffset + upperBound) && yo < (yoffset + upperBound); xo += len, yo += len) {
            //Convert BF16 to F32
            var xv = x.getVector(ShortVector.SPECIES_128, xo)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            //Convert BF16 to F32
            var yv = y.getVector(ShortVector.SPECIES_128, yo)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            var res = xv.add(yv.mul(beta));

            //Turn back into BF16 and save
            y.intoTensor((ShortVector) res.reinterpretAsInts()
                            .lanewise(VectorOperators.ASHR, BF16_BYTE_SHIFT)
                            .convertShape(VectorOperators.I2S, ShortVector.SPECIES_128, 0)
                    ,yo);
        }

        for (; xo < (xoffset + limit) && yo < (yoffset + limit); xo++, yo++) {
            float v = x.get(xo) + beta * y.get(yo);
            y.set(v, yo);
        }
    }

    void sxpbyBF16_512(float beta, BFloat16BufferTensor x, BFloat16BufferTensor y, int xoffset, int yoffset, int limit) {
        int upperBound = FloatVector.SPECIES_512.loopBound(limit);
        Preconditions.checkArgument(upperBound == limit);

        int xo = xoffset;
        int yo = yoffset;

        int len = FloatVector.SPECIES_512.length();

        for (; xo < (xoffset + upperBound) && yo < (yoffset + upperBound); xo += len, yo += len) {
            //Convert BF16 to F32
            var xv = x.getVector(ShortVector.SPECIES_256, xo)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            //Convert BF16 to F32
            var yv = y.getVector(ShortVector.SPECIES_256, yo)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            var res = xv.add(yv.mul(beta));

            //Turn back into BF16 and save
            y.intoTensor((ShortVector) res.reinterpretAsInts()
                            .lanewise(VectorOperators.ASHR, BF16_BYTE_SHIFT)
                            .convertShape(VectorOperators.I2S, ShortVector.SPECIES_256, 0)
                    ,yo);
        }

        for (; xo < (xoffset + limit) && yo < (yoffset + limit); xo++, yo++) {
            float v = x.get(xo) + beta * y.get(yo);
            y.set(v, yo);
        }
    }
}
