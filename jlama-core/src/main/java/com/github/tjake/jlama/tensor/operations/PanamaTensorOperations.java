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
package com.github.tjake.jlama.tensor.operations;

import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.BFloat16BufferTensor;
import com.github.tjake.jlama.tensor.FloatBufferTensor;
import com.github.tjake.jlama.tensor.Q4ByteBufferTensor;
import com.github.tjake.jlama.tensor.Q8ByteBufferTensor;
import com.github.tjake.jlama.tensor.TensorCache;
import com.github.tjake.jlama.util.MachineSpec;
import com.google.common.base.Preconditions;
import jdk.incubator.vector.*;

public final class PanamaTensorOperations implements TensorOperations {
    static final ByteVector Q4_BYTE_SUB_128 = ByteVector.broadcast(ByteVector.SPECIES_128, 8);
    static final ByteVector Q4_BYTE_MASK_128 = ByteVector.broadcast(ByteVector.SPECIES_128, 0xF);
    static final ByteVector Q4_BYTE_SHIFT_128 = ByteVector.broadcast(ByteVector.SPECIES_128, 4);

    static final ByteVector Q4_BYTE_SUB_64 = ByteVector.broadcast(ByteVector.SPECIES_64, 8);
    static final ByteVector Q4_BYTE_MASK_64 = ByteVector.broadcast(ByteVector.SPECIES_64, 0xF);
    static final ByteVector Q4_BYTE_SHIFT_64 = ByteVector.broadcast(ByteVector.SPECIES_64, 4);

    static final IntVector BF16_BYTE_SHIFT_512 = IntVector.broadcast(IntVector.SPECIES_512, 16);
    static final FloatVector F32_ROUND_UP_512 = FloatVector.broadcast(FloatVector.SPECIES_512, 0.5f);

    static final IntVector BF16_BYTE_SHIFT_256 = IntVector.broadcast(IntVector.SPECIES_256, 16);
    static final FloatVector F32_ROUND_UP_256 = FloatVector.broadcast(FloatVector.SPECIES_256, 0.5f);

    static final FloatVector F32_ROUND_UP_128 = FloatVector.broadcast(FloatVector.SPECIES_128, 0.5f);

    static final VectorMask<Byte> BYTE_MASK_32 =
            VectorMask.fromValues(ByteVector.SPECIES_64, true, true, true, true, false, false, false, false);

    private final MachineSpec.Type vectorType;

    public PanamaTensorOperations(MachineSpec.Type vectorType) {
        this.vectorType = vectorType;
    }

    @Override
    public String name() {
        return "Panama Vector Operations";
    }

    @Override
    public boolean requiresOffHeapTensor() {
        return true;
    }

    @Override
    public float dotProduct(AbstractTensor a, AbstractTensor b, int aoffset, int boffset, int limit) {
        Preconditions.checkArgument(limit % 2 == 0, "Limit must be a multiple of 2");

        return switch (a.dType()) {
            case F32 -> switch (b.dType()) {
                case F32 -> dotProductF32((FloatBufferTensor) a, (FloatBufferTensor) b, aoffset, boffset, limit);
                case I8 -> switch (vectorType) {
                    case AVX_512 -> dotProductF32I8_512(
                            (FloatBufferTensor) a, (Q8ByteBufferTensor) b, aoffset, boffset, limit);
                    case AVX_256 -> dotProductF32I8_256(
                            (FloatBufferTensor) a, (Q8ByteBufferTensor) b, aoffset, boffset, limit);
                    default -> throw new UnsupportedOperationException(MachineSpec.VECTOR_TYPE.name());
                };
                    // case Q5 -> dotProductF32Q5((FloatBufferTensor) a, (Q5ByteBufferTensor) b, aoffset, boffset,
                    // limit);
                case Q4 -> switch (vectorType) {
                    case AVX_512 -> dotProductF32Q4_512(
                            (FloatBufferTensor) a, (Q4ByteBufferTensor) b, aoffset, boffset, limit);
                    case AVX_256 -> dotProductF32Q4_256(
                            (FloatBufferTensor) a, (Q4ByteBufferTensor) b, aoffset, boffset, limit);
                    case ARM_128 -> dotProductF32Q4_arm(
                            (FloatBufferTensor) a, (Q4ByteBufferTensor) b, aoffset, boffset, limit);
                    default -> throw new UnsupportedOperationException(MachineSpec.VECTOR_TYPE.name());
                };
                default -> throw new UnsupportedOperationException(b.dType().name());
            };
            case I8 -> switch (b.dType()) {
                case I8 -> switch (vectorType) {
                    case AVX_512 -> dotProductI8_512(
                            (Q8ByteBufferTensor) a, (Q8ByteBufferTensor) b, aoffset, boffset, limit);
                    case AVX_256 -> dotProductI8_256(
                            (Q8ByteBufferTensor) a, (Q8ByteBufferTensor) b, aoffset, boffset, limit);
                    default -> throw new UnsupportedOperationException();
                };
                case Q4 -> switch (vectorType) {
                    case AVX_512 -> QDotProductI8Q4_512(
                            (Q8ByteBufferTensor) a, (Q4ByteBufferTensor) b, aoffset, boffset, limit);
                    case AVX_256 -> QDotProductI8Q4_256(
                            (Q8ByteBufferTensor) a, (Q4ByteBufferTensor) b, aoffset, boffset, limit);
                    default -> throw new UnsupportedOperationException();
                };
                default -> throw new UnsupportedOperationException(b.dType().name());
            };
            case BF16 -> switch (b.dType()) {
                case F32 -> switch (vectorType) {
                    case AVX_512 -> dotProductBF16F32_512(
                            (BFloat16BufferTensor) a, (FloatBufferTensor) b, aoffset, boffset, limit);
                    case AVX_256 -> dotProductBF16F32_256(
                            (BFloat16BufferTensor) a, (FloatBufferTensor) b, aoffset, boffset, limit);
                    default -> throw new UnsupportedOperationException(MachineSpec.VECTOR_TYPE.name());
                };
                case I8 -> switch (vectorType) {
                    case AVX_512 -> dotProductBF16I8_512(
                            (BFloat16BufferTensor) a, (Q8ByteBufferTensor) b, aoffset, boffset, limit);
                    case AVX_256 -> dotProductBF16I8_256(
                            (BFloat16BufferTensor) a, (Q8ByteBufferTensor) b, aoffset, boffset, limit);
                    default -> throw new UnsupportedOperationException(MachineSpec.VECTOR_TYPE.name());
                };
                case Q4 -> switch (vectorType) {
                    case AVX_512 -> dotProductBF16Q4_512(
                            (BFloat16BufferTensor) a, (Q4ByteBufferTensor) b, aoffset, boffset, limit);
                    case AVX_256 -> dotProductBF16Q4_256(
                            (BFloat16BufferTensor) a, (Q4ByteBufferTensor) b, aoffset, boffset, limit);
                    default -> throw new UnsupportedOperationException(MachineSpec.VECTOR_TYPE.name());
                };
                case BF16 -> switch (vectorType) {
                    case AVX_512 -> dotProductBF16_512(
                            (BFloat16BufferTensor) a, (BFloat16BufferTensor) b, aoffset, boffset, limit);
                    case AVX_256 -> dotProductBF16_256(
                            (BFloat16BufferTensor) a, (BFloat16BufferTensor) b, aoffset, boffset, limit);
                    default -> throw new UnsupportedOperationException(MachineSpec.VECTOR_TYPE.name());
                };
                default -> throw new UnsupportedOperationException(b.dType().name());
            };
            default -> throw new UnsupportedOperationException();
        };
    }

    @Override
    public AbstractTensor quantize(AbstractTensor t, DType qtype, int offset, int length) {
        Preconditions.checkArgument(t.dims() == 1);

        return switch (t.dType()) {
            case F32 -> switch (qtype) {
                case I8 -> switch (vectorType) {
                    case AVX_512 -> quantizeQ8_512((FloatBufferTensor) t, offset, length);
                    case AVX_256 -> quantizeQ8_256((FloatBufferTensor) t, offset, length);
                    case ARM_128 -> quantizeQ8_arm((FloatBufferTensor) t, offset, length);
                    default -> throw new UnsupportedOperationException();
                };
                default -> throw new UnsupportedOperationException();
            };
            default -> throw new UnsupportedOperationException();
        };
    }

    public Q8ByteBufferTensor quantizeQ8_512(FloatBufferTensor ft, int offset, int length) {
        Preconditions.checkArgument(length % Q8ByteBufferTensor.BLOCK_SIZE == 0 && ft.dims() == 1);

        // Up to caller to release
        Q8ByteBufferTensor qft = (Q8ByteBufferTensor) TensorCache.instance.get(DType.I8, ft.shape());

        for (int i = offset; i < offset + length; i += Q8ByteBufferTensor.BLOCK_SIZE) {
            FloatVector fv0 = ft.getVector(FloatVector.SPECIES_512, i);
            FloatVector fv1 = ft.getVector(FloatVector.SPECIES_512, i + 16);

            // Compute max abs
            var maxAbs0 = fv0.abs();
            var maxAbs1 = fv1.abs();

            float maxScalar = maxAbs0.max(maxAbs1).reduceLanes(VectorOperators.MAX);

            // Quantize these floats
            float d = maxScalar / 127f;
            float id = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;

            var vid = FloatVector.broadcast(FloatVector.SPECIES_512, id);
            var fvq0 = fv0.mul(vid).add(F32_ROUND_UP_512); // rounding
            var fvq1 = fv1.mul(vid).add(F32_ROUND_UP_512); // rounding

            // Squash to bytes (rounds internally)
            var bvq0 = fvq0.convertShape(VectorOperators.F2B, ByteVector.SPECIES_128, 0)
                    .reinterpretAsBytes();
            var bvq1 = fvq1.convertShape(VectorOperators.F2B, ByteVector.SPECIES_128, 0)
                    .reinterpretAsBytes();

            qft.intoTensor(bvq0, i);
            qft.intoTensor(bvq1, i + 16);
            qft.getBlockF().set(d, (int) (i * Q8ByteBufferTensor.I_BLOCK_SIZE));
        }

        return qft;
    }

    public Q8ByteBufferTensor quantizeQ8_256(FloatBufferTensor ft, int offset, int length) {
        Preconditions.checkArgument(length % Q8ByteBufferTensor.BLOCK_SIZE == 0 && ft.dims() == 1);

        // Up to caller to release
        Q8ByteBufferTensor qft = (Q8ByteBufferTensor) TensorCache.instance.get(DType.I8, ft.shape());

        for (int i = offset; i < offset + length; i += Q8ByteBufferTensor.BLOCK_SIZE) {
            FloatVector fv0 = ft.getVector(FloatVector.SPECIES_256, i);
            FloatVector fv1 = ft.getVector(FloatVector.SPECIES_256, i + 8);
            FloatVector fv2 = ft.getVector(FloatVector.SPECIES_256, i + 16);
            FloatVector fv3 = ft.getVector(FloatVector.SPECIES_256, i + 24);

            // Compute max abs
            var maxAbs0 = fv0.abs();
            var maxAbs1 = fv1.abs();
            var maxAbs2 = fv2.abs();
            var maxAbs3 = fv3.abs();

            var m0 = maxAbs0.max(maxAbs1);
            var m1 = maxAbs2.max(maxAbs3);
            float maxScalar = m0.max(m1).reduceLanes(VectorOperators.MAX);

            // Quantize these floats
            float d = maxScalar / 127f;
            float id = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;

            var vid = FloatVector.broadcast(FloatVector.SPECIES_256, id);
            var fvq0 = fv0.mul(vid).add(F32_ROUND_UP_256); // rounding
            var fvq1 = fv1.mul(vid).add(F32_ROUND_UP_256); // rounding
            var fvq2 = fv2.mul(vid).add(F32_ROUND_UP_256); // rounding
            var fvq3 = fv3.mul(vid).add(F32_ROUND_UP_256); // rounding

            // Squash to bytes (rounds internally)
            var bvq0 = fvq0.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0)
                    .reinterpretAsBytes();
            var bvq1 = fvq1.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0)
                    .reinterpretAsBytes();
            var bvq2 = fvq2.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0)
                    .reinterpretAsBytes();
            var bvq3 = fvq3.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0)
                    .reinterpretAsBytes();

            qft.intoTensor(bvq0, i);
            qft.intoTensor(bvq1, i + 8);
            qft.intoTensor(bvq2, i + 16);
            qft.intoTensor(bvq3, i + 24);

            qft.getBlockF().set(d, (int) (i * Q8ByteBufferTensor.I_BLOCK_SIZE));
        }

        return qft;
    }

    public Q8ByteBufferTensor quantizeQ8_arm(FloatBufferTensor ft, int offset, int length) {
        Preconditions.checkArgument(length % Q8ByteBufferTensor.BLOCK_SIZE == 0 && ft.dims() == 1);

        // Up to caller to release
        Q8ByteBufferTensor qft = (Q8ByteBufferTensor) TensorCache.instance.get(DType.I8, ft.shape());

        for (int i = offset; i < offset + length; i += Q8ByteBufferTensor.BLOCK_SIZE) {
            FloatVector fv0 = ft.getVector(FloatVector.SPECIES_128, i);
            FloatVector fv1 = ft.getVector(FloatVector.SPECIES_128, i + 4);
            FloatVector fv2 = ft.getVector(FloatVector.SPECIES_128, i + 8);
            FloatVector fv3 = ft.getVector(FloatVector.SPECIES_128, i + 12);
            FloatVector fv4 = ft.getVector(FloatVector.SPECIES_128, i + 16);
            FloatVector fv5 = ft.getVector(FloatVector.SPECIES_128, i + 20);
            FloatVector fv6 = ft.getVector(FloatVector.SPECIES_128, i + 24);
            FloatVector fv7 = ft.getVector(FloatVector.SPECIES_128, i + 28);

            // Compute max abs
            var maxAbs0 = fv0.abs();
            var maxAbs1 = fv1.abs();
            var maxAbs2 = fv2.abs();
            var maxAbs3 = fv3.abs();
            var maxAbs4 = fv4.abs();
            var maxAbs5 = fv5.abs();
            var maxAbs6 = fv6.abs();
            var maxAbs7 = fv7.abs();

            var m0 = maxAbs0.max(maxAbs1);
            var m1 = maxAbs2.max(maxAbs3);
            var m2 = maxAbs4.max(maxAbs5);
            var m3 = maxAbs6.max(maxAbs7);

            var m4 = m0.max(m1);
            var m5 = m2.max(m3);

            float maxScalar = m4.max(m5).reduceLanes(VectorOperators.MAX);

            // Quantize these floats
            float d = maxScalar / 127f;
            float id = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;

            var vid = FloatVector.broadcast(FloatVector.SPECIES_128, id);
            var fvq0 = fv0.mul(vid).add(F32_ROUND_UP_128); // rounding
            var fvq1 = fv1.mul(vid).add(F32_ROUND_UP_128); // rounding
            var fvq2 = fv2.mul(vid).add(F32_ROUND_UP_128); // rounding
            var fvq3 = fv3.mul(vid).add(F32_ROUND_UP_128); // rounding
            var fvq4 = fv4.mul(vid).add(F32_ROUND_UP_128); // rounding
            var fvq5 = fv5.mul(vid).add(F32_ROUND_UP_128); // rounding
            var fvq6 = fv6.mul(vid).add(F32_ROUND_UP_128); // rounding
            var fvq7 = fv7.mul(vid).add(F32_ROUND_UP_128); // rounding

            // Squash to bytes (rounds internally)
            var bvq0 = fvq0.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0)
                    .reinterpretAsBytes();
            var bvq1 = fvq1.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0)
                    .reinterpretAsBytes();
            var bvq2 = fvq2.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0)
                    .reinterpretAsBytes();
            var bvq3 = fvq3.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0)
                    .reinterpretAsBytes();
            var bvq4 = fvq4.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0)
                    .reinterpretAsBytes();
            var bvq5 = fvq5.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0)
                    .reinterpretAsBytes();
            var bvq6 = fvq6.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0)
                    .reinterpretAsBytes();
            var bvq7 = fvq7.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0)
                    .reinterpretAsBytes();

            qft.intoTensor(bvq0, i, BYTE_MASK_32);
            qft.intoTensor(bvq1, i + 4, BYTE_MASK_32);
            qft.intoTensor(bvq2, i + 8, BYTE_MASK_32);
            qft.intoTensor(bvq3, i + 12, BYTE_MASK_32);
            qft.intoTensor(bvq4, i + 16, BYTE_MASK_32);
            qft.intoTensor(bvq5, i + 20, BYTE_MASK_32);
            qft.intoTensor(bvq6, i + 24, BYTE_MASK_32);
            qft.intoTensor(bvq7, i + 28, BYTE_MASK_32);

            qft.getBlockF().set(d, (int) (i * Q8ByteBufferTensor.I_BLOCK_SIZE));
        }

        return qft;
    }

    public float dotProductI8_256(Q8ByteBufferTensor a, Q8ByteBufferTensor b, int aoffset, int boffset, int limit) {
        Preconditions.checkArgument(aoffset % Q8ByteBufferTensor.BLOCK_SIZE == 0
                && boffset % Q8ByteBufferTensor.BLOCK_SIZE == 0
                && limit % Q8ByteBufferTensor.BLOCK_SIZE == 0);

        int slen = Q8ByteBufferTensor.BLOCK_SIZE;

        int blocksNeeded = limit / Q8ByteBufferTensor.BLOCK_SIZE;

        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_256);

        for (int i = 0; i < blocksNeeded; i += FloatVector.SPECIES_256.length()) {
            var ablock =
                    a.getBlockF().getVector(FloatVector.SPECIES_256, (int) (Q8ByteBufferTensor.I_BLOCK_SIZE * aoffset));
            var bblock =
                    b.getBlockF().getVector(FloatVector.SPECIES_256, (int) (Q8ByteBufferTensor.I_BLOCK_SIZE * boffset));

            var scales = ablock.mul(bblock);
            for (int j = 0; j < FloatVector.SPECIES_256.length(); j++, aoffset += slen, boffset += slen) {
                var scale = FloatVector.broadcast(FloatVector.SPECIES_256, scales.lane(j));

                var af1 = a.getVector(ByteVector.SPECIES_128, aoffset)
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0)
                        .reinterpretAsShorts();

                var af2 = a.getVector(ByteVector.SPECIES_128, aoffset + 16)
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0)
                        .reinterpretAsShorts();

                var bf1 = b.getVector(ByteVector.SPECIES_128, boffset)
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0)
                        .reinterpretAsShorts();

                var bf2 = b.getVector(ByteVector.SPECIES_128, boffset + 16)
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0)
                        .reinterpretAsShorts();

                var isum = af1.mul(bf1);
                isum = isum.add(af2.mul(bf2));

                acc = scale.fma(isum.convert(VectorOperators.S2F, 0), acc);
                acc = scale.fma(isum.convert(VectorOperators.S2F, 1), acc);
            }
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    public float dotProductI8_512(Q8ByteBufferTensor a, Q8ByteBufferTensor b, int aoffset, int boffset, int limit) {
        Preconditions.checkArgument(aoffset % Q8ByteBufferTensor.BLOCK_SIZE == 0
                && boffset % Q8ByteBufferTensor.BLOCK_SIZE == 0
                && limit % Q8ByteBufferTensor.BLOCK_SIZE == 0);

        int slen = Q8ByteBufferTensor.BLOCK_SIZE;

        int blocksNeeded = limit / Q8ByteBufferTensor.BLOCK_SIZE;

        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_512);

        for (int i = 0; i < blocksNeeded; i += FloatVector.SPECIES_512.length()) {
            var ablock =
                    a.getBlockF().getVector(FloatVector.SPECIES_512, (int) (Q8ByteBufferTensor.I_BLOCK_SIZE * aoffset));
            var bblock =
                    b.getBlockF().getVector(FloatVector.SPECIES_512, (int) (Q8ByteBufferTensor.I_BLOCK_SIZE * boffset));

            var scales = ablock.mul(bblock);
            for (int j = 0; j < FloatVector.SPECIES_512.length(); j++, aoffset += slen, boffset += slen) {
                var scale = FloatVector.broadcast(FloatVector.SPECIES_512, scales.lane(j));

                var af1 = a.getVector(ByteVector.SPECIES_256, aoffset)
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_512, 0)
                        .reinterpretAsShorts();

                var bf1 = b.getVector(ByteVector.SPECIES_256, boffset)
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_512, 0)
                        .reinterpretAsShorts();

                var isum = af1.mul(bf1);

                acc = scale.fma(isum.convert(VectorOperators.S2F, 0), acc);
                acc = scale.fma(isum.convert(VectorOperators.S2F, 1), acc);
            }
        }
        return acc.reduceLanes(VectorOperators.ADD);
    }

    public float QDotProductI8Q4_256(Q8ByteBufferTensor a, Q4ByteBufferTensor b, int aoffset, int boffset, int limit) {
        Preconditions.checkArgument(aoffset % Q8ByteBufferTensor.BLOCK_SIZE == 0
                && boffset % Q8ByteBufferTensor.BLOCK_SIZE == 0
                && limit % Q8ByteBufferTensor.BLOCK_SIZE == 0);

        final int blockSize = Q8ByteBufferTensor.BLOCK_SIZE;
        final int blocksNeeded = limit / Q8ByteBufferTensor.BLOCK_SIZE;

        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_256);

        // First take the scaling factors of both tensors and multiply them in SIMD
        for (int i = 0; i < blocksNeeded; i += FloatVector.SPECIES_256.length()) {
            final var ablock =
                    a.getBlockF().getVector(FloatVector.SPECIES_256, (int) (Q8ByteBufferTensor.I_BLOCK_SIZE * aoffset));
            final var bblock =
                    b.getBlockF().getVector(FloatVector.SPECIES_256, (int) (Q8ByteBufferTensor.I_BLOCK_SIZE * boffset));

            final var scales = ablock.mul(bblock);
            // Now for each scalar fetch the corresponding block of data and dot product them
            for (int j = 0; j < FloatVector.SPECIES_256.length(); j++, aoffset += blockSize, boffset += blockSize) {
                final var scale = FloatVector.broadcast(FloatVector.SPECIES_256, scales.lane(j));

                final var af0 = a.getVector(ByteVector.SPECIES_128, aoffset)
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0)
                        .reinterpretAsShorts();

                final var af1 = a.getVector(ByteVector.SPECIES_128, aoffset + 16)
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0)
                        .reinterpretAsShorts();

                // Make 16 bytes -> 32 4bit -> 32 bytes -> 32 32F
                final var bf0 = b.getVector(ByteVector.SPECIES_128, boffset);

                // Convert the first 4 bits into bytes
                final var low0 = bf0.lanewise(VectorOperators.AND, Q4_BYTE_MASK_128)
                        .sub(Q4_BYTE_SUB_128)
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);

                final var high0 = bf0.lanewise(VectorOperators.ASHR, Q4_BYTE_SHIFT_128)
                        .lanewise(VectorOperators.AND, Q4_BYTE_MASK_128)
                        .sub(Q4_BYTE_SUB_128)
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);

                var isum = low0.mul(af0);
                isum = isum.add(high0.mul(af1));

                final var r0 = isum.convertShape(VectorOperators.S2F, FloatVector.SPECIES_256, 0)
                        .reinterpretAsFloats();
                final var r1 = isum.convertShape(VectorOperators.S2F, FloatVector.SPECIES_256, 1)
                        .reinterpretAsFloats();

                acc = scale.fma(r0, acc);
                acc = scale.fma(r1, acc);
            }
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    private float QDotProductI8Q4_512(Q8ByteBufferTensor a, Q4ByteBufferTensor b, int aoffset, int boffset, int limit) {
        Preconditions.checkArgument(aoffset % Q8ByteBufferTensor.BLOCK_SIZE == 0
                && boffset % Q8ByteBufferTensor.BLOCK_SIZE == 0
                && limit % Q8ByteBufferTensor.BLOCK_SIZE == 0);

        final int blockSize = Q8ByteBufferTensor.BLOCK_SIZE;
        final int blocksNeeded = limit / Q8ByteBufferTensor.BLOCK_SIZE;

        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_512);

        // First take the scaling factors of both tensors and multiply them in SIMD
        for (int i = 0; i < blocksNeeded; i += FloatVector.SPECIES_512.length()) {
            final var ablock =
                    a.getBlockF().getVector(FloatVector.SPECIES_512, (int) (Q8ByteBufferTensor.I_BLOCK_SIZE * aoffset));
            final var bblock =
                    b.getBlockF().getVector(FloatVector.SPECIES_512, (int) (Q8ByteBufferTensor.I_BLOCK_SIZE * boffset));

            final var scales = ablock.mul(bblock);
            // Now for each scalar fetch the corresponding block of data and dot product them
            for (int j = 0; j < FloatVector.SPECIES_512.length(); j++, aoffset += blockSize, boffset += blockSize) {
                final var scale = FloatVector.broadcast(FloatVector.SPECIES_512, scales.lane(j));

                final var af = a.getVector(ByteVector.SPECIES_256, aoffset)
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_512, 0)
                        .reinterpretAsShorts();

                // Make 16 bytes -> 32 4bit -> 32 bytes -> 32 32F
                final var bf0 = b.getVector(ByteVector.SPECIES_128, boffset);

                // Convert the first 4 bits into bytes
                final var low0 = bf0.lanewise(VectorOperators.AND, Q4_BYTE_MASK_128)
                        .sub(Q4_BYTE_SUB_128)
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);

                final var high0 = bf0.lanewise(VectorOperators.ASHR, Q4_BYTE_SHIFT_128)
                        .lanewise(VectorOperators.AND, Q4_BYTE_MASK_128)
                        .sub(Q4_BYTE_SUB_128)
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);

                var isum = low0.mul(af.castShape(ShortVector.SPECIES_256, 0));
                isum = isum.add(high0.mul(af.castShape(ShortVector.SPECIES_256, 1)));

                final var r0 = isum.convertShape(VectorOperators.S2F, FloatVector.SPECIES_512, 0)
                        .reinterpretAsFloats();

                acc = scale.fma(r0, acc);
            }
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    private float dotProductF32I8_256(FloatBufferTensor a, Q8ByteBufferTensor b, int aoffset, int boffset, int limit) {
        Preconditions.checkArgument(
                boffset % Q8ByteBufferTensor.BLOCK_SIZE == 0 && limit % Q8ByteBufferTensor.BLOCK_SIZE == 0);

        int alim = aoffset + limit;
        int blim = boffset + limit;
        int slen = ByteVector.SPECIES_64.length();

        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_256);

        // Unroll 4x
        for (; aoffset < alim && boffset < blim; aoffset += slen * 4, boffset += slen * 4) {
            FloatVector scale = FloatVector.broadcast(FloatVector.SPECIES_256, b.getFactorForIndex(boffset));
            var af = a.getVector(FloatVector.SPECIES_256, aoffset);
            var bf = b.getVector(ByteVector.SPECIES_64, boffset)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0);
            acc = acc.add(af.mul(bf.mul(scale)));

            af = a.getVector(FloatVector.SPECIES_256, aoffset + slen);
            bf = b.getVector(ByteVector.SPECIES_64, boffset + slen)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0);
            acc = acc.add(af.mul(bf.mul(scale)));

            af = a.getVector(FloatVector.SPECIES_256, aoffset + slen + slen);
            bf = b.getVector(ByteVector.SPECIES_64, boffset + slen + slen)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0);
            acc = acc.add(af.mul(bf.mul(scale)));

            af = a.getVector(FloatVector.SPECIES_256, aoffset + slen + slen + slen);
            bf = b.getVector(ByteVector.SPECIES_64, boffset + slen + slen + slen)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0);
            acc = acc.add(af.mul(bf.mul(scale)));
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    public float dotProductF32I8_512(FloatBufferTensor a, Q8ByteBufferTensor b, int aoffset, int boffset, int limit) {
        Preconditions.checkArgument(
                boffset % Q8ByteBufferTensor.BLOCK_SIZE == 0 && limit % Q8ByteBufferTensor.BLOCK_SIZE == 0);

        int alim = aoffset + limit;
        int blim = boffset + limit;
        int slen = Q8ByteBufferTensor.BLOCK_SIZE;

        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_512);

        // Unroll 4x
        for (; aoffset < alim && boffset < blim; aoffset += slen, boffset += slen) {
            FloatVector scale = FloatVector.broadcast(FloatVector.SPECIES_512, b.getFactorForIndex(boffset));
            var af = a.getVector(FloatVector.SPECIES_512, aoffset);
            var bf = b.getVector(ByteVector.SPECIES_128, boffset)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0)
                    .mul(scale);

            acc = af.fma(bf, acc);

            af = a.getVector(FloatVector.SPECIES_512, aoffset + 16);
            bf = b.getVector(ByteVector.SPECIES_128, boffset + 16)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0)
                    .mul(scale);

            acc = af.fma(bf, acc);
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    private float dotProductF32Q4_arm(FloatBufferTensor a, Q4ByteBufferTensor b, int aoffset, int boffset, int limit) {
        Preconditions.checkArgument(
                boffset % Q4ByteBufferTensor.BLOCK_SIZE == 0 && limit % Q4ByteBufferTensor.BLOCK_SIZE == 0);

        int alim = aoffset + limit;
        int blim = boffset + limit;
        int slen = Q4ByteBufferTensor.BLOCK_SIZE;

        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_128);

        for (; aoffset < alim && boffset < blim; aoffset += slen, boffset += slen) {
            FloatVector scale = FloatVector.broadcast(FloatVector.SPECIES_128, b.getFactorForIndex(boffset));

            // BLOCK_SIZE Floats
            var af0 = a.getVector(FloatVector.SPECIES_128, aoffset);
            var af1 = a.getVector(FloatVector.SPECIES_128, aoffset + 4);
            var af2 = a.getVector(FloatVector.SPECIES_128, aoffset + 8);
            var af3 = a.getVector(FloatVector.SPECIES_128, aoffset + 12);

            var af4 = a.getVector(FloatVector.SPECIES_128, aoffset + Q4ByteBufferTensor.HALF_BLOCK);
            var af5 = a.getVector(FloatVector.SPECIES_128, aoffset + Q4ByteBufferTensor.HALF_BLOCK + 4);
            var af6 = a.getVector(FloatVector.SPECIES_128, aoffset + Q4ByteBufferTensor.HALF_BLOCK + 8);
            var af7 = a.getVector(FloatVector.SPECIES_128, aoffset + Q4ByteBufferTensor.HALF_BLOCK + 12);

            // Make 8 bytes -> 16 4bit -> 16 bytes -> 16 32F
            var bf0 = b.getVector(ByteVector.SPECIES_64, boffset);
            var bf1 = b.getVector(ByteVector.SPECIES_64, boffset + 16);

            // Convert the first 4 bits into bytes
            var low = bf0.lanewise(VectorOperators.AND, Q4_BYTE_MASK_64).sub(Q4_BYTE_SUB_64);
            var high = bf0.lanewise(VectorOperators.ASHR, Q4_BYTE_SHIFT_64)
                    .lanewise(VectorOperators.AND, Q4_BYTE_MASK_64)
                    .sub(Q4_BYTE_SUB_64);

            var low0 = low.castShape(FloatVector.SPECIES_128, 0).mul(scale);
            var low1 = low.castShape(FloatVector.SPECIES_128, 1).mul(scale);
            var high0 = high.castShape(FloatVector.SPECIES_128, 0).mul(scale);
            var high1 = high.castShape(FloatVector.SPECIES_128, 1).mul(scale);

            var nlow = bf1.lanewise(VectorOperators.AND, Q4_BYTE_MASK_64).sub(Q4_BYTE_SUB_64);
            var nhigh = bf1.lanewise(VectorOperators.ASHR, Q4_BYTE_SHIFT_64)
                    .lanewise(VectorOperators.AND, Q4_BYTE_MASK_64)
                    .sub(Q4_BYTE_SUB_64);

            var low2 = nlow.castShape(FloatVector.SPECIES_128, 0).mul(scale);
            var low3 = nlow.castShape(FloatVector.SPECIES_128, 1).mul(scale);
            var high2 = nhigh.castShape(FloatVector.SPECIES_128, 0).mul(scale);
            var high3 = nhigh.castShape(FloatVector.SPECIES_128, 1).mul(scale);

            acc = acc.add(af0.mul(low0));
            acc = acc.add(af1.mul(low1));
            acc = acc.add(af2.mul(low2));
            acc = acc.add(af3.mul(low3));

            acc = acc.add(af4.mul(high0));
            acc = acc.add(af5.mul(high1));
            acc = acc.add(af6.mul(high2));
            acc = acc.add(af7.mul(high3));
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    private float dotProductF32Q4_256(FloatBufferTensor a, Q4ByteBufferTensor b, int aoffset, int boffset, int limit) {
        Preconditions.checkArgument(
                boffset % Q4ByteBufferTensor.BLOCK_SIZE == 0 && limit % Q4ByteBufferTensor.BLOCK_SIZE == 0);

        int alim = aoffset + limit;
        int blim = boffset + limit;
        int slen = Q4ByteBufferTensor.BLOCK_SIZE;

        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_256);

        for (; aoffset < alim && boffset < blim; aoffset += slen, boffset += slen) {
            FloatVector scale = FloatVector.broadcast(FloatVector.SPECIES_256, b.getFactorForIndex(boffset));
            // BLOCK_SIZE Floats
            var af0 = a.getVector(FloatVector.SPECIES_256, aoffset);
            var af1 = a.getVector(FloatVector.SPECIES_256, aoffset + 8);

            var af2 = a.getVector(FloatVector.SPECIES_256, aoffset + Q4ByteBufferTensor.HALF_BLOCK);
            var af3 = a.getVector(FloatVector.SPECIES_256, aoffset + Q4ByteBufferTensor.HALF_BLOCK + 8);

            // Make 8 bytes -> 16 4bit -> 16 bytes -> 16 32F
            var bf0 = b.getVector(ByteVector.SPECIES_64, boffset);
            var bf1 = b.getVector(ByteVector.SPECIES_64, boffset + 16);

            // Convert the first 4 bits into bytes
            var low0 = bf0.lanewise(VectorOperators.AND, Q4_BYTE_MASK_64)
                    .sub(Q4_BYTE_SUB_64)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0)
                    .mul(scale);

            var high0 = bf0.lanewise(VectorOperators.ASHR, Q4_BYTE_SHIFT_64)
                    .lanewise(VectorOperators.AND, Q4_BYTE_MASK_64)
                    .sub(Q4_BYTE_SUB_64)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0)
                    .mul(scale);

            // Convert the first 4 bits into bytes
            var low1 = bf1.lanewise(VectorOperators.AND, Q4_BYTE_MASK_64)
                    .sub(Q4_BYTE_SUB_64)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0)
                    .mul(scale);

            var high1 = bf1.lanewise(VectorOperators.ASHR, Q4_BYTE_SHIFT_64)
                    .lanewise(VectorOperators.AND, Q4_BYTE_MASK_64)
                    .sub(Q4_BYTE_SUB_64)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0)
                    .mul(scale);

            acc = af0.fma(low0, acc);
            acc = af1.fma(low1, acc);
            acc = af2.fma(high0, acc);
            acc = af3.fma(high1, acc);
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    private float dotProductF32Q4_512(FloatBufferTensor a, Q4ByteBufferTensor b, int aoffset, int boffset, int limit) {
        Preconditions.checkArgument(
                boffset % Q4ByteBufferTensor.BLOCK_SIZE == 0 && limit % Q4ByteBufferTensor.BLOCK_SIZE == 0);

        int alim = aoffset + limit;
        int blim = boffset + limit;
        int slen = Q4ByteBufferTensor.BLOCK_SIZE;

        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_512);
        FloatVector scale = FloatVector.zero(FloatVector.SPECIES_512);
        for (; aoffset < alim && boffset < blim; aoffset += slen, boffset += slen) {
            scale = scale.broadcast(FloatVector.SPECIES_512, b.getFactorForIndex(boffset));
            // BLOCK_SIZE Floats
            var af0 = a.getVector(FloatVector.SPECIES_512, aoffset);
            var af1 = a.getVector(FloatVector.SPECIES_512, aoffset + Q4ByteBufferTensor.HALF_BLOCK);

            // Make 16 bytes -> 32 4bit -> 32 bytes -> 32 32F
            var bf0 = b.getVector(ByteVector.SPECIES_128, boffset);

            // Convert the first 4 bits into bytes
            var low0 = bf0.lanewise(VectorOperators.AND, Q4_BYTE_MASK_128)
                    .sub(Q4_BYTE_SUB_128)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0)
                    .mul(scale);

            var high0 = bf0.lanewise(VectorOperators.ASHR, Q4_BYTE_SHIFT_128)
                    .lanewise(VectorOperators.AND, Q4_BYTE_MASK_128)
                    .sub(Q4_BYTE_SUB_128)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0)
                    .mul(scale);

            acc = af0.fma(low0, acc);
            acc = af1.fma(high0, acc);
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    private float dotProductBF16Q4_512(
            BFloat16BufferTensor a, Q4ByteBufferTensor b, int aoffset, int boffset, int limit) {
        Preconditions.checkArgument(
                boffset % Q4ByteBufferTensor.BLOCK_SIZE == 0 && limit % Q4ByteBufferTensor.BLOCK_SIZE == 0);

        int alim = aoffset + limit;
        int blim = boffset + limit;
        int slen = Q4ByteBufferTensor.BLOCK_SIZE;

        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_512);

        // Unroll 4x
        for (; aoffset < alim && boffset < blim; aoffset += slen, boffset += slen) {
            FloatVector scale = FloatVector.broadcast(FloatVector.SPECIES_512, b.getFactorForIndex(boffset));
            // BLOCK_SIZE Floats
            var af0 = a.getVector(ShortVector.SPECIES_256, aoffset)
                    .convertShape(VectorOperators.ZERO_EXTEND_S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_512)
                    .reinterpretAsFloats();

            var af1 = a.getVector(ShortVector.SPECIES_256, aoffset + Q4ByteBufferTensor.HALF_BLOCK)
                    .convertShape(VectorOperators.ZERO_EXTEND_S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_512)
                    .reinterpretAsFloats();

            // Make 16 bytes -> 32 4bit -> 32 bytes -> 32 32F
            var bf0 = b.getVector(ByteVector.SPECIES_128, boffset);

            // Convert the first 4 bits into bytes
            var low0 = bf0.lanewise(VectorOperators.AND, Q4_BYTE_MASK_128)
                    .sub(Q4_BYTE_SUB_128)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0);

            // Convert the second 4 bits into bytes
            var high0 = bf0.lanewise(VectorOperators.ASHR, Q4_BYTE_SHIFT_128)
                    .lanewise(VectorOperators.AND, Q4_BYTE_MASK_128)
                    .sub(Q4_BYTE_SUB_128)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0);

            acc = acc.add(af0.mul(low0.mul(scale)));
            acc = acc.add(af1.mul(high0.mul(scale)));
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    private float dotProductBF16Q4_256(
            BFloat16BufferTensor a, Q4ByteBufferTensor b, int aoffset, int boffset, int limit) {
        Preconditions.checkArgument(
                boffset % Q4ByteBufferTensor.BLOCK_SIZE == 0 && limit % Q4ByteBufferTensor.BLOCK_SIZE == 0);

        int alim = aoffset + limit;
        int blim = boffset + limit;
        int slen = Q4ByteBufferTensor.BLOCK_SIZE;

        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_256);

        // Unroll 4x
        for (; aoffset < alim && boffset < blim; aoffset += slen, boffset += slen) {
            FloatVector scale = FloatVector.broadcast(FloatVector.SPECIES_256, b.getFactorForIndex(boffset));
            // BLOCK_SIZE Floats
            var af0 = a.getVector(ShortVector.SPECIES_128, aoffset)
                    .convertShape(VectorOperators.ZERO_EXTEND_S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_256)
                    .reinterpretAsFloats();

            var af1 = a.getVector(ShortVector.SPECIES_128, aoffset + 8)
                    .convertShape(VectorOperators.ZERO_EXTEND_S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_256)
                    .reinterpretAsFloats();

            var af2 = a.getVector(ShortVector.SPECIES_128, aoffset + 8 + 8)
                    .convertShape(VectorOperators.ZERO_EXTEND_S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_256)
                    .reinterpretAsFloats();

            var af3 = a.getVector(ShortVector.SPECIES_128, aoffset + 8 + 8 + 8)
                    .convertShape(VectorOperators.ZERO_EXTEND_S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_256)
                    .reinterpretAsFloats();

            // Make 8 bytes -> 16 4bit -> 16 bytes -> 16 32F
            var bf0 = b.getVector(ByteVector.SPECIES_64, boffset);
            var bf16 = b.getVector(ByteVector.SPECIES_64, boffset + Q4ByteBufferTensor.HALF_BLOCK);

            // Convert the first 4 bits into bytes
            var low0 = bf0.lanewise(VectorOperators.AND, Q4_BYTE_MASK_64)
                    .sub(Q4_BYTE_SUB_64)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0);

            // Convert the second 4 bits into bytes
            var high0 = bf0.lanewise(VectorOperators.ASHR, Q4_BYTE_SHIFT_64)
                    .lanewise(VectorOperators.AND, Q4_BYTE_MASK_64)
                    .sub(Q4_BYTE_SUB_64)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0);

            // Convert the first 4 bits into bytes
            var low1 = bf16.lanewise(VectorOperators.AND, Q4_BYTE_MASK_64)
                    .sub(Q4_BYTE_SUB_64)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0);

            // Convert the second 4 bits into bytes
            var high1 = bf16.lanewise(VectorOperators.ASHR, Q4_BYTE_SHIFT_64)
                    .lanewise(VectorOperators.AND, Q4_BYTE_MASK_64)
                    .sub(Q4_BYTE_SUB_64)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0);

            acc = acc.add(af0.mul(low0.mul(scale)));
            acc = acc.add(af1.mul(low1.mul(scale)));
            acc = acc.add(af2.mul(high0.mul(scale)));
            acc = acc.add(af3.mul(high1.mul(scale)));
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    public float dotProductBF16I8_256(
            BFloat16BufferTensor a, Q8ByteBufferTensor b, final int aoffset, final int boffset, int limit) {
        int ao = aoffset;
        int bo = boffset;
        final int alim = aoffset + limit;
        final int blim = boffset + limit;
        final int slen = ByteVector.SPECIES_64.length();

        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_256);

        // Unroll 4x
        for (; ao < alim && bo < blim; ao += slen, bo += slen) {
            FloatVector scale = FloatVector.broadcast(FloatVector.SPECIES_256, b.getFactorForIndex(bo));

            // Convert BF16 to F32
            var af = a.getVector(ShortVector.SPECIES_128, ao)
                    .convertShape(VectorOperators.ZERO_EXTEND_S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_256)
                    .reinterpretAsFloats()
                    .mul(scale);

            var bf = b.getVector(ByteVector.SPECIES_64, bo)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_256, 0);

            acc = af.fma(bf, acc);
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    public float dotProductBF16I8_512(
            BFloat16BufferTensor a, Q8ByteBufferTensor b, final int aoffset, final int boffset, int limit) {
        Preconditions.checkArgument(a.dType() == DType.BF16 && b.dType() == DType.I8);

        int ao = aoffset;
        int bo = boffset;
        final int alim = aoffset + limit;
        final int blim = boffset + limit;
        final int slen = ByteVector.SPECIES_128.length();

        FloatVector acc = FloatVector.SPECIES_512.zero().reinterpretAsFloats();

        // Unroll 4x
        for (; ao < alim && bo < blim; ao += slen, bo += slen) {
            FloatVector scale = FloatVector.broadcast(FloatVector.SPECIES_512, b.getFactorForIndex(bo));

            // Convert BF16 to F32
            var af = a.getVector(ShortVector.SPECIES_256, ao)
                    .convertShape(VectorOperators.ZERO_EXTEND_S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_512)
                    .reinterpretAsFloats()
                    .mul(scale);

            var bf = b.getVector(ByteVector.SPECIES_128, bo)
                    .convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0);

            acc = af.fma(bf, acc);
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    private float dotProductBF16F32_512(
            BFloat16BufferTensor a, FloatBufferTensor b, int aoffset, int boffset, int limit) {
        int alim = aoffset + limit;
        int blim = boffset + limit;
        int slen = ByteVector.SPECIES_128.length();

        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_512);

        // Unroll 4x
        for (; aoffset < alim && boffset < blim; aoffset += slen, boffset += slen) {

            // Convert BF16 to F32
            var af = a.getVector(ShortVector.SPECIES_256, aoffset)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_512)
                    .reinterpretAsFloats();

            FloatVector bf = b.getVector(FloatVector.SPECIES_512, boffset);

            acc = acc.add(af.mul(bf));
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    private float dotProductBF16F32_256(
            BFloat16BufferTensor a, FloatBufferTensor b, int aoffset, int boffset, int limit) {
        int alim = aoffset + limit;
        int blim = boffset + limit;
        int slen = ByteVector.SPECIES_64.length();

        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_256);

        // Unroll 4x
        for (; aoffset < alim && boffset < blim; aoffset += slen, boffset += slen) {

            // Convert BF16 to F32
            var af = a.getVector(ShortVector.SPECIES_128, aoffset)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_256)
                    .reinterpretAsFloats();

            FloatVector bf = b.getVector(FloatVector.SPECIES_256, boffset);

            acc = acc.add(af.mul(bf));
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    private float dotProductBF16_256(
            BFloat16BufferTensor a, BFloat16BufferTensor b, int aoffset, int boffset, int limit) {
        int alim = aoffset + limit;
        int blim = boffset + limit;
        int slen = ByteVector.SPECIES_64.length();

        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_256);

        // Unroll 4x
        for (; aoffset < alim && boffset < blim; aoffset += slen, boffset += slen) {

            // Convert BF16 to F32
            var af = a.getVector(ShortVector.SPECIES_128, aoffset)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_256)
                    .reinterpretAsFloats();

            // Convert BF16 to F32
            var bf = b.getVector(ShortVector.SPECIES_128, boffset)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_256)
                    .reinterpretAsFloats();

            acc = acc.add(af.mul(bf));
        }

        return acc.reduceLanes(VectorOperators.ADD);
    }

    private float dotProductBF16_512(
            BFloat16BufferTensor a, BFloat16BufferTensor b, int aoffset, int boffset, int limit) {
        int alim = aoffset + limit;
        int blim = boffset + limit;
        int slen = ByteVector.SPECIES_128.length();

        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_512);

        // Unroll 4x
        for (; aoffset < alim && boffset < blim; aoffset += slen, boffset += slen) {

            // Convert BF16 to F32
            var af = a.getVector(ShortVector.SPECIES_256, aoffset)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_512)
                    .reinterpretAsFloats();

            // Convert BF16 to F32
            var bf = b.getVector(ShortVector.SPECIES_256, boffset)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_512)
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
        for (; ao < alim && bo < blim; ao += slen, bo += slen) {
            FloatVector va = a.getVector(FloatVector.SPECIES_PREFERRED, ao);
            FloatVector vb = b.getVector(FloatVector.SPECIES_PREFERRED, bo);
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
    public void maccumulate(AbstractTensor a, AbstractTensor b, int offset, int limit) {
        Preconditions.checkArgument(a.dType() == b.dType());
        Preconditions.checkArgument(limit % 8 == 0);

        switch (a.dType()) {
            case F32:
                maccumulateF32((FloatBufferTensor) a, (FloatBufferTensor) b, offset, limit);
                break;
            default:
                throw new UnsupportedOperationException();
        }
    }

    void maccumulateF32(FloatBufferTensor a, FloatBufferTensor b, int offset, int limit) {
        int upperBound = offset + FloatVector.SPECIES_PREFERRED.loopBound(limit);
        int i = offset;

        for (; i < upperBound; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector va = a.getVector(FloatVector.SPECIES_PREFERRED, i);
            FloatVector vb = b.getVector(FloatVector.SPECIES_PREFERRED, i);
            a.intoTensor(va.mul(vb), i);
        }

        // tail
        for (; i < offset + limit; i++) {
            a.set(a.get(i) * b.get(i), i);
        }
    }

    @Override
    public void accumulate(AbstractTensor a, AbstractTensor b, int offset, int limit) {
        Preconditions.checkArgument(a.dType() == b.dType());
        Preconditions.checkArgument(limit % 8 == 0);

        switch (a.dType()) {
            case F32:
                accumulateF32((FloatBufferTensor) a, (FloatBufferTensor) b, offset, limit);
                break;
            case BF16:
                switch (vectorType) {
                    case AVX_512:
                        accumulateBF16_512((BFloat16BufferTensor) a, (BFloat16BufferTensor) b, offset, limit);
                        break;
                    case AVX_256:
                        accumulateBF16_256((BFloat16BufferTensor) a, (BFloat16BufferTensor) b, offset, limit);
                        break;
                    default:
                        throw new UnsupportedOperationException();
                }
                break;
            default:
                throw new UnsupportedOperationException();
        }
    }

    void accumulateF32(FloatBufferTensor a, FloatBufferTensor b, int offset, int limit) {
        int upperBound = offset + FloatVector.SPECIES_PREFERRED.loopBound(limit);
        int i = offset;

        for (; i < upperBound; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector va = a.getVector(FloatVector.SPECIES_PREFERRED, i);
            FloatVector vb = b.getVector(FloatVector.SPECIES_PREFERRED, i);
            a.intoTensor(va.add(vb), i);
        }

        // tail
        for (; i < offset + limit; i++) {
            a.set(a.get(i) + b.get(i), i);
        }
    }

    void accumulateBF16_256(BFloat16BufferTensor a, BFloat16BufferTensor b, int offset, int limit) {
        int upperBound = offset + FloatVector.SPECIES_256.loopBound(limit);

        int i = offset;
        for (; i < upperBound; i += FloatVector.SPECIES_256.length()) {

            // Convert BF16 to F32
            var af = a.getVector(ShortVector.SPECIES_128, i)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_256)
                    .reinterpretAsFloats();

            // Convert BF16 to F32
            var bf = b.getVector(ShortVector.SPECIES_128, i)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_256)
                    .reinterpretAsFloats();

            var res = af.add(bf)
                    .reinterpretAsInts()
                    .lanewise(VectorOperators.ASHR, BF16_BYTE_SHIFT_256)
                    .convertShape(VectorOperators.I2S, ShortVector.SPECIES_128, 0);

            a.intoTensor((ShortVector) res, i);
        }

        // tail
        for (; i < offset + limit; i++) {
            a.set(a.get(i) + b.get(i), i);
        }
    }

    void accumulateBF16_512(BFloat16BufferTensor a, BFloat16BufferTensor b, int offset, int limit) {
        int upperBound = offset + FloatVector.SPECIES_512.loopBound(limit);

        int i = offset;
        for (; i < upperBound; i += FloatVector.SPECIES_512.length()) {

            // Convert BF16 to F32
            var af = a.getVector(ShortVector.SPECIES_256, i)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_512)
                    .reinterpretAsFloats();

            // Convert BF16 to F32
            var bf = b.getVector(ShortVector.SPECIES_256, i)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_512)
                    .reinterpretAsFloats();

            var res = af.add(bf)
                    .reinterpretAsInts()
                    .lanewise(VectorOperators.ASHR, BF16_BYTE_SHIFT_512)
                    .convertShape(VectorOperators.I2S, ShortVector.SPECIES_256, 0);

            a.intoTensor((ShortVector) res, i);
        }

        // tail
        for (; i < offset + limit; i++) {
            a.set(a.get(i) + b.get(i), i);
        }
    }

    @Override
    public void scale(float factor, AbstractTensor a, int offset, int length) {
        switch (a.dType()) {
            case F32:
                scaleF32(factor, (FloatBufferTensor) a, offset, length);
                break;
            case BF16:
                switch (vectorType) {
                    case AVX_512:
                        scaleBF16_512(factor, (BFloat16BufferTensor) a, offset, length);
                        break;
                    case AVX_256:
                        scaleBF16_256(factor, (BFloat16BufferTensor) a, offset, length);
                        break;
                    default:
                        throw new UnsupportedOperationException();
                }
                break;
            default:
                throw new UnsupportedOperationException();
        }
    }

    public void scaleF32(float factor, FloatBufferTensor a, int offset, int length) {
        int upperBound = FloatVector.SPECIES_PREFERRED.loopBound(length) + offset;
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

    public void scaleBF16_512(float factor, BFloat16BufferTensor a, int offset, int length) {
        int upperBound = FloatVector.SPECIES_512.loopBound(length) + offset;
        int i = offset;

        FloatVector sf = FloatVector.broadcast(FloatVector.SPECIES_512, factor);
        for (; i < upperBound; i += FloatVector.SPECIES_512.length()) {
            var va = a.getVector(ShortVector.SPECIES_256, i)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_512)
                    .reinterpretAsFloats();

            var res = va.mul(sf)
                    .reinterpretAsInts()
                    .lanewise(VectorOperators.ASHR, BF16_BYTE_SHIFT_512)
                    .convertShape(VectorOperators.I2S, ShortVector.SPECIES_256, 0);

            a.intoTensor((ShortVector) res, i);
        }

        // tail
        for (; i < (offset + length); i++) {
            a.set(a.get(i) + factor, i);
        }
    }

    public void scaleBF16_256(float factor, BFloat16BufferTensor a, int offset, int length) {
        int upperBound = FloatVector.SPECIES_256.loopBound(length) + offset;
        int i = offset;

        FloatVector sf = FloatVector.broadcast(FloatVector.SPECIES_256, factor);
        for (; i < upperBound; i += FloatVector.SPECIES_256.length()) {
            var va = a.getVector(ShortVector.SPECIES_128, i)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_256)
                    .reinterpretAsFloats();

            var res = va.mul(sf)
                    .reinterpretAsInts()
                    .lanewise(VectorOperators.ASHR, BF16_BYTE_SHIFT_256)
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
        Preconditions.checkArgument(limit % 2 == 0);

        switch (x.dType()) {
            case F32:
                saxpyF32(alpha, (FloatBufferTensor) x, (FloatBufferTensor) y, xoffset, yoffset, limit);
                break;
            case BF16:
                switch (vectorType) {
                    case AVX_512:
                        saxpyBF16_512(
                                alpha, (BFloat16BufferTensor) x, (BFloat16BufferTensor) y, xoffset, yoffset, limit);
                        break;
                    case AVX_256:
                        saxpyBF16_256(
                                alpha, (BFloat16BufferTensor) x, (BFloat16BufferTensor) y, xoffset, yoffset, limit);
                        break;
                    default:
                        throw new UnsupportedOperationException();
                }
                break;
            default:
                throw new UnsupportedOperationException();
        }
    }

    void saxpyF32(float alpha, FloatBufferTensor x, FloatBufferTensor y, int xoffset, int yoffset, int limit) {
        int upperBound = FloatVector.SPECIES_PREFERRED.loopBound(limit);
        int xo = xoffset;
        int yo = yoffset;
        for (;
                xo < (xoffset + upperBound) && yo < (yoffset + upperBound);
                xo += FloatVector.SPECIES_PREFERRED.length(), yo += FloatVector.SPECIES_PREFERRED.length()) {
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

    void saxpyBF16_256(
            float alpha, BFloat16BufferTensor a, BFloat16BufferTensor b, int aoffset, int boffset, int limit) {
        int upperBound = FloatVector.SPECIES_256.loopBound(limit);
        Preconditions.checkArgument(upperBound == limit);

        int ao = aoffset;
        int bo = boffset;
        int len = FloatVector.SPECIES_256.length();

        for (; ao < (aoffset + upperBound) && bo < (boffset + upperBound); ao += len, bo += len) {
            // Convert BF16 to F32
            var af = a.getVector(ShortVector.SPECIES_128, ao)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_256)
                    .reinterpretAsFloats();

            // Convert BF16 to F32
            var bf = b.getVector(ShortVector.SPECIES_128, bo)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_256)
                    .reinterpretAsFloats();

            var r = bf.add(af.mul(alpha))
                    .reinterpretAsInts()
                    .lanewise(VectorOperators.ASHR, BF16_BYTE_SHIFT_256)
                    .convertShape(VectorOperators.I2S, ShortVector.SPECIES_128, 0);

            b.intoTensor((ShortVector) r, bo);
        }

        // tail
        for (; ao < (aoffset + limit) && bo < (boffset + limit); ao++, bo++) {
            float v = a.get(ao) + alpha * b.get(bo);
            b.set(v, bo);
        }
    }

    void saxpyBF16_512(
            float alpha, BFloat16BufferTensor a, BFloat16BufferTensor b, int aoffset, int boffset, int limit) {
        int upperBound = FloatVector.SPECIES_512.loopBound(limit);
        Preconditions.checkArgument(upperBound == limit);

        int ao = aoffset;
        int bo = boffset;
        int len = FloatVector.SPECIES_512.length();

        for (; ao < (aoffset + upperBound) && bo < (boffset + upperBound); ao += len, bo += len) {
            // Convert BF16 to F32
            var af = a.getVector(ShortVector.SPECIES_256, ao)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_512)
                    .reinterpretAsFloats();

            // Convert BF16 to F32
            var bf = b.getVector(ShortVector.SPECIES_256, bo)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_512)
                    .reinterpretAsFloats();

            var r = bf.add(af.mul(alpha))
                    .reinterpretAsInts()
                    .lanewise(VectorOperators.ASHR, BF16_BYTE_SHIFT_512)
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
        Preconditions.checkArgument(limit % 2 == 0);

        switch (x.dType()) {
            case F32:
                sxpbyF32(beta, (FloatBufferTensor) x, (FloatBufferTensor) y, xoffset, yoffset, limit);
                break;
            case BF16:
                switch (vectorType) {
                    case AVX_512:
                        sxpbyBF16_512(
                                beta, (BFloat16BufferTensor) x, (BFloat16BufferTensor) y, xoffset, yoffset, limit);
                        break;
                    case AVX_256:
                        sxpbyBF16_256(
                                beta, (BFloat16BufferTensor) x, (BFloat16BufferTensor) y, xoffset, yoffset, limit);
                        break;
                    default:
                        throw new UnsupportedOperationException();
                }
                break;
            default:
                throw new UnsupportedOperationException();
        }
    }

    void sxpbyF32(float beta, FloatBufferTensor x, FloatBufferTensor y, int xoffset, int yoffset, int limit) {
        int upperBound = FloatVector.SPECIES_PREFERRED.loopBound(limit);
        int xo = xoffset;
        int yo = yoffset;
        for (;
                xo < (xoffset + upperBound) && yo < (yoffset + upperBound);
                xo += FloatVector.SPECIES_PREFERRED.length(), yo += FloatVector.SPECIES_PREFERRED.length()) {
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

    void sxpbyBF16_256(
            float beta, BFloat16BufferTensor x, BFloat16BufferTensor y, int xoffset, int yoffset, int limit) {
        int upperBound = FloatVector.SPECIES_256.loopBound(limit);
        Preconditions.checkArgument(upperBound == limit);

        int xo = xoffset;
        int yo = yoffset;

        int len = FloatVector.SPECIES_256.length();

        for (; xo < (xoffset + upperBound) && yo < (yoffset + upperBound); xo += len, yo += len) {
            // Convert BF16 to F32
            var xv = x.getVector(ShortVector.SPECIES_128, xo)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_256)
                    .reinterpretAsFloats();

            // Convert BF16 to F32
            var yv = y.getVector(ShortVector.SPECIES_128, yo)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_256)
                    .reinterpretAsFloats();

            var res = xv.add(yv.mul(beta));

            // Turn back into BF16 and save
            y.intoTensor(
                    (ShortVector) res.reinterpretAsInts()
                            .lanewise(VectorOperators.ASHR, BF16_BYTE_SHIFT_256)
                            .convertShape(VectorOperators.I2S, ShortVector.SPECIES_128, 0),
                    yo);
        }

        for (; xo < (xoffset + limit) && yo < (yoffset + limit); xo++, yo++) {
            float v = x.get(xo) + beta * y.get(yo);
            y.set(v, yo);
        }
    }

    void sxpbyBF16_512(
            float beta, BFloat16BufferTensor x, BFloat16BufferTensor y, int xoffset, int yoffset, int limit) {
        int upperBound = FloatVector.SPECIES_512.loopBound(limit);
        Preconditions.checkArgument(upperBound == limit);

        int xo = xoffset;
        int yo = yoffset;

        int len = FloatVector.SPECIES_512.length();

        for (; xo < (xoffset + upperBound) && yo < (yoffset + upperBound); xo += len, yo += len) {
            // Convert BF16 to F32
            var xv = x.getVector(ShortVector.SPECIES_256, xo)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_512)
                    .reinterpretAsFloats();

            // Convert BF16 to F32
            var yv = y.getVector(ShortVector.SPECIES_256, yo)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_512)
                    .reinterpretAsFloats();

            var res = xv.add(yv.mul(beta));

            // Turn back into BF16 and save
            y.intoTensor(
                    (ShortVector) res.reinterpretAsInts()
                            .lanewise(VectorOperators.ASHR, BF16_BYTE_SHIFT_512)
                            .convertShape(VectorOperators.I2S, ShortVector.SPECIES_256, 0),
                    yo);
        }

        for (; xo < (xoffset + limit) && yo < (yoffset + limit); xo++, yo++) {
            float v = x.get(xo) + beta * y.get(yo);
            y.set(v, yo);
        }
    }
}
