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
import com.github.tjake.jlama.util.BiIntConsumer;
import com.github.tjake.jlama.util.MachineSpec;
import com.github.tjake.jlama.util.PhysicalCoreExecutor;
import com.google.common.base.Preconditions;
import jdk.incubator.vector.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class PanamaTensorOperations implements TensorOperations {
    private static final Logger logger = LoggerFactory.getLogger(PanamaTensorOperations.class);
    static final ByteVector Q4_BYTE_SUB_128 = ByteVector.broadcast(ByteVector.SPECIES_128, 8);
    static final ByteVector Q4_BYTE_MASK_128 = ByteVector.broadcast(ByteVector.SPECIES_128, 0xF);
    static final ByteVector Q4_BYTE_SHIFT_128 = ByteVector.broadcast(ByteVector.SPECIES_128, 4);

    static final ByteVector Q4_BYTE_SUB_64 = ByteVector.broadcast(ByteVector.SPECIES_64, 8);
    static final ByteVector Q4_BYTE_MASK_64 = ByteVector.broadcast(ByteVector.SPECIES_64, 0xF);
    static final ByteVector Q4_BYTE_SHIFT_64 = ByteVector.broadcast(ByteVector.SPECIES_64, 4);

    static final IntVector BF16_BYTE_SHIFT = IntVector.broadcast(IntVector.SPECIES_PREFERRED, 16);

    static final IntVector BF16_BYTE_SHIFT_512 = IntVector.broadcast(IntVector.SPECIES_512, 16);
    static final FloatVector F32_ROUND_UP_512 = FloatVector.broadcast(FloatVector.SPECIES_512, 0.5f);

    static final IntVector BF16_BYTE_SHIFT_256 = IntVector.broadcast(IntVector.SPECIES_256, 16);
    static final FloatVector F32_ROUND_UP_256 = FloatVector.broadcast(FloatVector.SPECIES_256, 0.5f);

    static final IntVector BF16_BYTE_SHIFT_128 = IntVector.broadcast(IntVector.SPECIES_128, 16);
    static final FloatVector F32_ROUND_UP_128 = FloatVector.broadcast(FloatVector.SPECIES_128, 0.5f);

    static final VectorMask<Byte> BYTE_MASK_32 = VectorMask.fromValues(
        ByteVector.SPECIES_64,
        true,
        true,
        true,
        true,
        false,
        false,
        false,
        false
    );

    private final MachineSpec.Type vectorType;

    public PanamaTensorOperations(MachineSpec.Type vectorType) {
        this.vectorType = vectorType;
    }

    @Override
    public String name() {
        return "Panama Vector Operations";
    }

    public int parallelSplitSize() {
        return PhysicalCoreExecutor.instance.get().getCoreCount();
    }

    /**
     *  multiplies matrices on cpu
     *  with column major ordering
     *
     *  m×k * k×n → m×n
     *  k×m * k×n → m×n if aᵀ
     *  m×k * n×k → m×n if bᵀ
     *  k×m * n×k → m×n if aᵀ and bᵀ
     *
     *  In Jlama we use row major ordering
     *  So: k×m * n×k → m×n
     *  EmbeddingxBatch * WeightsxEmbedding → BATCHxEmbedding
     */
    @Override
    public void batchDotProduct(
        AbstractTensor result,
        AbstractTensor a,
        AbstractTensor b,
        int aColumnOffset,
        int bColumnOffset,
        int columnLength,
        int rOffset,
        int bRowOffset,
        int rowChunkSize
    ) {
        Preconditions.checkArgument(a.dims() == 2 && b.dims() == 2 && result.dims() == 2);
        Preconditions.checkArgument(a.shape().dim(0) == result.shape().dim(0), "BAD M");
        Preconditions.checkArgument(rOffset == 0 || rOffset >= bRowOffset, "Result offset must be >= b row offset");
        // Preconditions.checkArgument(b.shape().dim(0) == result.shape().dim(1), "BAD N");
        // This check breaks for GQA
        // Preconditions.checkArgument(a.shape().dim(1) == b.shape().dim(1), "BAD K" + a.shape() + " " + b.shape() + " "
        // + columnLength);

        int M = a.shape().dim(0);
        int N = rowChunkSize; // b.shape().dim(0);
        int K = columnLength; // a.shape().dim(1);

        Gemmer gemm = switch (a.dType()) {
            case F32 -> switch (b.dType()) {
                case F32 -> new GemmerF32(K, a, b, result, aColumnOffset, bColumnOffset, rOffset);
                case BF16 -> new GemmerF32BF16(K, a, b, result, aColumnOffset, bColumnOffset, rOffset);
                case Q4 -> switch (vectorType) {
                    case AVX_256 -> new GemmerF32Q4_256(K, a, b, result, aColumnOffset, bColumnOffset, rOffset);
                    case AVX_512 -> new GemmerF32Q4_512(K, a, b, result, aColumnOffset, bColumnOffset, rOffset);
                    default -> throw new UnsupportedOperationException(vectorType.name());
                };
                default -> throw new UnsupportedOperationException(b.dType().name());
            };
            case I8 -> switch (b.dType()) {
                case Q4 -> switch (vectorType) {
                    case AVX_256 -> new GemmerI8Q4_256(K, a, b, result, aColumnOffset, bColumnOffset, rOffset);
                    case AVX_512 -> new GemmerI8Q4_512(K, a, b, result, aColumnOffset, bColumnOffset, rOffset);
                    case ARM_128 -> new GemmerI8Q4_arm(K, a, b, result, aColumnOffset, bColumnOffset, rOffset);
                    default -> throw new UnsupportedOperationException(vectorType.name());
                };
                default -> throw new UnsupportedOperationException(b.dType().name());
            };
            case BF16 -> switch (b.dType()) {
                case BF16 -> new GemmerBF16(K, a, b, result, aColumnOffset, bColumnOffset, rOffset);
                default -> throw new UnsupportedOperationException(b.dType().name());
            };
            default -> throw new UnsupportedOperationException(a.dType().name() + " " + b.dType().name());
        };

        gemm.matmul(0, M, bRowOffset, bRowOffset + N);
    }

    private class GemmerF32Q4_256 extends Gemmer {
        final BiIntConsumer matmul1x1;
        final BiIntConsumer matmul1x4;
        final BiIntConsumer matmul3x4;
        final BiIntConsumer matmul4x1;

        final Q4ByteBufferTensor b;
        final FloatBufferTensor a;

        GemmerF32Q4_256(int k, AbstractTensor ta, AbstractTensor tb, AbstractTensor c, int ith, int nth, int rOffset) {
            super(k, ta, tb, c, ith, nth, rOffset);

            this.a = (FloatBufferTensor) ta;
            this.b = (Q4ByteBufferTensor) tb;

            this.matmul1x1 = initMatmul1x1();
            this.matmul1x4 = initMatmul1x4();
            this.matmul3x4 = null;
            this.matmul4x1 = null;
        }

        @Override
        protected int pickKernel(int m0, int m, int n0, int n) {
            short mc, nc;
            /*if (m - m0 >= 3 && n - n0 >= 4)
            {
                mc = 3;
                nc = 4;
                kernel(m0, m, 3, n0, n, 4, matmul3x4);
            }
            else if (m - m0 >= 4 && n - n0 >= 1)
            {
                mc = 4;
                nc = 1;
                kernel(m0, m, 4, n0, n, 1, matmul4x1);
            }
            else if (m - m0 >= 1 && n - n0 >= 2) {
                mc = 1;
                nc = 2;
                kernel(m0, m, 1, n0, n, 2, matmul1x4);
            } else*/ {
                mc = 1;
                nc = 1;
                kernel(m0, m, 1, n0, n, 1, matmul1x1);
            }

            return (mc << 4) | nc;
        }

        protected BiIntConsumer initMatmul1x1() {
            return (i, j) -> {
                int aoffset = aColumnOffset;
                int boffset = bColumnOffset;
                int alim = aoffset + k;
                int blim = boffset + k;
                int slen = Q4ByteBufferTensor.BLOCK_SIZE;
                FloatVector acc = FloatVector.zero(FloatVector.SPECIES_256);

                for (; aoffset < alim && boffset < blim; aoffset += slen, boffset += slen) {
                    FloatVector scale = FloatVector.broadcast(FloatVector.SPECIES_256, b.getFactorForIndex(j, boffset));

                    // Make 8 bytes -> 16 4bit -> 16 bytes -> 16 32F
                    var b0 = b.getVector(ByteVector.SPECIES_128, j, boffset);
                    var b0lo = b0.and(Q4_BYTE_MASK_128).sub(Q4_BYTE_SUB_128);
                    var b0hi = b0.lanewise(VectorOperators.LSHR, Q4_BYTE_SHIFT_128).sub(Q4_BYTE_SUB_128);

                    // BLOCK_SIZE Floats
                    var af0 = a.getVector(FloatVector.SPECIES_256, i, aoffset).mul(b0lo.castShape(FloatVector.SPECIES_256, 0));
                    ;
                    var af1 = a.getVector(FloatVector.SPECIES_256, i, aoffset + 8).mul(b0lo.castShape(FloatVector.SPECIES_256, 1));
                    ;
                    var af2 = a.getVector(FloatVector.SPECIES_256, i, aoffset + Q4ByteBufferTensor.HALF_BLOCK)
                        .mul(b0hi.castShape(FloatVector.SPECIES_256, 0));
                    var af3 = a.getVector(FloatVector.SPECIES_256, i, aoffset + Q4ByteBufferTensor.HALF_BLOCK + 8)
                        .mul(b0hi.castShape(FloatVector.SPECIES_256, 1));

                    acc = af0.add(af1).add(af2).add(af3).fma(scale, acc);
                }

                c.set(acc.reduceLanes(VectorOperators.ADD), i, j + rOffset);
            };
        }

        protected BiIntConsumer initMatmul1x4() {
            return (i, j) -> {
                int aoffset = aColumnOffset;
                int boffset = bColumnOffset;
                int alim = aoffset + k;
                int blim = boffset + k;
                int slen = Q4ByteBufferTensor.BLOCK_SIZE;
                FloatVector acc0 = FloatVector.zero(FloatVector.SPECIES_256);
                FloatVector acc1 = FloatVector.zero(FloatVector.SPECIES_256);

                for (; aoffset < alim && boffset < blim; aoffset += slen, boffset += slen) {
                    FloatVector scale0 = FloatVector.broadcast(FloatVector.SPECIES_256, b.getFactorForIndex(j + 0, boffset));
                    FloatVector scale1 = FloatVector.broadcast(FloatVector.SPECIES_256, b.getFactorForIndex(j + 1, boffset));

                    // BLOCK_SIZE Floats
                    var af0 = a.getVector(FloatVector.SPECIES_256, i, aoffset);
                    var af1 = a.getVector(FloatVector.SPECIES_256, i, aoffset + 8);
                    var af2 = a.getVector(FloatVector.SPECIES_256, i, aoffset + Q4ByteBufferTensor.HALF_BLOCK);
                    var af3 = a.getVector(FloatVector.SPECIES_256, i, aoffset + Q4ByteBufferTensor.HALF_BLOCK + 8);

                    {
                        // Make 8 bytes -> 16 4bit -> 16 bytes -> 16 32F
                        var bf0 = b.getVector(ByteVector.SPECIES_128, j + 0, boffset);
                        var lo0 = bf0.and(Q4_BYTE_MASK_128).sub(Q4_BYTE_SUB_128);
                        var hi0 = bf0.lanewise(VectorOperators.LSHR, Q4_BYTE_SHIFT_128).sub(Q4_BYTE_SUB_128);

                        var af0l = af0.mul(lo0.castShape(FloatVector.SPECIES_256, 0));
                        var af1l = af1.mul(lo0.castShape(FloatVector.SPECIES_256, 1));
                        var af2l = af2.mul(hi0.castShape(FloatVector.SPECIES_256, 0));
                        var af3l = af3.mul(hi0.castShape(FloatVector.SPECIES_256, 1));

                        acc0 = af0l.add(af1l).add(af2l).add(af3l).fma(scale0, acc0);
                    }

                    {
                        // Make 8 bytes -> 16 4bit -> 16 bytes -> 16 32F
                        var bf0 = b.getVector(ByteVector.SPECIES_128, j + 1, boffset);
                        var lo0 = bf0.and(Q4_BYTE_MASK_128).sub(Q4_BYTE_SUB_128);
                        var hi0 = bf0.lanewise(VectorOperators.LSHR, Q4_BYTE_SHIFT_128).sub(Q4_BYTE_SUB_128);

                        var af0l = af0.mul(lo0.castShape(FloatVector.SPECIES_256, 0));
                        var af1l = af1.mul(lo0.castShape(FloatVector.SPECIES_256, 1));
                        var af2l = af2.mul(hi0.castShape(FloatVector.SPECIES_256, 0));
                        var af3l = af3.mul(hi0.castShape(FloatVector.SPECIES_256, 1));

                        acc1 = af0l.add(af1l).add(af2l).add(af3l).fma(scale1, acc1);
                    }

                }

                c.set(acc0.reduceLanes(VectorOperators.ADD), i, j + 0 + rOffset);
                c.set(acc1.reduceLanes(VectorOperators.ADD), i, j + 1 + rOffset);
                // c.set(acc2.reduceLanes(VectorOperators.ADD), i, j + 2);
                // c.set(acc3.reduceLanes(VectorOperators.ADD), i, j + 3);
            };
        }
    }

    private class GemmerF32Q4_512 extends Gemmer {
        final BiIntConsumer matmul1x1;
        final BiIntConsumer matmul1x4;
        final BiIntConsumer matmul3x4;
        final BiIntConsumer matmul4x1;

        final Q4ByteBufferTensor b;
        final FloatBufferTensor a;

        GemmerF32Q4_512(int k, AbstractTensor ta, AbstractTensor tb, AbstractTensor c, int ith, int nth, int rOffset) {
            super(k, ta, tb, c, ith, nth, rOffset);

            this.a = (FloatBufferTensor) ta;
            this.b = (Q4ByteBufferTensor) tb;

            this.matmul1x1 = initMatmul1x1();
            this.matmul1x4 = initMatmul1x4();
            this.matmul3x4 = null;
            this.matmul4x1 = initMatmul4x1();
        }

        @Override
        protected int pickKernel(int m0, int m, int n0, int n) {
            short mc, nc;
            /*if (m - m0 >= 3 && n - n0 >= 4)
            {
                mc = 3;
                nc = 4;
                kernel(m0, m, 3, n0, n, 4, matmul3x4);
            }
            else*/ if (m - m0 >= 4 && n - n0 >= 1) {
                mc = 4;
                nc = 1;
                kernel(m0, m, 4, n0, n, 1, matmul4x1);
            } else if (m - m0 >= 1 && n - n0 >= 4) {
                mc = 1;
                nc = 4;
                kernel(m0, m, mc, n0, n, nc, matmul1x4);
            } else {
                mc = 1;
                nc = 1;
                kernel(m0, m, 1, n0, n, 1, matmul1x1);
            }

            return (mc << 4) | nc;
        }

        protected BiIntConsumer initMatmul1x1() {
            return (i, j) -> {
                int aoffset = aColumnOffset;
                int boffset = bColumnOffset;
                int alim = aoffset + k;
                int blim = boffset + k;
                int slen = Q4ByteBufferTensor.BLOCK_SIZE;

                FloatVector acc = FloatVector.zero(FloatVector.SPECIES_512);
                FloatVector scale;

                for (; aoffset < alim && boffset < blim; aoffset += slen, boffset += slen) {
                    scale = FloatVector.broadcast(FloatVector.SPECIES_512, b.getFactorForIndex(j, boffset));
                    // BLOCK_SIZE Floats
                    var af0 = a.getVector(FloatVector.SPECIES_512, i, aoffset);
                    var af1 = a.getVector(FloatVector.SPECIES_512, i, aoffset + Q4ByteBufferTensor.HALF_BLOCK);

                    // Make 16 bytes -> 32 4bit -> 32 bytes -> 32 32F
                    var bf0 = b.getVector(ByteVector.SPECIES_128, j, boffset);

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

                c.set(acc.reduceLanes(VectorOperators.ADD), i, j + rOffset);
            };
        }

        protected final BiIntConsumer initMatmul4x1() {
            return (i, j) -> {
                int aoffset = aColumnOffset;
                int boffset = bColumnOffset;
                int alim = aoffset + k;
                int blim = boffset + k;
                int slen = Q4ByteBufferTensor.BLOCK_SIZE;

                FloatVector acc0 = FloatVector.zero(FloatVector.SPECIES_512);
                FloatVector acc1 = FloatVector.zero(FloatVector.SPECIES_512);
                FloatVector acc2 = FloatVector.zero(FloatVector.SPECIES_512);
                FloatVector acc3 = FloatVector.zero(FloatVector.SPECIES_512);

                FloatVector scale;

                for (; aoffset < alim && boffset < blim; aoffset += slen, boffset += slen) {
                    scale = FloatVector.broadcast(FloatVector.SPECIES_512, b.getFactorForIndex(j, boffset));

                    // Make 16 bytes -> 32 4bit -> 32 bytes -> 32 32F
                    var bf0 = b.getVector(ByteVector.SPECIES_128, j, boffset);

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

                    // BLOCK_SIZE Floats
                    var af00 = a.getVector(FloatVector.SPECIES_512, i, aoffset);
                    var af01 = a.getVector(FloatVector.SPECIES_512, i, aoffset + Q4ByteBufferTensor.HALF_BLOCK);

                    var af10 = a.getVector(FloatVector.SPECIES_512, i + 1, aoffset);
                    var af11 = a.getVector(FloatVector.SPECIES_512, i + 1, aoffset + Q4ByteBufferTensor.HALF_BLOCK);

                    var af20 = a.getVector(FloatVector.SPECIES_512, i + 2, aoffset);
                    var af21 = a.getVector(FloatVector.SPECIES_512, i + 2, aoffset + Q4ByteBufferTensor.HALF_BLOCK);

                    var af30 = a.getVector(FloatVector.SPECIES_512, i + 3, aoffset);
                    var af31 = a.getVector(FloatVector.SPECIES_512, i + 3, aoffset + Q4ByteBufferTensor.HALF_BLOCK);

                    acc0 = af00.fma(low0, acc0);
                    acc0 = af01.fma(high0, acc0);
                    acc1 = af10.fma(low0, acc1);
                    acc1 = af11.fma(high0, acc1);
                    acc2 = af20.fma(low0, acc2);
                    acc2 = af21.fma(high0, acc2);
                    acc3 = af30.fma(low0, acc3);
                    acc3 = af31.fma(high0, acc3);
                }

                c.set(acc0.reduceLanes(VectorOperators.ADD), i + 0, j + rOffset);
                c.set(acc1.reduceLanes(VectorOperators.ADD), i + 1, j + rOffset);
                c.set(acc2.reduceLanes(VectorOperators.ADD), i + 2, j + rOffset);
                c.set(acc3.reduceLanes(VectorOperators.ADD), i + 3, j + rOffset);
            };
        }

        protected BiIntConsumer initMatmul1x4() {
            return (i, j) -> {
                int aoffset = aColumnOffset;
                int boffset = bColumnOffset;
                int alim = aoffset + k;
                int blim = boffset + k;
                int slen = Q4ByteBufferTensor.BLOCK_SIZE;

                FloatVector acc0 = FloatVector.zero(FloatVector.SPECIES_512);
                FloatVector acc1 = FloatVector.zero(FloatVector.SPECIES_512);
                FloatVector acc2 = FloatVector.zero(FloatVector.SPECIES_512);
                FloatVector acc3 = FloatVector.zero(FloatVector.SPECIES_512);

                for (; aoffset < alim && boffset < blim; aoffset += slen, boffset += slen) {
                    // BLOCK_SIZE Floats
                    var af0 = a.getVector(FloatVector.SPECIES_512, i, aoffset);
                    var af1 = a.getVector(FloatVector.SPECIES_512, i, aoffset + Q4ByteBufferTensor.HALF_BLOCK);

                    {
                        // Make 16 bytes -> 32 4bit -> 32 bytes -> 32 32F
                        var scale0 = FloatVector.broadcast(FloatVector.SPECIES_512, b.getFactorForIndex(j + 0, boffset));
                        var bf0 = b.getVector(ByteVector.SPECIES_128, j + 0, boffset);

                        // Convert the first 4 bits into bytes
                        var low0 = bf0.lanewise(VectorOperators.AND, Q4_BYTE_MASK_128)
                            .sub(Q4_BYTE_SUB_128)
                            .convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0)
                            .mul(scale0);

                        var high0 = bf0.lanewise(VectorOperators.ASHR, Q4_BYTE_SHIFT_128)
                            .lanewise(VectorOperators.AND, Q4_BYTE_MASK_128)
                            .sub(Q4_BYTE_SUB_128)
                            .convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0)
                            .mul(scale0);

                        acc0 = af0.fma(low0, acc0);
                        acc0 = af1.fma(high0, acc0);
                    }

                    {
                        // Make 16 bytes -> 32 4bit -> 32 bytes -> 32 32F
                        var scale0 = FloatVector.broadcast(FloatVector.SPECIES_512, b.getFactorForIndex(j + 1, boffset));
                        var bf0 = b.getVector(ByteVector.SPECIES_128, j + 1, boffset);

                        // Convert the first 4 bits into bytes
                        var low0 = bf0.lanewise(VectorOperators.AND, Q4_BYTE_MASK_128)
                            .sub(Q4_BYTE_SUB_128)
                            .convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0)
                            .mul(scale0);

                        var high0 = bf0.lanewise(VectorOperators.ASHR, Q4_BYTE_SHIFT_128)
                            .lanewise(VectorOperators.AND, Q4_BYTE_MASK_128)
                            .sub(Q4_BYTE_SUB_128)
                            .convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0)
                            .mul(scale0);

                        acc1 = af0.fma(low0, acc1);
                        acc1 = af1.fma(high0, acc1);
                    }

                    {
                        // Make 16 bytes -> 32 4bit -> 32 bytes -> 32 32F
                        var scale0 = FloatVector.broadcast(FloatVector.SPECIES_512, b.getFactorForIndex(j + 2, boffset));
                        var bf0 = b.getVector(ByteVector.SPECIES_128, j + 2, boffset);

                        // Convert the first 4 bits into bytes
                        var low0 = bf0.lanewise(VectorOperators.AND, Q4_BYTE_MASK_128)
                            .sub(Q4_BYTE_SUB_128)
                            .convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0)
                            .mul(scale0);

                        var high0 = bf0.lanewise(VectorOperators.ASHR, Q4_BYTE_SHIFT_128)
                            .lanewise(VectorOperators.AND, Q4_BYTE_MASK_128)
                            .sub(Q4_BYTE_SUB_128)
                            .convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0)
                            .mul(scale0);

                        acc2 = af0.fma(low0, acc2);
                        acc2 = af1.fma(high0, acc2);
                    }

                    {
                        // Make 16 bytes -> 32 4bit -> 32 bytes -> 32 32F
                        var scale0 = FloatVector.broadcast(FloatVector.SPECIES_512, b.getFactorForIndex(j + 3, boffset));
                        var bf0 = b.getVector(ByteVector.SPECIES_128, j + 3, boffset);

                        // Convert the first 4 bits into bytes
                        var low0 = bf0.lanewise(VectorOperators.AND, Q4_BYTE_MASK_128)
                            .sub(Q4_BYTE_SUB_128)
                            .convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0)
                            .mul(scale0);

                        var high0 = bf0.lanewise(VectorOperators.ASHR, Q4_BYTE_SHIFT_128)
                            .lanewise(VectorOperators.AND, Q4_BYTE_MASK_128)
                            .sub(Q4_BYTE_SUB_128)
                            .convertShape(VectorOperators.B2F, FloatVector.SPECIES_512, 0)
                            .mul(scale0);

                        acc3 = af0.fma(low0, acc3);
                        acc3 = af1.fma(high0, acc3);
                    }
                }

                c.set(acc0.reduceLanes(VectorOperators.ADD), i, j + 0 + rOffset);
                c.set(acc1.reduceLanes(VectorOperators.ADD), i, j + 1 + rOffset);
                c.set(acc2.reduceLanes(VectorOperators.ADD), i, j + 2 + rOffset);
                c.set(acc3.reduceLanes(VectorOperators.ADD), i, j + 3 + rOffset);
            };
        }
    }

    private class GemmerI8Q4_arm extends Gemmer {
        final BiIntConsumer matmul1x1;
        final BiIntConsumer matmul1x4;
        final BiIntConsumer matmul3x4;
        final BiIntConsumer matmul4x1;

        final Q8ByteBufferTensor a;
        final Q4ByteBufferTensor b;

        GemmerI8Q4_arm(int k, AbstractTensor ta, AbstractTensor tb, AbstractTensor c, int aColumnOffset, int bColumnOffset, int rOffset) {
            super(k, ta, tb, c, aColumnOffset, bColumnOffset, rOffset);

            this.a = (Q8ByteBufferTensor) ta;
            this.b = (Q4ByteBufferTensor) tb;

            this.matmul1x1 = initMatmul1x1();
            this.matmul1x4 = null;
            this.matmul3x4 = null;
            this.matmul4x1 = null;
        }

        @Override
        protected int pickKernel(int m0, int m, int n0, int n) {
            short mc, nc;
            /*if (m - m0 >= 3 && n - n0 >= 4)
            {
                mc = 3;
                nc = 4;
                kernel(m0, m, 3, n0, n, 4, matmul3x4);
            }
            else if (m - m0 >= 4 && n - n0 >= 1)
            {
                mc = 4;
                nc = 1;
                kernel(m0, m, 4, n0, n, 1, matmul4x1);
            }
            else if (m - m0 >= 1 && n - n0 >= 4)
            {
                mc = 1;
                nc = 4;
                kernel(m0, m, 1, n0, n, 4, matmul1x4);
            }
            else*/
            {
                mc = 1;
                nc = 1;
                kernel(m0, m, 1, n0, n, 1, matmul1x1);
            }

            return (mc << 4) | nc;
        }

        protected BiIntConsumer initMatmul1x1() {
            return (i, j) -> {
                final int blockSize = Q8ByteBufferTensor.BLOCK_SIZE;
                final int blocksNeeded = k / Q8ByteBufferTensor.BLOCK_SIZE;

                int aoffset = aColumnOffset;
                int boffset = bColumnOffset;

                FloatVector acc = FloatVector.zero(FloatVector.SPECIES_128);

                // First take the scaling factors of both tensors and multiply them in SIMD
                for (int bi = 0; bi < blocksNeeded; bi += FloatVector.SPECIES_128.length()) {
                    final var ablock = a.getBlockF()
                        .getVector(FloatVector.SPECIES_128, i, (int) (Q8ByteBufferTensor.I_BLOCK_SIZE * aoffset));
                    final var bblock = b.getBlockF()
                        .getVector(FloatVector.SPECIES_128, j, (int) (Q8ByteBufferTensor.I_BLOCK_SIZE * boffset));

                    final var scales = ablock.mul(bblock);
                    // Now for each scalar fetch the corresponding block of data and dot product them
                    for (int k = 0; k < FloatVector.SPECIES_128.length(); k++, aoffset += blockSize, boffset += blockSize) {
                        var scale = FloatVector.broadcast(FloatVector.SPECIES_128, scales.lane(k));

                        var ab0 = a.getVector(ByteVector.SPECIES_128, i, aoffset);
                        var ab1 = a.getVector(ByteVector.SPECIES_128, i, aoffset + 16);

                        var af0 = ab0.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
                        var af1 = ab0.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 1);
                        var af2 = ab1.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 0);
                        var af3 = ab1.convertShape(VectorOperators.B2S, ShortVector.SPECIES_128, 1);

                        // Make 8 bytes -> 16 4bit -> 16 bytes -> 16 32F
                        var bf0 = b.getVector(ByteVector.SPECIES_64, j, boffset);
                        var bf1 = b.getVector(ByteVector.SPECIES_64, j, boffset + 16);

                        // Convert the first 4 bits into bytes
                        var low = bf0.lanewise(VectorOperators.AND, Q4_BYTE_MASK_64).sub(Q4_BYTE_SUB_64);
                        var high = bf0.lanewise(VectorOperators.ASHR, Q4_BYTE_SHIFT_64)
                            .lanewise(VectorOperators.AND, Q4_BYTE_MASK_64)
                            .sub(Q4_BYTE_SUB_64);

                        var low0 = low.castShape(ShortVector.SPECIES_128, 0);
                        var high0 = high.castShape(ShortVector.SPECIES_128, 0);

                        var nlow = bf1.lanewise(VectorOperators.AND, Q4_BYTE_MASK_64).sub(Q4_BYTE_SUB_64);
                        var nhigh = bf1.lanewise(VectorOperators.ASHR, Q4_BYTE_SHIFT_64)
                            .lanewise(VectorOperators.AND, Q4_BYTE_MASK_64)
                            .sub(Q4_BYTE_SUB_64);

                        var low2 = nlow.castShape(ShortVector.SPECIES_128, 0);
                        var high2 = nhigh.castShape(ShortVector.SPECIES_128, 0);

                        ShortVector tacc = ShortVector.zero(ShortVector.SPECIES_128);
                        tacc = tacc.add(af0.mul(low0));
                        tacc = tacc.add(af1.mul(low2));

                        tacc = tacc.add(af2.mul(high0));
                        tacc = tacc.add(af3.mul(high2));

                        acc = acc.add(tacc.convertShape(VectorOperators.S2F, FloatVector.SPECIES_128, 0).mul(scale));
                        acc = acc.add(tacc.convertShape(VectorOperators.S2F, FloatVector.SPECIES_128, 1).mul(scale));
                    }
                }

                c.set(acc.reduceLanes(VectorOperators.ADD), i, j + rOffset);
            };
        }
    }

    private class GemmerI8Q4_256 extends Gemmer {
        final BiIntConsumer matmul1x1;
        final BiIntConsumer matmul1x4;
        final BiIntConsumer matmul3x4;
        final BiIntConsumer matmul4x1;

        final Q8ByteBufferTensor a;
        final Q4ByteBufferTensor b;

        GemmerI8Q4_256(int k, AbstractTensor ta, AbstractTensor tb, AbstractTensor c, int aColumnOffset, int bColumnOffset, int rOffset) {
            super(k, ta, tb, c, aColumnOffset, bColumnOffset, rOffset);

            this.a = (Q8ByteBufferTensor) ta;
            this.b = (Q4ByteBufferTensor) tb;

            this.matmul1x1 = initMatmul1x1();
            this.matmul1x4 = null;
            this.matmul3x4 = null;
            this.matmul4x1 = null;
        }

        @Override
        protected int pickKernel(int m0, int m, int n0, int n) {
            short mc, nc;
            /*if (m - m0 >= 3 && n - n0 >= 4)
            {
                mc = 3;
                nc = 4;
                kernel(m0, m, 3, n0, n, 4, matmul3x4);
            }
            else if (m - m0 >= 4 && n - n0 >= 1)
            {
                mc = 4;
                nc = 1;
                kernel(m0, m, 4, n0, n, 1, matmul4x1);
            }
            else if (m - m0 >= 1 && n - n0 >= 4)
            {
                mc = 1;
                nc = 4;
                kernel(m0, m, 1, n0, n, 4, matmul1x4);
            }
            else*/
            {
                mc = 1;
                nc = 1;
                kernel(m0, m, 1, n0, n, 1, matmul1x1);
            }

            return (mc << 4) | nc;
        }

        protected BiIntConsumer initMatmul1x1() {
            return (i, j) -> {
                final int blockSize = Q8ByteBufferTensor.BLOCK_SIZE;
                final int blocksNeeded = k / Q8ByteBufferTensor.BLOCK_SIZE;

                int aoffset = aColumnOffset;
                int boffset = bColumnOffset;

                FloatVector acc = FloatVector.zero(FloatVector.SPECIES_256);

                // First take the scaling factors of both tensors and multiply them in SIMD
                for (int bi = 0; bi < blocksNeeded; bi += FloatVector.SPECIES_256.length()) {
                    final var ablock = a.getBlockF()
                        .getVector(FloatVector.SPECIES_256, i, (int) (Q8ByteBufferTensor.I_BLOCK_SIZE * aoffset));
                    final var bblock = b.getBlockF()
                        .getVector(FloatVector.SPECIES_256, j, (int) (Q8ByteBufferTensor.I_BLOCK_SIZE * boffset));
                    final var scales = ablock.mul(bblock);

                    // Now for each scalar fetch the corresponding block of data and dot product them
                    for (int k = 0; k < FloatVector.SPECIES_256.length(); k++, aoffset += blockSize, boffset += blockSize) {
                        final var scale = FloatVector.broadcast(FloatVector.SPECIES_256, scales.lane(k));

                        final var ai = a.getVector(ByteVector.SPECIES_256, i, aoffset);
                        final var af0 = ai.convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);
                        final var af1 = ai.convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 1);

                        // Make 8 bytes -> 16 4bit -> 16 bytes -> 16 32F
                        var b0 = b.getVector(ByteVector.SPECIES_128, j, boffset);
                        var b0low = b0.and(Q4_BYTE_MASK_128).sub(Q4_BYTE_SUB_128);
                        var b0hi = b0.lanewise(VectorOperators.LSHR, Q4_BYTE_SHIFT_128).sub(Q4_BYTE_SUB_128);

                        var isum = b0low.convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0).mul(af0);
                        isum = isum.add(b0hi.convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0).mul(af1));

                        var r0 = isum.convertShape(VectorOperators.S2F, FloatVector.SPECIES_256, 0);
                        var r1 = isum.convertShape(VectorOperators.S2F, FloatVector.SPECIES_256, 1);

                        acc = scale.fma(r0.add(r1), acc);
                    }
                }

                c.set(acc.reduceLanes(VectorOperators.ADD), i, j + rOffset);
            };
        }
    }

    private class GemmerI8Q4_512 extends Gemmer {
        final BiIntConsumer matmul1x1;
        final BiIntConsumer matmul1x4;
        final BiIntConsumer matmul3x4;

        final Q8ByteBufferTensor a;
        final Q4ByteBufferTensor b;

        GemmerI8Q4_512(int k, AbstractTensor ta, AbstractTensor tb, AbstractTensor c, int ith, int nth, int rOffset) {
            super(k, ta, tb, c, ith, nth, rOffset);

            this.a = (Q8ByteBufferTensor) ta;
            this.b = (Q4ByteBufferTensor) tb;

            this.matmul1x1 = initMatmul1x1();
            this.matmul1x4 = initMatmul1x4();
            this.matmul3x4 = initMatmul3x4();
        }

        @Override
        protected int pickKernel(int m0, int m, int n0, int n) {
            short mc, nc;
            if (m - m0 >= 2 && n - n0 >= 2) {
                mc = 2;
                nc = 2;
                kernel(m0, m, 2, n0, n, 2, matmul3x4);
            } else if (m - m0 >= 1 && n - n0 >= 4) {
                mc = 1;
                nc = 4;
                kernel(m0, m, 1, n0, n, 4, matmul1x4);
            } else {
                mc = 1;
                nc = 1;
                kernel(m0, m, 1, n0, n, 1, matmul1x1);
            }

            return (mc << 4) | nc;
        }

        protected BiIntConsumer initMatmul1x1() {
            return (i, j) -> {
                final int blockSize = Q8ByteBufferTensor.BLOCK_SIZE;

                int aoffset = aColumnOffset;
                int boffset = bColumnOffset;

                FloatVector acc = FloatVector.zero(FloatVector.SPECIES_512);

                // First take the scaling factors of both tensors and multiply them in SIMD
                // Now for each scalar fetch the corresponding block of data and dot product them
                for (int l = 0; l < k; l += blockSize, aoffset += blockSize, boffset += blockSize) {
                    final var scale = FloatVector.broadcast(
                        FloatVector.SPECIES_512,
                        a.getFactorForIndex(i, aoffset) * b.getFactorForIndex(j, boffset)
                    );

                    final var af = a.getVector(ByteVector.SPECIES_256, i, aoffset)
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_512, 0)
                        .reinterpretAsShorts();

                    // Make 16 bytes -> 32 4bit -> 32 bytes -> 32 32F
                    final var bf0 = b.getVector(ByteVector.SPECIES_128, j, boffset);

                    // Convert the first 4 bits into bytes
                    final var low0 = bf0.lanewise(VectorOperators.AND, Q4_BYTE_MASK_128)
                        .sub(Q4_BYTE_SUB_128)
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);

                    final var high0 = bf0.lanewise(VectorOperators.ASHR, Q4_BYTE_SHIFT_128)
                        .lanewise(VectorOperators.AND, Q4_BYTE_MASK_128)
                        .sub(Q4_BYTE_SUB_128)
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);

                    var isum = low0.mul(af.castShape(ShortVector.SPECIES_256, 0)).add(high0.mul(af.castShape(ShortVector.SPECIES_256, 1)));

                    var r0 = isum.convertShape(VectorOperators.S2F, FloatVector.SPECIES_512, 0);

                    acc = scale.fma(r0, acc);
                }

                c.set(acc.reduceLanes(VectorOperators.ADD), i, j + rOffset);
            };
        }

        protected BiIntConsumer initMatmul1x4() {
            return (i, j) -> {
                final int blockSize = Q8ByteBufferTensor.BLOCK_SIZE;
                final int blocksNeeded = k / Q8ByteBufferTensor.BLOCK_SIZE;

                int aoffset = aColumnOffset;
                int boffset = bColumnOffset;

                FloatVector acc0 = FloatVector.zero(FloatVector.SPECIES_512);
                FloatVector acc1 = FloatVector.zero(FloatVector.SPECIES_512);
                FloatVector acc2 = FloatVector.zero(FloatVector.SPECIES_512);
                FloatVector acc3 = FloatVector.zero(FloatVector.SPECIES_512);

                // First take the scaling factors of both tensors and multiply them in SIMD
                for (int bi = 0; bi < blocksNeeded; bi += FloatVector.SPECIES_512.length()) {
                    // Now for each scalar fetch the corresponding block of data and dot product them
                    for (int k = 0; k < FloatVector.SPECIES_512.length(); k++, aoffset += blockSize, boffset += blockSize) {
                        float as = a.getFactorForIndex(i + 0, aoffset);
                        final var scale0 = FloatVector.broadcast(FloatVector.SPECIES_512, as * b.getFactorForIndex(j + 0, boffset));
                        final var scale1 = FloatVector.broadcast(FloatVector.SPECIES_512, as * b.getFactorForIndex(j + 1, boffset));
                        final var scale2 = FloatVector.broadcast(FloatVector.SPECIES_512, as * b.getFactorForIndex(j + 2, boffset));
                        final var scale3 = FloatVector.broadcast(FloatVector.SPECIES_512, as * b.getFactorForIndex(j + 3, boffset));

                        var af = a.getVector(ByteVector.SPECIES_256, i, aoffset)
                            .convertShape(VectorOperators.B2S, ShortVector.SPECIES_512, 0);
                        var af0 = af.castShape(ShortVector.SPECIES_256, 0);
                        var af1 = af.castShape(ShortVector.SPECIES_256, 1);

                        // Make 16 bytes -> 32 4bit -> 32 bytes -> 32 32F
                        final var bf0 = b.getVector(ByteVector.SPECIES_128, j + 0, boffset);
                        final var bf1 = b.getVector(ByteVector.SPECIES_128, j + 1, boffset);
                        final var bf2 = b.getVector(ByteVector.SPECIES_128, j + 2, boffset);
                        final var bf3 = b.getVector(ByteVector.SPECIES_128, j + 3, boffset);

                        // Convert the first 4 bits into bytes
                        final var r0 = bf0.lanewise(VectorOperators.AND, Q4_BYTE_MASK_128)
                            .sub(Q4_BYTE_SUB_128)
                            .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0)
                            .mul(af0)
                            .add(
                                bf0.lanewise(VectorOperators.ASHR, Q4_BYTE_SHIFT_128)
                                    .lanewise(VectorOperators.AND, Q4_BYTE_MASK_128)
                                    .sub(Q4_BYTE_SUB_128)
                                    .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0)
                                    .mul(af1)
                            )
                            .convertShape(VectorOperators.S2F, FloatVector.SPECIES_512, 0);

                        final var r1 = bf1.lanewise(VectorOperators.AND, Q4_BYTE_MASK_128)
                            .sub(Q4_BYTE_SUB_128)
                            .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0)
                            .mul(af0)
                            .add(
                                bf1.lanewise(VectorOperators.ASHR, Q4_BYTE_SHIFT_128)
                                    .lanewise(VectorOperators.AND, Q4_BYTE_MASK_128)
                                    .sub(Q4_BYTE_SUB_128)
                                    .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0)
                                    .mul(af1)
                            )
                            .convertShape(VectorOperators.S2F, FloatVector.SPECIES_512, 0);

                        final var r2 = bf2.lanewise(VectorOperators.AND, Q4_BYTE_MASK_128)
                            .sub(Q4_BYTE_SUB_128)
                            .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0)
                            .mul(af0)
                            .add(
                                bf2.lanewise(VectorOperators.ASHR, Q4_BYTE_SHIFT_128)
                                    .lanewise(VectorOperators.AND, Q4_BYTE_MASK_128)
                                    .sub(Q4_BYTE_SUB_128)
                                    .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0)
                                    .mul(af1)
                            )
                            .convertShape(VectorOperators.S2F, FloatVector.SPECIES_512, 0);

                        final var r3 = bf3.lanewise(VectorOperators.AND, Q4_BYTE_MASK_128)
                            .sub(Q4_BYTE_SUB_128)
                            .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0)
                            .mul(af0)
                            .add(
                                bf3.lanewise(VectorOperators.ASHR, Q4_BYTE_SHIFT_128)
                                    .lanewise(VectorOperators.AND, Q4_BYTE_MASK_128)
                                    .sub(Q4_BYTE_SUB_128)
                                    .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0)
                                    .mul(af1)
                            )
                            .convertShape(VectorOperators.S2F, FloatVector.SPECIES_512, 0);

                        acc0 = scale0.fma(r0, acc0);
                        acc1 = scale1.fma(r1, acc1);
                        acc2 = scale2.fma(r2, acc2);
                        acc3 = scale3.fma(r3, acc3);
                    }
                }

                float r0 = acc0.reduceLanes(VectorOperators.ADD);
                float r1 = acc1.reduceLanes(VectorOperators.ADD);
                float r2 = acc2.reduceLanes(VectorOperators.ADD);
                float r3 = acc3.reduceLanes(VectorOperators.ADD);

                c.set(r0, i, j + 0 + rOffset);
                c.set(r1, i, j + 1 + rOffset);
                c.set(r2, i, j + 2 + rOffset);
                c.set(r3, i, j + 3 + rOffset);
            };
        }

        protected BiIntConsumer initMatmul3x4() {
            return (i, j) -> {
                final int blockSize = Q8ByteBufferTensor.BLOCK_SIZE;

                int aoffset = aColumnOffset;
                int boffset = bColumnOffset;

                FloatVector acc00 = FloatVector.zero(FloatVector.SPECIES_512);
                FloatVector acc01 = FloatVector.zero(FloatVector.SPECIES_512);
                FloatVector acc10 = FloatVector.zero(FloatVector.SPECIES_512);
                FloatVector acc11 = FloatVector.zero(FloatVector.SPECIES_512);

                // First take the scaling factors of both tensors and multiply them in SIMD

                // Now for each scalar fetch the corresponding block of data and dot product them
                for (int l = 0; l < k; l += blockSize, aoffset += blockSize, boffset += blockSize) {
                    float as0 = a.getFactorForIndex(i + 0, aoffset);
                    float as1 = a.getFactorForIndex(i + 1, aoffset);
                    float bs0 = b.getFactorForIndex(j + 0, boffset);
                    float bs1 = b.getFactorForIndex(j + 1, boffset);

                    var scale00 = FloatVector.broadcast(FloatVector.SPECIES_512, as0 * bs0);
                    var scale01 = FloatVector.broadcast(FloatVector.SPECIES_512, as0 * bs1);
                    var scale10 = FloatVector.broadcast(FloatVector.SPECIES_512, as1 * bs0);
                    var scale11 = FloatVector.broadcast(FloatVector.SPECIES_512, as1 * bs1);

                    var af0 = a.getVector(ByteVector.SPECIES_256, i + 0, aoffset)
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_512, 0);

                    var af1 = a.getVector(ByteVector.SPECIES_256, i + 1, aoffset)
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_512, 0);

                    var af0low = af0.castShape(ShortVector.SPECIES_256, 0);
                    var af0high = af0.castShape(ShortVector.SPECIES_256, 1);

                    var af1low = af1.castShape(ShortVector.SPECIES_256, 0);
                    var af1high = af1.castShape(ShortVector.SPECIES_256, 1);

                    // Make 16 bytes -> 32 4bit -> 32 bytes -> 32 32F
                    var bf0 = b.getVector(ByteVector.SPECIES_128, j + 0, boffset);
                    var bf1 = b.getVector(ByteVector.SPECIES_128, j + 1, boffset);

                    var low0 = bf0.lanewise(VectorOperators.AND, Q4_BYTE_MASK_128)
                        .sub(Q4_BYTE_SUB_128)
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);

                    var high0 = bf0.lanewise(VectorOperators.ASHR, Q4_BYTE_SHIFT_128)
                        .lanewise(VectorOperators.AND, Q4_BYTE_MASK_128)
                        .sub(Q4_BYTE_SUB_128)
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);

                    var low1 = bf1.lanewise(VectorOperators.AND, Q4_BYTE_MASK_128)
                        .sub(Q4_BYTE_SUB_128)
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);

                    var high1 = bf1.lanewise(VectorOperators.ASHR, Q4_BYTE_SHIFT_128)
                        .lanewise(VectorOperators.AND, Q4_BYTE_MASK_128)
                        .sub(Q4_BYTE_SUB_128)
                        .convertShape(VectorOperators.B2S, ShortVector.SPECIES_256, 0);

                    // Convert the first 4 bits into bytes
                    final var r00 = low0.mul(af0low).add(high0.mul(af0high)).convertShape(VectorOperators.S2F, FloatVector.SPECIES_512, 0);

                    final var r01 = low1.mul(af0low).add(high1.mul(af0high)).convertShape(VectorOperators.S2F, FloatVector.SPECIES_512, 0);

                    final var r10 = low0.mul(af1low).add(high0.mul(af1high)).convertShape(VectorOperators.S2F, FloatVector.SPECIES_512, 0);

                    final var r11 = low1.mul(af1low).add(high1.mul(af1high)).convertShape(VectorOperators.S2F, FloatVector.SPECIES_512, 0);

                    acc00 = scale00.fma(r00, acc00);
                    acc01 = scale01.fma(r01, acc01);
                    acc10 = scale10.fma(r10, acc10);
                    acc11 = scale11.fma(r11, acc11);
                }

                float r00 = acc00.reduceLanes(VectorOperators.ADD);
                float r01 = acc01.reduceLanes(VectorOperators.ADD);
                float r10 = acc10.reduceLanes(VectorOperators.ADD);
                float r11 = acc11.reduceLanes(VectorOperators.ADD);

                c.set(r00, i + 0, j + 0 + rOffset);
                c.set(r01, i + 0, j + 1 + rOffset);
                c.set(r10, i + 1, j + 0 + rOffset);
                c.set(r11, i + 1, j + 1 + rOffset);
            };
        }
    }

    private class GemmerF32 extends Gemmer {

        final BiIntConsumer matmul1x1;
        final BiIntConsumer matmul1x4;
        final BiIntConsumer matmul3x4;
        final BiIntConsumer matmul4x1;

        GemmerF32(int k, AbstractTensor a, AbstractTensor b, AbstractTensor c, int ith, int nth, int rOffset) {
            super(k, a, b, c, ith, nth, rOffset);

            this.matmul1x1 = initMatmul1x1();
            this.matmul1x4 = initMatmul1x4();
            this.matmul3x4 = initMatmul3x4();
            this.matmul4x1 = initMatmul4x1();
        }

        @Override
        protected int pickKernel(int m0, int m, int n0, int n) {
            short mc, nc;
            if (m - m0 >= 3 && n - n0 >= 4) {
                mc = 3;
                nc = 4;
                kernel(m0, m, 3, n0, n, 4, matmul3x4);
            } else if (m - m0 >= 4 && n - n0 >= 1) {
                mc = 4;
                nc = 1;
                kernel(m0, m, 4, n0, n, 1, matmul4x1);
            } else if (m - m0 >= 1 && n - n0 >= 4) {
                mc = 1;
                nc = 4;
                kernel(m0, m, 1, n0, n, 4, matmul1x4);
            } else {
                mc = 1;
                nc = 1;
                kernel(m0, m, 1, n0, n, 1, matmul1x1);
            }

            return (mc << 4) | nc;
        }

        protected BiIntConsumer initMatmul1x1() {
            return (i, j) -> {
                FloatVector vc = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                int aoffset = aColumnOffset;
                int boffset = bColumnOffset;
                int alim = aColumnOffset + k;
                int blim = bColumnOffset + k;

                for (; aoffset < alim || boffset < blim; aoffset += FloatVector.SPECIES_PREFERRED.length(), boffset +=
                    FloatVector.SPECIES_PREFERRED.length()) {
                    FloatVector va = a.getVector(FloatVector.SPECIES_PREFERRED, i, aoffset).reinterpretAsFloats();
                    FloatVector vb = b.getVector(FloatVector.SPECIES_PREFERRED, j, boffset).reinterpretAsFloats();
                    vc = va.fma(vb, vc);
                }
                c.set(vc.reduceLanes(VectorOperators.ADD), i, j + rOffset);
            };
        }

        protected BiIntConsumer initMatmul1x4() {
            return (i, j) -> {
                FloatVector vc0 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc1 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc2 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc3 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

                int aoffset = aColumnOffset;
                int boffset = bColumnOffset;
                int alim = aColumnOffset + k;
                int blim = bColumnOffset + k;

                for (; aoffset < alim || boffset < blim; aoffset += FloatVector.SPECIES_PREFERRED.length(), boffset +=
                    FloatVector.SPECIES_PREFERRED.length()) {
                    FloatVector va = a.getVector(FloatVector.SPECIES_PREFERRED, i, aoffset).reinterpretAsFloats();
                    FloatVector vb0 = b.getVector(FloatVector.SPECIES_PREFERRED, j + 0, boffset).reinterpretAsFloats();
                    FloatVector vb1 = b.getVector(FloatVector.SPECIES_PREFERRED, j + 1, boffset).reinterpretAsFloats();
                    FloatVector vb2 = b.getVector(FloatVector.SPECIES_PREFERRED, j + 2, boffset).reinterpretAsFloats();
                    FloatVector vb3 = b.getVector(FloatVector.SPECIES_PREFERRED, j + 3, boffset).reinterpretAsFloats();
                    vc0 = va.fma(vb0, vc0);
                    vc1 = va.fma(vb1, vc1);
                    vc2 = va.fma(vb2, vc2);
                    vc3 = va.fma(vb3, vc3);
                }

                c.set(vc0.reduceLanes(VectorOperators.ADD), i, j + 0 + rOffset);
                c.set(vc1.reduceLanes(VectorOperators.ADD), i, j + 1 + rOffset);
                c.set(vc2.reduceLanes(VectorOperators.ADD), i, j + 2 + rOffset);
                c.set(vc3.reduceLanes(VectorOperators.ADD), i, j + 3 + rOffset);
            };
        }

        protected BiIntConsumer initMatmul3x4() {
            return (i, j) -> {
                FloatVector vc00 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc01 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc02 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc03 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc10 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc11 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc12 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc13 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc20 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc21 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc22 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc23 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

                int aoffset = aColumnOffset;
                int boffset = bColumnOffset;
                int alim = aColumnOffset + k;
                int blim = bColumnOffset + k;

                for (; aoffset < alim || boffset < blim; aoffset += FloatVector.SPECIES_PREFERRED.length(), boffset +=
                    FloatVector.SPECIES_PREFERRED.length()) {
                    FloatVector vb0 = b.getVector(FloatVector.SPECIES_PREFERRED, j + 0, boffset).reinterpretAsFloats();
                    FloatVector vb1 = b.getVector(FloatVector.SPECIES_PREFERRED, j + 1, boffset).reinterpretAsFloats();
                    FloatVector vb2 = b.getVector(FloatVector.SPECIES_PREFERRED, j + 2, boffset).reinterpretAsFloats();
                    FloatVector vb3 = b.getVector(FloatVector.SPECIES_PREFERRED, j + 3, boffset).reinterpretAsFloats();

                    FloatVector va = a.getVector(FloatVector.SPECIES_PREFERRED, i + 0, aoffset).reinterpretAsFloats();
                    vc00 = va.fma(vb0, vc00);
                    vc01 = va.fma(vb1, vc01);
                    vc02 = va.fma(vb2, vc02);
                    vc03 = va.fma(vb3, vc03);

                    FloatVector va1 = a.getVector(FloatVector.SPECIES_PREFERRED, i + 1, aoffset).reinterpretAsFloats();
                    vc10 = va1.fma(vb0, vc10);
                    vc11 = va1.fma(vb1, vc11);
                    vc12 = va1.fma(vb2, vc12);
                    vc13 = va1.fma(vb3, vc13);

                    FloatVector va2 = a.getVector(FloatVector.SPECIES_PREFERRED, i + 2, aoffset).reinterpretAsFloats();
                    vc20 = va2.fma(vb0, vc20);
                    vc21 = va2.fma(vb1, vc21);
                    vc22 = va2.fma(vb2, vc22);
                    vc23 = va2.fma(vb3, vc23);
                }

                c.set(vc00.reduceLanes(VectorOperators.ADD), i + 0, j + 0 + rOffset);
                c.set(vc01.reduceLanes(VectorOperators.ADD), i + 0, j + 1 + rOffset);
                c.set(vc02.reduceLanes(VectorOperators.ADD), i + 0, j + 2 + rOffset);
                c.set(vc03.reduceLanes(VectorOperators.ADD), i + 0, j + 3 + rOffset);

                c.set(vc10.reduceLanes(VectorOperators.ADD), i + 1, j + 0 + rOffset);
                c.set(vc11.reduceLanes(VectorOperators.ADD), i + 1, j + 1 + rOffset);
                c.set(vc12.reduceLanes(VectorOperators.ADD), i + 1, j + 2 + rOffset);
                c.set(vc13.reduceLanes(VectorOperators.ADD), i + 1, j + 3 + rOffset);

                c.set(vc20.reduceLanes(VectorOperators.ADD), i + 2, j + 0 + rOffset);
                c.set(vc21.reduceLanes(VectorOperators.ADD), i + 2, j + 1 + rOffset);
                c.set(vc22.reduceLanes(VectorOperators.ADD), i + 2, j + 2 + rOffset);
                c.set(vc23.reduceLanes(VectorOperators.ADD), i + 2, j + 3 + rOffset);
            };
        }

        protected BiIntConsumer initMatmul4x1() {
            return (i, j) -> {
                FloatVector vc0 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc1 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc2 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc3 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);

                int aoffset = aColumnOffset;
                int boffset = bColumnOffset;
                int alim = aColumnOffset + k;
                int blim = bColumnOffset + k;

                for (; aoffset < alim || boffset < blim; aoffset += FloatVector.SPECIES_PREFERRED.length(), boffset +=
                    FloatVector.SPECIES_PREFERRED.length()) {
                    FloatVector va0 = a.getVector(FloatVector.SPECIES_PREFERRED, i + 0, aoffset).reinterpretAsFloats();
                    FloatVector va1 = a.getVector(FloatVector.SPECIES_PREFERRED, i + 1, aoffset).reinterpretAsFloats();
                    FloatVector va2 = a.getVector(FloatVector.SPECIES_PREFERRED, i + 2, aoffset).reinterpretAsFloats();
                    FloatVector va3 = a.getVector(FloatVector.SPECIES_PREFERRED, i + 3, aoffset).reinterpretAsFloats();
                    FloatVector vb0 = b.getVector(FloatVector.SPECIES_PREFERRED, j, boffset).reinterpretAsFloats();

                    vc0 = va0.fma(vb0, vc0);
                    vc1 = va1.fma(vb0, vc1);
                    vc2 = va2.fma(vb0, vc2);
                    vc3 = va3.fma(vb0, vc3);
                }

                c.set(vc0.reduceLanes(VectorOperators.ADD), i + 0, j + rOffset);
                c.set(vc1.reduceLanes(VectorOperators.ADD), i + 1, j + rOffset);
                c.set(vc2.reduceLanes(VectorOperators.ADD), i + 2, j + rOffset);
                c.set(vc3.reduceLanes(VectorOperators.ADD), i + 3, j + rOffset);
            };
        }
    }

    private class GemmerBF16 extends Gemmer {

        final BiIntConsumer matmul1x1;
        /*final BiIntConsumer matmul1x4;
        final BiIntConsumer matmul3x4;
        final BiIntConsumer matmul4x1;*/

        final BFloat16BufferTensor a;
        final BFloat16BufferTensor b;

        GemmerBF16(int k, AbstractTensor ta, AbstractTensor tb, AbstractTensor c, int ith, int nth, int rOffset) {
            super(k, ta, tb, c, ith, nth, rOffset);

            this.matmul1x1 = initMatmul1x1();
            /*this.matmul1x4 = initMatmul1x4();
            this.matmul3x4 = initMatmul3x4();
            this.matmul4x1 = initMatmul4x1();*/

            this.a = (BFloat16BufferTensor) ta;
            this.b = (BFloat16BufferTensor) tb;
        }

        @Override
        protected int pickKernel(int m0, int m, int n0, int n) {
            short mc, nc;
            /*if (m - m0 >= 3 && n - n0 >= 4) {
                mc = 3;
                nc = 4;
                kernel(m0, m, 3, n0, n, 4, matmul3x4);
            } else if (m - m0 >= 4 && n - n0 >= 1) {
                mc = 4;
                nc = 1;
                kernel(m0, m, 4, n0, n, 1, matmul4x1);
            } else if (m - m0 >= 1 && n - n0 >= 4) {
                mc = 1;
                nc = 4;
                kernel(m0, m, 1, n0, n, 4, matmul1x4);
            } else {*/
            mc = 1;
            nc = 1;
            kernel(m0, m, 1, n0, n, 1, matmul1x1);
            // }

            return (mc << 4) | nc;
        }

        protected BiIntConsumer initMatmul1x1() {
            return (i, j) -> {
                FloatVector vc = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                int aoffset = aColumnOffset;
                int boffset = bColumnOffset;
                int alim = aColumnOffset + k;
                int blim = bColumnOffset + k;
                int slen = ShortVector.SPECIES_PREFERRED.length();
                for (; aoffset < alim && boffset < blim; aoffset += slen, boffset += slen) {
                    ShortVector sa = a.getVector(ShortVector.SPECIES_PREFERRED, i, aoffset);
                    FloatVector va0 = sa.convertShape(VectorOperators.S2I, IntVector.SPECIES_PREFERRED, 0)
                        .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                        .reinterpretAsFloats();

                    FloatVector va1 = sa.convertShape(VectorOperators.S2I, IntVector.SPECIES_PREFERRED, 1)
                        .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                        .reinterpretAsFloats();

                    ShortVector sb = b.getVector(ShortVector.SPECIES_PREFERRED, j, boffset);
                    FloatVector vb0 = sb.convertShape(VectorOperators.S2I, IntVector.SPECIES_PREFERRED, 0)
                        .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                        .reinterpretAsFloats();

                    FloatVector vb1 = sb.convertShape(VectorOperators.S2I, IntVector.SPECIES_PREFERRED, 1)
                        .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                        .reinterpretAsFloats();

                    vc = va0.fma(vb0, vc);
                    vc = va1.fma(vb1, vc);
                }
                float res = vc.reduceLanes(VectorOperators.ADD);
                c.set(res, i, j + rOffset);
            };
        }

        /*protected BiIntConsumer initMatmul1x4() {
            return (i, j) -> {
                FloatVector vc0 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc1 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc2 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc3 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        
                int aoffset = aColumnOffset;
                int boffset = bColumnOffset;
                int alim = aColumnOffset + k;
                int blim = bColumnOffset + k;
        
                for (;
                     aoffset < alim || boffset < blim;
                     aoffset += FloatVector.SPECIES_PREFERRED.length(),
                             boffset += FloatVector.SPECIES_PREFERRED.length()) {
                    FloatVector va = a.getVector(FloatVector.SPECIES_PREFERRED, i, aoffset)
                            .reinterpretAsFloats();
                    FloatVector vb0 = b.getVector(FloatVector.SPECIES_PREFERRED, j + 0, boffset)
                            .reinterpretAsFloats();
                    FloatVector vb1 = b.getVector(FloatVector.SPECIES_PREFERRED, j + 1, boffset)
                            .reinterpretAsFloats();
                    FloatVector vb2 = b.getVector(FloatVector.SPECIES_PREFERRED, j + 2, boffset)
                            .reinterpretAsFloats();
                    FloatVector vb3 = b.getVector(FloatVector.SPECIES_PREFERRED, j + 3, boffset)
                            .reinterpretAsFloats();
                    vc0 = va.fma(vb0, vc0);
                    vc1 = va.fma(vb1, vc1);
                    vc2 = va.fma(vb2, vc2);
                    vc3 = va.fma(vb3, vc3);
                }
        
                c.set(vc0.reduceLanes(VectorOperators.ADD), i, j + 0);
                c.set(vc1.reduceLanes(VectorOperators.ADD), i, j + 1);
                c.set(vc2.reduceLanes(VectorOperators.ADD), i, j + 2);
                c.set(vc3.reduceLanes(VectorOperators.ADD), i, j + 3);
            };
        }
        
        protected BiIntConsumer initMatmul3x4() {
            return (i, j) -> {
                FloatVector vc00 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc01 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc02 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc03 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc10 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc11 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc12 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc13 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc20 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc21 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc22 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc23 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        
                int aoffset = aColumnOffset;
                int boffset = bColumnOffset;
                int alim = aColumnOffset + k;
                int blim = bColumnOffset + k;
        
                for (;
                     aoffset < alim || boffset < blim;
                     aoffset += FloatVector.SPECIES_PREFERRED.length(),
                             boffset += FloatVector.SPECIES_PREFERRED.length()) {
                    FloatVector vb0 = b.getVector(FloatVector.SPECIES_PREFERRED, j + 0, boffset)
                            .reinterpretAsFloats();
                    FloatVector vb1 = b.getVector(FloatVector.SPECIES_PREFERRED, j + 1, boffset)
                            .reinterpretAsFloats();
                    FloatVector vb2 = b.getVector(FloatVector.SPECIES_PREFERRED, j + 2, boffset)
                            .reinterpretAsFloats();
                    FloatVector vb3 = b.getVector(FloatVector.SPECIES_PREFERRED, j + 3, boffset)
                            .reinterpretAsFloats();
        
                    FloatVector va = a.getVector(FloatVector.SPECIES_PREFERRED, i + 0, aoffset)
                            .reinterpretAsFloats();
                    vc00 = va.fma(vb0, vc00);
                    vc01 = va.fma(vb1, vc01);
                    vc02 = va.fma(vb2, vc02);
                    vc03 = va.fma(vb3, vc03);
        
                    FloatVector va1 = a.getVector(FloatVector.SPECIES_PREFERRED, i + 1, aoffset)
                            .reinterpretAsFloats();
                    vc10 = va1.fma(vb0, vc10);
                    vc11 = va1.fma(vb1, vc11);
                    vc12 = va1.fma(vb2, vc12);
                    vc13 = va1.fma(vb3, vc13);
        
                    FloatVector va2 = a.getVector(FloatVector.SPECIES_PREFERRED, i + 2, aoffset)
                            .reinterpretAsFloats();
                    vc20 = va2.fma(vb0, vc20);
                    vc21 = va2.fma(vb1, vc21);
                    vc22 = va2.fma(vb2, vc22);
                    vc23 = va2.fma(vb3, vc23);
                }
        
                c.set(vc00.reduceLanes(VectorOperators.ADD), i + 0, j + 0);
                c.set(vc01.reduceLanes(VectorOperators.ADD), i + 0, j + 1);
                c.set(vc02.reduceLanes(VectorOperators.ADD), i + 0, j + 2);
                c.set(vc03.reduceLanes(VectorOperators.ADD), i + 0, j + 3);
        
                c.set(vc10.reduceLanes(VectorOperators.ADD), i + 1, j + 0);
                c.set(vc11.reduceLanes(VectorOperators.ADD), i + 1, j + 1);
                c.set(vc12.reduceLanes(VectorOperators.ADD), i + 1, j + 2);
                c.set(vc13.reduceLanes(VectorOperators.ADD), i + 1, j + 3);
        
                c.set(vc20.reduceLanes(VectorOperators.ADD), i + 2, j + 0);
                c.set(vc21.reduceLanes(VectorOperators.ADD), i + 2, j + 1);
                c.set(vc22.reduceLanes(VectorOperators.ADD), i + 2, j + 2);
                c.set(vc23.reduceLanes(VectorOperators.ADD), i + 2, j + 3);
            };
        }
        
        protected BiIntConsumer initMatmul4x1() {
            return (i, j) -> {
                FloatVector vc0 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc1 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc2 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                FloatVector vc3 = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
        
                int aoffset = aColumnOffset;
                int boffset = bColumnOffset;
                int alim = aColumnOffset + k;
                int blim = bColumnOffset + k;
        
                for (;
                     aoffset < alim || boffset < blim;
                     aoffset += FloatVector.SPECIES_PREFERRED.length(),
                             boffset += FloatVector.SPECIES_PREFERRED.length()) {
                    FloatVector va0 = a.getVector(FloatVector.SPECIES_PREFERRED, i + 0, aoffset)
                            .reinterpretAsFloats();
                    FloatVector va1 = a.getVector(FloatVector.SPECIES_PREFERRED, i + 1, aoffset)
                            .reinterpretAsFloats();
                    FloatVector va2 = a.getVector(FloatVector.SPECIES_PREFERRED, i + 2, aoffset)
                            .reinterpretAsFloats();
                    FloatVector va3 = a.getVector(FloatVector.SPECIES_PREFERRED, i + 3, aoffset)
                            .reinterpretAsFloats();
                    FloatVector vb0 = b.getVector(FloatVector.SPECIES_PREFERRED, j, boffset)
                            .reinterpretAsFloats();
        
                    vc0 = va0.fma(vb0, vc0);
                    vc1 = va1.fma(vb0, vc1);
                    vc2 = va2.fma(vb0, vc2);
                    vc3 = va3.fma(vb0, vc3);
                }
        
                c.set(vc0.reduceLanes(VectorOperators.ADD), i + 0, j);
                c.set(vc1.reduceLanes(VectorOperators.ADD), i + 1, j);
                c.set(vc2.reduceLanes(VectorOperators.ADD), i + 2, j);
                c.set(vc3.reduceLanes(VectorOperators.ADD), i + 3, j);
            };
        }*/
    }

    private class GemmerF32BF16 extends Gemmer {

        final BiIntConsumer matmul1x1;
        /*final BiIntConsumer matmul1x4;
        final BiIntConsumer matmul3x4;
        final BiIntConsumer matmul4x1;*/

        final FloatBufferTensor a;
        final BFloat16BufferTensor b;

        GemmerF32BF16(int k, AbstractTensor ta, AbstractTensor tb, AbstractTensor c, int ith, int nth, int rOffset) {
            super(k, ta, tb, c, ith, nth, rOffset);

            this.matmul1x1 = initMatmul1x1();
            /*this.matmul1x4 = initMatmul1x4();
            this.matmul3x4 = initMatmul3x4();
            this.matmul4x1 = initMatmul4x1();*/
            this.a = (FloatBufferTensor) ta;
            this.b = (BFloat16BufferTensor) tb;
        }

        @Override
        protected int pickKernel(int m0, int m, int n0, int n) {
            short mc, nc;
            /*if (m - m0 >= 3 && n - n0 >= 4) {
                mc = 3;
                nc = 4;
                kernel(m0, m, 3, n0, n, 4, matmul3x4);
            } else if (m - m0 >= 4 && n - n0 >= 1) {
                mc = 4;
                nc = 1;
                kernel(m0, m, 4, n0, n, 1, matmul4x1);
            } else if (m - m0 >= 1 && n - n0 >= 4) {
                mc = 1;
                nc = 4;
                kernel(m0, m, 1, n0, n, 4, matmul1x4);
            } else {*/
            mc = 1;
            nc = 1;
            kernel(m0, m, 1, n0, n, 1, matmul1x1);
            // }

            return (mc << 4) | nc;
        }

        protected BiIntConsumer initMatmul1x1() {
            return (i, j) -> {
                FloatVector vc = FloatVector.zero(FloatVector.SPECIES_PREFERRED);
                int aoffset = aColumnOffset;
                int boffset = bColumnOffset;
                int alim = aColumnOffset + k;
                int blim = bColumnOffset + k;
                int slen = ShortVector.SPECIES_PREFERRED.length();
                for (; aoffset < alim && boffset < blim; aoffset += slen, boffset += slen) {
                    FloatVector va0 = a.getVector(FloatVector.SPECIES_PREFERRED, i, aoffset);
                    FloatVector va1 = a.getVector(FloatVector.SPECIES_PREFERRED, i, aoffset + FloatVector.SPECIES_PREFERRED.length());

                    ShortVector sb = b.getVector(ShortVector.SPECIES_PREFERRED, j, boffset);
                    FloatVector vb0 = sb.convertShape(VectorOperators.ZERO_EXTEND_S2I, IntVector.SPECIES_PREFERRED, 0)
                        .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                        .reinterpretAsFloats();

                    FloatVector vb1 = sb.convertShape(VectorOperators.ZERO_EXTEND_S2I, IntVector.SPECIES_PREFERRED, 1)
                        .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                        .reinterpretAsFloats();

                    vc = va0.fma(vb0, vc);
                    vc = va1.fma(vb1, vc);
                }
                float res = vc.reduceLanes(VectorOperators.ADD);
                c.set(res, i, j + rOffset);
            };
        }
    }

    private abstract class Gemmer {
        final int k;
        final AbstractTensor a;
        final AbstractTensor b;
        final AbstractTensor c;
        final int aColumnOffset;
        final int bColumnOffset;
        final int rOffset;

        // The id of each thread is called ith and the number of threads is called nth.
        Gemmer(int k, AbstractTensor a, AbstractTensor b, AbstractTensor c, int aColumnOffset, int bColumnOffset, int rOffset) {
            this.k = k;
            this.a = a;
            this.b = b;
            this.c = c;
            this.aColumnOffset = aColumnOffset;
            this.bColumnOffset = bColumnOffset;
            this.rOffset = rOffset;
        }

        void matmul(int m0, int m, int n0, int n) {
            mnpack(m0, m, n0, n);
        }

        private void mnpack(int m0, int m, int n0, int n) {
            if (m - m0 <= 0 || n - n0 <= 0) return;

            int r = pickKernel(m0, m, n0, n);
            int mc, nc, mp, np;
            mc = r >> 4;
            nc = r & 0xF;

            mp = m0 + (m - m0) / mc * mc;
            np = n0 + (n - n0) / nc * nc;
            mnpack(mp, m, n0, np);
            mnpack(m0, mp, np, n);
        }

        protected abstract int pickKernel(int m0, int m, int n0, int n);

        void kernel(int m0, int m, int RM, int n0, int n, int RN, BiIntConsumer action) {
            int ytiles = (m - m0) / RM;
            int xtiles = (n - n0) / RN;
            int tiles = ytiles * xtiles;

            for (int job = 0; job < tiles; ++job) {
                int i = m0 + job / xtiles * RM;
                int j = n0 + job % xtiles * RN;

                action.accept(i, j);
            }
        }
    }

    @Override
    public AbstractTensor quantize(AbstractTensor t, DType qtype, int offset, int length) {
        Preconditions.checkArgument(t.dims() == 2 && length % Q8ByteBufferTensor.BLOCK_SIZE == 0);

        return switch (t.dType()) {
            case F32 -> switch (qtype) {
                case I8 -> switch (vectorType) {
                    case AVX_512 -> quantizeQ8_512((FloatBufferTensor) t, offset, length);
                    case AVX_256 -> quantizeQ8_256((FloatBufferTensor) t, offset, length);
                    case ARM_128 -> quantizeQ8_arm((FloatBufferTensor) t, offset, length);
                    default -> throw new UnsupportedOperationException();
                };
                case BF16 -> quantizeBF16((FloatBufferTensor) t, offset, length);
                default -> throw new UnsupportedOperationException("F32 => " + qtype);
            };
            case BF16 -> switch (qtype) {
                case I8 -> switch (vectorType) {
                    case AVX_512 -> quantizeBF16_Q8_512((BFloat16BufferTensor) t, offset, length);
                    case AVX_256 -> quantizeBF16_Q8_256((BFloat16BufferTensor) t, offset, length);
                    case ARM_128 -> quantizeBF16_Q8_arm((BFloat16BufferTensor) t, offset, length);
                    default -> throw new UnsupportedOperationException();
                };
                case F32 -> quantizeBF16_F32((BFloat16BufferTensor) t, offset, length);
                default -> throw new UnsupportedOperationException("BF16 => " + qtype);
            };
            default -> throw new UnsupportedOperationException("" + t.dType());
        };
    }

    public BFloat16BufferTensor quantizeBF16(FloatBufferTensor ft, final int offset, int length) {

        // Need this till we have a proper quantization
        https: // github.com/pytorch/pytorch/blob/7c1fbc7fe9cb8ddd5c913b4b3a9e94d00cb055ee/aten/src/ATen/cpu/vec/vec256/vec256_bfloat16.h#L47
        if (true) return new BFloat16BufferTensor(ft);

        // Up to caller to release
        BFloat16BufferTensor qft = (BFloat16BufferTensor) TensorCache.instance.get(DType.BF16, ft.shape());
        int batchSize = ft.shape().first();
        for (int b = 0; b < batchSize; b++) {
            for (int i = offset; i < offset + length; i += ShortVector.SPECIES_PREFERRED.length()) {
                var r0 = ft.getVector(FloatVector.SPECIES_PREFERRED, b, i)
                    .reinterpretAsInts()
                    .lanewise(VectorOperators.ASHR, BF16_BYTE_SHIFT)
                    .convertShape(VectorOperators.I2S, ShortVector.SPECIES_PREFERRED, 0);

                var r1 = ft.getVector(FloatVector.SPECIES_PREFERRED, b, i + FloatVector.SPECIES_PREFERRED.length())
                    .reinterpretAsInts()
                    .lanewise(VectorOperators.ASHR, BF16_BYTE_SHIFT)
                    .convertShape(VectorOperators.I2S, ShortVector.SPECIES_PREFERRED, -1);

                VectorMask<Short> mask = VectorMask.fromLong(
                    ShortVector.SPECIES_PREFERRED,
                    (1L << FloatVector.SPECIES_PREFERRED.length()) - 1
                );
                mask = mask.not(); // Invert the mask to select the second half

                var r = r0.blend(r1, mask);

                qft.intoTensor((ShortVector) r, b, i);
            }
        }

        return qft;
    }

    public FloatBufferTensor quantizeBF16_F32(BFloat16BufferTensor ft, final int offset, int length) {

        // Up to caller to release
        FloatBufferTensor qft = (FloatBufferTensor) TensorCache.instance.get(DType.F32, ft.shape());
        int batchSize = ft.shape().first();
        for (int b = 0; b < batchSize; b++) {
            for (int i = offset; i < offset + length; i += ShortVector.SPECIES_PREFERRED.length()) {
                var sa = ft.getVector(ShortVector.SPECIES_PREFERRED, b, i);
                var af0 = sa.convertShape(VectorOperators.S2I, IntVector.SPECIES_PREFERRED, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

                var af1 = sa.convertShape(VectorOperators.S2I, IntVector.SPECIES_PREFERRED, 1)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

                qft.intoTensor(af0, b, i);
                qft.intoTensor(af1, b, i + FloatVector.SPECIES_PREFERRED.length());
            }
        }

        return qft;
    }

    public Q8ByteBufferTensor quantizeQ8_512(FloatBufferTensor ft, final int offset, int length) {

        // Up to caller to release
        Q8ByteBufferTensor qft = (Q8ByteBufferTensor) TensorCache.instance.get(DType.I8, ft.shape());
        int batchSize = ft.shape().first();
        for (int b = 0; b < batchSize; b++) {
            for (int i = offset; i < offset + length; i += Q8ByteBufferTensor.BLOCK_SIZE) {
                FloatVector fv0 = ft.getVector(FloatVector.SPECIES_512, b, i);
                FloatVector fv1 = ft.getVector(FloatVector.SPECIES_512, b, i + 16);

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
                var bvq0 = fvq0.convertShape(VectorOperators.F2B, ByteVector.SPECIES_128, 0).reinterpretAsBytes();
                var bvq1 = fvq1.convertShape(VectorOperators.F2B, ByteVector.SPECIES_128, 0).reinterpretAsBytes();

                qft.intoTensor(bvq0, b, i);
                qft.intoTensor(bvq1, b, i + 16);
                try {
                    qft.getBlockF().set(d, b, (int) (i * Q8ByteBufferTensor.I_BLOCK_SIZE));
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }

        return qft;
    }

    public Q8ByteBufferTensor quantizeQ8_256(FloatBufferTensor ft, int offset, int length) {

        // Up to caller to release
        Q8ByteBufferTensor qft = (Q8ByteBufferTensor) TensorCache.instance.get(DType.I8, ft.shape());

        int batchSize = ft.shape().first();
        for (int b = 0; b < batchSize; b++) {
            for (int i = offset; i < offset + length; i += Q8ByteBufferTensor.BLOCK_SIZE) {
                FloatVector fv0 = ft.getVector(FloatVector.SPECIES_256, b, i);
                FloatVector fv1 = ft.getVector(FloatVector.SPECIES_256, b, i + 8);
                FloatVector fv2 = ft.getVector(FloatVector.SPECIES_256, b, i + 16);
                FloatVector fv3 = ft.getVector(FloatVector.SPECIES_256, b, i + 24);

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
                var bvq0 = fvq0.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0).reinterpretAsBytes();
                var bvq1 = fvq1.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0).reinterpretAsBytes();
                var bvq2 = fvq2.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0).reinterpretAsBytes();
                var bvq3 = fvq3.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0).reinterpretAsBytes();

                qft.intoTensor(bvq0, b, i);
                qft.intoTensor(bvq1, b, i + 8);
                qft.intoTensor(bvq2, b, i + 16);
                qft.intoTensor(bvq3, b, i + 24);

                qft.getBlockF().set(d, b, (int) (i * Q8ByteBufferTensor.I_BLOCK_SIZE));
            }
        }

        return qft;
    }

    public Q8ByteBufferTensor quantizeQ8_arm(FloatBufferTensor ft, int offset, int length) {

        // Up to caller to release
        Q8ByteBufferTensor qft = (Q8ByteBufferTensor) TensorCache.instance.get(DType.I8, ft.shape());

        int batchSize = ft.shape().first();
        for (int b = 0; b < batchSize; b++) {
            for (int i = offset; i < offset + length; i += Q8ByteBufferTensor.BLOCK_SIZE) {
                FloatVector fv0 = ft.getVector(FloatVector.SPECIES_128, b, i + 0);
                FloatVector fv1 = ft.getVector(FloatVector.SPECIES_128, b, i + 4);
                FloatVector fv2 = ft.getVector(FloatVector.SPECIES_128, b, i + 8);
                FloatVector fv3 = ft.getVector(FloatVector.SPECIES_128, b, i + 12);
                FloatVector fv4 = ft.getVector(FloatVector.SPECIES_128, b, i + 16);
                FloatVector fv5 = ft.getVector(FloatVector.SPECIES_128, b, i + 20);
                FloatVector fv6 = ft.getVector(FloatVector.SPECIES_128, b, i + 24);
                FloatVector fv7 = ft.getVector(FloatVector.SPECIES_128, b, i + 28);

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
                var bvq0 = fvq0.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0).reinterpretAsBytes();
                var bvq1 = fvq1.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0).reinterpretAsBytes();
                var bvq2 = fvq2.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0).reinterpretAsBytes();
                var bvq3 = fvq3.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0).reinterpretAsBytes();
                var bvq4 = fvq4.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0).reinterpretAsBytes();
                var bvq5 = fvq5.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0).reinterpretAsBytes();
                var bvq6 = fvq6.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0).reinterpretAsBytes();
                var bvq7 = fvq7.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0).reinterpretAsBytes();

                qft.intoTensor(bvq0, BYTE_MASK_32, b, i + 0);
                qft.intoTensor(bvq1, BYTE_MASK_32, b, i + 4);
                qft.intoTensor(bvq2, BYTE_MASK_32, b, i + 8);
                qft.intoTensor(bvq3, BYTE_MASK_32, b, i + 12);
                qft.intoTensor(bvq4, BYTE_MASK_32, b, i + 16);
                qft.intoTensor(bvq5, BYTE_MASK_32, b, i + 20);
                qft.intoTensor(bvq6, BYTE_MASK_32, b, i + 24);
                qft.intoTensor(bvq7, BYTE_MASK_32, b, i + 28);

                qft.getBlockF().set(d, b, (int) (i * Q8ByteBufferTensor.I_BLOCK_SIZE));
            }
        }

        return qft;
    }

    public Q8ByteBufferTensor quantizeBF16_Q8_512(BFloat16BufferTensor ft, final int offset, int length) {

        // Up to caller to release
        Q8ByteBufferTensor qft = (Q8ByteBufferTensor) TensorCache.instance.get(DType.I8, ft.shape());
        int batchSize = ft.shape().first();
        for (int b = 0; b < batchSize; b++) {
            for (int i = offset; i < offset + length; i += Q8ByteBufferTensor.BLOCK_SIZE) {
                ShortVector sv = ft.getVector(ShortVector.SPECIES_512, b, i);
                FloatVector fv0 = sv.convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_512)
                    .reinterpretAsFloats();
                FloatVector fv1 = sv.convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 1)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_512)
                    .reinterpretAsFloats();

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
                var bvq0 = fvq0.convertShape(VectorOperators.F2B, ByteVector.SPECIES_128, 0).reinterpretAsBytes();
                var bvq1 = fvq1.convertShape(VectorOperators.F2B, ByteVector.SPECIES_128, 0).reinterpretAsBytes();

                qft.intoTensor(bvq0, b, i);
                qft.intoTensor(bvq1, b, i + 16);
                try {
                    qft.getBlockF().set(d, b, (int) (i * Q8ByteBufferTensor.I_BLOCK_SIZE));
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }

        return qft;
    }

    public Q8ByteBufferTensor quantizeBF16_Q8_256(BFloat16BufferTensor ft, int offset, int length) {

        // Up to caller to release
        Q8ByteBufferTensor qft = (Q8ByteBufferTensor) TensorCache.instance.get(DType.I8, ft.shape());

        int batchSize = ft.shape().first();
        for (int b = 0; b < batchSize; b++) {
            for (int i = offset; i < offset + length; i += Q8ByteBufferTensor.BLOCK_SIZE) {
                ShortVector sv = ft.getVector(ShortVector.SPECIES_256, b, i);
                FloatVector fv0 = sv.convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_256)
                    .reinterpretAsFloats();
                FloatVector fv1 = sv.convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 1)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_256)
                    .reinterpretAsFloats();

                sv = ft.getVector(ShortVector.SPECIES_256, b, i + 16);
                FloatVector fv2 = sv.convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_256)
                    .reinterpretAsFloats();
                FloatVector fv3 = sv.convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 1)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_256)
                    .reinterpretAsFloats();

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
                var bvq0 = fvq0.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0).reinterpretAsBytes();
                var bvq1 = fvq1.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0).reinterpretAsBytes();
                var bvq2 = fvq2.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0).reinterpretAsBytes();
                var bvq3 = fvq3.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0).reinterpretAsBytes();

                qft.intoTensor(bvq0, b, i);
                qft.intoTensor(bvq1, b, i + 8);
                qft.intoTensor(bvq2, b, i + 16);
                qft.intoTensor(bvq3, b, i + 24);

                qft.getBlockF().set(d, b, (int) (i * Q8ByteBufferTensor.I_BLOCK_SIZE));
            }
        }

        return qft;
    }

    public Q8ByteBufferTensor quantizeBF16_Q8_arm(BFloat16BufferTensor ft, int offset, int length) {

        // Up to caller to release
        Q8ByteBufferTensor qft = (Q8ByteBufferTensor) TensorCache.instance.get(DType.I8, ft.shape());

        int batchSize = ft.shape().first();
        for (int b = 0; b < batchSize; b++) {
            for (int i = offset; i < offset + length; i += Q8ByteBufferTensor.BLOCK_SIZE) {
                ShortVector sv = ft.getVector(ShortVector.SPECIES_128, b, i);
                FloatVector fv0 = sv.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_128)
                    .reinterpretAsFloats();
                FloatVector fv1 = sv.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 1)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_128)
                    .reinterpretAsFloats();

                sv = ft.getVector(ShortVector.SPECIES_128, b, i + 8);
                FloatVector fv2 = sv.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_128)
                    .reinterpretAsFloats();
                FloatVector fv3 = sv.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 1)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_128)
                    .reinterpretAsFloats();

                sv = ft.getVector(ShortVector.SPECIES_128, b, i + 16);
                FloatVector fv4 = sv.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_128)
                    .reinterpretAsFloats();
                FloatVector fv5 = sv.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 1)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_128)
                    .reinterpretAsFloats();

                sv = ft.getVector(ShortVector.SPECIES_128, b, i + 24);
                FloatVector fv6 = sv.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_128)
                    .reinterpretAsFloats();
                FloatVector fv7 = sv.convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 1)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_128)
                    .reinterpretAsFloats();

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
                var bvq0 = fvq0.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0).reinterpretAsBytes();
                var bvq1 = fvq1.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0).reinterpretAsBytes();
                var bvq2 = fvq2.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0).reinterpretAsBytes();
                var bvq3 = fvq3.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0).reinterpretAsBytes();
                var bvq4 = fvq4.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0).reinterpretAsBytes();
                var bvq5 = fvq5.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0).reinterpretAsBytes();
                var bvq6 = fvq6.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0).reinterpretAsBytes();
                var bvq7 = fvq7.convertShape(VectorOperators.F2B, ByteVector.SPECIES_64, 0).reinterpretAsBytes();

                qft.intoTensor(bvq0, BYTE_MASK_32, b, i + 0);
                qft.intoTensor(bvq1, BYTE_MASK_32, b, i + 4);
                qft.intoTensor(bvq2, BYTE_MASK_32, b, i + 8);
                qft.intoTensor(bvq3, BYTE_MASK_32, b, i + 12);
                qft.intoTensor(bvq4, BYTE_MASK_32, b, i + 16);
                qft.intoTensor(bvq5, BYTE_MASK_32, b, i + 20);
                qft.intoTensor(bvq6, BYTE_MASK_32, b, i + 24);
                qft.intoTensor(bvq7, BYTE_MASK_32, b, i + 28);

                qft.getBlockF().set(d, b, (int) (i * Q8ByteBufferTensor.I_BLOCK_SIZE));
            }
        }

        return qft;
    }

    @Override
    public void maccumulate(AbstractTensor aBatch, AbstractTensor bBatch, int offset, int limit) {
        Preconditions.checkArgument(aBatch.dType() == bBatch.dType());
        Preconditions.checkArgument(limit % 8 == 0);

        boolean isBatch = bBatch.shape().first() > 1;
        for (int ai = 0; ai < aBatch.shape().first(); ai++) {
            AbstractTensor a = aBatch.slice(ai);
            AbstractTensor b = isBatch ? bBatch.slice(ai) : bBatch;
            switch (a.dType()) {
                case F32:
                    maccumulateF32((FloatBufferTensor) a, (FloatBufferTensor) b, offset, limit);
                    break;
                case BF16:
                    maccumulateBF16((BFloat16BufferTensor) a, (BFloat16BufferTensor) b, offset, limit);
                    break;
                default:
                    throw new UnsupportedOperationException(a.dType().name());
            }
        }
    }

    void maccumulateF32(FloatBufferTensor a, FloatBufferTensor b, int offset, int limit) {
        int upperBound = offset + FloatVector.SPECIES_PREFERRED.loopBound(limit);
        int i = offset;

        for (; i < upperBound; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector va = a.getVector(FloatVector.SPECIES_PREFERRED, 0, i);
            FloatVector vb = b.getVector(FloatVector.SPECIES_PREFERRED, 0, i);
            a.intoTensor(va.mul(vb), 0, i);
        }

        // tail
        for (; i < offset + limit; i++) {
            a.set(a.get(0, i) * b.get(0, i), 0, i);
        }
    }

    void maccumulateBF16(BFloat16BufferTensor a, BFloat16BufferTensor b, int offset, int limit) {
        int upperBound = offset + ShortVector.SPECIES_PREFERRED.loopBound(limit);
        int i = offset;

        int half = ShortVector.SPECIES_PREFERRED.length() / 2;

        for (; i < upperBound; i += ShortVector.SPECIES_PREFERRED.length()) {
            // Convert BF16 to F32
            var sa = a.getVector(ShortVector.SPECIES_PREFERRED, 0, i);
            var af0 = sa.convertShape(VectorOperators.S2I, IntVector.SPECIES_PREFERRED, 0)
                .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                .reinterpretAsFloats();

            var af1 = sa.convertShape(VectorOperators.S2I, IntVector.SPECIES_PREFERRED, 1)
                .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                .reinterpretAsFloats();

            // Convert BF16 to F32
            var sb = b.getVector(ShortVector.SPECIES_PREFERRED, 0, i);
            var bf0 = sb.convertShape(VectorOperators.S2I, IntVector.SPECIES_PREFERRED, 0)
                .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                .reinterpretAsFloats();

            var bf1 = sb.convertShape(VectorOperators.S2I, IntVector.SPECIES_PREFERRED, 1)
                .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                .reinterpretAsFloats();

            var r0 = af0.mul(bf0)
                .reinterpretAsInts()
                .lanewise(VectorOperators.ASHR, BF16_BYTE_SHIFT)
                .convertShape(VectorOperators.I2S, ShortVector.SPECIES_PREFERRED, 0);

            var r1 = af1.mul(bf1)
                .reinterpretAsInts()
                .lanewise(VectorOperators.ASHR, BF16_BYTE_SHIFT)
                .convertShape(VectorOperators.I2S, ShortVector.SPECIES_PREFERRED, -1);

            VectorMask<Short> mask = VectorMask.fromLong(ShortVector.SPECIES_PREFERRED, (1L << half) - 1);
            mask = mask.not(); // Invert the mask to select the second half

            var r = r0.blend(r1, mask);

            a.intoTensor((ShortVector) r, 0, i);
        }

        // tail
        for (; i < offset + limit; i++) {
            a.set(a.get(0, i) * b.get(0, i), 0, i);
        }
    }

    @Override
    public void accumulate(AbstractTensor aBatch, AbstractTensor bBatch, int offset, int limit) {

        boolean isBatch = bBatch.shape().first() > 1;
        for (int ai = 0; ai < aBatch.shape().first(); ai++) {
            AbstractTensor a = aBatch.slice(ai);
            AbstractTensor b = isBatch ? bBatch.slice(ai) : bBatch;
            switch (a.dType()) {
                case F32:
                    switch (b.dType()) {
                        case F32:
                            accumulateF32((FloatBufferTensor) a, (FloatBufferTensor) b, offset, limit);
                            break;
                        case Q4:
                            switch (vectorType) {
                                case AVX_512:
                                case AVX_256:
                                    accumulateF32Q4_256((FloatBufferTensor) a, (Q4ByteBufferTensor) b, offset, limit);
                                    break;
                                case ARM_128:
                                    accumulateF32Q4_arm((FloatBufferTensor) a, (Q4ByteBufferTensor) b, offset, limit);
                                    break;
                                default:
                                    throw new UnsupportedOperationException();
                            }
                            break;
                        case BF16:
                            switch (vectorType) {
                                case AVX_512:
                                case AVX_256:
                                    accumulateF32BF16_256((FloatBufferTensor) a, (BFloat16BufferTensor) b, offset, limit);
                                    break;
                                case ARM_128:
                                    accumulateF32BF16_arm((FloatBufferTensor) a, (BFloat16BufferTensor) b, offset, limit);
                                    break;
                                default:
                                    throw new UnsupportedOperationException();
                            }
                            break;
                        default:
                            throw new UnsupportedOperationException("F32 => " + b.dType());
                    }
                    break;
                case BF16:
                    switch (b.dType()) {
                        case BF16:
                            switch (vectorType) {
                                case AVX_512:
                                    accumulateBF16_512((BFloat16BufferTensor) a, (BFloat16BufferTensor) b, offset, limit);
                                    break;
                                case AVX_256:
                                    accumulateBF16_256((BFloat16BufferTensor) a, (BFloat16BufferTensor) b, offset, limit);
                                    break;
                                case ARM_128:
                                    accumulateBF16_arm((BFloat16BufferTensor) a, (BFloat16BufferTensor) b, offset, limit);
                                    break;
                                default:
                                    throw new UnsupportedOperationException();
                            }
                            break;
                        default:
                            throw new UnsupportedOperationException();
                    }
                    break;
                default:
                    throw new UnsupportedOperationException("" + a.dType());
            }
        }
    }

    private void accumulateF32Q4_arm(FloatBufferTensor a, Q4ByteBufferTensor b, int offset, int limit) {

        int aoffset = offset;
        int boffset = offset;
        int alim = offset + FloatVector.SPECIES_128.loopBound(limit);
        int slen = Q4ByteBufferTensor.BLOCK_SIZE;

        // Now for each scalar fetch the corresponding block of data and dot product them
        for (; aoffset < alim; aoffset += slen, boffset += slen) {
            var scale = FloatVector.broadcast(FloatVector.SPECIES_128, b.getFactorForIndex(0, boffset));

            var af0 = a.getVector(FloatVector.SPECIES_128, 0, aoffset);
            var af1 = a.getVector(FloatVector.SPECIES_128, 0, aoffset + 4);
            var af2 = a.getVector(FloatVector.SPECIES_128, 0, aoffset + 8);
            var af3 = a.getVector(FloatVector.SPECIES_128, 0, aoffset + 12);
            var af4 = a.getVector(FloatVector.SPECIES_128, 0, aoffset + 16);
            var af5 = a.getVector(FloatVector.SPECIES_128, 0, aoffset + 20);
            var af6 = a.getVector(FloatVector.SPECIES_128, 0, aoffset + 24);
            var af7 = a.getVector(FloatVector.SPECIES_128, 0, aoffset + 28);

            // Make 8 bytes -> 16 4bit -> 16 bytes -> 16 32F
            var bf0 = b.getVector(ByteVector.SPECIES_64, 0, boffset);
            var bf1 = b.getVector(ByteVector.SPECIES_64, 0, boffset + 16);

            // Convert the first 4 bits into bytes
            var low = bf0.lanewise(VectorOperators.AND, Q4_BYTE_MASK_64).sub(Q4_BYTE_SUB_64);
            var high = bf0.lanewise(VectorOperators.ASHR, Q4_BYTE_SHIFT_64)
                .lanewise(VectorOperators.AND, Q4_BYTE_MASK_64)
                .sub(Q4_BYTE_SUB_64);

            var low0 = low.castShape(ShortVector.SPECIES_128, 0);
            var lowf0 = low0.convertShape(VectorOperators.S2F, FloatVector.SPECIES_128, 0);
            var lowf1 = low0.convertShape(VectorOperators.S2F, FloatVector.SPECIES_128, 1);
            var high0 = high.castShape(ShortVector.SPECIES_128, 0);
            var highf0 = high0.convertShape(VectorOperators.S2F, FloatVector.SPECIES_128, 0);
            var highf1 = high0.convertShape(VectorOperators.S2F, FloatVector.SPECIES_128, 1);

            var nlow = bf1.lanewise(VectorOperators.AND, Q4_BYTE_MASK_64).sub(Q4_BYTE_SUB_64);
            var nhigh = bf1.lanewise(VectorOperators.ASHR, Q4_BYTE_SHIFT_64)
                .lanewise(VectorOperators.AND, Q4_BYTE_MASK_64)
                .sub(Q4_BYTE_SUB_64);

            var low2 = nlow.castShape(ShortVector.SPECIES_128, 0);
            var low2f0 = low2.convertShape(VectorOperators.S2F, FloatVector.SPECIES_128, 0);
            var low2f1 = low2.convertShape(VectorOperators.S2F, FloatVector.SPECIES_128, 1);

            var high2 = nhigh.castShape(ShortVector.SPECIES_128, 0);
            var high2f0 = high2.convertShape(VectorOperators.S2F, FloatVector.SPECIES_128, 0);
            var high2f1 = high2.convertShape(VectorOperators.S2F, FloatVector.SPECIES_128, 1);

            a.intoTensor(af0.add(lowf0.mul(scale)), 0, aoffset);
            a.intoTensor(af1.add(lowf1.mul(scale)), 0, aoffset + 4);
            a.intoTensor(af2.add(low2f0.mul(scale)), 0, aoffset + 8);
            a.intoTensor(af3.add(low2f1.mul(scale)), 0, aoffset + 12);
            a.intoTensor(af4.add(highf0.mul(scale)), 0, aoffset + 16);
            a.intoTensor(af5.add(highf1.mul(scale)), 0, aoffset + 20);
            a.intoTensor(af6.add(high2f0.mul(scale)), 0, aoffset + 24);
            a.intoTensor(af7.add(high2f1.mul(scale)), 0, aoffset + 28);
        }
    }

    void accumulateF32(FloatBufferTensor a, FloatBufferTensor b, int offset, int limit) {
        int upperBound = offset + FloatVector.SPECIES_PREFERRED.loopBound(limit);
        int i = offset;

        for (; i < upperBound; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector va = a.getVector(FloatVector.SPECIES_PREFERRED, 0, i);
            FloatVector vb = b.getVector(FloatVector.SPECIES_PREFERRED, 0, i);
            a.intoTensor(va.add(vb), 0, i);
        }

        // tail
        for (; i < offset + limit; i++) {
            a.set(a.get(0, i) + b.get(0, i), 0, i);
        }
    }

    void accumulateF32Q4_256(FloatBufferTensor a, Q4ByteBufferTensor b, int offset, int limit) {
        int aoffset = offset;
        int boffset = offset;
        int alim = offset + FloatVector.SPECIES_256.loopBound(limit);

        int slen = Q4ByteBufferTensor.BLOCK_SIZE;

        for (; aoffset < alim; aoffset += slen, boffset += slen) {
            FloatVector scale = FloatVector.broadcast(FloatVector.SPECIES_256, b.getFactorForIndex(0, boffset));

            // Make 8 bytes -> 16 4bit -> 16 bytes -> 16 32F
            var wBytes = b.getVector(ByteVector.SPECIES_128, 0, boffset);
            var loBytes = wBytes.and((byte) 0xF).sub((byte) 8);
            var hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4).sub((byte) 8);

            // BLOCK_SIZE Floats
            var af0 = a.getVector(FloatVector.SPECIES_256, 0, aoffset).add(loBytes.castShape(FloatVector.SPECIES_256, 0).mul(scale));
            var af1 = a.getVector(FloatVector.SPECIES_256, 0, aoffset + 8).add(loBytes.castShape(FloatVector.SPECIES_256, 1).mul(scale));
            var af2 = a.getVector(FloatVector.SPECIES_256, 0, aoffset + Q4ByteBufferTensor.HALF_BLOCK)
                .add(hiBytes.castShape(FloatVector.SPECIES_256, 0).mul(scale));
            var af3 = a.getVector(FloatVector.SPECIES_256, 0, aoffset + Q4ByteBufferTensor.HALF_BLOCK + 8)
                .add(hiBytes.castShape(FloatVector.SPECIES_256, 1).mul(scale));

            a.intoTensor(af0, 0, aoffset);
            a.intoTensor(af1, 0, aoffset + 8);
            a.intoTensor(af2, 0, aoffset + Q4ByteBufferTensor.HALF_BLOCK);
            a.intoTensor(af3, 0, aoffset + Q4ByteBufferTensor.HALF_BLOCK + 8);
        }
    }

    void accumulateF32BF16_256(FloatBufferTensor a, BFloat16BufferTensor b, int offset, int limit) {
        int upperBound = offset + FloatVector.SPECIES_256.loopBound(limit);

        int i = offset;
        for (; i < upperBound; i += FloatVector.SPECIES_256.length()) {

            // F32
            var af = a.getVector(FloatVector.SPECIES_256, 0, i);

            // Convert BF16 to F32
            var bf = b.getVector(ShortVector.SPECIES_128, 0, i)
                .convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0)
                .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_256)
                .reinterpretAsFloats();

            var res = af.add(bf);
            a.intoTensor(res, 0, i);
        }

        // tail
        for (; i < offset + limit; i++) {
            a.set(a.get(0, i) + b.get(0, i), 0, i);
        }
    }

    void accumulateF32BF16_arm(FloatBufferTensor a, BFloat16BufferTensor b, int offset, int limit) {
        int upperBound = offset + FloatVector.SPECIES_128.loopBound(limit);

        int i = offset;
        for (; i < upperBound; i += FloatVector.SPECIES_128.length()) {

            // F32
            var af = a.getVector(FloatVector.SPECIES_128, 0, i);

            // Convert BF16 to F32
            var bf = b.getVector(ShortVector.SPECIES_64, 0, i)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_128)
                    .reinterpretAsFloats();

            var res = af.add(bf);
            a.intoTensor(res, 0, i);
        }

        // tail
        for (; i < offset + limit; i++) {
            a.set(a.get(0, i) + b.get(0, i), 0, i);
        }
    }

    void accumulateBF16_arm(BFloat16BufferTensor a, BFloat16BufferTensor b, int offset, int limit) {
        int upperBound = offset + FloatVector.SPECIES_128.loopBound(limit);

        int i = offset;
        for (; i < upperBound; i += FloatVector.SPECIES_128.length()) {

            // Convert BF16 to F32
            var af = a.getVector(ShortVector.SPECIES_64, 0, i)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_128)
                    .reinterpretAsFloats();

            // Convert BF16 to F32
            var bf = b.getVector(ShortVector.SPECIES_64, 0, i)
                    .convertShape(VectorOperators.S2I, IntVector.SPECIES_128, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_128)
                    .reinterpretAsFloats();

            var res = af.add(bf)
                    .reinterpretAsInts()
                    .lanewise(VectorOperators.ASHR, BF16_BYTE_SHIFT_128)
                    .convertShape(VectorOperators.I2S, ShortVector.SPECIES_64, 0);

            a.intoTensor((ShortVector) res, 0, i);
        }

        // tail
        for (; i < offset + limit; i++) {
            a.set(a.get(0, i) + b.get(0, i), 0, i);
        }
    }

    void accumulateBF16_256(BFloat16BufferTensor a, BFloat16BufferTensor b, int offset, int limit) {
        int upperBound = offset + FloatVector.SPECIES_256.loopBound(limit);

        int i = offset;
        for (; i < upperBound; i += FloatVector.SPECIES_256.length()) {

            // Convert BF16 to F32
            var af = a.getVector(ShortVector.SPECIES_128, 0, i)
                .convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0)
                .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_256)
                .reinterpretAsFloats();

            // Convert BF16 to F32
            var bf = b.getVector(ShortVector.SPECIES_128, 0, i)
                .convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0)
                .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_256)
                .reinterpretAsFloats();

            var res = af.add(bf)
                .reinterpretAsInts()
                .lanewise(VectorOperators.ASHR, BF16_BYTE_SHIFT_256)
                .convertShape(VectorOperators.I2S, ShortVector.SPECIES_128, 0);

            a.intoTensor((ShortVector) res, 0, i);
        }

        // tail
        for (; i < offset + limit; i++) {
            a.set(a.get(0, i) + b.get(0, i), 0, i);
        }
    }

    void accumulateBF16_512(BFloat16BufferTensor a, BFloat16BufferTensor b, int offset, int limit) {
        int upperBound = offset + FloatVector.SPECIES_512.loopBound(limit);

        int i = offset;
        for (; i < upperBound; i += FloatVector.SPECIES_512.length()) {

            // Convert BF16 to F32
            var af = a.getVector(ShortVector.SPECIES_256, 0, i)
                .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_512)
                .reinterpretAsFloats();

            // Convert BF16 to F32
            var bf = b.getVector(ShortVector.SPECIES_256, 0, i)
                .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_512)
                .reinterpretAsFloats();

            var res = af.add(bf)
                .reinterpretAsInts()
                .lanewise(VectorOperators.ASHR, BF16_BYTE_SHIFT_512)
                .convertShape(VectorOperators.I2S, ShortVector.SPECIES_256, 0);

            a.intoTensor((ShortVector) res, 0, i);
        }

        // tail
        for (; i < offset + limit; i++) {
            a.set(a.get(0, i) + b.get(0, i), 0, i);
        }
    }

    @Override
    public void scale(float factor, AbstractTensor aBatch, int offset, int length) {

        for (int ai = 0; ai < aBatch.shape().first(); ai++) {
            AbstractTensor a = aBatch.slice(ai);
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
    }

    public void scaleF32(float factor, FloatBufferTensor a, int offset, int length) {
        int upperBound = FloatVector.SPECIES_PREFERRED.loopBound(length) + offset;
        int i = offset;

        FloatVector sf = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, factor);
        for (; i < upperBound; i += FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector va = a.getVector(FloatVector.SPECIES_PREFERRED, 0, i);
            a.intoTensor(va.mul(sf), 0, i);
        }

        // tail
        for (; i < (offset + length); i++) {
            a.set(a.get(0, i) * factor, 0, i);
        }
    }

    public void scaleBF16_512(float factor, BFloat16BufferTensor a, int offset, int length) {
        int upperBound = FloatVector.SPECIES_512.loopBound(length) + offset;
        int i = offset;

        FloatVector sf = FloatVector.broadcast(FloatVector.SPECIES_512, factor);
        for (; i < upperBound; i += FloatVector.SPECIES_512.length()) {
            var va = a.getVector(ShortVector.SPECIES_256, 0, i)
                .convertShape(VectorOperators.S2I, IntVector.SPECIES_512, 0)
                .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_512)
                .reinterpretAsFloats();

            var res = va.mul(sf)
                .reinterpretAsInts()
                .lanewise(VectorOperators.ASHR, BF16_BYTE_SHIFT_512)
                .convertShape(VectorOperators.I2S, ShortVector.SPECIES_256, 0);

            a.intoTensor((ShortVector) res, 0, i);
        }

        // tail
        for (; i < (offset + length); i++) {
            a.set(a.get(0, i) * factor, 0, i);
        }
    }

    public void scaleBF16_256(float factor, BFloat16BufferTensor a, int offset, int length) {
        int upperBound = FloatVector.SPECIES_256.loopBound(length) + offset;
        int i = offset;

        FloatVector sf = FloatVector.broadcast(FloatVector.SPECIES_256, factor);
        for (; i < upperBound; i += FloatVector.SPECIES_256.length()) {
            var va = a.getVector(ShortVector.SPECIES_128, 0, i)
                .convertShape(VectorOperators.S2I, IntVector.SPECIES_256, 0)
                .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT_256)
                .reinterpretAsFloats();

            var res = va.mul(sf)
                .reinterpretAsInts()
                .lanewise(VectorOperators.ASHR, BF16_BYTE_SHIFT_256)
                .convertShape(VectorOperators.I2S, ShortVector.SPECIES_128, 0);

            a.intoTensor((ShortVector) res, 0, i);
        }

        // tail
        for (; i < (offset + length); i++) {
            a.set(a.get(0, i) * factor, 0, i);
        }
    }

    @Override
    public void saxpy(float alpha, AbstractTensor x, AbstractTensor y, int xoffset, int yoffset, int limit) {
        Preconditions.checkArgument(y.shape().first() == 1);
        Preconditions.checkArgument(x.dType() == y.dType() || x.dType() == DType.BF16 && y.dType() == DType.F32);
        Preconditions.checkArgument(limit % 2 == 0);

        switch (x.dType()) {
            case F32:
                saxpyF32(alpha, (FloatBufferTensor) x, (FloatBufferTensor) y, xoffset, yoffset, limit);
                break;
            case BF16:
                switch (y.dType()) {
                    case F32:
                        saxpyBF16F32(alpha, (BFloat16BufferTensor) x, (FloatBufferTensor) y, xoffset, yoffset, limit);
                        break;
                    case BF16:
                        saxpyBF16(alpha, (BFloat16BufferTensor) x, (BFloat16BufferTensor) y, xoffset, yoffset, limit);
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
        FloatVector av = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, alpha);
        for (; xo < (xoffset + upperBound) && yo < (yoffset + upperBound); xo += FloatVector.SPECIES_PREFERRED.length(), yo +=
            FloatVector.SPECIES_PREFERRED.length()) {
            FloatVector vx = x.getVector(FloatVector.SPECIES_PREFERRED, 0, xo);
            FloatVector vy = y.getVector(FloatVector.SPECIES_PREFERRED, 0, yo);
            FloatVector res = vx.fma(av, vy);
            y.intoTensor(res, 0, yo);
        }

        // tail
        for (; xo < (xoffset + limit) && yo < (yoffset + limit); xo++, yo++) {
            float v = y.get(0, yo) + (alpha * x.get(0, xo));
            y.set(v, 0, yo);
        }
    }

    @Override
    public void saxpy(
        AbstractTensor alpha,
        AbstractTensor x,
        AbstractTensor y,
        int xoffset,
        int yoffset,
        int limit,
        int aOffset,
        int xOffset,
        int batchSize
    ) {
        Preconditions.checkArgument(limit % 2 == 0);

        switch (x.dType()) {
            case F32:
                saxpyF32(alpha, (FloatBufferTensor) x, (FloatBufferTensor) y, xoffset, yoffset, limit, aOffset, xOffset, batchSize);
                break;
            case BF16:
                switch (y.dType()) {
                    case F32:
                        saxpyBF16F32(alpha, x, y, xoffset, yoffset, limit, aOffset, xOffset, batchSize);
                        break;
                    case BF16:
                        saxpyBF16(alpha, x, y, xoffset, yoffset, limit, aOffset, xOffset, batchSize);
                        break;
                    default:
                        throw new UnsupportedOperationException();
                }
                break;
            default:
                throw new UnsupportedOperationException();
        }
    }

    public void saxpyF32(
        AbstractTensor alpha,
        FloatBufferTensor x,
        FloatBufferTensor y,
        int xoffset,
        int yoffset,
        int limit,
        int aOffset,
        int xOffset,
        int batchSize
    ) {
        int upperBound = FloatVector.SPECIES_PREFERRED.loopBound(limit);

        // Use Nearest multiple of 4
        int aLimit = batchSize - (batchSize % 4);
        int a = aOffset;
        int xi = xOffset;
        aLimit += aOffset;

        for (; a < aLimit; a += 4, xi += 4) {
            int xo = xoffset;
            int yo = yoffset;

            FloatVector a0 = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, alpha.get(0, a + 0));
            FloatVector a1 = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, alpha.get(0, a + 1));
            FloatVector a2 = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, alpha.get(0, a + 2));
            FloatVector a3 = FloatVector.broadcast(FloatVector.SPECIES_PREFERRED, alpha.get(0, a + 3));

            for (; xo < (xoffset + upperBound) && yo < (yoffset + upperBound); xo += FloatVector.SPECIES_PREFERRED.length(), yo +=
                FloatVector.SPECIES_PREFERRED.length()) {
                FloatVector x0 = x.getVector(FloatVector.SPECIES_PREFERRED, xi + 0, xo);
                FloatVector x1 = x.getVector(FloatVector.SPECIES_PREFERRED, xi + 1, xo);
                FloatVector x2 = x.getVector(FloatVector.SPECIES_PREFERRED, xi + 2, xo);
                FloatVector x3 = x.getVector(FloatVector.SPECIES_PREFERRED, xi + 3, xo);

                FloatVector vy = y.getVector(FloatVector.SPECIES_PREFERRED, 0, yo);

                FloatVector r0 = x0.fma(a0, vy);
                r0 = x1.fma(a1, r0);
                r0 = x2.fma(a2, r0);
                r0 = x3.fma(a3, r0);

                y.intoTensor(r0, 0, yo);
            }
        }

        // tail
        for (; a < aOffset + batchSize; a++, xi++) {
            saxpyF32(alpha.get(0, a), (FloatBufferTensor) x.slice(xi), y, xoffset, yoffset, limit);
        }
    }

    public void saxpyBF16(
        AbstractTensor alpha,
        AbstractTensor xt,
        AbstractTensor yt,
        int xoffset,
        int yoffset,
        int limit,
        int aOffset,
        int xOffset,
        int batchSize
    ) {

        BFloat16BufferTensor x = (BFloat16BufferTensor) xt;
        BFloat16BufferTensor y = (BFloat16BufferTensor) yt;

        int batchLimit = aOffset + batchSize;
        for (int a = aOffset, xi = xOffset; a < batchLimit; a++, xi++) {
            saxpyBF16(alpha.get(0, a), (BFloat16BufferTensor) x.slice(xi), y, xoffset, yoffset, limit);
        }
    }

    public void saxpyBF16F32(
        AbstractTensor alpha,
        AbstractTensor xt,
        AbstractTensor yt,
        int xoffset,
        int yoffset,
        int limit,
        int aOffset,
        int xOffset,
        int batchSize
    ) {

        BFloat16BufferTensor x = (BFloat16BufferTensor) xt;
        FloatBufferTensor y = (FloatBufferTensor) yt;

        int batchLimit = aOffset + batchSize;
        for (int a = aOffset, xi = xOffset; a < batchLimit; a++, xi++) {
            saxpyBF16F32(alpha.get(0, a), (BFloat16BufferTensor) x.slice(xi), y, xoffset, yoffset, limit);
        }
    }

    void saxpyBF16(float alpha, BFloat16BufferTensor a, BFloat16BufferTensor b, int aoffset, int boffset, int limit) {
        int upperBound = ShortVector.SPECIES_PREFERRED.loopBound(limit);
        Preconditions.checkArgument(upperBound == limit);

        int ao = aoffset;
        int bo = boffset;
        int len = ShortVector.SPECIES_PREFERRED.length();
        int half = ShortVector.SPECIES_PREFERRED.length() / 2;

        for (; ao < (aoffset + upperBound) && bo < (boffset + upperBound); ao += len, bo += len) {
            // Convert BF16 to F32
            var sa = a.getVector(ShortVector.SPECIES_PREFERRED, 0, ao);
            var af0 = sa.convertShape(VectorOperators.S2I, IntVector.SPECIES_PREFERRED, 0)
                .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                .reinterpretAsFloats();

            var af1 = sa.convertShape(VectorOperators.S2I, IntVector.SPECIES_PREFERRED, 1)
                .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                .reinterpretAsFloats();

            // Convert BF16 to F32
            var sb = b.getVector(ShortVector.SPECIES_PREFERRED, 0, bo);
            var bf0 = sb.convertShape(VectorOperators.S2I, IntVector.SPECIES_PREFERRED, 0)
                .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                .reinterpretAsFloats();

            var bf1 = sb.convertShape(VectorOperators.S2I, IntVector.SPECIES_PREFERRED, 1)
                .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                .reinterpretAsFloats();

            var r0 = bf0.add(af0.mul(alpha))
                .reinterpretAsInts()
                .lanewise(VectorOperators.ASHR, BF16_BYTE_SHIFT)
                .convertShape(VectorOperators.I2S, ShortVector.SPECIES_PREFERRED, 0);

            var r1 = bf1.add(af1.mul(alpha))
                .reinterpretAsInts()
                .lanewise(VectorOperators.ASHR, BF16_BYTE_SHIFT)
                .convertShape(VectorOperators.I2S, ShortVector.SPECIES_PREFERRED, -1);

            VectorMask<Short> mask = VectorMask.fromLong(ShortVector.SPECIES_PREFERRED, (1L << half) - 1);
            mask = mask.not(); // Invert the mask to select the second half

            var r = r0.blend(r1, mask);

            b.intoTensor((ShortVector) r, 0, bo);
        }
    }

    void saxpyBF16F32(float alpha, BFloat16BufferTensor a, FloatBufferTensor b, int aoffset, int boffset, int limit) {
        int upperBound = ShortVector.SPECIES_PREFERRED.loopBound(limit);
        Preconditions.checkArgument(upperBound == limit);

        int ao = aoffset;
        int bo = boffset;
        int len = ShortVector.SPECIES_PREFERRED.length();

        for (; ao < (aoffset + upperBound) && bo < (boffset + upperBound); ao += len, bo += len) {
            // Convert BF16 to F32
            var sa = a.getVector(ShortVector.SPECIES_PREFERRED, 0, ao);
            var af0 = sa.convertShape(VectorOperators.S2I, IntVector.SPECIES_PREFERRED, 0)
                .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                .reinterpretAsFloats();

            var af1 = sa.convertShape(VectorOperators.S2I, IntVector.SPECIES_PREFERRED, 1)
                .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                .reinterpretAsFloats();

            // Convert BF16 to F32
            var bf0 = b.getVector(FloatVector.SPECIES_PREFERRED, 0, bo);
            var bf1 = b.getVector(FloatVector.SPECIES_PREFERRED, 0, bo + FloatVector.SPECIES_PREFERRED.length());

            var r0 = bf0.add(af0.mul(alpha));
            var r1 = bf1.add(af1.mul(alpha));

            b.intoTensor(r0, 0, bo);
            b.intoTensor(r1, 0, bo + FloatVector.SPECIES_PREFERRED.length());
        }
    }
}
