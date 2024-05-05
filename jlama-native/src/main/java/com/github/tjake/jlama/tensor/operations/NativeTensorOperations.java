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
import com.github.tjake.jlama.tensor.Q4ByteBufferTensor;
import com.github.tjake.jlama.tensor.Q8ByteBufferTensor;
import com.github.tjake.jlama.tensor.operations.cnative.NativeSimd;
import com.github.tjake.jlama.util.MachineSpec;
import com.github.tjake.jlama.util.PhysicalCoreExecutor;
import com.github.tjake.jlama.util.RuntimeSupport;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class NativeTensorOperations implements TensorOperations {

    private static final int MAX_BATCH_SIZE = 4;
    private static final ThreadLocal<MemorySegment[]> tmpArr = ThreadLocal.withInitial(() -> new MemorySegment[] {
        Arena.global().allocateArray(ValueLayout.ADDRESS, MAX_BATCH_SIZE),
        Arena.global().allocateArray(ValueLayout.ADDRESS, MAX_BATCH_SIZE),
        Arena.global().allocateArray(ValueLayout.ADDRESS, MAX_BATCH_SIZE),
    });

    private static final Logger logger = LoggerFactory.getLogger(NativeTensorOperations.class);
    public static final int HAS_F16C = NativeSimd.HAS_F16C();
    public static final int HAS_AVX2 = NativeSimd.HAS_AVX2();

    private static final TensorOperations delegate;

    static {
        TensorOperations tmp;
        try {
            tmp = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE);
        } catch (Throwable t) {
            tmp = new NaiveTensorOperations();
        }
        delegate = tmp;
    }

    final int flags;

    public NativeTensorOperations() {
        int f = 0;

        if (RuntimeSupport.isLinux()) f |= HAS_F16C;

        if (MachineSpec.VECTOR_TYPE == MachineSpec.Type.AVX_512) f |= HAS_AVX2;

        this.flags = f;
        checkLib();
    }

    NativeTensorOperations(int flags) {
        this.flags = flags;
    }

    @Override
    public String name() {
        return "Native SIMD Operations";
    }

    private void checkLib() {
        NativeSimd.dot_product_f32$MH();
    }

    @Override
    public boolean requiresOffHeapTensor() {
        return true;
    }

    public int parallelSplitSize() {
        return PhysicalCoreExecutor.instance.get().getCoreCount();
    }

    @Override
    public float dotProduct(AbstractTensor a, AbstractTensor b, int aoffset, int boffset, int limit) {
        aoffset = a.getOffset(aoffset);
        boffset = b.getOffset(boffset);

        return switch (a.dType()) {
            case F32 -> switch (b.dType()) {
                case F32 -> NativeSimd.dot_product_f32(
                        flags, a.getMemorySegment(), aoffset, b.getMemorySegment(), boffset, limit);
                case I8 -> NativeSimd.dot_product_f32_q8(
                        flags,
                        a.getMemorySegment(),
                        aoffset,
                        ((Q8ByteBufferTensor) b).getBlockF().getMemorySegment(),
                        b.getMemorySegment(),
                        boffset,
                        limit);
                case Q4 -> NativeSimd.dot_product_f32_q4(
                        flags,
                        a.getMemorySegment(),
                        aoffset,
                        ((Q4ByteBufferTensor) b).getBlockF().getMemorySegment(),
                        b.getMemorySegment(),
                        boffset,
                        limit);
                default -> throw new UnsupportedOperationException(b.dType().name());
            };
            case I8 -> switch (b.dType()) {
                case Q4 -> NativeSimd.dot_product_q8_q4(
                        flags,
                        ((Q8ByteBufferTensor) a).getBlockF().getMemorySegment(),
                        a.getMemorySegment(),
                        aoffset,
                        ((Q4ByteBufferTensor) b).getBlockF().getMemorySegment(),
                        b.getMemorySegment(),
                        boffset,
                        limit);
                    // case I8 -> NativeSimd.dot_product_q8(flags,
                    // ((Q8ByteBufferTensor)a).getBlockF().getMemorySegment(), a.getMemorySegment(), aoffset,
                    // ((Q8ByteBufferTensor)b).getBlockF().getMemorySegment(), b.getMemorySegment(), boffset, limit);
                default -> throw new UnsupportedOperationException(b.dType().name());
            };
            default -> throw new UnsupportedOperationException(a.dType().name());
        };
    }

    @Override
    public void batchDotProduct(AbstractTensor result, AbstractTensor a, AbstractTensor b, int aColumnOffset, int bColumnOffset, int columnLength, int bRowOffset, int rowChunkSize) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public void dotProductChunk(
            AbstractTensor r,
            AbstractTensor a,
            AbstractTensor b,
            int offset,
            int limit,
            int chunkStart,
            int chunkSize) {
        int aoffset = a.getOffset(offset);
        int boffset = b.getOffset(0, offset);
        int roffset = r.getOffset(chunkStart);

        switch (a.dType()) {
            case F32:
                switch (b.dType()) {
                    case F32:
                        NativeSimd.dot_product_f32_chunked(
                                flags,
                                r.getMemorySegment(),
                                roffset,
                                a.getMemorySegment(),
                                aoffset,
                                b.getMemorySegment(),
                                boffset,
                                limit,
                                chunkStart,
                                chunkSize);
                        break;
                    case I8:
                        NativeSimd.dot_product_f32_q8_chunked(
                                flags,
                                r.getMemorySegment(),
                                roffset,
                                a.getMemorySegment(),
                                aoffset,
                                ((Q8ByteBufferTensor) b).getBlockF().getMemorySegment(),
                                b.getMemorySegment(),
                                boffset,
                                limit,
                                chunkStart,
                                chunkSize);
                        break;
                    case Q4:
                        NativeSimd.dot_product_f32_q4_chunked(
                                flags,
                                r.getMemorySegment(),
                                roffset,
                                a.getMemorySegment(),
                                aoffset,
                                ((Q4ByteBufferTensor) b).getBlockF().getMemorySegment(),
                                b.getMemorySegment(),
                                boffset,
                                limit,
                                chunkStart,
                                chunkSize);
                        break;
                    default:
                        throw new UnsupportedOperationException(b.dType().name());
                }
                break;
            case I8:
                switch (b.dType()) {
                    case Q4:
                        NativeSimd.dot_product_q8_q4_chunked(
                                flags,
                                r.getMemorySegment(),
                                roffset,
                                ((Q8ByteBufferTensor) a).getBlockF().getMemorySegment(),
                                a.getMemorySegment(),
                                aoffset,
                                ((Q4ByteBufferTensor) b).getBlockF().getMemorySegment(),
                                b.getMemorySegment(),
                                boffset,
                                limit,
                                chunkStart,
                                chunkSize);
                        break;
                    default:
                        throw new UnsupportedOperationException(b.dType().name());
                }
                break;
            default:
                throw new UnsupportedOperationException(a.dType().name());
        }
    }

    @Override
    public void dotProductBatchChunk(
            AbstractTensor[] r,
            AbstractTensor a,
            AbstractTensor[] b,
            int offset,
            int limit,
            int chunkStart,
            int chunkSize) {
        int aoffset = a.getOffset(offset);
        int boffset = b[0].getOffset(0, offset);
        int roffset = r[0].getOffset(chunkStart);

        MemorySegment[] tmp = tmpArr.get();
        MemorySegment ra = tmp[0];
        MemorySegment rb = tmp[1];
        MemorySegment rc = tmp[2];

        for (int i = 0; i < r.length; i++) {
            ra.setAtIndex(ValueLayout.ADDRESS, i, r[i].getMemorySegment());
            rb.setAtIndex(ValueLayout.ADDRESS, i, b[i].getMemorySegment());
        }

        switch (a.dType()) {
            case F32:
                switch (b[0].dType()) {
                    case F32:
                        NativeSimd.dot_product_f32_batch_chunked(
                                flags,
                                r.length,
                                ra,
                                roffset,
                                a.getMemorySegment(),
                                aoffset,
                                rb,
                                boffset,
                                limit,
                                chunkStart,
                                chunkSize);
                        break;
                    case I8:
                        for (int i = 0; i < r.length; i++)
                            rc.setAtIndex(
                                    ValueLayout.ADDRESS,
                                    i,
                                    ((Q8ByteBufferTensor) b[i]).getBlockF().getMemorySegment());
                        NativeSimd.dot_product_f32_q8_batch_chunked(
                                flags,
                                r.length,
                                ra,
                                roffset,
                                a.getMemorySegment(),
                                aoffset,
                                rc,
                                rb,
                                boffset,
                                limit,
                                chunkStart,
                                chunkSize);
                        break;
                    case Q4:
                        for (int i = 0; i < r.length; i++)
                            rc.setAtIndex(
                                    ValueLayout.ADDRESS,
                                    i,
                                    ((Q4ByteBufferTensor) b[i]).getBlockF().getMemorySegment());
                        NativeSimd.dot_product_f32_q4_batch_chunked(
                                flags,
                                r.length,
                                ra,
                                roffset,
                                a.getMemorySegment(),
                                aoffset,
                                rc,
                                rb,
                                boffset,
                                limit,
                                chunkStart,
                                chunkSize);
                        break;
                    default:
                        throw new UnsupportedOperationException(b[0].dType().name());
                }
                break;
            case I8:
                switch (b[0].dType()) {
                    case Q4:
                        for (int i = 0; i < r.length; i++)
                            rc.setAtIndex(
                                    ValueLayout.ADDRESS,
                                    i,
                                    ((Q4ByteBufferTensor) b[i]).getBlockF().getMemorySegment());
                        NativeSimd.dot_product_q8_q4_batch_chunked(
                                flags,
                                r.length,
                                ra,
                                roffset,
                                ((Q8ByteBufferTensor) a).getBlockF().getMemorySegment(),
                                a.getMemorySegment(),
                                aoffset,
                                rc,
                                rb,
                                boffset,
                                limit,
                                chunkStart,
                                chunkSize);
                        break;
                    default:
                        throw new UnsupportedOperationException(b[0].dType().name());
                }
                break;
            default:
                throw new UnsupportedOperationException(a.dType().name());
        }
    }

    @Override
    public void accumulate(AbstractTensor a, AbstractTensor b, int offset, int length) {
        delegate.accumulate(a, b, offset, length);
    }

    @Override
    public void maccumulate(AbstractTensor a, AbstractTensor b, int offset, int length) {
        delegate.maccumulate(a, b, offset, length);
    }

    @Override
    public void saxpy(float alpha, AbstractTensor x, AbstractTensor y, int xoffset, int yoffset, int limit) {
        delegate.saxpy(alpha, x, y, xoffset, yoffset, limit);
    }

    @Override
    public void sxpby(float beta, AbstractTensor x, AbstractTensor y, int xoffset, int yoffset, int limit) {
        delegate.sxpby(beta, x, y, xoffset, yoffset, limit);
    }

    @Override
    public void scale(float factor, AbstractTensor x, int offset, int length) {
        delegate.scale(factor, x, offset, length);
    }

    @Override
    public AbstractTensor quantize(AbstractTensor t, DType qtype, int offset, int length) {
        return delegate.quantize(t, qtype, offset, length);
    }
}
