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
        NativeSimd.gemm_f32$MH();
    }

    @Override
    public boolean requiresOffHeapTensor() {
        return true;
    }

    public int parallelSplitSize() {
        return 128;
    }

    @Override
    public void batchDotProduct(
            AbstractTensor result,
            AbstractTensor at,
            AbstractTensor bt,
            int aColumnOffset,
            int bColumnOffset,
            int columnLength,
            int bRowOffset,
            int rowChunkSize) {

        int M = at.shape().dim(0);
        int N = rowChunkSize; // b.shape().dim(0);
        int K = columnLength; // a.shape().dim(1);

        switch (at.dType()) {
            case F32:
                switch (bt.dType()) {
                    case F32:
                        NativeSimd.gemm_f32(
                                flags,
                                at.getMemorySegment(),
                                at.getOffset(0, aColumnOffset),
                                bt.getMemorySegment(),
                                bt.getOffset(0, bColumnOffset),
                                result.getMemorySegment(),
                                result.shape().sparseOffset(),
                                M,
                                bRowOffset,
                                N,
                                K,
                                at.getStride(),
                                bt.getStride(),
                                result.getStride());
                        break;
                    case Q4:
                        switch (MachineSpec.VECTOR_TYPE) {
                            case ARM_128:
                                throw new UnsupportedOperationException("F32 Q4 Unsupported on Arm");
                            default:
                                Q4ByteBufferTensor b = (Q4ByteBufferTensor) bt;
                                NativeSimd.gemm_f32_q4(
                                        flags,
                                        at.getMemorySegment(),
                                        at.getOffset(0, aColumnOffset),
                                        b.getBlockF().getMemorySegment(),
                                        b.getMemorySegment(),
                                        b.getMemorySegmentOffset(b.getOffset(0, bColumnOffset)),
                                        result.getMemorySegment(),
                                        result.shape().sparseOffset(),
                                        M,
                                        bRowOffset,
                                        N,
                                        K,
                                        at.getStride(),
                                        b.getMemorySegmentOffset(b.getStride()),
                                        b.getBlockF().getStride(),
                                        result.getStride());
                        }
                        break;
                    default:
                        throw new UnsupportedOperationException(
                                at.dType().name() + " " + bt.dType().name());
                }
                break;
            case I8:
                switch (bt.dType()) {
                    case Q4:
                        Q8ByteBufferTensor a = (Q8ByteBufferTensor) at;
                        Q4ByteBufferTensor b = (Q4ByteBufferTensor) bt;
                        NativeSimd.gemm_q8_q4(
                                flags,
                                a.getBlockF().getMemorySegment(),
                                a.getMemorySegment(),
                                a.getOffset(0, aColumnOffset),
                                b.getBlockF().getMemorySegment(),
                                b.getMemorySegment(),
                                b.getMemorySegmentOffset(b.getOffset(0, bColumnOffset)),
                                result.getMemorySegment(),
                                result.shape().sparseOffset(),
                                M,
                                bRowOffset,
                                N,
                                K,
                                a.getStride(),
                                a.getBlockF().getStride(),
                                b.getMemorySegmentOffset(b.getStride()),
                                b.getBlockF().getStride(),
                                result.getStride());
                        break;
                    default:
                        throw new UnsupportedOperationException(
                                at.dType().name() + " " + bt.dType().name());
                }
                break;
            default:
                throw new UnsupportedOperationException(at.dType().name());
        }
    }

    @Override
    public void dotProductBatchChunk(
            AbstractTensor[] r,
            AbstractTensor a,
            AbstractTensor[] b,
            int columnOffset,
            int columnLength,
            int bRowOffset,
            int rowChunkSize) {

        MemorySegment[] tmp = tmpArr.get();
        MemorySegment ra = tmp[0];
        MemorySegment rb = tmp[1];
        MemorySegment rc = tmp[2];

        for (int i = 0; i < r.length; i++) {
            ra.setAtIndex(ValueLayout.ADDRESS, i, r[i].getMemorySegment());
            rb.setAtIndex(ValueLayout.ADDRESS, i, b[i].getMemorySegment());
        }

        int M = a.shape().dim(0);
        int N = rowChunkSize; // b.shape().dim(0);
        int K = columnLength; // a.shape().dim(1);

        switch (a.dType()) {
            case F32:
                switch (b[0].dType()) {
                    case F32:
                        NativeSimd.gemm_f32_batch(
                                flags,
                                r.length,
                                a.getMemorySegment(),
                                a.getOffset(0, columnOffset),
                                rb,
                                b[0].getOffset(0, columnOffset),
                                ra,
                                r[0].shape().sparseOffset(),
                                M,
                                bRowOffset,
                                N,
                                K,
                                a.getStride(),
                                b[0].getStride(),
                                r[0].getStride());
                        break;
                    case Q4:
                        switch (MachineSpec.VECTOR_TYPE) {
                            case ARM_128:
                                throw new UnsupportedOperationException("F32 Q4 Unsupported on Arm");
                            default:
                                Q4ByteBufferTensor bt = (Q4ByteBufferTensor) b[0];
                                for (int i = 0; i < r.length; i++)
                                    rc.setAtIndex(
                                            ValueLayout.ADDRESS,
                                            i,
                                            ((Q4ByteBufferTensor) b[i])
                                                    .getBlockF()
                                                    .getMemorySegment());
                                NativeSimd.gemm_f32_q4_batch(
                                        flags,
                                        r.length,
                                        a.getMemorySegment(),
                                        a.getOffset(0, columnOffset),
                                        rc,
                                        rb,
                                        b[0].getMemorySegmentOffset(b[0].getOffset(0, columnOffset)),
                                        ra,
                                        r[0].shape().sparseOffset(),
                                        M,
                                        bRowOffset,
                                        N,
                                        K,
                                        a.getStride(),
                                        b[0].getMemorySegmentOffset(b[0].getStride()),
                                        bt.getBlockF().getStride(),
                                        r[0].getStride());
                        }
                        break;
                    default:
                        throw new UnsupportedOperationException(
                                a.dType().name() + " " + b[0].dType().name());
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

                        Q8ByteBufferTensor at = (Q8ByteBufferTensor) a;
                        Q4ByteBufferTensor bt = (Q4ByteBufferTensor) b[0];
                        NativeSimd.gemm_q8_q4_batch(
                                flags,
                                r.length,
                                at.getBlockF().getMemorySegment(),
                                a.getMemorySegment(),
                                a.getOffset(0, columnOffset),
                                rc,
                                rb,
                                bt.getMemorySegmentOffset(bt.getOffset(0, columnOffset)),
                                ra,
                                r[0].shape().sparseOffset(),
                                M,
                                bRowOffset,
                                N,
                                K,
                                a.getStride(),
                                at.getBlockF().getStride(),
                                bt.getMemorySegmentOffset(bt.getStride()),
                                bt.getBlockF().getStride(),
                                r[0].getStride());
                        break;
                    default:
                        throw new UnsupportedOperationException(
                                a.dType().name() + " " + b[0].dType().name());
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
    public void saxpy(
            AbstractTensor alpha,
            AbstractTensor x,
            AbstractTensor y,
            int xoffset,
            int yoffset,
            int limit,
            int batchSize) {
        delegate.saxpy(alpha, x, y, xoffset, yoffset, limit, batchSize);
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
