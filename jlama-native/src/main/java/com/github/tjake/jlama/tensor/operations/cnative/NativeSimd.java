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
package com.github.tjake.jlama.tensor.operations.cnative;

import java.lang.foreign.*;

public class NativeSimd {

    /**
     * {@snippet :
     * #define HAS_F16C 2
     * }
     */
    public static int HAS_F16C() {
        return (int) 2L;
    }

    /**
     * {@snippet :
     * #define HAS_AVX2 4
     * }
     */
    public static int HAS_AVX2() {
        return (int) 4L;
    }

    /**
     * {@snippet :
     * #define IS_M_SERIES_MAC 8
     * }
     */
    public static int IS_M_SERIES_MAC() {
        return (int) 8L;
    }

    /**
     * {@snippet :
     * #define Q8_BLOCK_SIZE 32
     * }
     */
    public static int Q8_BLOCK_SIZE() {
        return (int) 32L;
    }

    /**
     * {@snippet :
     * #define Q4_BLOCK_SIZE 32
     * }
     */
    public static int Q4_BLOCK_SIZE() {
        return (int) 32L;
    }

    /**
     * {@snippet :
     * void gemm_q8_q4(int flags, float* af, char* a, int aoffset, float* bf, char* b, int boffset, float* r, int roffset, int m, int n0, int n, int k, int lda, int ldaf, int ldb, int ldbf, int ldc);
     * }
     */
    public static void gemm_q8_q4(
        int flags,
        MemorySegment af,
        MemorySegment a,
        int aoffset,
        MemorySegment bf,
        MemorySegment b,
        int boffset,
        MemorySegment r,
        int roffset,
        int m,
        int n0,
        int n,
        int k,
        int lda,
        int ldaf,
        int ldb,
        int ldbf,
        int ldc
    ) {
        throw new UnsupportedOperationException("Not implemented for this JDK version");
    }

    /**
     * {@snippet :
     * void gemm_q8_q4_batch(int flags, int batch_num, float* af, char* a, int aoffset, float** bf, char** b, int boffset, float** r, int roffset, int m, int n0, int n, int k, int lda, int ldaf, int ldb, int ldbf, int ldc);
     * }
     */
    public static void gemm_q8_q4_batch(
        int flags,
        int batch_num,
        MemorySegment af,
        MemorySegment a,
        int aoffset,
        MemorySegment bf,
        MemorySegment b,
        int boffset,
        MemorySegment r,
        int roffset,
        int m,
        int n0,
        int n,
        int k,
        int lda,
        int ldaf,
        int ldb,
        int ldbf,
        int ldc
    ) {
        throw new UnsupportedOperationException("Not implemented for this JDK version");
    }

    /**
     * {@snippet :
     * void gemm_f32(int flags, float* a, int aoffset, float* b, int boffset, float* r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc);
     * }
     */
    public static void gemm_f32(
        int flags,
        MemorySegment a,
        int aoffset,
        MemorySegment b,
        int boffset,
        MemorySegment r,
        int roffset,
        int m,
        int n0,
        int n,
        int k,
        int lda,
        int ldb,
        int ldc
    ) {
        throw new UnsupportedOperationException("Not implemented for this JDK version");
    }

    /**
     * {@snippet :
     * void gemm_f32_batch(int flags, int batch_num, float* a, int aoffset, float** b, int boffset, float** r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc);
     * }
     */
    public static void gemm_f32_batch(
        int flags,
        int batch_num,
        MemorySegment a,
        int aoffset,
        MemorySegment b,
        int boffset,
        MemorySegment r,
        int roffset,
        int m,
        int n0,
        int n,
        int k,
        int lda,
        int ldb,
        int ldc
    ) {
        throw new UnsupportedOperationException("Not implemented for this JDK version");
    }

    /**
     * {@snippet :
     * void gemm_f32_q4(int flags, float* a, int aoffset, float* bf, char* b, int boffset, float* r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldbf, int ldc);
     * }
     */
    public static void gemm_f32_q4(
        int flags,
        MemorySegment a,
        int aoffset,
        MemorySegment bf,
        MemorySegment b,
        int boffset,
        MemorySegment r,
        int roffset,
        int m,
        int n0,
        int n,
        int k,
        int lda,
        int ldb,
        int ldbf,
        int ldc
    ) {
        throw new UnsupportedOperationException("Not implemented for this JDK version");
    }

    /**
     * {@snippet :
     * void gemm_f32_q4_batch(int flags, int batch_num, float* a, int aoffset, float** bf, char** b, int boffset, float** r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldbf, int ldc);
     * }
     */
    public static void gemm_f32_q4_batch(
        int flags,
        int batch_num,
        MemorySegment a,
        int aoffset,
        MemorySegment bf,
        MemorySegment b,
        int boffset,
        MemorySegment r,
        int roffset,
        int m,
        int n0,
        int n,
        int k,
        int lda,
        int ldb,
        int ldbf,
        int ldc
    ) {
        throw new UnsupportedOperationException("Not implemented for this JDK version");
    }

    /**
     * {@snippet :
     * void gemm_bf16(int flags, short* a, int aoffset, short* b, int boffset, short* cr, float* r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc);
     * }
     */
    public static void gemm_bf16(
        int flags,
        MemorySegment a,
        int aoffset,
        MemorySegment b,
        int boffset,
        MemorySegment cr,
        MemorySegment r,
        int roffset,
        int m,
        int n0,
        int n,
        int k,
        int lda,
        int ldb,
        int ldc
    ) {
        throw new UnsupportedOperationException("Not implemented for this JDK version");
    }

    /**
     * {@snippet :
     * void gemm_bf16_batch(int flags, int batch_num, short* a, int aoffset, short** b, int boffset, short** cr, float** r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc);
     * }
     */
    public static void gemm_bf16_batch(
        int flags,
        int batch_num,
        MemorySegment a,
        int aoffset,
        MemorySegment b,
        int boffset,
        MemorySegment cr,
        MemorySegment r,
        int roffset,
        int m,
        int n0,
        int n,
        int k,
        int lda,
        int ldb,
        int ldc
    ) {
        throw new UnsupportedOperationException("Not implemented for this JDK version");
    }

    /**
     * {@snippet :
     * void gemm_f32_bf16(int flags, float* a, int aoffset, short* b, int boffset, short* cr, float* r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc);
     * }
     */
    public static void gemm_f32_bf16(
        int flags,
        MemorySegment a,
        int aoffset,
        MemorySegment b,
        int boffset,
        MemorySegment cr,
        MemorySegment r,
        int roffset,
        int m,
        int n0,
        int n,
        int k,
        int lda,
        int ldb,
        int ldc
    ) {
        throw new UnsupportedOperationException("Not implemented for this JDK version");
    }

    /**
     * {@snippet :
     * void gemm_f32_bf16_batch(int flags, int batch_num, float* a, int aoffset, short** b, int boffset, short** cr, float** r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc);
     * }
     */
    public static void gemm_f32_bf16_batch(
        int flags,
        int batch_num,
        MemorySegment a,
        int aoffset,
        MemorySegment b,
        int boffset,
        MemorySegment cr,
        MemorySegment r,
        int roffset,
        int m,
        int n0,
        int n,
        int k,
        int lda,
        int ldb,
        int ldc
    ) {
        throw new UnsupportedOperationException("Not implemented for this JDK version");
    }
}
