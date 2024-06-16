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

import static java.lang.foreign.ValueLayout.*;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

public class NativeSimd {

    public static final OfByte C_CHAR = JAVA_BYTE;
    public static final OfShort C_SHORT = JAVA_SHORT;
    public static final OfInt C_INT = JAVA_INT;
    public static final OfLong C_LONG = JAVA_LONG;
    public static final OfLong C_LONG_LONG = JAVA_LONG;
    public static final OfFloat C_FLOAT = JAVA_FLOAT;
    public static final OfDouble C_DOUBLE = JAVA_DOUBLE;
    public static final AddressLayout C_POINTER = RuntimeHelper.POINTER;
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

    public static MethodHandle gemm_q8_q4$MH() {
        return RuntimeHelper.requireNonNull(constants$0.const$1, "gemm_q8_q4");
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
            int ldc) {
        var mh$ = gemm_q8_q4$MH();
        try {
            mh$.invokeExact(flags, af, a, aoffset, bf, b, boffset, r, roffset, m, n0, n, k, lda, ldaf, ldb, ldbf, ldc);
        } catch (Throwable ex$) {
            throw new AssertionError("should not reach here", ex$);
        }
    }

    public static MethodHandle gemm_q8_q4_batch$MH() {
        return RuntimeHelper.requireNonNull(constants$0.const$3, "gemm_q8_q4_batch");
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
            int ldc) {
        var mh$ = gemm_q8_q4_batch$MH();
        try {
            mh$.invokeExact(
                    flags, batch_num, af, a, aoffset, bf, b, boffset, r, roffset, m, n0, n, k, lda, ldaf, ldb, ldbf,
                    ldc);
        } catch (Throwable ex$) {
            throw new AssertionError("should not reach here", ex$);
        }
    }

    public static MethodHandle gemm_f32$MH() {
        return RuntimeHelper.requireNonNull(constants$0.const$5, "gemm_f32");
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
            int ldc) {
        var mh$ = gemm_f32$MH();
        try {
            mh$.invokeExact(flags, a, aoffset, b, boffset, r, roffset, m, n0, n, k, lda, ldb, ldc);
        } catch (Throwable ex$) {
            throw new AssertionError("should not reach here", ex$);
        }
    }

    public static MethodHandle gemm_f32_batch$MH() {
        return RuntimeHelper.requireNonNull(constants$1.const$1, "gemm_f32_batch");
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
            int ldc) {
        var mh$ = gemm_f32_batch$MH();
        try {
            mh$.invokeExact(flags, batch_num, a, aoffset, b, boffset, r, roffset, m, n0, n, k, lda, ldb, ldc);
        } catch (Throwable ex$) {
            throw new AssertionError("should not reach here", ex$);
        }
    }

    public static MethodHandle gemm_f32_q4$MH() {
        return RuntimeHelper.requireNonNull(constants$1.const$3, "gemm_f32_q4");
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
            int ldc) {
        var mh$ = gemm_f32_q4$MH();
        try {
            mh$.invokeExact(flags, a, aoffset, bf, b, boffset, r, roffset, m, n0, n, k, lda, ldb, ldbf, ldc);
        } catch (Throwable ex$) {
            throw new AssertionError("should not reach here", ex$);
        }
    }

    public static MethodHandle gemm_f32_q4_batch$MH() {
        return RuntimeHelper.requireNonNull(constants$1.const$5, "gemm_f32_q4_batch");
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
            int ldc) {
        var mh$ = gemm_f32_q4_batch$MH();
        try {
            mh$.invokeExact(flags, batch_num, a, aoffset, bf, b, boffset, r, roffset, m, n0, n, k, lda, ldb, ldbf, ldc);
        } catch (Throwable ex$) {
            throw new AssertionError("should not reach here", ex$);
        }
    }
}
