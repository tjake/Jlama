// Generated by jextract

package com.github.tjake.jlama.tensor.operations.gpunative;

import java.lang.invoke.MethodHandle;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;
import java.lang.foreign.*;
import static java.lang.foreign.ValueLayout.*;
public class NativeGPU  {

    public static final OfByte C_CHAR = Constants$root.C_CHAR$LAYOUT;
    public static final OfShort C_SHORT = Constants$root.C_SHORT$LAYOUT;
    public static final OfInt C_INT = Constants$root.C_INT$LAYOUT;
    public static final OfLong C_LONG = Constants$root.C_LONG_LONG$LAYOUT;
    public static final OfLong C_LONG_LONG = Constants$root.C_LONG_LONG$LAYOUT;
    public static final OfFloat C_FLOAT = Constants$root.C_FLOAT$LAYOUT;
    public static final OfDouble C_DOUBLE = Constants$root.C_DOUBLE$LAYOUT;
    public static final OfAddress C_POINTER = Constants$root.C_POINTER$LAYOUT;
    public static MethodHandle init_gpu$MH() {
        return RuntimeHelper.requireNonNull(constants$0.init_gpu$MH,"init_gpu");
    }
    /**
     * {@snippet :
     * void init_gpu(long* results);
     * }
     */
    public static void init_gpu(MemorySegment results) {
        var mh$ = init_gpu$MH();
        try {
            mh$.invokeExact(results);
        } catch (Throwable ex$) {
            throw new AssertionError("should not reach here", ex$);
        }
    }
    public static MethodHandle register_tensor$MH() {
        return RuntimeHelper.requireNonNull(constants$0.register_tensor$MH,"register_tensor");
    }
    /**
     * {@snippet :
     * long register_tensor(char* data, int size);
     * }
     */
    public static long register_tensor(MemorySegment data, int size) {
        var mh$ = register_tensor$MH();
        try {
            return (long)mh$.invokeExact(data, size);
        } catch (Throwable ex$) {
            throw new AssertionError("should not reach here", ex$);
        }
    }
    public static MethodHandle register_scratch_buffers$MH() {
        return RuntimeHelper.requireNonNull(constants$0.register_scratch_buffers$MH,"register_scratch_buffers");
    }
    /**
     * {@snippet :
     * long register_scratch_buffers(int params_size, int input_size, int result_size);
     * }
     */
    public static long register_scratch_buffers(int params_size, int input_size, int result_size) {
        var mh$ = register_scratch_buffers$MH();
        try {
            return (long)mh$.invokeExact(params_size, input_size, result_size);
        } catch (Throwable ex$) {
            throw new AssertionError("should not reach here", ex$);
        }
    }
    public static MethodHandle register_shader$MH() {
        return RuntimeHelper.requireNonNull(constants$0.register_shader$MH,"register_shader");
    }
    /**
     * {@snippet :
     * long register_shader(char* data, int size);
     * }
     */
    public static long register_shader(MemorySegment data, int size) {
        var mh$ = register_shader$MH();
        try {
            return (long)mh$.invokeExact(data, size);
        } catch (Throwable ex$) {
            throw new AssertionError("should not reach here", ex$);
        }
    }
    public static MethodHandle gpu_gemm$MH() {
        return RuntimeHelper.requireNonNull(constants$0.gpu_gemm$MH,"gpu_gemm");
    }
    /**
     * {@snippet :
     * void gpu_gemm(long scratch_id, long shader, void* a, void* a2, int aoffset, int alimit, long bid, long bid2, int boffset, int blimit, float* r, int roffset, int rlimit, int m, int n0, int n, int k, int lda, int ldb, int ldc, int m1_optimized);
     * }
     */
    public static void gpu_gemm(long scratch_id, long shader, MemorySegment a, MemorySegment a2, int aoffset, int alimit, long bid, long bid2, int boffset, int blimit, MemorySegment r, int roffset, int rlimit, int m, int n0, int n, int k, int lda, int ldb, int ldc, int m1_optimized) {
        var mh$ = gpu_gemm$MH();
        try {
            mh$.invokeExact(scratch_id, shader, a, a2, aoffset, alimit, bid, bid2, boffset, blimit, r, roffset, rlimit, m, n0, n, k, lda, ldb, ldc, m1_optimized);
        } catch (Throwable ex$) {
            throw new AssertionError("should not reach here", ex$);
        }
    }
    public static MethodHandle gpu_gemm_batch$MH() {
        return RuntimeHelper.requireNonNull(constants$0.gpu_gemm_batch$MH,"gpu_gemm_batch");
    }
    /**
     * {@snippet :
     * void gpu_gemm_batch(long shader, int batch_num, void* a, void* a2, int aoffset, long* bid, int boffset, float** r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc);
     * }
     */
    public static void gpu_gemm_batch(long shader, int batch_num, MemorySegment a, MemorySegment a2, int aoffset, MemorySegment bid, int boffset, MemorySegment r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc) {
        var mh$ = gpu_gemm_batch$MH();
        try {
            mh$.invokeExact(shader, batch_num, a, a2, aoffset, bid, boffset, r, roffset, m, n0, n, k, lda, ldb, ldc);
        } catch (Throwable ex$) {
            throw new AssertionError("should not reach here", ex$);
        }
    }
}


