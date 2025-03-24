package com.github.tjake.jlama.tensor.operations.gpunative;

import java.lang.foreign.MemorySegment;

public class NativeGPU {

    public static void init_gpu(MemorySegment results) {
        throw new UnsupportedOperationException("Not implemented for this JDK version: " + Runtime.version().toString());
    }

    public static long register_tensor(MemorySegment data, int size) {
        throw new UnsupportedOperationException("Not implemented for this JDK version: " + Runtime.version().toString());
    }

    public static long register_scratch_buffers(int params_size, int input_size, int result_size) {
        throw new UnsupportedOperationException("Not implemented for this JDK version: " + Runtime.version().toString());
    }

    public static void free_working_tensor(long id) {
        throw new UnsupportedOperationException("Not implemented for this JDK version: " + Runtime.version().toString());
    }

    public static long register_shader(MemorySegment data, int size) {
        throw new UnsupportedOperationException("Not implemented for this JDK version: " + Runtime.version().toString());
    }

    public static void gpu_gemm(long scratch_id, long shader, MemorySegment a, int aoffset, int alimit, long bid, int boffset, int blimit, MemorySegment r, int roffset, int rlimit, int m, int n0, int n, int k, int lda, int ldb, int ldc) {
        throw new UnsupportedOperationException("Not implemented for this JDK version: " + Runtime.version().toString());
    }

    public static void gpu_gemm_batch(long shader, MemorySegment a, int aoffset, MemorySegment[] b, int boffset, MemorySegment[] r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc) {
        throw new UnsupportedOperationException("Not implemented for this JDK version: " + Runtime.version().toString());
    }
}
