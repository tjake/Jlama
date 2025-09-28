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

    public static void gpu_gemm(
        long scratch_id,
        long shader,
        MemorySegment a,
        MemorySegment a2,
        int aoffset,
        int alimit,
        long bid,
        long bid2,
        int boffset,
        int blimit,
        MemorySegment r,
        int roffset,
        int rlimit,
        int m,
        int n0,
        int n,
        int k,
        int lda,
        int ldb,
        int ldc,
        int m1_optimized
    ) {
        throw new UnsupportedOperationException("Not implemented for this JDK version: " + Runtime.version().toString());
    }

    public static void gpu_gemm_batch(
        long shader,
        MemorySegment a,
        int aoffset,
        MemorySegment[] b,
        int boffset,
        MemorySegment[] r,
        int roffset,
        int m,
        int n0,
        int n,
        int k,
        int lda,
        int ldb,
        int ldc
    ) {
        throw new UnsupportedOperationException("Not implemented for this JDK version: " + Runtime.version().toString());
    }
}
