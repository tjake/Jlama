// Generated by jextract

package com.github.tjake.jlama.tensor.operations.gpunative;

import java.lang.invoke.*;
import java.lang.foreign.*;
import java.nio.ByteOrder;
import java.util.*;
import java.util.function.*;
import java.util.stream.*;

import static java.lang.foreign.ValueLayout.*;
import static java.lang.foreign.MemoryLayout.PathElement.*;

public class NativeGPU {

    NativeGPU() {
        // Should not be called directly
    }

    static final Arena LIBRARY_ARENA = Arena.ofAuto();
    static final boolean TRACE_DOWNCALLS = Boolean.getBoolean("jextract.trace.downcalls");

    static void traceDowncall(String name, Object... args) {
         String traceArgs = Arrays.stream(args)
                       .map(Object::toString)
                       .collect(Collectors.joining(", "));
         System.out.printf("%s(%s)\n", name, traceArgs);
    }

    static MemorySegment findOrThrow(String symbol) {
        return SYMBOL_LOOKUP.find(symbol)
            .orElseThrow(() -> new UnsatisfiedLinkError("unresolved symbol: " + symbol));
    }

    static MethodHandle upcallHandle(Class<?> fi, String name, FunctionDescriptor fdesc) {
        try {
            return MethodHandles.lookup().findVirtual(fi, name, fdesc.toMethodType());
        } catch (ReflectiveOperationException ex) {
            throw new AssertionError(ex);
        }
    }

    static MemoryLayout align(MemoryLayout layout, long align) {
        return switch (layout) {
            case PaddingLayout p -> p;
            case ValueLayout v -> v.withByteAlignment(align);
            case GroupLayout g -> {
                MemoryLayout[] alignedMembers = g.memberLayouts().stream()
                        .map(m -> align(m, align)).toArray(MemoryLayout[]::new);
                yield g instanceof StructLayout ?
                        MemoryLayout.structLayout(alignedMembers) : MemoryLayout.unionLayout(alignedMembers);
            }
            case SequenceLayout s -> MemoryLayout.sequenceLayout(s.elementCount(), align(s.elementLayout(), align));
        };
    }

    static final SymbolLookup SYMBOL_LOOKUP = SymbolLookup.loaderLookup()
            .or(Linker.nativeLinker().defaultLookup());

    public static final ValueLayout.OfBoolean C_BOOL = ValueLayout.JAVA_BOOLEAN;
    public static final ValueLayout.OfByte C_CHAR = ValueLayout.JAVA_BYTE;
    public static final ValueLayout.OfShort C_SHORT = ValueLayout.JAVA_SHORT;
    public static final ValueLayout.OfInt C_INT = ValueLayout.JAVA_INT;
    public static final ValueLayout.OfLong C_LONG_LONG = ValueLayout.JAVA_LONG;
    public static final ValueLayout.OfFloat C_FLOAT = ValueLayout.JAVA_FLOAT;
    public static final ValueLayout.OfDouble C_DOUBLE = ValueLayout.JAVA_DOUBLE;
    public static final AddressLayout C_POINTER = ValueLayout.ADDRESS
            .withTargetLayout(MemoryLayout.sequenceLayout(java.lang.Long.MAX_VALUE, JAVA_BYTE));
    public static final ValueLayout.OfLong C_LONG = ValueLayout.JAVA_LONG;

    private static class init_gpu {
        public static final FunctionDescriptor DESC = FunctionDescriptor.ofVoid(
            NativeGPU.C_POINTER
        );

        public static final MemorySegment ADDR = NativeGPU.findOrThrow("init_gpu");

        public static final MethodHandle HANDLE = Linker.nativeLinker().downcallHandle(ADDR, DESC);
    }

    /**
     * Function descriptor for:
     * {@snippet lang=c :
     * void init_gpu(long *results)
     * }
     */
    public static FunctionDescriptor init_gpu$descriptor() {
        return init_gpu.DESC;
    }

    /**
     * Downcall method handle for:
     * {@snippet lang=c :
     * void init_gpu(long *results)
     * }
     */
    public static MethodHandle init_gpu$handle() {
        return init_gpu.HANDLE;
    }

    /**
     * Address for:
     * {@snippet lang=c :
     * void init_gpu(long *results)
     * }
     */
    public static MemorySegment init_gpu$address() {
        return init_gpu.ADDR;
    }

    /**
     * {@snippet lang=c :
     * void init_gpu(long *results)
     * }
     */
    public static void init_gpu(MemorySegment results) {
        var mh$ = init_gpu.HANDLE;
        try {
            if (TRACE_DOWNCALLS) {
                traceDowncall("init_gpu", results);
            }
            mh$.invokeExact(results);
        } catch (Throwable ex$) {
           throw new AssertionError("should not reach here", ex$);
        }
    }

    private static class register_tensor {
        public static final FunctionDescriptor DESC = FunctionDescriptor.of(
            NativeGPU.C_LONG,
            NativeGPU.C_POINTER,
            NativeGPU.C_INT
        );

        public static final MemorySegment ADDR = NativeGPU.findOrThrow("register_tensor");

        public static final MethodHandle HANDLE = Linker.nativeLinker().downcallHandle(ADDR, DESC);
    }

    /**
     * Function descriptor for:
     * {@snippet lang=c :
     * long register_tensor(const char *data, int size)
     * }
     */
    public static FunctionDescriptor register_tensor$descriptor() {
        return register_tensor.DESC;
    }

    /**
     * Downcall method handle for:
     * {@snippet lang=c :
     * long register_tensor(const char *data, int size)
     * }
     */
    public static MethodHandle register_tensor$handle() {
        return register_tensor.HANDLE;
    }

    /**
     * Address for:
     * {@snippet lang=c :
     * long register_tensor(const char *data, int size)
     * }
     */
    public static MemorySegment register_tensor$address() {
        return register_tensor.ADDR;
    }

    /**
     * {@snippet lang=c :
     * long register_tensor(const char *data, int size)
     * }
     */
    public static long register_tensor(MemorySegment data, int size) {
        var mh$ = register_tensor.HANDLE;
        try {
            if (TRACE_DOWNCALLS) {
                traceDowncall("register_tensor", data, size);
            }
            return (long)mh$.invokeExact(data, size);
        } catch (Throwable ex$) {
           throw new AssertionError("should not reach here", ex$);
        }
    }

    private static class register_scratch_buffers {
        public static final FunctionDescriptor DESC = FunctionDescriptor.of(
            NativeGPU.C_LONG,
            NativeGPU.C_INT,
            NativeGPU.C_INT,
            NativeGPU.C_INT
        );

        public static final MemorySegment ADDR = NativeGPU.findOrThrow("register_scratch_buffers");

        public static final MethodHandle HANDLE = Linker.nativeLinker().downcallHandle(ADDR, DESC);
    }

    /**
     * Function descriptor for:
     * {@snippet lang=c :
     * long register_scratch_buffers(int params_size, int input_size, int result_size)
     * }
     */
    public static FunctionDescriptor register_scratch_buffers$descriptor() {
        return register_scratch_buffers.DESC;
    }

    /**
     * Downcall method handle for:
     * {@snippet lang=c :
     * long register_scratch_buffers(int params_size, int input_size, int result_size)
     * }
     */
    public static MethodHandle register_scratch_buffers$handle() {
        return register_scratch_buffers.HANDLE;
    }

    /**
     * Address for:
     * {@snippet lang=c :
     * long register_scratch_buffers(int params_size, int input_size, int result_size)
     * }
     */
    public static MemorySegment register_scratch_buffers$address() {
        return register_scratch_buffers.ADDR;
    }

    /**
     * {@snippet lang=c :
     * long register_scratch_buffers(int params_size, int input_size, int result_size)
     * }
     */
    public static long register_scratch_buffers(int params_size, int input_size, int result_size) {
        var mh$ = register_scratch_buffers.HANDLE;
        try {
            if (TRACE_DOWNCALLS) {
                traceDowncall("register_scratch_buffers", params_size, input_size, result_size);
            }
            return (long)mh$.invokeExact(params_size, input_size, result_size);
        } catch (Throwable ex$) {
           throw new AssertionError("should not reach here", ex$);
        }
    }

    private static class register_shader {
        public static final FunctionDescriptor DESC = FunctionDescriptor.of(
            NativeGPU.C_LONG,
            NativeGPU.C_POINTER,
            NativeGPU.C_INT
        );

        public static final MemorySegment ADDR = NativeGPU.findOrThrow("register_shader");

        public static final MethodHandle HANDLE = Linker.nativeLinker().downcallHandle(ADDR, DESC);
    }

    /**
     * Function descriptor for:
     * {@snippet lang=c :
     * long register_shader(const char *data, int size)
     * }
     */
    public static FunctionDescriptor register_shader$descriptor() {
        return register_shader.DESC;
    }

    /**
     * Downcall method handle for:
     * {@snippet lang=c :
     * long register_shader(const char *data, int size)
     * }
     */
    public static MethodHandle register_shader$handle() {
        return register_shader.HANDLE;
    }

    /**
     * Address for:
     * {@snippet lang=c :
     * long register_shader(const char *data, int size)
     * }
     */
    public static MemorySegment register_shader$address() {
        return register_shader.ADDR;
    }

    /**
     * {@snippet lang=c :
     * long register_shader(const char *data, int size)
     * }
     */
    public static long register_shader(MemorySegment data, int size) {
        var mh$ = register_shader.HANDLE;
        try {
            if (TRACE_DOWNCALLS) {
                traceDowncall("register_shader", data, size);
            }
            return (long)mh$.invokeExact(data, size);
        } catch (Throwable ex$) {
           throw new AssertionError("should not reach here", ex$);
        }
    }

    private static class gpu_gemm {
        public static final FunctionDescriptor DESC = FunctionDescriptor.ofVoid(
            NativeGPU.C_LONG,
            NativeGPU.C_LONG,
            NativeGPU.C_POINTER,
            NativeGPU.C_POINTER,
            NativeGPU.C_INT,
            NativeGPU.C_INT,
            NativeGPU.C_LONG,
            NativeGPU.C_LONG,
            NativeGPU.C_INT,
            NativeGPU.C_INT,
            NativeGPU.C_POINTER,
            NativeGPU.C_INT,
            NativeGPU.C_INT,
            NativeGPU.C_INT,
            NativeGPU.C_INT,
            NativeGPU.C_INT,
            NativeGPU.C_INT,
            NativeGPU.C_INT,
            NativeGPU.C_INT,
            NativeGPU.C_INT
        );

        public static final MemorySegment ADDR = NativeGPU.findOrThrow("gpu_gemm");

        public static final MethodHandle HANDLE = Linker.nativeLinker().downcallHandle(ADDR, DESC);
    }

    /**
     * Function descriptor for:
     * {@snippet lang=c :
     * void gpu_gemm(long scratch_id, long shader, const void *a, const void *a2, int aoffset, int alimit, long bid, long bid2, int boffset, int blimit, float *r, int roffset, int rlimit, int m, int n0, int n, int k, int lda, int ldb, int ldc)
     * }
     */
    public static FunctionDescriptor gpu_gemm$descriptor() {
        return gpu_gemm.DESC;
    }

    /**
     * Downcall method handle for:
     * {@snippet lang=c :
     * void gpu_gemm(long scratch_id, long shader, const void *a, const void *a2, int aoffset, int alimit, long bid, long bid2, int boffset, int blimit, float *r, int roffset, int rlimit, int m, int n0, int n, int k, int lda, int ldb, int ldc)
     * }
     */
    public static MethodHandle gpu_gemm$handle() {
        return gpu_gemm.HANDLE;
    }

    /**
     * Address for:
     * {@snippet lang=c :
     * void gpu_gemm(long scratch_id, long shader, const void *a, const void *a2, int aoffset, int alimit, long bid, long bid2, int boffset, int blimit, float *r, int roffset, int rlimit, int m, int n0, int n, int k, int lda, int ldb, int ldc)
     * }
     */
    public static MemorySegment gpu_gemm$address() {
        return gpu_gemm.ADDR;
    }

    /**
     * {@snippet lang=c :
     * void gpu_gemm(long scratch_id, long shader, const void *a, const void *a2, int aoffset, int alimit, long bid, long bid2, int boffset, int blimit, float *r, int roffset, int rlimit, int m, int n0, int n, int k, int lda, int ldb, int ldc)
     * }
     */
    public static void gpu_gemm(long scratch_id, long shader, MemorySegment a, MemorySegment a2, int aoffset, int alimit, long bid, long bid2, int boffset, int blimit, MemorySegment r, int roffset, int rlimit, int m, int n0, int n, int k, int lda, int ldb, int ldc) {
        var mh$ = gpu_gemm.HANDLE;
        try {
            if (TRACE_DOWNCALLS) {
                traceDowncall("gpu_gemm", scratch_id, shader, a, a2, aoffset, alimit, bid, bid2, boffset, blimit, r, roffset, rlimit, m, n0, n, k, lda, ldb, ldc);
            }
            mh$.invokeExact(scratch_id, shader, a, a2, aoffset, alimit, bid, bid2, boffset, blimit, r, roffset, rlimit, m, n0, n, k, lda, ldb, ldc);
        } catch (Throwable ex$) {
           throw new AssertionError("should not reach here", ex$);
        }
    }

    private static class gpu_gemm_batch {
        public static final FunctionDescriptor DESC = FunctionDescriptor.ofVoid(
            NativeGPU.C_LONG,
            NativeGPU.C_INT,
            NativeGPU.C_POINTER,
            NativeGPU.C_POINTER,
            NativeGPU.C_INT,
            NativeGPU.C_POINTER,
            NativeGPU.C_INT,
            NativeGPU.C_POINTER,
            NativeGPU.C_INT,
            NativeGPU.C_INT,
            NativeGPU.C_INT,
            NativeGPU.C_INT,
            NativeGPU.C_INT,
            NativeGPU.C_INT,
            NativeGPU.C_INT,
            NativeGPU.C_INT
        );

        public static final MemorySegment ADDR = NativeGPU.findOrThrow("gpu_gemm_batch");

        public static final MethodHandle HANDLE = Linker.nativeLinker().downcallHandle(ADDR, DESC);
    }

    /**
     * Function descriptor for:
     * {@snippet lang=c :
     * void gpu_gemm_batch(long shader, int batch_num, const void *a, const void *a2, int aoffset, const long *bid, int boffset, float **r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc)
     * }
     */
    public static FunctionDescriptor gpu_gemm_batch$descriptor() {
        return gpu_gemm_batch.DESC;
    }

    /**
     * Downcall method handle for:
     * {@snippet lang=c :
     * void gpu_gemm_batch(long shader, int batch_num, const void *a, const void *a2, int aoffset, const long *bid, int boffset, float **r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc)
     * }
     */
    public static MethodHandle gpu_gemm_batch$handle() {
        return gpu_gemm_batch.HANDLE;
    }

    /**
     * Address for:
     * {@snippet lang=c :
     * void gpu_gemm_batch(long shader, int batch_num, const void *a, const void *a2, int aoffset, const long *bid, int boffset, float **r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc)
     * }
     */
    public static MemorySegment gpu_gemm_batch$address() {
        return gpu_gemm_batch.ADDR;
    }

    /**
     * {@snippet lang=c :
     * void gpu_gemm_batch(long shader, int batch_num, const void *a, const void *a2, int aoffset, const long *bid, int boffset, float **r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc)
     * }
     */
    public static void gpu_gemm_batch(long shader, int batch_num, MemorySegment a, MemorySegment a2, int aoffset, MemorySegment bid, int boffset, MemorySegment r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc) {
        var mh$ = gpu_gemm_batch.HANDLE;
        try {
            if (TRACE_DOWNCALLS) {
                traceDowncall("gpu_gemm_batch", shader, batch_num, a, a2, aoffset, bid, boffset, r, roffset, m, n0, n, k, lda, ldb, ldc);
            }
            mh$.invokeExact(shader, batch_num, a, a2, aoffset, bid, boffset, r, roffset, m, n0, n, k, lda, ldb, ldc);
        } catch (Throwable ex$) {
           throw new AssertionError("should not reach here", ex$);
        }
    }
}

