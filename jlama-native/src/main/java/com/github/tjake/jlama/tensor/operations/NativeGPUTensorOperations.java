package com.github.tjake.jlama.tensor.operations;

import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.operations.gpunative.NativeGPU;
import com.github.tjake.jlama.tensor.operations.util.JarSupport;
import com.github.tjake.jlama.util.MachineSpec;
import com.google.common.io.Resources;
import com.google.common.primitives.Ints;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.LongBuffer;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

public class NativeGPUTensorOperations implements TensorOperations {

    private static final Logger logger = LoggerFactory.getLogger(NativeGPUTensorOperations.class);

    static {
        if (!JarSupport.maybeLoadLibrary("webgpu_dawn")) System.loadLibrary("webgpu_dawn");
        if (!JarSupport.maybeLoadLibrary("jlamagpu")) System.loadLibrary("jlamagpu");
    }

    private final ConcurrentMap<String, Long> tensorCache  = new ConcurrentHashMap<>();

    private long maxBindBytes;
    private long gemm_f32_id;
    private int params_size;

    private static final TensorOperations delegate;

    static {
        TensorOperations tmp;
        try {
            tmp = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE);
            //tmp = new NativeSimdTensorOperations();
        } catch (Throwable t) {
            logger.warn("Native SIMD operations not available. Consider adding 'com.github.tjake:jlama-native' to the classpath");
            try {
                tmp = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE);
            } catch (Throwable t2) {
                tmp = new NaiveTensorOperations();
            }
        }
        delegate = tmp;
    }

    public NativeGPUTensorOperations() {
        checkLib();
    }

    /**
     * For GPUs we want to actually send the entire batch to the GPU
     * vs sending it in chunks.
     */
    @Override
    public int parallelSplitSize() {
        return 1;
    }

    @Override
    public void registerModelTensor(AbstractTensor t) {

        if (t.getMemorySegment().byteSize() >= maxBindBytes)
        {
            logger.info("Tensor {} is too large to bind, using fallback operations", t);
            return;
        }

        tensorCache.computeIfAbsent(t.getUid(), s -> {
            synchronized (tensorCache) {
                Long id = NativeGPU.register_tensor(t.getMemorySegment(), (int) t.getMemorySegment().byteSize());
                logger.info("Registering tensor {} as {}", t, id);
                return id;
            }
        });
    }

    private final ThreadLocal<Long> gpuBuffers = ThreadLocal.withInitial(() -> {
        // Allocate a 1MB scratch buffers
        synchronized (tensorCache) {
            return NativeGPU.register_scratch_buffers(params_size, 1 << 21, 1 << 21);
        }
    });



    @Override
    public String name() {
        return "Native GPU Operations";
    }

    private void checkLib() {
        // Check if the native library is loaded
        try {

            LongBuffer lb = ByteBuffer.allocateDirect(Long.BYTES * 3)
                    .order(ByteOrder.LITTLE_ENDIAN)
                    .asLongBuffer();
            NativeGPU.init_gpu(MemorySegment.ofBuffer(lb));

            if (lb.get(0) == -1) {
                throw new RuntimeException("Failed to initialize GPU");
            }

            maxBindBytes = lb.get(0);
            logger.info("Native GPU Operations loaded with {} memory and {} groups", lb.get(0), lb.get(1));
            params_size = Ints.checkedCast(lb.get(2));

            byte[] shader = Resources.readLines(Resources.getResource("gemm_f32_v4.wgsl"), StandardCharsets.UTF_8).stream()
                    .reduce((a, b) -> a + "\n" + b)
                    .map(f -> f.trim().getBytes(StandardCharsets.UTF_8))
                    .orElseThrow(() -> new RuntimeException("Failed to load shader"));
            ByteBuffer shaderBuffer = ByteBuffer.allocateDirect(shader.length + 1);
            shaderBuffer.put(shader);
            shaderBuffer.flip();
            gemm_f32_id = NativeGPU.register_shader(MemorySegment.ofBuffer(shaderBuffer), shader.length + 1);

            logger.info("Registered shader {}", gemm_f32_id);

        } catch (Throwable t) {
            logger.error("Failed to load native GPU operations", t);
            throw new RuntimeException(t);
        }
    }

    @Override
    public void batchDotProduct(
            AbstractTensor result,
            AbstractTensor at,
            AbstractTensor bt,
            int aColumnOffset,
            int bColumnOffset,
            int columnLength,
            int rRowOffset,
            int bRowOffset,
            int rowChunkSize
    ) {
        Long btId = tensorCache.get(bt.getUid());

        if (btId != null && at.dType() == DType.F32 && bt.dType() == DType.F32 && result.dType() == DType.F32) {
            long scratchId = gpuBuffers.get();

            int M = at.shape().dim(0);
            int N = rowChunkSize; // b.shape().dim(0);
            int K = columnLength; // a.shape().dim(1);

            int aOffset = at.getOffset(0, aColumnOffset);
            int aLimit = at.getOffset(M, aColumnOffset);
            if (aLimit > at.size()) aLimit = (int) at.size();

            int bOffset = bt.getOffset(bRowOffset + bt.shape().sparseRowOffset(), bColumnOffset);
            int bLimit = bt.getOffset(bRowOffset + N + bt.shape().sparseRowOffset(), bColumnOffset);
            if (bLimit > bt.size()) bLimit = (int) bt.size();

            // Adjusts for both sparse columns and rows this goes negative because we subtract the row offset
            // And the row offsets need to add to the result offset
            int rOffset = result.shape().sparseColumnOffset() - bt.shape().sparseRowOffset() - rRowOffset;
            int rLimit = (int) result.size();

            if (result.size() * 4 > 1 << 21)
                throw new RuntimeException("Result scratch is too small");

            int adjBRowOffset = bRowOffset - bt.shape().sparseRowOffset();
            NativeGPU.gemm(
                    scratchId,
                    gemm_f32_id,
                    at.getMemorySegment(),
                    at.getMemorySegmentOffset(aOffset),
                    at.getMemorySegmentOffset(aLimit),
                    btId,
                    bt.getMemorySegmentOffset(bOffset),
                    bt.getMemorySegmentOffset(bLimit),
                    result.getMemorySegment(),
                    rOffset,
                    result.getMemorySegmentOffset(rLimit),
                    M,
                    adjBRowOffset,
                    N,
                    K,
                    at.getStride(),
                    bt.getStride(),
                    result.getStride());

        } else {
            delegate.batchDotProduct(
                    result, at, bt, aColumnOffset, bColumnOffset, columnLength, rRowOffset, bRowOffset, rowChunkSize);
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
            int rowChunkSize
    ) {

        Long btId = tensorCache.get(b[0].getUid());

        if (btId != null && a.dType() == DType.F32 && b[0].dType() == DType.F32 && r[0].dType() == DType.F32) {
            for (int i = 0; i < r.length; i++) {
                batchDotProduct(r[i], a, b[i], columnOffset, columnOffset, columnLength, 0, bRowOffset, rowChunkSize);
            }
        } else {
            delegate.dotProductBatchChunk(r, a, b, columnOffset, columnLength, bRowOffset, rowChunkSize);
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
            int rOffset,
            int xOffset,
            int batchSize
    ) {
        delegate.saxpy(alpha, x, y, xoffset, yoffset, limit, rOffset, xOffset, batchSize);
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
