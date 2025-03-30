package com.github.tjake.jlama.tensor.operations;

import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.Q4ByteBufferTensor;
import com.github.tjake.jlama.tensor.operations.gpunative.NativeGPU;
import com.github.tjake.jlama.tensor.operations.util.JarSupport;
import com.github.tjake.jlama.util.MachineSpec;
import com.google.common.io.Resources;
import com.google.common.primitives.Ints;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.LongBuffer;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

public class NativeGPUTensorOperations implements TensorOperations {

    private static final Logger logger = LoggerFactory.getLogger(NativeGPUTensorOperations.class);
    private static final int MAX_SCRATCH_SIZE = 1 << 24;

    static {
        if (!JarSupport.maybeLoadLibrary("webgpu_dawn")) System.loadLibrary("webgpu_dawn");
        if (!JarSupport.maybeLoadLibrary("jlamagpu")) System.loadLibrary("jlamagpu");
    }

    private final ConcurrentMap<String, Long> tensorCache  = new ConcurrentHashMap<>();

    private long maxBindBytes;
    private long gemm_f32_id;
    private long gemm_bf16_id;
    private long gemm_q4_id;

    private int params_size;

    private static final TensorOperations delegate;

    static {
        TensorOperations tmp;
        try {
            tmp = new NativeSimdTensorOperations();
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
                logger.debug("Registering GPU tensor {} as {}", t, id);
                return id;
            }
        });

        if (t.dType() == DType.Q4) {
            Q4ByteBufferTensor q4 = (Q4ByteBufferTensor) t;
            tensorCache.computeIfAbsent(q4.getBlockF().getUid(), s -> {
                synchronized (tensorCache) {
                    Long id = NativeGPU.register_tensor(q4.getBlockF().getMemorySegment(), (int) q4.getBlockF().getMemorySegment().byteSize());
                    logger.debug("Registering GPU tensor {} as {}", q4.getBlockF(), id);
                    return id;
                }
            });
        }
    }

    private final ThreadLocal<Long> gpuBuffers = ThreadLocal.withInitial(() -> {
        // Allocate a 1MB scratch buffers
        synchronized (tensorCache) {
            return NativeGPU.register_scratch_buffers(params_size, MAX_SCRATCH_SIZE, MAX_SCRATCH_SIZE);
        }
    });

    @Override
    public String name() {
        return "Native GPU Operations";
    }

    @Override
    public DType preferredWorkingQuantizedType() {
        return DType.F32;
    }

    private static long registerShader(String name) throws IOException {
        byte[] shader = Resources.readLines(Resources.getResource(name), StandardCharsets.UTF_8).stream()
                .reduce((a, b) -> a + "\n" + b)
                .map(f -> f.trim().getBytes(StandardCharsets.UTF_8))
                .orElseThrow(() -> new RuntimeException("Failed to load shader"));
        ByteBuffer shaderBuffer = ByteBuffer.allocateDirect(shader.length + 1);
        shaderBuffer.put(shader);
        shaderBuffer.flip();
        long id = NativeGPU.register_shader(MemorySegment.ofBuffer(shaderBuffer), shader.length + 1);
        logger.debug("Registered shader {} as {}", name, id);
        return id;
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

            gemm_f32_id = registerShader("gemm_f32_v4.wgsl");
            gemm_bf16_id = registerShader("gemm_bf16_v4.wgsl");
            gemm_q4_id = registerShader("gemm_q4.wgsl");

        } catch (Throwable t) {
            logger.error("Failed to load native GPU operations", t);
            throw new RuntimeException(t);
        }
    }

    private boolean gpuSupported(Long btId, DType atype, DType btype, DType rtype ) {
        return btId != null && atype == DType.F32 && (btype == DType.F32 || btype == DType.BF16 || btype == DType.Q4) && rtype == DType.F32;
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

        if (gpuSupported(btId, at.dType(), bt.dType(), result.dType())) {
            long scratchId = gpuBuffers.get();
            long shaderId = switch (bt.dType()) {
                case F32 -> gemm_f32_id;
                case BF16 -> gemm_bf16_id;
                case Q4 -> gemm_q4_id;
                default -> throw new RuntimeException("Unsupported type: " + bt.dType());
            };

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

            if (rLimit * result.dType().size() > MAX_SCRATCH_SIZE)
                throw new RuntimeException("Result scratch is too small: " + rLimit * result.dType().size() + " > " + MAX_SCRATCH_SIZE);

            if (aLimit * at.dType().size() > MAX_SCRATCH_SIZE)
                throw new RuntimeException("input scratch is too small: " + aLimit * at.dType().size() + " > " + MAX_SCRATCH_SIZE);

            int adjBRowOffset = bRowOffset - bt.shape().sparseRowOffset();
            long bid2 = bt.dType() == DType.Q4 ? tensorCache.get(((Q4ByteBufferTensor) bt).getBlockF().getUid()) : -1;
            NativeGPU.gpu_gemm(
                    scratchId,
                    shaderId,
                    at.getMemorySegment(),
                    at.getMemorySegmentOffset(aOffset),
                    at.getMemorySegmentOffset(aLimit),
                    btId,
                    bid2,
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

            // I REALLY need to redo this terrible API.
            // rRowOffset > 0 effectively means we are decoupling the result from the bRowOffset
            // This happens when doing attention because the b weights are not contiguous
            if (rRowOffset > 0 || rowChunkSize < 1024) {
                delegate.batchDotProduct(
                        result,
                        at,
                        bt,
                        aColumnOffset,
                        bColumnOffset,
                        columnLength,
                        rRowOffset,
                        bRowOffset,
                        rowChunkSize);
            } else {
                // We know we have split size of 1 so we can just re-process this using the delegate's split size
                VectorMath.pchunk(
                        0,
                        rowChunkSize,
                        (chunkStart, chunkSize) -> {
                            delegate.batchDotProduct(
                                    result,
                                    at,
                                    bt,
                                    aColumnOffset,
                                    bColumnOffset,
                                    columnLength,
                                    0,
                                    chunkStart,
                                    chunkSize);
                        },
                        delegate.parallelSplitSize());
            }
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
        if (gpuSupported(btId, a.dType(), b[0].dType(), r[0].dType())) {
            for (int i = 0; i < r.length; i++) {
                batchDotProduct(r[i], a, b[i], columnOffset, columnOffset, columnLength, 0, bRowOffset, rowChunkSize);
            }
        } else {
            VectorMath.pchunk(bRowOffset, rowChunkSize, (chunkStart, chunkSize) -> { delegate.dotProductBatchChunk(
                    r, a, b, columnOffset, columnLength, chunkStart, chunkSize); },
                    delegate.parallelSplitSize());
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
