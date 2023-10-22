package com.github.tjake.jlama.tensor.operations;

import com.github.tjake.jlama.safetensors.DType;
import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.Q4ByteBufferTensor;
import com.github.tjake.jlama.tensor.Q8ByteBufferTensor;
import com.github.tjake.jlama.tensor.operations.cnative.NativeSimd;
import com.github.tjake.jlama.util.MachineSpec;
import com.github.tjake.jlama.util.PhysicalCoreExecutor;
import com.github.tjake.jlama.util.RuntimeSupport;

public class NativeTensorOperations implements TensorOperations {
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

        if (RuntimeSupport.isLinux())
            f |= HAS_F16C;

        if (MachineSpec.VECTOR_TYPE == MachineSpec.Type.AVX_512)
            f |= HAS_AVX2;

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
    public float dotProduct(AbstractTensor a, AbstractTensor b, int aoffset, int boffset, int limit)
    {
        return switch (a.dType()) {
            case F32 -> switch (b.dType()) {
                case F32 -> NativeSimd.dot_product_f32(flags, a.getMemorySegment(), aoffset, b.getMemorySegment(), boffset, limit);
                case I8 -> NativeSimd.dot_product_f32_q8(flags, a.getMemorySegment(), aoffset, ((Q8ByteBufferTensor)b).getBlockF().getMemorySegment(), b.getMemorySegment(), boffset, limit);
                case Q4 -> NativeSimd.dot_product_f32_q4(flags, a.getMemorySegment(), aoffset, ((Q4ByteBufferTensor)b).getBlockF().getMemorySegment(), b.getMemorySegment(), boffset, limit);
                default -> throw new UnsupportedOperationException(b.dType().name());
            };
            case I8 -> switch (b.dType()) {
                case Q4 -> NativeSimd.dot_product_q8_q4(flags, ((Q8ByteBufferTensor)a).getBlockF().getMemorySegment(), a.getMemorySegment(), aoffset, ((Q4ByteBufferTensor)b).getBlockF().getMemorySegment(), b.getMemorySegment(), boffset, limit);
                //case I8 -> NativeSimd.dot_product_q8(flags, ((Q8ByteBufferTensor)a).getBlockF().getMemorySegment(), a.getMemorySegment(), aoffset, ((Q8ByteBufferTensor)b).getBlockF().getMemorySegment(), b.getMemorySegment(), boffset, limit);
                default -> throw new UnsupportedOperationException(b.dType().name());
            };
            default -> throw new UnsupportedOperationException(a.dType().name());
        };
    }

    @Override
    public void dotProductChunk(AbstractTensor r, AbstractTensor a, AbstractTensor b, int limit, int chunkStart, int chunkSize) {
        switch (a.dType()) {
            case F32: switch (b.dType()) {
                case F32: NativeSimd.dot_product_f32_chunked(flags, r.getMemorySegment(), a.getMemorySegment(), 0, b.getMemorySegment(), 0, limit, chunkStart, chunkSize); break;
                case I8: NativeSimd.dot_product_f32_q8_chunked(flags, r.getMemorySegment(), a.getMemorySegment(), 0, ((Q8ByteBufferTensor)b).getBlockF().getMemorySegment(), b.getMemorySegment(), 0, limit, chunkStart, chunkSize); break;
                case Q4: NativeSimd.dot_product_f32_q4_chunked(flags, r.getMemorySegment(), a.getMemorySegment(), 0, ((Q4ByteBufferTensor)b).getBlockF().getMemorySegment(), b.getMemorySegment(), 0, limit, chunkStart, chunkSize); break;
                default: throw new UnsupportedOperationException(b.dType().name());
            }
            break;
            case I8: switch (b.dType()) {
                case Q4: NativeSimd.dot_product_q8_q4_chunked(flags, r.getMemorySegment(), ((Q8ByteBufferTensor)a).getBlockF().getMemorySegment(), a.getMemorySegment(), 0, ((Q4ByteBufferTensor)b).getBlockF().getMemorySegment(), b.getMemorySegment(), 0, limit, chunkStart, chunkSize);
                break;
                default: throw new UnsupportedOperationException(b.dType().name());
            }
            break;
            default: throw new UnsupportedOperationException(a.dType().name());
        }
    }


    @Override
    public void accumulate(AbstractTensor a, AbstractTensor b) {
         delegate.accumulate(a, b);
    }

    @Override
    public void maccumulate(AbstractTensor a, AbstractTensor b) {
        delegate.maccumulate(a, b);
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
    public AbstractTensor quantize(AbstractTensor t, DType qtype) {
        return delegate.quantize(t, qtype);
    }
}
