package com.github.tjake.jlama.tensor.operations;

import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.Q4ByteBufferTensor;
import com.github.tjake.jlama.tensor.Q8ByteBufferTensor;
import com.github.tjake.jlama.tensor.operations.cnative.NativeSimd;
import com.github.tjake.jlama.util.MachineSpec;
import com.github.tjake.jlama.util.RuntimeSupport;

public class NativeTensorOperations implements TensorOperations {

    static String OS = System.getProperty("os.name").toLowerCase();
    public static final int HAS_F16C = NativeSimd.HAS_F16C();
    public static final int HAS_AVX2 = NativeSimd.HAS_AVX2();

    private static final NaiveTensorOperations delegate = new NaiveTensorOperations();

    final int flags;


    public NativeTensorOperations() {
        int f = 0;

        if (RuntimeSupport.isLinux())
            f |= HAS_F16C;

        if (MachineSpec.VECTOR_TYPE == MachineSpec.Type.AVX_512)
            f |= HAS_AVX2;

        this.flags = f;
    }

    public NativeTensorOperations(int flags) {
        this.flags = flags;
    }

    @Override
    public boolean requiresOffHeapTensor() {
        return true;
    }

    @Override
    public float dotProduct(AbstractTensor a, AbstractTensor b, int aoffset, int boffset, int limit)
    {
        return switch (a.dType()) {
            case F32 -> switch (b.dType()) {
                case F32 -> NativeSimd.dot_product_f32(flags, a.getMemorySegment(), aoffset, b.getMemorySegment(), boffset, limit);
                case I8 -> NativeSimd.dot_product_f32_q8(flags, a.getMemorySegment(), aoffset, ((Q8ByteBufferTensor)b).getBlockF().getMemorySegment(), b.getMemorySegment(), boffset, limit);
                case Q4 -> NativeSimd.dot_product_f32_q4(flags, a.getMemorySegment(), aoffset, ((Q4ByteBufferTensor)b).getBlockF().getMemorySegment(), b.getMemorySegment(), boffset, limit);
                default -> throw new UnsupportedOperationException();
            };
            case F16 -> switch (b.dType()) {
                case F16 -> NativeSimd.dot_product_f16(flags, a.getMemorySegment(), aoffset, b.getMemorySegment(), boffset, limit);
                case I8 -> NativeSimd.dot_product_f16_q8(flags, a.getMemorySegment(), aoffset, ((Q8ByteBufferTensor)b).getBlockF().getMemorySegment(), b.getMemorySegment(), boffset, limit);
                default -> throw new UnsupportedOperationException();
            };
            default -> throw new UnsupportedOperationException();
        };
    }

    @Override
    public void accumulate(AbstractTensor a, AbstractTensor b)
    {
         switch (a.dType()) {
             case F16:
                 switch (b.dType()) {
                     case F16: NativeSimd.accumulate_f16(flags, a.getMemorySegment(), b.getMemorySegment(), a.size()); break;
                     default: throw new UnsupportedOperationException();
                 }
                 break;
             case F32:
                 switch (b.dType()) {
                     case F32: NativeSimd.accumulate_f32(flags, a.getMemorySegment(), b.getMemorySegment(), a.size()); break;
                     default: throw new UnsupportedOperationException();
                 }
                 break;
             default: throw new UnsupportedOperationException();
        }
    }

    @Override
    public void saxpy(float alpha, AbstractTensor x, AbstractTensor y, int xoffset, int yoffset, int limit)
    {
        delegate.saxpy(alpha, x, y, xoffset, yoffset, limit);
    }

    @Override
    public void sxpby(float beta, AbstractTensor x, AbstractTensor y, int xoffset, int yoffset, int limit)
    {
        delegate.sxpby(beta, x, y, xoffset, yoffset, limit);
    }

    @Override
    public void scale(float factor, AbstractTensor x, int offset, int length)
    {
        delegate.scale(factor, x, offset, length);
    }
}
