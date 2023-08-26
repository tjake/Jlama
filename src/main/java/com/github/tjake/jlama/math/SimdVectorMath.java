package com.github.tjake.jlama.math;

import com.github.tjake.jlama.math.panama.VectorNativeSimd;
import com.github.tjake.jlama.model.AbstractTensor;
import com.google.common.base.Preconditions;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

class SimdVectorMath {
     static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    public static float dotProduct(AbstractTensor a, AbstractTensor b, int aoffset, int boffset, int limit) {
        Preconditions.checkArgument(a.dType() == b.dType());
        Preconditions.checkArgument(limit % 8 == 0);

        return switch (a.dType()) {
            case F32 -> dotProductF32(a, b, aoffset, boffset, limit);
            case F16 -> dotProductF16(a, b, aoffset, boffset, limit);
            default -> throw new UnsupportedOperationException();
        };
    }

    private static float dotProductF16(AbstractTensor a, AbstractTensor b, int aoffset, int boffset, int limit) {
        return VectorNativeSimd.dot_product(a.getMemorySegment(), aoffset, b.getMemorySegment(), boffset, limit);
    }

    private static float dotProductF32(AbstractTensor a, AbstractTensor b, int aoffset, int boffset, int limit) {
         FloatVector acc = FloatVector.zero(SPECIES);
         int upperBound = SPECIES.loopBound(limit);
         int ao = aoffset;
         int bo = boffset;
         for (; ao < (aoffset + upperBound) && bo < (boffset + upperBound); ao += SPECIES.length(), bo += SPECIES.length()) {
             FloatVector va = a.getFloatVector(ao);
             FloatVector vb = b.getFloatVector(bo);
             acc = acc.add(va.mul(vb));
         }
         // reduce
         float res = acc.reduceLanes(VectorOperators.ADD);
         // tail
         for (; ao < (aoffset + limit) && bo < (boffset + limit); ao++, bo++) {
             res += a.get(ao) * b.get(bo);
         }
         return res;
     }

    public static void accumulate(AbstractTensor a, AbstractTensor b) {
        Preconditions.checkArgument(a.dType() == b.dType());
        Preconditions.checkArgument(a.size() % 8 == 0);

        switch (a.dType()) {
            case F32 -> accumulateF32(a, b);
            case F16 -> VectorNativeSimd.accumulate(a.getMemorySegment(), b.getMemorySegment(), a.size());
            default -> throw new UnsupportedOperationException();
        }
    }

    static void accumulateF32(AbstractTensor a, AbstractTensor b) {
         int upperBound = SPECIES.loopBound(a.size());
         int i = 0;
         float[] af = a.getFloatArray();
         float[] bf = b.getFloatArray();
         for (; i < upperBound; i += SPECIES.length()) {
             FloatVector va = a.getFloatVector(i);
             FloatVector vb = b.getFloatVector(i);
             va.add(vb).intoArray(af, a.getArrayOffset() + i);
         }

         // tail
         for (; i < a.size(); i++) {
             af[i + a.getArrayOffset()] += bf[i + b.getArrayOffset()];
         }
     }

    public static void saxpy(float alpha, AbstractTensor x, AbstractTensor y, int xoffset, int yoffset, int limit) {
        Preconditions.checkArgument(x.dType() == y.dType());
        Preconditions.checkArgument(limit % 8 == 0);

        switch (x.dType()) {
            case F32 -> saxpyF32(alpha, x, y, xoffset, yoffset, limit);
            case F16 -> saxpyF16(alpha, x, y, xoffset, yoffset, limit);
        }

    }

    public static void saxpyF16(float alpha, AbstractTensor x, AbstractTensor y, int xoffset, int yoffset, int limit) {
        VectorNativeSimd.saxpy(alpha, x.getMemorySegment(), xoffset, y.getMemorySegment(), yoffset, limit);
    }

    public static void saxpyF32(float alpha, AbstractTensor x, AbstractTensor y, int xoffset, int yoffset, int limit) {
        int upperBound = SPECIES.loopBound(limit);
        float[] xf = x.getFloatArray();
        float[] yf = y.getFloatArray();
        int xo = xoffset;
        int yo = yoffset;
        for (; xo < (xoffset + upperBound) && yo < (yoffset + upperBound); xo += SPECIES.length(), yo += SPECIES.length()) {
            FloatVector vx = x.getFloatVector(xo);
            FloatVector vy = y.getFloatVector(yo);
            vy.add(vx.mul(alpha)).intoArray(yf, y.getArrayOffset() + yo);
        }

        // tail
        for (; xo < (xoffset + limit) && yo < (yoffset + limit); xo++, yo++) {
            yf[yo + y.getArrayOffset()] += (alpha * xf[xo + x.getArrayOffset()]);
        }
    }
}
