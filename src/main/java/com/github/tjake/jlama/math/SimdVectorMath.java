package com.github.tjake.jlama.math;

import com.github.tjake.jlama.model.Tensor;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

class SimdVectorMath {
     static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

     static float dotProduct(Tensor a, Tensor b, int aoffset, int boffset, int limit) {
         FloatVector acc = FloatVector.zero(SPECIES);
         float[] af = a.getFloatArray();
         float[] bf = b.getFloatArray();
         int upperBound = SPECIES.loopBound(limit);
         int ao = aoffset;
         int bo = boffset;
         for (; ao < (aoffset + upperBound) && bo < (boffset + upperBound); ao += SPECIES.length(), bo += SPECIES.length()) {
             FloatVector va = a.getVector(ao);
             FloatVector vb = b.getVector(bo);
             acc = acc.add(va.mul(vb));
         }
         // reduce
         float res = acc.reduceLanes(VectorOperators.ADD);
         // tail
         for (; ao < (aoffset + limit) && bo < (boffset + limit); ao++, bo++) {
             res += af[ao + a.getArrayOffset()] * bf[bo + b.getArrayOffset()];
         }
         return res;
     }

     static void accumulate(Tensor a, Tensor b) {
         int upperBound = SPECIES.loopBound(a.size());
         int i = 0;
         float[] af = a.getFloatArray();
         float[] bf = b.getFloatArray();
         for (; i < upperBound; i += SPECIES.length()) {
             FloatVector va = a.getVector(i);
             FloatVector vb = b.getVector(i);
             va.add(vb).intoArray(af, a.getArrayOffset() + i);
         }

         // tail
         for (; i < a.size(); i++) {
             af[i + a.getArrayOffset()] += bf[i + b.getArrayOffset()];
         }
     }

    public static void saxpy(float alpha, Tensor x, Tensor y, int xoffset, int yoffset, int limit) {
        int upperBound = SPECIES.loopBound(limit);
        float[] xf = x.getFloatArray();
        float[] yf = y.getFloatArray();
        int xo = xoffset;
        int yo = yoffset;
        for (; xo < (xoffset + upperBound) && yo < (yoffset + upperBound); xo += SPECIES.length(), yo += SPECIES.length()) {
            FloatVector vx = x.getVector(xo);
            FloatVector vy = y.getVector(yo);
            vy.add(vx.mul(alpha)).intoArray(yf, y.getArrayOffset() + yo);
        }

        // tail
        for (; xo < (xoffset + limit) && yo < (yoffset + limit); xo++, yo++) {
            yf[yo + y.getArrayOffset()] += (alpha * xf[xo + x.getArrayOffset()]);
        }
    }
}
