package com.github.tjake.jlama.math;

import com.github.tjake.jlama.model.Tensor;
import com.google.common.base.Preconditions;
import jdk.incubator.vector.VectorOperators;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class VectorMath {

    private static final Logger logger = LoggerFactory.getLogger(VectorMath.class);

    public static boolean hasVectorAPI = hasVectorAPI();

    private static boolean hasVectorAPI() {
        try {
            VectorOperators.ADD.name();
            logger.info("Java 20+ Vector API Available");
            return true;
        } catch (Throwable t) {
            logger.warn("Java SIMD Vector API *not* available. Missing --add-modules=jdk.incubator.vector?");
            return false;
        }
    }

    public static void scale(float[] t, int toffset, int tlimit, float scalar)
    {
        for (int i = toffset; i < (toffset + tlimit); i++)
            t[i] *= scalar;
    }

    public static float dotProduct(Tensor a, Tensor b, int limit)
    {
        return dotProduct(a, b, 0, 0, limit);
    }

    // a[0..n] += b[0..n]
    public static void accumulate(Tensor a, Tensor b) {
        Preconditions.checkArgument(a.size() == b.size() && a.dims() == b.dims() && a.dims() == 1);
        Preconditions.checkArgument(a.hasArray());

        if (hasVectorAPI) {
            SimdVectorMath.accumulate(a, b);
            return;
        }

        float[] aArr = a.getFloatArray();
        int aoffset = a.getArrayOffset();

        float[] bArr = b.getFloatArray();
        int boffset = b.getArrayOffset();

        for (int ao = aoffset, bo = boffset; ao < (aoffset + aArr.length) && bo < (boffset + bArr.length); ao++, bo++) {
            aArr[ao] += bArr[bo];
        }
    }

    public static float dotProduct(Tensor a, Tensor b, int aoffset, int boffset, int limit) {
        Preconditions.checkArgument(a.dims() == b.dims());

        if (hasVectorAPI)
            return SimdVectorMath.dotProduct(a, b, aoffset, boffset, limit);

        float[] aArr = a.getFloatArray();
        aoffset += a.getArrayOffset();

        float[] bArr = b.getFloatArray();
        boffset += b.getArrayOffset();

        float s = 0;
        for (int ao = aoffset, bo = boffset; ao < (aoffset + limit) && bo < (boffset + limit); ao++, bo++) {
            s += aArr[ao] * bArr[bo];
        }

        return s;
    }

    // Computes a constant times a vector plus a vector (single-precision).
    // On return, the contents of vector Y are replaced with the result. The value computed is (alpha * X[i]) + Y[i].
    public static void saxpy(float alpha, Tensor x, Tensor y, int xoffset, int yoffset, int limit) {
        Preconditions.checkArgument(x.dims() == y.dims());
        if (hasVectorAPI) {
            SimdVectorMath.saxpy(alpha, x, y, xoffset, yoffset, limit);
            return;
        }

        for (int xo = xoffset, yo = yoffset; xo < (xoffset + limit) && yo < (yoffset + limit) ; xo++, yo++) {
            float v = (alpha * x.get(xo)) + y.get(yo);
            y.set(v, yo);
        }
    }

    // y = x + b*y variant
    public static void sxpby(float beta, Tensor x, Tensor y, int xoffset, int yoffset, int limit) {
        Preconditions.checkArgument(x.dims() == y.dims());

        for (int xo = xoffset, yo = yoffset; xo < (xoffset + limit) && yo < (yoffset + limit) ; xo++, yo++) {
            float v = x.get(xo) + beta * y.get(yo);
            y.set(v, yo);
        }
    }

    public static void softMax(Tensor t) {
        float[] x = t.getFloatArray();
        int offset = t.getArrayOffset();
        int size = t.size();

        // find max value (for numerical stability)
        float max_val = x[offset];
        for (int i = offset + 1; i < size; i++) {
            if (x[i] > max_val) {
                max_val = x[i];
            }
        }
        // exp and sum
        float sum = 0.0f;
        for (int i = offset; i < size; i++) {
            x[i] = (float)StrictMath.exp(x[i] - max_val);
            sum += x[i];
        }
        // normalize
        for (int i = 0; i < size; i++) {
            x[i] /= sum;
        }
    }


    // https://pytorch.org/docs/stable/generated/torch.polar.html
    public static float[] polar(float abs, float angle) {
        float r = (float) StrictMath.cos(angle) * abs;
        float theta = (float) StrictMath.sin(angle) * abs ;
        return new float[] {r, theta};
    }

    public static float[] outerProduct(float[] xs, float[] ys) {
        int n = xs.length;
        int m = ys.length;
        float[] result = new float[n * m];
        int idx = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                result[idx++] = xs[i] * ys[j];
            }
        }
        return result;
    }

    public static float[][] precomputeFreqsCis(int dim, int end, double theta) {
        float[] freqs = new float[dim / 2];
        float step = 0.0f;
        for (int i = 0; i < freqs.length; i++, step += 2.0)
            freqs[i] = (float) (1.0 / StrictMath.pow(theta, step / dim));

        float[] t = new float[end];
        for (int i = 0; i < end; i++)
            t[i] = i;

        float[] freqs_cis = outerProduct(t, freqs);

        float[][] r = new float[freqs_cis.length][];
        for (int i = 0; i < freqs_cis.length; i++)
            r[i] = polar(1.0f, freqs_cis[i]);

        return r;
    }
}
