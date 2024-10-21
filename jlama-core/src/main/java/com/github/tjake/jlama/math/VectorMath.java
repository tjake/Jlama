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
package com.github.tjake.jlama.math;

import com.github.tjake.jlama.tensor.AbstractTensor;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;
import com.github.tjake.jlama.util.BiIntConsumer;
import com.github.tjake.jlama.util.PhysicalCoreExecutor;
import com.google.common.base.Preconditions;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;

import net.jafama.FastMath;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class VectorMath {

    private static final Logger logger = LoggerFactory.getLogger(VectorMath.class);

    public static void pfor(int start, int end, IntConsumer action) {
        PhysicalCoreExecutor.instance.get().execute(() -> IntStream.range(start, end).parallel().forEach(action));
    }

    public static void pchunk(int offset, int length, BiIntConsumer action) {
        int splits = Math.min(length, TensorOperationsProvider.get().parallelSplitSize());
        int chunkSize = length / splits;
        int remainder = 0;

        // Non optimal case, just run in parallel
        if (splits == 1) {
            splits = length;
            chunkSize = 1;
        } else if (length % chunkSize != 0) {
            remainder = length % chunkSize;
        }

        int fsplits = splits;
        int fchunkSize = chunkSize;
        int fremainder = remainder;

        PhysicalCoreExecutor.instance.get()
            .execute(
                () -> IntStream.range(0, fsplits)
                    .parallel()
                    .forEach(
                        i -> action.accept(
                            offset + (i * fchunkSize),
                            fremainder > 0 && i == fsplits - 1 ? fchunkSize + fremainder : fchunkSize
                        )
                    )
            );
    }

    public static void softMax(AbstractTensor x, int offset, int length) {
        Preconditions.checkArgument(x.shape().first() == 1);
        long size = offset + length;

        // find max value (for numerical stability)
        float max_val = x.get(0, offset);
        for (int i = offset + 1; i < size; i++) {
            if (x.get(0, i) > max_val) {
                max_val = x.get(0, i);
            }
        }
        // exp and sum
        float sum = 0.0f;
        for (int i = offset; i < size; i++) {
            x.set((float) FastMath.exp(x.get(0, i) - max_val), 0, i);
            sum += x.get(0, i);
        }
        // normalize
        for (int i = 0; i < size; i++) {
            x.set(x.get(0, i) / sum, 0, i);
        }
    }

    public static void l1normalize(float[] x) {
        float sum = 0.0f;
        for (int i = 0; i < x.length; i++)
            sum += FastMath.abs(x[i]);

        for (int i = 0; i < x.length; i++)
            x[i] /= sum;
    }

    public static void l2normalize(AbstractTensor x) {
        float sum = 0.0f;
        for (int i = 0; i < x.shape().last(); i++) {
            float v = x.get(0, i);
            sum += v * v;
        }
        double magnitude = FastMath.sqrt(sum);
        for (int i = 0; i < x.shape().last(); i++)
            x.set((float) (x.get(0, i) / magnitude), 0, i);
    }

    public static void l2normalize(float[] x) {
        float sum = 0.0f;
        for (int i = 0; i < x.length; i++)
            sum += x[i] * x[i];

        double magnitude = FastMath.sqrt(sum);
        for (int i = 0; i < x.length; i++)
            x[i] /= magnitude;
    }

    public static float cosineSimilarity(float[] a, float[] b) {
        float dotProduct = 0.0f;
        float aMagnitude = 0.0f;
        float bMagnitude = 0.0f;
        for (int i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            aMagnitude += a[i] * a[i];
            bMagnitude += b[i] * b[i];
        }

        return (float) (dotProduct / (FastMath.sqrt(aMagnitude) * FastMath.sqrt(bMagnitude)));
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

    public static float[][] precomputeFreqsCis(int dim, int end, double theta, double scaling_factor) {
        float[] freqs = new float[dim / 2];
        float step = 0.0f;
        for (int i = 0; i < freqs.length; i++, step += 2.0)
            freqs[i] = (float) ((1.0 / FastMath.pow(theta, step / dim)) / scaling_factor);

        float[] t = new float[end];
        for (int i = 0; i < end; i++)
            t[i] = i;

        float[] freqs_cis = outerProduct(t, freqs);

        float[][] r = new float[freqs_cis.length][];
        for (int i = 0; i < freqs_cis.length; i++)
            r[i] = new float[] { (float) FastMath.cos(freqs_cis[i]), (float) FastMath.sin(freqs_cis[i]) };

        return r;
    }
}
