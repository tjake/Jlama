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
package com.github.tjake.jlama.microbench;

import com.github.tjake.jlama.tensor.FloatBufferTensor;
import com.github.tjake.jlama.tensor.Q4ByteBufferTensor;
import com.github.tjake.jlama.tensor.operations.PanamaTensorOperations;
import com.github.tjake.jlama.util.MachineSpec;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import jdk.incubator.vector.*;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

@Warmup(iterations = 1, time = 5)
@Measurement(iterations = 3, time = 5)
@Fork(warmups = 1, value = 1, jvmArgsPrepend = { "--add-modules=jdk.incubator.vector", "--enable-preview" })
public class VectorPerfBench {
    private static final PanamaTensorOperations ops = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE);

    private static final int SIZE = 8192;
    private static final IntVector BF16_BYTE_SHIFT_512 = IntVector.broadcast(IntVector.SPECIES_512, 16);
    private static final IntVector BF16_BYTE_SHIFT_128 = IntVector.broadcast(IntVector.SPECIES_128, 16);

    public static short float32ToBFloat16(float f) {
        return (short) (Float.floatToIntBits(f) >> 16);
    }

    @State(Scope.Benchmark)
    public static class Parameters {

        final short[] s1 = new short[SIZE];
        final short[] s2 = new short[SIZE];

        final int[] i1 = new int[SIZE];
        final int[] i2 = new int[SIZE];

        final float[] f1 = new float[SIZE];
        final float[] f2 = new float[SIZE];

        final byte[] b1 = new byte[SIZE];
        final byte[] b2 = new byte[SIZE];

        final Q4ByteBufferTensor q4;
        final FloatBufferTensor fb1 = new FloatBufferTensor(SIZE);
        final FloatBufferTensor fb2 = new FloatBufferTensor(SIZE);

        public Parameters() {
            for (int i = 0; i < SIZE; i++) {
                s1[i] = float32ToBFloat16(ThreadLocalRandom.current().nextFloat());
                s2[i] = float32ToBFloat16(ThreadLocalRandom.current().nextFloat());
                i1[i] = ThreadLocalRandom.current().nextInt();
                i2[i] = ThreadLocalRandom.current().nextInt();
                f1[i] = ThreadLocalRandom.current().nextFloat();
                f2[i] = ThreadLocalRandom.current().nextFloat();
                ThreadLocalRandom.current().nextBytes(b1);
                ThreadLocalRandom.current().nextBytes(b2);

                fb1.set(ThreadLocalRandom.current().nextFloat(), i);
                fb2.set(ThreadLocalRandom.current().nextFloat(), i);
            }

            q4 = new Q4ByteBufferTensor(fb2);
        }
    }

    @Benchmark
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @BenchmarkMode(Mode.Throughput)
    public void bfloatDot512(Parameters p, Blackhole bh) {
        bh.consume(ops.dotProduct(p.fb1, p.q4, SIZE));
    }

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}
