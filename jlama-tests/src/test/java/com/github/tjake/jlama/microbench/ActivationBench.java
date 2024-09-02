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

import com.github.tjake.jlama.tensor.Float16BufferTensor;
import com.github.tjake.jlama.tensor.FloatBufferTensor;
import com.github.tjake.jlama.tensor.Q8ByteBufferTensor;
import com.github.tjake.jlama.tensor.operations.PanamaTensorOperations;
import com.github.tjake.jlama.tensor.operations.TensorOperations;
import com.github.tjake.jlama.util.MachineSpec;
import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

@Warmup(iterations = 1, time = 5)
@Measurement(iterations = 1, time = 5)
@Fork(warmups = 1, value = 1, jvmArgsAppend = { "--add-modules=jdk.incubator.vector", "--add-exports", "java.base/sun.nio.ch=ALL-UNNAMED",
    "-Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0" })
public class ActivationBench {
    private static final TensorOperations ops = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE);
    static int size = 1 << 20;
    static byte[] cacheKill = new byte[1 << 10]; // To Flush the L3 cache
    static FloatBufferTensor f1 = new FloatBufferTensor(size);
    static FloatBufferTensor f2 = new FloatBufferTensor(size);
    static Q8ByteBufferTensor b1;
    static Q8ByteBufferTensor b2;
    static Float16BufferTensor f161 = new Float16BufferTensor(size);
    static Float16BufferTensor f162 = new Float16BufferTensor(size);

    static {
        for (int i = 0; i < size; i++) {
            f1.set(ThreadLocalRandom.current().nextFloat(), i);
            f2.set(ThreadLocalRandom.current().nextFloat(), i);

            f161.set(ThreadLocalRandom.current().nextFloat(), i);
            f162.set(ThreadLocalRandom.current().nextFloat(), i);
        }

        b1 = new Q8ByteBufferTensor(f1);
        b2 = new Q8ByteBufferTensor(f2);
    }

    static void flushCache(Blackhole bh) {
        bh.consume(Arrays.hashCode(cacheKill));
    }

    @Benchmark
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @BenchmarkMode(Mode.Throughput)
    public void f32(Blackhole bh) {
        flushCache(bh);
        bh.consume(ops.dotProduct(f1, f2, 0, 0, size));
    }

    @Benchmark
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @BenchmarkMode(Mode.Throughput)
    public void f16(Blackhole bh) {
        flushCache(bh);
        bh.consume(ops.dotProduct(f161, f162, 0, 0, size));
    }

    @Benchmark
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @BenchmarkMode(Mode.Throughput)
    public void i8(Blackhole bh) {
        flushCache(bh);
        bh.consume(ops.dotProduct(b1, b2, 0, 0, size));
    }

    @Benchmark
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @BenchmarkMode(Mode.Throughput)
    public void f32i8(Blackhole bh) {
        flushCache(bh);
        bh.consume(ops.dotProduct(f1, b2, 0, 0, size));
    }

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}
