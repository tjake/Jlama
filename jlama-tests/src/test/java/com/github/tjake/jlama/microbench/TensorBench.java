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

import com.github.tjake.jlama.tensor.BFloat16BufferTensor;
import com.github.tjake.jlama.tensor.FloatBufferTensor;
import com.github.tjake.jlama.tensor.Q4ByteBufferTensor;
import com.github.tjake.jlama.tensor.Q8ByteBufferTensor;
import com.github.tjake.jlama.tensor.operations.NaiveTensorOperations;
import com.github.tjake.jlama.tensor.operations.PanamaTensorOperations;
import com.github.tjake.jlama.tensor.operations.TensorOperations;
import com.github.tjake.jlama.tensor.operations.TensorOperationsProvider;
import com.github.tjake.jlama.util.MachineSpec;

import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

@Warmup(iterations = 1, time = 5)
@Measurement(iterations = 3, time = 5)
@Fork(warmups = 1, value = 1, jvmArgsPrepend = { "--add-modules=jdk.incubator.vector", "--add-exports", "java.base/sun.nio.ch=ALL-UNNAMED",
    "-Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0", "--enable-preview", "-XX:+UnlockDiagnosticVMOptions",
    "-XX:CompilerDirectivesFile=inlinerules.json", "--enable-native-access=ALL-UNNAMED", "-XX:+AlignVector" })
public class TensorBench {
    private static final PanamaTensorOperations ops = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE);
    private static final TensorOperations n2ops = new NaiveTensorOperations();
    private static final TensorOperations nops = TensorOperationsProvider.get();

    private static final int SIZE = 2048;

    @State(Scope.Benchmark)
    public static class Parameters {

        final FloatBufferTensor f = new FloatBufferTensor(SIZE);
        final FloatBufferTensor f2 = new FloatBufferTensor(SIZE);
        final FloatBufferTensor r = new FloatBufferTensor(1, 1);
        final BFloat16BufferTensor bf;
        final Q8ByteBufferTensor q81;
        final Q8ByteBufferTensor q82;

        final Q4ByteBufferTensor q4;

        public Parameters() {

            try {
                float[] arr = new float[SIZE];

                for (int i = 0; i < SIZE; i++) {
                    f.set(ThreadLocalRandom.current().nextFloat(), 0, i);
                    f2.set(ThreadLocalRandom.current().nextFloat(), 0, i);
                    arr[i] = ThreadLocalRandom.current().nextFloat();
                }

                this.bf = new BFloat16BufferTensor(f);
                this.q81 = new Q8ByteBufferTensor(f);
                this.q82 = new Q8ByteBufferTensor(f2);

                this.q4 = new Q4ByteBufferTensor(f2);

            } catch (Throwable t) {
                throw new RuntimeException(t);
            }
        }
    }

    /* @Benchmark
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void a_aq8dotq4(Parameters p, Blackhole bh) {
        bh.consume(nops.dotProduct(p.q81, p.q4, 0, 0, SIZE));
    }
    
    @Benchmark
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void a_pq8dotq4(Parameters p, Blackhole bh) {
        bh.consume(ops.dotProduct(p.q81, p.q4, 0, 0, SIZE));
    }
    
    @Benchmark
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void a_q8dotq8(Parameters p, Blackhole bh) {
        bh.consume(ops.dotProduct(p.q81, p.q82, 0, 0, SIZE));
    }*/

    @Benchmark
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void native_f32dotq4(Parameters p, Blackhole bh) {
        bh.consume(nops.dotProduct(p.f, p.q4, 0, 0, SIZE));
    }

    @Benchmark
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void panama_f32dotq4(Parameters p, Blackhole bh) {
        bh.consume(ops.dotProduct(p.f, p.q4, 0, 0, SIZE));
    }

    /* @Benchmark
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void a_f32dotq8(Parameters p, Blackhole bh) {
        bh.consume(ops.dotProduct(p.f, p.q82, 0, 0, SIZE));
    }
    
    @Benchmark
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void f32dotf32(Parameters p, Blackhole bh) {
        bh.consume(ops.dotProduct(p.f, p.f2, 0, 0, SIZE));
    }
    
    @Benchmark
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void f32dotf32nops(Parameters p, Blackhole bh) {
        bh.consume(nops.dotProduct(p.f, p.f2, 0, 0, SIZE));
    }
    
    
    @Benchmark
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @BenchmarkMode(Mode.Throughput)
    @Threads(8)
    public void f32dotf32na(Parameters p, Blackhole bh) {
        bh.consume(n2ops.dotProduct(p.f, p.f2, 0, 0, SIZE));
    }*/

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(new String[] { "-prof", "gc", "TensorBench" });
    }
}
