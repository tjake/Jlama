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

import com.github.tjake.jlama.math.VectorMath;
import com.github.tjake.jlama.tensor.*;
import com.github.tjake.jlama.tensor.operations.NaiveTensorOperations;
import com.github.tjake.jlama.tensor.operations.TensorOperations;
import java.util.Collection;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.results.RunResult;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import org.openjdk.jmh.runner.options.TimeValue;

@Warmup(iterations = 1, time = 5)
@Measurement(iterations = 3, time = 5)
@Fork(warmups = 1, value = 1, jvmArgsPrepend = { "--add-modules=jdk.incubator.vector", "--enable-preview",
    "-Djlama.force_panama_tensor_operations=true" })
public class BatchBench {
    private static final TensorOperations ops = new NaiveTensorOperations(); // TensorOperationsProvider.get();

    private static final int BATCH_SIZE = 1024;
    private static final int SIZE = 1024;

    @State(Scope.Benchmark)
    public static class Parameters {

        final FloatBufferTensor fb1 = new FloatBufferTensor(BATCH_SIZE, SIZE);
        final FloatBufferTensor fb2 = new FloatBufferTensor(SIZE, SIZE);

        final FloatBufferTensor r = new FloatBufferTensor(BATCH_SIZE, SIZE);

        final Q8ByteBufferTensor qa8;
        final Q4ByteBufferTensor qb4;

        final BFloat16BufferTensor qa16;
        final BFloat16BufferTensor qb16;

        final BFloat16BufferTensor rbf16 = new BFloat16BufferTensor(BATCH_SIZE, SIZE);

        public Parameters() {
            for (int j = 0; j < BATCH_SIZE; j++) {
                for (int i = 0; i < SIZE; i++) {
                    fb1.set(ThreadLocalRandom.current().nextFloat(), j, i);
                    fb2.set(ThreadLocalRandom.current().nextFloat(), j, i);
                }
            }

            qa8 = new Q8ByteBufferTensor(fb1);
            qb4 = new Q4ByteBufferTensor(fb2);

            qa16 = new BFloat16BufferTensor(fb1);
            qb16 = new BFloat16BufferTensor(fb2);
        }
    }

    @Benchmark
    @OutputTimeUnit(TimeUnit.SECONDS)
    @BenchmarkMode(Mode.Throughput)
    public void dotBatchF32(Parameters p, Blackhole bh) {
        VectorMath.pchunk(0, BATCH_SIZE, (start, len) -> { ops.dotProductChunk(p.r, p.fb1, p.fb2, 0, SIZE, start, len); });
        bh.consume(p.r);
    }

    @Benchmark
    @OutputTimeUnit(TimeUnit.SECONDS)
    @BenchmarkMode(Mode.Throughput)
    public void dotBatchBF16(Parameters p, Blackhole bh) {
        VectorMath.pchunk(0, BATCH_SIZE, (start, len) -> { ops.dotProductChunk(p.rbf16, p.qa16, p.qb16, 0, SIZE, start, len); });
        bh.consume(p.r);
    }

    @Benchmark
    @OutputTimeUnit(TimeUnit.SECONDS)
    @BenchmarkMode(Mode.Throughput)
    public void slowF32(Parameters p, Blackhole bh) {
        int alim = p.fb1.shape().dim(0);
        int blim = p.fb2.shape().dim(0);
        for (int i = 0; i < alim; i++) {
            AbstractTensor a = p.fb1.slice(true, i);
            for (int j = 0; j < blim; j++) {
                float d = ops.dotProduct(a, p.fb2.slice(true, j), 0, 0, SIZE);
                p.r.set(d, i, j);
            }
        }
        bh.consume(p.r);
    }

    @Benchmark
    @OutputTimeUnit(TimeUnit.SECONDS)
    @BenchmarkMode(Mode.Throughput)
    public void dotBatchI8Q4(Parameters p, Blackhole bh) {
        VectorMath.pchunk(0, BATCH_SIZE, (start, len) -> { ops.dotProductChunk(p.r, p.qa8, p.qb4, 0, SIZE, start, len); });
        bh.consume(p.r);
    }

    @Benchmark
    @OutputTimeUnit(TimeUnit.SECONDS)
    @BenchmarkMode(Mode.Throughput)
    public void slowI8Q4(Parameters p, Blackhole bh) {
        int alim = p.qa8.shape().dim(0);
        int blim = p.qb4.shape().dim(0);
        for (int i = 0; i < alim; i++) {
            AbstractTensor a = p.qa8.slice(true, i);
            for (int j = 0; j < blim; j++) {
                float d = ops.dotProduct(a, p.qb4.slice(true, j), 0, 0, SIZE);
                p.r.set(d, i, j);
            }
        }
        bh.consume(p.r);
    }

    @Benchmark
    @OutputTimeUnit(TimeUnit.SECONDS)
    @BenchmarkMode(Mode.Throughput)
    public void dotBatchF32Q4(Parameters p, Blackhole bh) {
        VectorMath.pchunk(0, BATCH_SIZE, (start, len) -> { ops.dotProductChunk(p.r, p.fb1, p.qb4, 0, SIZE, start, len); });
        bh.consume(p.r);
    }

    @Benchmark
    @OutputTimeUnit(TimeUnit.SECONDS)
    @BenchmarkMode(Mode.Throughput)
    public void slowF32Q4(Parameters p, Blackhole bh) {
        int alim = p.qa8.shape().dim(0);
        int blim = p.qb4.shape().dim(0);
        for (int i = 0; i < alim; i++) {
            AbstractTensor a = p.fb1.slice(true, i);
            for (int j = 0; j < blim; j++) {
                float d = ops.dotProduct(a, p.qb4.slice(true, j), 0, 0, SIZE);
                p.r.set(d, i, j);
            }
        }
        bh.consume(p.r);
    }

    public static void main(String[] args) throws Exception {

        Options opt = new OptionsBuilder().include(BatchBench.class.getSimpleName() + ".dotBatchBF")
            .forks(1)
            .warmupBatchSize(1)
            .measurementBatchSize(1)
            .warmupIterations(1)
            .measurementIterations(3)
            .warmupTime(TimeValue.seconds(5))
            .measurementTime(TimeValue.seconds(5))
            .threads(1)
            .jvmArgs("--add-modules=jdk.incubator.vector", "--enable-preview", "-Djava.library.path=jlama-native/target/native-lib-only")
            .build();

        Collection<RunResult> results = new Runner(opt).run();

        double flopsPerIteration = 2.0 * BATCH_SIZE * SIZE * SIZE;

        for (RunResult r : results) {
            for (var b : r.getBenchmarkResults()) {
                double elapsedTime = TimeUnit.MILLISECONDS.toSeconds(b.getMetadata().getStopTime() - b.getMetadata().getMeasurementTime());

                // Calculate total number of floating-point operations
                double totalFlops = flopsPerIteration * b.getMetadata().getMeasurementOps();

                // Calculate GFLOPS
                double gflops = totalFlops / (elapsedTime * 1_000_000_000.0);
                double gflops2 = b.getPrimaryResult().getScore() * flopsPerIteration / 1_000_000_000.0;
                System.out.println(b.getPrimaryResult().getLabel() + " " + gflops + " GFLOPS" + gflops2);
            }
        }
    }
}
