package com.github.tjake.jlama.microbench;

import com.github.tjake.jlama.tensor.AbstractTensor;import com.github.tjake.jlama.tensor.FloatBufferTensor;
import com.github.tjake.jlama.tensor.Q4ByteBufferTensor;
import com.github.tjake.jlama.tensor.operations.NaiveTensorOperations;
import com.github.tjake.jlama.tensor.operations.PanamaTensorOperations;
import com.github.tjake.jlama.util.MachineSpec;
import jdk.incubator.vector.IntVector;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;

@Warmup(iterations = 1, time = 5)
@Measurement(iterations = 3, time = 5)
@Fork(
        warmups = 1,
        value = 1,
        jvmArgsPrepend = {"--add-modules=jdk.incubator.vector", "--enable-preview"})
public class BatchBench {
    private static final PanamaTensorOperations ops = new PanamaTensorOperations(MachineSpec.VECTOR_TYPE);

    private static final int BATCH_SIZE = 1;
    private static final int SIZE = 4096;

    @State(Scope.Benchmark)
    public static class Parameters {

        final FloatBufferTensor fb1 = new FloatBufferTensor(BATCH_SIZE, SIZE);
        final FloatBufferTensor fb1a;
        final FloatBufferTensor fb2 = new FloatBufferTensor(SIZE, SIZE);

        final FloatBufferTensor r = new FloatBufferTensor(BATCH_SIZE, SIZE);
        final FloatBufferTensor ra;



        public Parameters() {
            for (int j = 0; j < BATCH_SIZE; j++) {
                for (int i = 0; i < SIZE; i++) {
                    fb1.set(ThreadLocalRandom.current().nextFloat(),j, i);
                    fb2.set(ThreadLocalRandom.current().nextFloat(),j, i);
                }
            }

            fb1a = (FloatBufferTensor) fb1.transpose();
            ra = (FloatBufferTensor) r.transpose();
        }
    }

    @Benchmark
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @BenchmarkMode(Mode.Throughput)
    public void dotBatch(Parameters p, Blackhole bh) {
        ops.batchDotProduct(p.r, p.fb1, p.fb2, 0, 0, SIZE);
        bh.consume(p.r);
    }


    @Benchmark
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @BenchmarkMode(Mode.Throughput)
    public void slow(Parameters p, Blackhole bh) {
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

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}
