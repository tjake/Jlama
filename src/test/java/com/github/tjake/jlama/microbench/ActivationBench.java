package com.github.tjake.jlama.microbench;

import com.github.tjake.jlama.math.SimdVectorMath;
import com.github.tjake.jlama.model.ByteBufferTensor;
import com.github.tjake.jlama.model.FloatBufferTensor;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;

@Warmup(iterations = 1, time = 5)
@Measurement(iterations = 1, time = 5)
@Fork(warmups = 1, value = 1, jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--add-exports", "java.base/sun.nio.ch=ALL-UNNAMED"})
public class ActivationBench {
    static {
        System.setProperty("jdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK", "0");
    }

    static FloatBufferTensor f1 = new FloatBufferTensor(128);
    static FloatBufferTensor f2 = new FloatBufferTensor(128);

    static ByteBufferTensor b1;
    static ByteBufferTensor b2;
    static {
        for (int i = 0; i < 128; i++) {
            f1.set(ThreadLocalRandom.current().nextFloat(), i);
            f2.set(ThreadLocalRandom.current().nextFloat(), i);
        }

        b1 = new ByteBufferTensor(f1);
        b2 = new ByteBufferTensor(f2);
    }

    /*@Benchmark
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @BenchmarkMode(Mode.Throughput)
    public void f32(Blackhole bh) {
        bh.consume(SimdVectorMath.dotProduct(f1, f2, 0, 0, 128));
    }*/

    @Benchmark
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @BenchmarkMode(Mode.Throughput)
    public void i8(Blackhole bh) {
        bh.consume(SimdVectorMath.dotProduct(b1, b2, 0, 0, 128));
    }

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}
