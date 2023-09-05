package com.github.tjake.jlama.microbench;

import com.github.tjake.jlama.math.SimdVectorMath;
import com.github.tjake.jlama.model.Q8ByteBufferTensor;
import com.github.tjake.jlama.model.Float16BufferTensor;
import com.github.tjake.jlama.model.FloatBufferTensor;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.Arrays;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;

@Warmup(iterations = 1, time = 5)
@Measurement(iterations = 1, time = 5)
@Fork(warmups = 1, value = 1, jvmArgsAppend = {"--add-modules=jdk.incubator.vector", "--add-exports", "java.base/sun.nio.ch=ALL-UNNAMED",
"-Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0", "-Djava.library.path=target/native-lib-only/"})
public class ActivationBench {
    static {
        System.setProperty("jdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK", "0");
    }

    static int size = 1<<20;
    static byte[] cacheKill = new byte[1 << 10]; //To Flush the L3 cache
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
        bh.consume(SimdVectorMath.dotProduct(f1, f2, 0, 0, size));
    }

    @Benchmark
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @BenchmarkMode(Mode.Throughput)
    public void f16(Blackhole bh) {
        flushCache(bh);
        bh.consume(SimdVectorMath.dotProduct(f161, f162, 0, 0, size));
    }

    @Benchmark
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @BenchmarkMode(Mode.Throughput)
    public void i8(Blackhole bh) {
        flushCache(bh);
        bh.consume(SimdVectorMath.dotProduct(b1, b2, 0, 0, size));
    }

    @Benchmark
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @BenchmarkMode(Mode.Throughput)
    public void f32i8(Blackhole bh) {
        flushCache(bh);
        bh.consume(SimdVectorMath.dotProduct(f1, b2, 0, 0, size));
    }

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}
