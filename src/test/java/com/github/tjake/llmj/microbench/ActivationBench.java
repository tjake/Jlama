package com.github.tjake.llmj.microbench;

import com.github.tjake.llmj.math.ActivationFunction;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.concurrent.TimeUnit;

@Warmup(iterations = 1, time = 5)
@Measurement(iterations = 1, time = 5)
@Fork(warmups = 1, value = 1)
public class ActivationBench {

    @Benchmark
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @BenchmarkMode(Mode.Throughput)
    public void silu2(Blackhole bh) {
        bh.consume(ActivationFunction.eval(ActivationFunction.Type.SILU2, 2.0f));
    }

    @Benchmark
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @BenchmarkMode(Mode.Throughput)
    public void silu1(Blackhole bh) {
        bh.consume(ActivationFunction.eval(ActivationFunction.Type.SILU, 2.0f));
    }

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(new String[]{"ActivationBench"});
    }

}
