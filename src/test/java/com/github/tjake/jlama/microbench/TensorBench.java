package com.github.tjake.jlama.microbench;

import com.github.tjake.jlama.model.AbstractTensor;
import com.github.tjake.jlama.model.FloatBufferTensor;
import com.github.tjake.jlama.model.TensorCache;
import com.github.tjake.jlama.safetensors.DType;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.concurrent.TimeUnit;

@Warmup(iterations = 1, time = 5)
@Measurement(iterations = 3, time = 5)
@Fork(warmups = 1, value = 1)
public class TensorBench {

    @State(Scope.Benchmark)
    public static class Parameters {
        TensorCache cache = new TensorCache(4096 * 4096);
    }

    @Benchmark
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @BenchmarkMode(Mode.Throughput)
    public void allocateTensorDirectly(Parameters params, Blackhole bh) {
        try(FloatBufferTensor b = new FloatBufferTensor(1024, 1024)) {
            bh.consume(b);
        }
    }

    @Benchmark
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @BenchmarkMode(Mode.Throughput)
    public void allocateTensorCached(Parameters params, Blackhole bh) {
        try(AbstractTensor b = params.cache.get(DType.F32, 1024, 1024)) {
            bh.consume(b);
        }
    }

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}
