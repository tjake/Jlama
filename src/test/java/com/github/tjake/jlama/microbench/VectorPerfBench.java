package com.github.tjake.jlama.microbench;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;


@Warmup(iterations = 1, time = 5)
@Measurement(iterations = 3, time = 5)
@Fork(warmups = 1, value = 1, jvmArgsPrepend = {
        "--add-modules=jdk.incubator.vector",
        "--enable-preview"})
public class VectorPerfBench
{
    private static final int SIZE = 8192;
    private static final IntVector BF16_BYTE_SHIFT = IntVector.broadcast(IntVector.SPECIES_512, 16);

    public static short float32ToBFloat16(float f) {
        return (short) (Float.floatToIntBits(f) >> 16);
    }
    @State(Scope.Benchmark)
    public static class Parameters {
        final short[] s1 = new short[SIZE];
        final short[] s2 = new short[SIZE];

        public Parameters() {
            for (int i = 0; i < SIZE; i++) {
                s1[i] = float32ToBFloat16(ThreadLocalRandom.current().nextFloat());
                s2[i] = float32ToBFloat16(ThreadLocalRandom.current().nextFloat());
            }
        }
    }

    @Benchmark
    @OutputTimeUnit(TimeUnit.MILLISECONDS)
    @BenchmarkMode(Mode.Throughput)
    public void bfloatDot(Parameters p, Blackhole bh) {
        FloatVector acc = FloatVector.zero(FloatVector.SPECIES_512);
        for (int i = 0; i < SIZE; i += FloatVector.SPECIES_512.length()) {

            var f1 = ShortVector.fromArray(ShortVector.SPECIES_256, p.s1, i)
                    .convertShape(VectorOperators.ZERO_EXTEND_S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            var f2 = ShortVector.fromArray(ShortVector.SPECIES_256, p.s2, i)
                    .convertShape(VectorOperators.ZERO_EXTEND_S2I, IntVector.SPECIES_512, 0)
                    .lanewise(VectorOperators.LSHL, BF16_BYTE_SHIFT)
                    .reinterpretAsFloats();

            acc = acc.add(f1.mul(f2));
        }

        bh.consume(acc.reduceLanes(VectorOperators.ADD));
    }

    public static void main(String[] args) throws Exception {
        org.openjdk.jmh.Main.main(args);
    }
}
