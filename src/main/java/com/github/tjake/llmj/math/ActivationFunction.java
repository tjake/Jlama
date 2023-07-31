package com.github.tjake.llmj.math;

public class ActivationFunction {

    public enum Type {
        SILU,
        SILU2,
        GELU
    }

    public static float eval(Type t, float x) {
        return switch (t) {
            case SILU -> x / (1 + (float) exp(-x));
            case SILU2 -> (float) (x * (1.0f / (1.0f + exp(-x))));
            case GELU -> (float) (x / (1 + exp(-1.702f * x)));
        };
    }

    // https://martin.ankerl.com/2007/02/11/optimized-exponential-functions-for-java/
    private static double exp(float val) {
        final long tmp = (long) (1512775 * val) + (1072693248 - 60801);
        return Double.longBitsToDouble(tmp << 32);
    }
}
