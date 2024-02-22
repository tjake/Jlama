package com.github.tjake.jlama.math;

public class ActivationFunction {

    public enum Type {
        SILU,
        GELU
    }

    public static float eval(Type t, float x) {
        return switch (t) {
            case SILU -> (float) (x * (1.0f / (1.0f + exp(-x))));
            case GELU -> (float) ( 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3)))));
        };
    }

    // https://martin.ankerl.com/2007/02/11/optimized-exponential-functions-for-java/
    public static double exp(float val) {
        final long tmp = (long) (1512775 * val) + (1072693248 - 60801);
        return Double.longBitsToDouble(tmp << 32);
    }
}
