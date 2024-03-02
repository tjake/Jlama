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
package com.github.tjake.jlama.math;

public class ActivationFunction {

    public enum Type {
        SILU,
        GELU
    }

    public static float eval(Type t, float x) {
        return switch (t) {
            case SILU -> (float) (x * (1.0f / (1.0f + exp(-x))));
            case GELU -> (float) (0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3)))));
        };
    }

    // https://martin.ankerl.com/2007/02/11/optimized-exponential-functions-for-java/
    public static double exp(float val) {
        final long tmp = (long) (1512775 * val) + (1072693248 - 60801);
        return Double.longBitsToDouble(tmp << 32);
    }
}
