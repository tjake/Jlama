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

import net.jafama.FastMath;

public class ActivationFunction {

    public enum Type {
        SILU,
        GELU,
        TANH,
        GELU_PYTORCH_TANH
    }

    public static float eval(Type t, float x) {
        return switch (t) {
            case SILU -> (float) (x * (1.0f / (1.0f + FastMath.exp(-x))));
            case GELU, GELU_PYTORCH_TANH -> (float) (0.5 * x * (1 + FastMath.tanh(
                FastMath.sqrt(2 / Math.PI) * (x + 0.044715 * FastMath.pow(x, 3))
            )));
            case TANH -> (float) FastMath.tanh(x);
        };
    }
}
