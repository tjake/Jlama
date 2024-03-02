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
package com.github.tjake.jlama.tensor.operations.cnative;

import static java.lang.foreign.ValueLayout.*;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

final class constants$3 {

    // Suppresses default constructor, ensuring non-instantiability.
    private constants$3() {}

    static final FunctionDescriptor const$0 = FunctionDescriptor.ofVoid(
            JAVA_INT,
            RuntimeHelper.POINTER,
            JAVA_INT,
            RuntimeHelper.POINTER,
            RuntimeHelper.POINTER,
            JAVA_INT,
            RuntimeHelper.POINTER,
            RuntimeHelper.POINTER,
            JAVA_INT,
            JAVA_INT,
            JAVA_INT,
            JAVA_INT);
    static final MethodHandle const$1 = RuntimeHelper.downcallHandle("dot_product_q8_q4_chunked", constants$3.const$0);
    static final FunctionDescriptor const$2 = FunctionDescriptor.ofVoid(
            JAVA_INT,
            JAVA_INT,
            RuntimeHelper.POINTER,
            JAVA_INT,
            RuntimeHelper.POINTER,
            RuntimeHelper.POINTER,
            JAVA_INT,
            RuntimeHelper.POINTER,
            RuntimeHelper.POINTER,
            JAVA_INT,
            JAVA_INT,
            JAVA_INT,
            JAVA_INT);
    static final MethodHandle const$3 =
            RuntimeHelper.downcallHandle("dot_product_q8_q4_batch_chunked", constants$3.const$2);
}
