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

final class Constants$root {

    // Suppresses default constructor, ensuring non-instantiability.
    private Constants$root() {}

    static final OfBoolean C_BOOL$LAYOUT = JAVA_BOOLEAN;
    static final OfByte C_CHAR$LAYOUT = JAVA_BYTE;
    static final OfShort C_SHORT$LAYOUT = JAVA_SHORT;
    static final OfInt C_INT$LAYOUT = JAVA_INT;
    static final OfLong C_LONG$LAYOUT = JAVA_LONG;
    static final OfLong C_LONG_LONG$LAYOUT = JAVA_LONG;
    static final OfFloat C_FLOAT$LAYOUT = JAVA_FLOAT;
    static final OfDouble C_DOUBLE$LAYOUT = JAVA_DOUBLE;
    static final AddressLayout C_POINTER$LAYOUT = ADDRESS.withByteAlignment(8);
}
