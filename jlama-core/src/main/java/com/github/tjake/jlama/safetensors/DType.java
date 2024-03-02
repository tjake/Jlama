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
package com.github.tjake.jlama.safetensors;

public enum DType {
    // BOOL represents a boolean type.
    BOOL(1),
    // U8 represents an unsigned byte type.
    U8(1),
    // I8 represents a signed byte type.
    I8(1),
    // I16 represents a 16-bit signed integer type.
    I16(2),
    // U16 represents a 16-bit unsigned integer type.
    U16(2),
    // F16 represents a half-precision (16-bit) floating point type.
    F16(2),
    // BF16 represents a brain (16-bit) floating point type.
    BF16(2),
    // I32 represents a 32-bit signed integer type.
    I32(4),
    // U32 represents a 32-bit unsigned integer type.
    U32(4),
    // F32 represents a 32-bit floating point type.
    F32(4),
    // F64 represents a 64-bit floating point type.
    F64(8),
    // I64 represents a 64-bit signed integer type.
    I64(8),
    // U64 represents a 64-bit unsigned integer type.
    U64(8),

    // JLAMA specific types
    // Q4 represents a 4-bit quantized type.
    Q4(1),
    // Q5 represents a 5-bit quantized type.
    Q5(1);

    private final int size;

    DType(int size) {
        this.size = size;
    }

    public int size() {
        return size;
    }
}
