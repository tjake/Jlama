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
    U64(8);

    private final int size;

    private DType(int size)
    {
        this.size = size;
    }

    public int size()
    {
        return size;
    }
}
