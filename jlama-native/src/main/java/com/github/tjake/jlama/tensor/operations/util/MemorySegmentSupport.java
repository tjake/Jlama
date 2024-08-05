package com.github.tjake.jlama.tensor.operations.util;

import java.lang.foreign.*;
import java.util.function.Function;

public class MemorySegmentSupport {
    public static MemorySegment[] setupBatch(Function<Integer, MemorySegment> r, Function<Integer, MemorySegment> b, Function<Integer, MemorySegment> c, int limit) {
        throw new UnsupportedOperationException("Not implemented for this JDK version: " + Runtime.version().toString());
    }
}
