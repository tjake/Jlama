package com.github.tjake.jlama.tensor.operations.util;

import java.lang.foreign.*;
import java.util.function.Function;

import static java.lang.foreign.ValueLayout.*;

public class MemorySegmentSupport {
    private static final int MAX_BATCH_SIZE = 4;
    private static final ThreadLocal<MemorySegment[]> tmpArr = ThreadLocal.withInitial(() -> scratchMemorySegments(MAX_BATCH_SIZE));

    private static MemorySegment[] scratchMemorySegments(int batchSize) {
        return new MemorySegment[] {
            MemorySegment.allocateNative(
                    MemoryLayout.sequenceLayout(batchSize, ADDRESS), SegmentScope.global()),
            MemorySegment.allocateNative(
                    MemoryLayout.sequenceLayout(batchSize, ADDRESS), SegmentScope.global()),
            MemorySegment.allocateNative(
                    MemoryLayout.sequenceLayout(batchSize, ADDRESS), SegmentScope.global()),
        };
    }

    public static MemorySegment[] setupBatch(Function<Integer, MemorySegment> r, Function<Integer, MemorySegment> b, Function<Integer, MemorySegment> c, int limit) {
        MemorySegment[] tmp = tmpArr.get();
        MemorySegment ra = tmp[0];
        MemorySegment rb = tmp[1];
        MemorySegment rc = tmp[2];

        for (int i = 0; i < limit; i++) {
            ra.setAtIndex(ValueLayout.ADDRESS, i, r.apply(i));
            rb.setAtIndex(ValueLayout.ADDRESS, i, b.apply(i));
            rc.setAtIndex(ValueLayout.ADDRESS, i, c.apply(i));
        }

        return tmp;
    }
}
