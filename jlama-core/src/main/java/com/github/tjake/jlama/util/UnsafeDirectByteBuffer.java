package com.github.tjake.jlama.util;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import org.jctools.util.UnsafeAccess;

public class UnsafeDirectByteBuffer
{
    private static final long addressOffset;
    public static final int CACHE_LINE_SIZE = 64;

    public static final int PAGE_SIZE = UnsafeAccess.UNSAFE.pageSize();

    static {
        try {
            addressOffset = UnsafeAccess.UNSAFE.objectFieldOffset(Buffer.class
                    .getDeclaredField("address"));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public static long getAddress(ByteBuffer buffy) {
        return UnsafeAccess.UNSAFE.getLong(buffy, addressOffset);
    }

    /**
     * put byte and skip position update and boundary checks
     */
    public static void putByte(long address, int position, byte b) {
        UnsafeAccess.UNSAFE.putByte(address + (position << 0), b);
    }

    public static void putByte(long address, byte b) {
        UnsafeAccess.UNSAFE.putByte(address, b);
    }

    public static ByteBuffer allocateAlignedByteBuffer(int capacity, long align)
    {
        if (Long.bitCount(align) != 1) {
            throw new IllegalArgumentException("Alignment must be a power of 2");
        }
        // We over allocate by the alignment so we know we can have a large
        // enough aligned block of memory to use.
        ByteBuffer buffy = ByteBuffer.allocateDirect((int) (capacity + align));
        long address = getAddress(buffy);
        if ((address & (align - 1)) == 0) {
            // limit to the capacity specified
            buffy.limit(capacity);
            // set order to native while we are here.
            ByteBuffer slice = buffy.slice().order(ByteOrder.nativeOrder());
            // the slice is now an aligned buffer of the required capacity
            return slice;
        } else {
            int newPosition = (int) (align - (address & (align - 1)));
            buffy.position(newPosition);
            int newLimit = newPosition + capacity;
            // limit to the capacity specified
            buffy.limit(newLimit);
            // set order to native while we are here.
            ByteBuffer slice = buffy.slice().order(ByteOrder.nativeOrder());
            // the slice is now an aligned buffer of the required capacity
            return slice;
        }
    }

    public static boolean isPageAligned(ByteBuffer buffy) {
        return isPageAligned(getAddress(buffy));
    }

    /**
     * This assumes cache line is 64b
     */
    public static boolean isCacheAligned(ByteBuffer buffy) {
        return isCacheAligned(getAddress(buffy));
    }

    public static boolean isPageAligned(long address) {
        return (address & (PAGE_SIZE - 1)) == 0;
    }

    /**
     * This assumes cache line is 64b
     */
    public static boolean isCacheAligned(long address) {
        return (address & (CACHE_LINE_SIZE - 1)) == 0;
    }

    public static boolean isAligned(long address, long align) {
        if (Long.bitCount(align) != 1) {
            throw new IllegalArgumentException("Alignment must be a power of 2");
        }
        return (address & (align - 1)) == 0;
    }
}