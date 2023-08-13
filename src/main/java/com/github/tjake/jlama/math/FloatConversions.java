package com.github.tjake.jlama.math;

public class FloatConversions {

    static short bFloat16NaN = 0x7f81;

    /**
     * Convert this BFloat16 value to the nearest Float.
     *
     * Unlike Float16, since BFloat16 has the same size exponents as
     * Float32 it means that all we have to do is add some extra zeros
     * to the mantissa.
     *
     * From https://github.com/stripe-archive/agate/blob/master/core/src/main/scala/com/stripe/agate/tensor/BFloat16.scala
     */
    public static float bFloat16ToFloat32(short raw) {
        return Float.intBitsToFloat((raw & 0xffff) << 16);
    }

    public static short float32ToBFloat16(float n) {
        int nbits = Float.floatToRawIntBits(n);
        // 32 bits has 1 sign bit, 8 exponent bits, 23 mantissa bits
        int s = (nbits >>> 16) & 0x8000;
        int e = (nbits >>> 16) & 0x7f80;
        int m = (nbits & 0x7fffff);

        if (e != 0x7f80) {
            // handle normal and subnormal numbers (i.e. not sentinels).
            //
            // m1 will be in [0, 128]; 0 means we rounded down to 0, 128
            // means we rounded up, and will have a zero mantissa left (plus
            // one exponent bit).
            //
            // in any of these cases it turns out m1 has the correct
            // exponent + mantissa bits set. what luck!
            int m1 = round(m, 16);
            int e1 = e + m1;
            return (short)(s | e1);
        } else {
            // handle sentinels
            //
            // if m != 0, we need to be sure to return a NaN. otherwise,
            // truncating will preserve the correctly-signed infinity value.
            return m != 0 ? bFloat16NaN : (short) (nbits >>> 16);
        }
    }

    /**
     * Implement left bit-shift with rounding.
     *
     *   val shifts = ?
     *   val mask =  (1 << shifts) - 1
     *   val n = (? & mask)
     *   val res = round(n, shifts)
     *   assert(res <= (mask + 1))
     */
    static int round(int m, int shifts) {
        int mid = 1 << (shifts - 1);
        int mask = (1 << shifts) - 1;
        int mshift = m >> shifts;
        int masked = m & mask;
        int cmp = masked - mid;
        // we are losing more than 1/2
        if (cmp > 0) return mshift + 1;
            // we are losing < 1/2
        else if (cmp < 0) return mshift;
        else {
            // we are losing exactly 1/2
            // we round to the nearest even
            // 2.5 => 2, 3.5 => 4, 4.5 => 4
            // -2.5 => -2, -3.5 => -4, -4.5 => -4
            boolean isOdd = (mshift & 1) != 0;
            return isOdd ? mshift + 1 : mshift;
        }
    }


    /**
     * Convert a 16-bit floating-point number in ARM alternative half-precision format to a 32-bit floating-point number.
     *
     * Ported from https://github.com/Maratyszcza/FP16/blob/0a92994d729ff76a58f692d3028ca1b64b145d91/include/fp16/fp16.h#L255
     */
    public static float float16ToFloat32Alt(short raw) {
        long  w = Integer.toUnsignedLong(raw << 16);
        long  sign =  w & 0x80000000L;
        long  nonsign = w & 0x7FFFFFFF;

        int renorm_shift = Long.numberOfLeadingZeros(nonsign);

        renorm_shift = renorm_shift > (32+5) ? renorm_shift - (32+5) : 0;

        long zero_mask = (nonsign - 1) >> (32+31);

        return Float.intBitsToFloat((int)(sign | (((nonsign << renorm_shift >> 3) + ((0x70 - renorm_shift) << 23)) & ~zero_mask)));
    }
}
