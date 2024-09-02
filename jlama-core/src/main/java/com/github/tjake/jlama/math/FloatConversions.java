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
        return Float.intBitsToFloat(raw << 16);
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
            return (short) (s | e1);
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
        long w = Integer.toUnsignedLong(raw << 16);
        long sign = w & 0x80000000L;
        long nonsign = w & 0x7FFFFFFF;

        int renorm_shift = Long.numberOfLeadingZeros(nonsign);

        renorm_shift = renorm_shift > (32 + 5) ? renorm_shift - (32 + 5) : 0;

        long zero_mask = (nonsign - 1) >> (32 + 31);

        return Float.intBitsToFloat((int) (sign | (((nonsign << renorm_shift >> 3) + ((0x70 - renorm_shift) << 23)) & ~zero_mask)));
    }

    private static final short SIGN_MASK = (short) 0x8000;
    private static final short EXP_MASK = 0x7C00;
    private static final short NAN_VALUE = 0x7FFF;

    private static boolean IS_ZERO(short x) {
        return (x & 0x7FFF) == 0;
    }

    private static boolean IS_INVALID(short x) {
        return (x & EXP_MASK) == EXP_MASK;
    }

    private static boolean IS_NAN(short x) {
        return (x & 0x7FFF) > 0x7C00;
    }

    private static boolean IS_INF(short x) {
        return (x & 0x7FFF) == 0x7C00;
    }

    private static short MANTISSA(short x) {
        return (short) ((x & 1023) | ((x & 0x7C00) == 0 ? 0 : 1024));
    }

    private static short EXPONENT(short x) {
        return (short) ((x & 0x7C00) >> 10);
    }

    private static short SIGNED_INF_VALUE(short x) {
        return (short) ((x & SIGN_MASK) | 0x7C00);
    }

    public static short subIeeeFloat16(short a, short b) {
        if (((a ^ b) & 0x8000) != 0) {
            return addIeeeFloat16(a, (short) (b ^ 0x8000));
        }

        short sign = (short) (a & 0x8000);
        a = (short) (a << 1);
        b = (short) (b << 1);

        if (a < b) {
            short x = a;
            a = b;
            b = x;
            sign ^= 0x8000;
        }

        short ax = (short) (a & 0xF800);
        short bx = (short) (b & 0xF800);

        if (a >= 0xF800 || b >= 0xF800) {
            if (a > 0xF800 || b > 0xF800 || a == b) {
                return NAN_VALUE;
            }
            short res = (short) (sign | 0x7C00);
            if (a == 0xF800) {
                return res;
            } else {
                return (short) (res ^ 0x8000);
            }
        }

        int exp_diff = ax - bx;
        short exp_part = ax;

        if (exp_diff != 0) {
            int shift = exp_diff >> 11;
            if (bx != 0) {
                b = (short) (((b & 2047) | 2048) >> shift);
            } else {
                b >>= (shift - 1);
            }
        } else {
            if (bx == 0) {
                short res = (short) ((a - b) >> 1);
                if (res == 0) {
                    return res;
                }
                return (short) (res | sign);
            } else {
                b = (short) ((b & 2047) | 2048);
            }
        }

        short r = (short) (a - b);

        if ((r & 0xF800) == exp_part) {
            return (short) ((r >> 1) | sign);
        }

        short am = (short) ((a & 2047) | 2048);
        short new_m = (short) (am - b);

        if (new_m == 0) {
            return 0;
        }

        while (exp_part != 0 && (new_m & 2048) == 0) {
            exp_part -= 0x800;
            if (exp_part != 0) {
                new_m <<= 1;
            }
        }

        return (short) ((((new_m & 2047) | exp_part) >> 1) | sign);
    }

    public static short addIeeeFloat16(short a, short b) {
        if (((a ^ b) & 0x8000) != 0) {
            return subIeeeFloat16(a, (short) (b ^ 0x8000));
        }
        short sign = (short) (a & 0x8000);
        a = (short) (a & 0x7FFF);
        b = (short) (b & 0x7FFF);
        if (a < b) {
            short x = a;
            a = b;
            b = x;
        }
        if (a >= 0x7C00 || b >= 0x7C00) {
            if (a > 0x7C00 || b > 0x7C00) {
                return NAN_VALUE;
            }
            return (short) (0x7C00 | sign);
        }
        short ax = (short) (a & 0x7C00);
        short bx = (short) (b & 0x7C00);
        short exp_diff = (short) (ax - bx);
        short exp_part = ax;
        if (exp_diff != 0) {
            short shift = (short) (exp_diff >> 10);
            if (bx != 0) {
                b = (short) (((b & 1023) | 1024) >> shift);
            } else {
                b = (short) (b >> (shift - 1));
            }
        } else {
            if (bx == 0) {
                return (short) ((a + b) | sign);
            } else {
                b = (short) ((b & 1023) | 1024);
            }
        }
        short r = (short) (a + b);
        if ((r & 0x7C00) != exp_part) {
            short am = (short) ((a & 1023) | 1024);
            short new_m = (short) ((am + b) >> 1);
            r = (short) ((exp_part + 0x400) | (1023 & new_m));
        }
        if ((r & 0xFFFF) >= 0x7C00) {
            return (short) (sign | 0x7C00);
        }
        return (short) (r | sign);
    }

    public static short mulIeeeFloat16(short a, short b) {
        int sign = (a ^ b) & SIGN_MASK;

        if (IS_INVALID(a) || IS_INVALID(b)) {
            if (IS_NAN(a) || IS_NAN(b) || IS_ZERO(a) || IS_ZERO(b)) {
                return NAN_VALUE;
            }
            return (short) (sign | 0x7C00);
        }

        if (IS_ZERO(a) || IS_ZERO(b)) {
            return 0;
        }

        short m1 = MANTISSA(a);
        short m2 = MANTISSA(b);

        long v = m1;
        v *= m2;
        int ax = EXPONENT(a);
        int bx = EXPONENT(b);
        ax += (ax == 0) ? 1 : 0;
        bx += (bx == 0) ? 1 : 0;
        int new_exp = ax + bx - 15;

        if ((v & ((long) 1 << 21)) != 0) {
            v >>= 11;
            new_exp++;
        } else if ((v & ((long) 1 << 20)) != 0) {
            v >>= 10;
        } else { // denormal
            new_exp -= 10;
            while (v >= 2048) {
                v >>= 1;
                new_exp++;
            }
        }

        if (new_exp <= 0) {
            v >>= (-new_exp + 1);
            new_exp = 0;
        } else if (new_exp >= 31) {
            return SIGNED_INF_VALUE((short) sign);
        }

        return (short) (sign | (new_exp << 10) | (v & 1023));
    }
}
