#include <stdio.h>
#include <immintrin.h>
#include <inttypes.h>
#include <math.h>
#include "vector_simd.h"

// https://github.com/Maratyszcza/FP16

static inline float f32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } f32;
    f32.as_bits = w;
    return f32.as_value;
}

static inline uint32_t f32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } f32;
    f32.as_value = f;
    return f32.as_bits;
}

static inline float f16_to_f32(short h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = f32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = f32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = f32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? f32_to_bits(denormalized_value) : f32_to_bits(normalized_value));
    return f32_from_bits(result);
}

static inline short f32_to_fp16(float f) {
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
#else
    const float scale_to_inf = f32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = f32_from_bits(UINT32_C(0x08800000));
#endif
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = f32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = f32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = f32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

float dot_product_f16_f16c_256(const short* a, int aoffset, const short* b, int boffset, int length) {
    __m256 sum = _mm256_setzero_ps();

    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;

    for(; ao < alim && bo < blim; ao += 8, bo += 8) {
        // Load and convert float16 data to float32 using F16C
        __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(a + ao)));
        __m256 vb = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(b + bo)));

        // Multiply and accumulate
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    // Horizontal sum of the vector to get dot product
    float result[8];
    _mm256_storeu_ps(result, sum);

    float dot = 0.0;
    for(int i = 0; i < 8; ++i) {
        dot += result[i];
    }

    return dot;
}

float dot_product_f16_f16c_512(const short* a, int aoffset, const short* b, int boffset, int length) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();

    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;

    for(; ao < alim && bo < blim; ao += 16, bo += 16) {
        // Load and convert float16 data to float32 using F16C
        __m512 va = _mm512_cvtph_ps(_mm256_load_si256((__m256i const*)(a + ao)));
        __m512 vb = _mm512_cvtph_ps(_mm256_load_si256((__m256i const*)(b + bo)));

        // Multiply and accumulate
        sum = _mm512_fmadd_ps(va, vb, sum);
    }

    // Horizontal sum of the vector to get dot product
    float result[16];
    _mm512_store_ps(result, sum);

    float dot = 0.0;
    for(int i = 0; i < 16; ++i) {
        dot += result[i];
    }

    return dot;
#else
    return dot_product_f16_f16c_256(a, aoffset, b, boffset, length);
#endif
}

float dot_product_f16_256(const short* a, int aoffset, const short* b, int boffset, int length) {
    __m256 sum = _mm256_setzero_ps();

    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;

    float atmp[8];
    float btmp[8];

    for(; ao < alim && bo < blim; ao += 8, bo += 8) {

        // Load and convert float16 data to float32
        for(int i = 0; i < 8; i++) {
            atmp[i] = f16_to_f32(*(a + ao + i));
            btmp[i] = f16_to_f32(*(b + bo + i));
        }

        __m256 va = _mm256_load_ps(atmp);
        __m256 vb = _mm256_load_ps(btmp);

        // Multiply and accumulate
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    // Horizontal sum of the vector to get dot product
    float result[8];
    _mm256_store_ps(result, sum);

    float dot = 0.0;
    for(int i = 0; i < 8; ++i) {
        dot += result[i];
    }

    return dot;
}

float dot_product_f16_512(const short* a, int aoffset, const short* b, int boffset, int length) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();

    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;

    float atmp[16];
    float btmp[16];

    for(; ao < alim && bo < blim; ao += 16, bo += 16) {

        // Load and convert float16 data to float32
        for(int i = 0; i < 16; i++) {
            atmp[i] = f16_to_f32(*(a + ao + i));
            btmp[i] = f16_to_f32(*(b + bo + i));
        }

        __m512 va = _mm512_loadu_ps(atmp);
        __m512 vb = _mm512_loadu_ps(btmp);

        // Multiply and accumulate
        sum = _mm512_fmadd_ps(va, vb, sum);
    }

    // Horizontal sum of the vector to get dot product
    float result[16];
    _mm512_storeu_ps(result, sum);

    float dot = 0.0;
    for(int i = 0; i < 16; ++i) {
        dot += result[i];
    }

    return dot;
#else
    return dot_product_f16_256(a, aoffset, b, boffset, length);
#endif
}


float dot_product_f16(int flags, const short* a, int aoffset, const short* b, int boffset, int length) {
    if ( (flags & HAS_F16C) != 0 ) {
       return ((flags & HAS_AVX2) != 0)
               ? dot_product_f16_f16c_512(a, aoffset, b, boffset, length)
               : dot_product_f16_f16c_256(a, aoffset, b, boffset, length);
    } else {
       return ((flags & HAS_AVX2) != 0)
                   ? dot_product_f16_512(a, aoffset, b, boffset, length)
                   : dot_product_f16_256(a, aoffset, b, boffset, length);
    }
}

float dot_product_f16_q8_f16c_256(const short* a, int aoffset, const float *bf, const char* b, int boffset, int length) {
    __m256 sum = _mm256_setzero_ps();

    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;

    for(; ao < alim && bo < blim; ao += 8, bo += 8) {
        int bf_idx = bo / Q8_BLOCK_SIZE;
        // broadcast the float32 version of 'factor' to all elements
        __m256 vb_f32 = _mm256_set1_ps(*(bf + bf_idx));

        // Load and convert float16 data to float32 using F16C
        __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(a + ao)));

        // Load 8 bytes into a 128-bit integer register
        __m128i int_vb = _mm_loadu_si128((__m128i const*)(b + bo));

        // Extend bytes to 32-bit integers
        __m256i int_vb_ext = _mm256_cvtepi8_epi32(int_vb);

        // Convert integers to floats
        __m256 vb = _mm256_cvtepi32_ps(int_vb_ext);

        // Perform the scaling
        __m256 vb_scaled = _mm256_mul_ps(vb_f32, vb);

        // Multiply and accumulate
        sum = _mm256_fmadd_ps(va, vb_scaled, sum);
    }

    // Horizontal sum of the vector to get dot product
    float result[8];
    _mm256_storeu_ps(result, sum);

    float dot = 0.0;
    for(int i = 0; i < 8; ++i) {
        dot += result[i];
    }

    return dot;
}


float dot_product_f16_q8_f16c_512(const short* a, int aoffset, const float *bf, const char* b, int boffset, int length) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();

    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;

    for(; ao < alim && bo < blim; ao += 16, bo += 16) {
        int bf_idx = bo / Q8_BLOCK_SIZE;
        // broadcast the float32 version of 'factor' to all elements
        __m512 vb_f32 = _mm512_set1_ps(*(bf + bf_idx));

        // Load and convert float16 data to float32 using F16C
        __m512 va = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i const*)(a + ao)));

        // Load 16 bytes into a 256-bit integer register
        __m128i int_vb = _mm_loadu_si128((__m128i const*)(b + bo));

        // Extend bytes to 32-bit integers
        __m512i int_vb_ext = _mm512_cvtepi8_epi32(int_vb);

        // Convert integers to floats
        __m512 vb = _mm512_cvtepi32_ps(int_vb_ext);

        // Perform the scaling
        __m512 vb_scaled = _mm512_mul_ps(vb_f32, vb);

        // Multiply and accumulate
        sum = _mm512_fmadd_ps(va, vb_scaled, sum);
    }

    // Horizontal sum of the vector to get dot product
    float result[16];
    _mm512_storeu_ps(result, sum);

    float dot = 0.0;
    for(int i = 0; i < 16; ++i) {
        dot += result[i];
    }

    return dot;
#else
    return dot_product_f16_q8_f16c_256(a, aoffset, bf, b, boffset, length);
#endif
}

float dot_product_f16_q8_256(const short* a, int aoffset, const float *bf, const char* b, int boffset, int length) {
    __m256 sum = _mm256_setzero_ps();

    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;
    float atmp[8];

    for(; ao < alim && bo < blim; ao += 8, bo += 8) {
        int bf_idx = bo / Q8_BLOCK_SIZE;
        // broadcast the float32 version of 'factor' to all elements
        __m256 vb_f32 = _mm256_set1_ps(*(bf + bf_idx));

        // Load and convert float16 data to float32
        for(int i = 0; i < 8; i++) {
            atmp[i] = f16_to_f32(*(a + ao + i));
        }
        __m256 va = _mm256_load_ps(atmp);

        // Load 8 bytes into a 128-bit integer register
        __m128i int_vb = _mm_loadu_si128((__m128i const*)(b + bo));

        // Extend bytes to 32-bit integers
        __m256i int_vb_ext = _mm256_cvtepi8_epi32(int_vb);

        // Convert integers to floats
        __m256 vb = _mm256_cvtepi32_ps(int_vb_ext);

        // Perform the scaling
        __m256 vb_scaled = _mm256_mul_ps(vb_f32, vb);

        // Multiply and accumulate
        sum = _mm256_fmadd_ps(va, vb_scaled, sum);
    }

    // Horizontal sum of the vector to get dot product
    float result[8];
    _mm256_storeu_ps(result, sum);

    float dot = 0.0;
    for(int i = 0; i < 8; ++i) {
        dot += result[i];
    }

    return dot;
}

float dot_product_f16_q8_512(const short* a, int aoffset, const float *bf, const char* b, int boffset, int length) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();

    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;
    float atmp[16];

    for(; ao < alim && bo < blim; ao += 16, bo += 16) {
        int bf_idx = bo / Q8_BLOCK_SIZE;
        // broadcast the float32 version of 'factor' to all elements
        __m512 vb_f32 = _mm512_set1_ps(*(bf + bf_idx));

        // Load and convert float16 data to float32
        for(int i = 0; i < 16; i++) {
            atmp[i] = f16_to_f32(*(a + ao + i));
        }
        __m512 va = _mm512_load_ps(atmp);

        // Load 16 bytes into a 256-bit integer register
        __m128i int_vb = _mm_loadu_si128((__m128i const*)(b + bo));

        // Extend bytes to 32-bit integers
        __m512i int_vb_ext = _mm512_cvtepi8_epi32(int_vb);

        // Convert integers to floats
        __m512 vb = _mm512_cvtepi32_ps(int_vb_ext);

        // Perform the scaling
        __m512 vb_scaled = _mm512_mul_ps(vb_f32, vb);

        // Multiply and accumulate
        sum = _mm512_fmadd_ps(va, vb_scaled, sum);
    }

    // Horizontal sum of the vector to get dot product
    float result[16];
    _mm512_storeu_ps(result, sum);

    float dot = 0.0;
    for(int i = 0; i < 16; ++i) {
        dot += result[i];
    }

    return dot;
#else
    return dot_product_f16_q8_256(a, aoffset, bf, b, boffset, length);
#endif
}


float dot_product_f16_q8(int flags, const short* a, int aoffset, const float *bf, const char* b, int boffset, int length) {
    if ( (flags & HAS_F16C) != 0 ) {
       return ((flags & HAS_AVX2) != 0)
               ? dot_product_f16_q8_f16c_512(a, aoffset, bf, b, boffset, length)
               : dot_product_f16_q8_f16c_256(a, aoffset, bf, b, boffset, length);
    } else {
       return ((flags & HAS_AVX2) != 0)
                   ? dot_product_f16_q8_512(a, aoffset, bf, b, boffset, length)
                   : dot_product_f16_q8_256(a, aoffset, bf, b, boffset, length);
    }
}


float dot_product_f16_q4(int flags, const short* a, int aoffset, const float *bf, const char* b, int boffset, int length) {
    return 0.0f;
}

float dot_product_f32_q8_256(const float* a, int aoffset, const float *bf, const char* b, int boffset, int length) {
    __m256 sum = _mm256_setzero_ps();

    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;

    for(; ao < alim && bo < blim; ao += 8, bo += 8) {
        int bf_idx = bo / Q8_BLOCK_SIZE;
        // broadcast the float32 version of 'factor' to all elements
        __m256 vb_f32 = _mm256_set1_ps(*(bf + bf_idx));

        // Load float32
        __m256 va = _mm256_load_ps(a + ao);

        // Load 8 bytes into a 128-bit integer register
        __m128i int_vb = _mm_loadu_si128((__m128i const*)(b + bo));

        // Extend bytes to 32-bit integers
        __m256i int_vb_ext = _mm256_cvtepi8_epi32(int_vb);

        // Convert integers to floats
        __m256 vb = _mm256_cvtepi32_ps(int_vb_ext);

        // Perform the scaling
        __m256 vb_scaled = _mm256_mul_ps(vb_f32, vb);

        // Multiply and accumulate
        sum = _mm256_fmadd_ps(va, vb_scaled, sum);
    }

    // Horizontal sum of the vector to get dot product
    float result[8];
    _mm256_storeu_ps(result, sum);

    float dot = 0.0;
    for(int i = 0; i < 8; ++i) {
        dot += result[i];
    }

    return dot;
}


float dot_product_f32_q8_512(const float* a, int aoffset, const float *bf, const char* b, int boffset, int length) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();

    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;

    for(; ao < alim && bo < blim; ao += 16, bo += 16) {
        int bf_idx = bo / Q8_BLOCK_SIZE;
        // broadcast the float32 version of 'factor' to all elements
        __m512 vb_f32 = _mm512_set1_ps(*(bf + bf_idx));

        // Load float32
        __m512 va = _mm512_load_ps(a + ao);

        // Load 16 bytes into a 256-bit integer register
        __m128i int_vb = _mm_loadu_si128((__m128i const*)(b + bo));

        // Extend bytes to 32-bit integers
        __m512i int_vb_ext = _mm512_cvtepi8_epi32(int_vb);

        // Convert integers to floats
        __m512 vb = _mm512_cvtepi32_ps(int_vb_ext);

        // Perform the scaling
        __m512 vb_scaled = _mm512_mul_ps(vb_f32, vb);

        // Multiply and accumulate
        sum = _mm512_fmadd_ps(va, vb_scaled, sum);
    }

    // Horizontal sum of the vector to get dot product
    float result[16];
    _mm512_storeu_ps(result, sum);

    float dot = 0.0;
    for(int i = 0; i < 16; ++i) {
        dot += result[i];
    }

    return dot;
#else
    return dot_product_f32_q8_256(a, aoffset, bf, b, boffset, length);
#endif
}



float dot_product_f32_q8(int flags, const float* a, int aoffset, const float *bf, const char* b, int boffset, int length) {
    return ((flags & HAS_AVX2) != 0)
           ? dot_product_f32_q8_512(a, aoffset, bf, b, boffset, length)
           : dot_product_f32_q8_256(a, aoffset, bf, b, boffset, length);
}



float dot_product_f32_256(const float* a, int aoffset, const float* b, int boffset, int length) {
    __m256 sum = _mm256_setzero_ps();

    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;

    for(; ao < alim && bo < blim; ao += 8, bo += 8) {
        // Load float32
        __m256 va = _mm256_loadu_ps(a + ao);
        __m256 vb = _mm256_loadu_ps(b + bo);

        // Multiply and accumulate
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    // Horizontal sum of the vector to get dot product
    float result[8];
    _mm256_storeu_ps(result, sum);

    float dot = 0.0;
    for(int i = 0; i < 8; ++i) {
        dot += result[i];
    }

    return dot;
}

float dot_product_f32_512(const float* a, int aoffset, const float* b, int boffset, int length) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();

    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;

    for(; ao < alim && bo < blim; ao += 16, bo += 16) {
        // Load float32
        __m512 va = _mm512_loadu_ps(a + ao);
        __m512 vb = _mm512_loadu_ps(b + bo);

        // Multiply and accumulate
        sum = _mm512_fmadd_ps(va, vb, sum);
    }

    // Horizontal sum of the vector to get dot product
    float result[16];
    _mm512_storeu_ps(result, sum);

    float dot = 0.0;
    for(int i = 0; i < 16; ++i) {
        dot += result[i];
    }

    return dot;
#else
    return dot_product_f32_256(a, aoffset, b, boffset, length);
#endif
}

float dot_product_f32(int flags, const float* a, int aoffset, const float* b, int boffset, int length) {
    return ((flags & HAS_AVX2) != 0)
           ? dot_product_f32_512(a, aoffset, b, boffset, length)
           : dot_product_f32_256(a, aoffset, b, boffset, length);
}

float dot_product_f32_q4_256(const float* a, int aoffset, const float *bf, const char* b, int boffset, int length) {
    __m256 sum = _mm256_setzero_ps();

    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;

    // perform a block at a time
    for(; ao < alim && bo < blim; ao += 32, bo += 16) {
        int bf_idx = (bo*2) / Q4_BLOCK_SIZE;
        // broadcast the float32 version of 'factor' to all elements
        __m256 vb_f32 = _mm256_set1_ps(*(bf + bf_idx));

        // Load float32
        __m256 va0 = _mm256_load_ps(a + ao);
        __m256 va1 = _mm256_load_ps(a + ao + 8);
        __m256 va2 = _mm256_load_ps(a + ao + 8 + 8);
        __m256 va3 = _mm256_load_ps(a + ao + 8 + 8 + 8);

        // Load 8 bytes into a 128-bit integer register
        __m128i int_vb0 = _mm_loadl_epi64((__m128i const*)(b + bo)); // Load lower 64 bits
        __m128i int_vb1 = _mm_loadl_epi64((__m128i const*)(b + bo + 8)); // Load lower 64 bits

        // Mask to keep the first 4 bits of each byte
        __m128i mask_first_4bits = _mm_set1_epi8(0xF);

        // Masked values
        __m128i first_4bits0 = _mm_and_si128(int_vb0, mask_first_4bits);
        __m128i first_4bits1 = _mm_and_si128(int_vb1, mask_first_4bits);

        // Shift first 4 bits to rightmost positions
        __m128i last_4bits0 = _mm_srli_epi16(int_vb0, 4);
        __m128i last_4bits1 = _mm_srli_epi16(int_vb1, 4);

        last_4bits0 = _mm_and_si128(last_4bits0, mask_first_4bits);
        last_4bits1 = _mm_and_si128(last_4bits1, mask_first_4bits);

        //Subtract 8 from each int
        __m128i eight = _mm_set1_epi8(8);
        first_4bits0 = _mm_sub_epi8(first_4bits0, eight);
        first_4bits1 = _mm_sub_epi8(first_4bits1, eight);

        last_4bits0 = _mm_sub_epi8(last_4bits0, eight);
        last_4bits1 = _mm_sub_epi8(last_4bits1, eight);

        // Extend these bytes to 32-bit integers (low and high)
        __m256i int_vb_ext_lo0 = _mm256_cvtepi8_epi32(first_4bits0);
        __m256i int_vb_ext_lo1 = _mm256_cvtepi8_epi32(first_4bits1);

        __m256i int_vb_ext_hi0 = _mm256_cvtepi8_epi32(last_4bits0);
        __m256i int_vb_ext_hi1 = _mm256_cvtepi8_epi32(last_4bits1);

        // Convert these 32-bit integers to floats
        __m256 float_vb_lo0 = _mm256_cvtepi32_ps(int_vb_ext_lo0);
        __m256 float_vb_lo1 = _mm256_cvtepi32_ps(int_vb_ext_lo1);

        __m256 float_vb_hi0 = _mm256_cvtepi32_ps(int_vb_ext_hi0);
        __m256 float_vb_hi1 = _mm256_cvtepi32_ps(int_vb_ext_hi1);

        // Perform the scaling
        __m256 vb_scaled_lo0 = _mm256_mul_ps(vb_f32, float_vb_lo0);
        __m256 vb_scaled_lo1 = _mm256_mul_ps(vb_f32, float_vb_lo1);
        __m256 vb_scaled_hi0 = _mm256_mul_ps(vb_f32, float_vb_hi0);
        __m256 vb_scaled_hi1 = _mm256_mul_ps(vb_f32, float_vb_hi1);

        // Multiply and accumulate
        sum = _mm256_fmadd_ps(va0, vb_scaled_lo0, sum);
        sum = _mm256_fmadd_ps(va1, vb_scaled_lo1, sum);
        sum = _mm256_fmadd_ps(va2, vb_scaled_hi0, sum);
        sum = _mm256_fmadd_ps(va3, vb_scaled_hi1, sum);
    }

    // Horizontal sum of the vector to get dot product
    __attribute__((aligned(16))) float result[8];
    _mm256_store_ps(result, sum);

    float dot = 0.0;
    for(int i = 0; i < 8; ++i) {
        dot += result[i];
    }

    return dot;
}

float dot_product_f32_q4_512(const float* a, int aoffset, const float *bf, const char* b, int boffset, int length) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();

    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;

    // perform a block at a time
    for(; ao < alim && bo < blim; ao += 32, bo += 16) {
        int bf_idx = (bo*2) / Q4_BLOCK_SIZE;
        // broadcast the float32 version of 'factor' to all elements
        __m512 vb_f32 = _mm512_set1_ps(*(bf + bf_idx));

        // Load float32
        __m512 va0 = _mm512_load_ps(a + ao);
        __m512 va1 = _mm512_load_ps(a + ao + 16);

        // Load 8 bytes into a 128-bit integer register
        __m128i int_vb0 = _mm_load_si128((__m128i const*)(b + bo)); // Load 128 bits

        // Mask to keep the first 4 bits of each byte
        __m128i mask_first_4bits = _mm_set1_epi8(0xF);

        // Masked values
        __m128i first_4bits0 = _mm_and_si128(int_vb0, mask_first_4bits);

        // Shift first 4 bits to rightmost positions
        __m128i last_4bits0 = _mm_srli_epi16(int_vb0, 4);

        last_4bits0 = _mm_and_si128(last_4bits0, mask_first_4bits);

        //Subtract 8 from each int
        __m128i eight = _mm_set1_epi8(8);
        first_4bits0 = _mm_sub_epi8(first_4bits0, eight);

        last_4bits0 = _mm_sub_epi8(last_4bits0, eight);

        // Extend these bytes to 32-bit integers (low and high)
        __m512i int_vb_ext_lo0 = _mm512_cvtepi8_epi32(first_4bits0);
        __m512i int_vb_ext_hi0 = _mm512_cvtepi8_epi32(last_4bits0);

        // Convert these 32-bit integers to floats
        __m512 float_vb_lo0 = _mm512_cvtepi32_ps(int_vb_ext_lo0);

        __m512 float_vb_hi0 = _mm512_cvtepi32_ps(int_vb_ext_hi0);

        // Perform the scaling
        __m512 vb_scaled_lo0 = _mm512_mul_ps(vb_f32, float_vb_lo0);
        __m512 vb_scaled_hi0 = _mm512_mul_ps(vb_f32, float_vb_hi0);

        // Multiply and accumulate
        sum = _mm512_fmadd_ps(va0, vb_scaled_lo0, sum);
        sum = _mm512_fmadd_ps(va1, vb_scaled_hi0, sum);
    }

    // Horizontal sum of the vector to get dot product
    __attribute__((aligned(16))) float result[16];
    _mm512_store_ps(result, sum);

    float dot = 0.0;
    for(int i = 0; i < 16; ++i) {
        dot += result[i];
    }

    return dot;
#else
    return dot_product_f32_q4_256(a, aoffset, bf, b, boffset, length);
#endif
}

float dot_product_f32_q4(int flags, const float* a, int aoffset, const float *bf, const char* b, int boffset, int length) {
    return ((flags & HAS_AVX2) != 0)
           ? dot_product_f32_q4_512(a, aoffset, bf, b, boffset, length)
           : dot_product_f32_q4_256(a, aoffset, bf, b, boffset, length);
}
