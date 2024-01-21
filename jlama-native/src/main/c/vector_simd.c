#include <stdio.h>
#if defined(__ARM_NEON__)
#include "sse2neon.h"
#else
#include <immintrin.h>
#endif
#include <inttypes.h>
#include <math.h>
#include "vector_simd.h"




////////////////////////////// F32 //////////////////////////

#if !defined(__ARM_NEON__)
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
        __m256 va = _mm256_loadu_ps(a + ao);

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
        __m512 va = _mm512_loadu_ps(a + ao);

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

#else

float dot_product_f32_q8_128(const float* a, int aoffset, const float *bf, const char* b, int boffset, int length) {
    __m128 sum = _mm_setzero_ps();

    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;

    for(; ao < alim && bo < blim; ao += 8, bo += 8) {
        int bf_idx = bo / Q8_BLOCK_SIZE;
        // broadcast the float32 version of 'factor' to all elements
        __m128 vb_f32 = _mm_set1_ps(*(bf + bf_idx));

        // Load float32
        __m128 va0 = _mm_loadu_ps(a + ao);
        __m128 va1 = _mm_loadu_ps(a + ao + 4);

        // Load 8 bytes into a 128-bit integer register
        __m128i int_vb0 = _mm_loadu_si32((__m128i const*)(b + bo));
        __m128i int_vb1 = _mm_loadu_si32((__m128i const*)(b + bo + 4));

        // Extend bytes to 32-bit integers
        __m128i int_vb0_ext = _mm_cvtepi8_epi32(int_vb0);
        __m128i int_vb1_ext = _mm_cvtepi8_epi32(int_vb1);

        // Convert integers to floats
        __m128 vb0 = _mm_cvtepi32_ps(int_vb0_ext);
        __m128 vb1 = _mm_cvtepi32_ps(int_vb1_ext);

        // Perform the scaling
        __m128 vb0_scaled = _mm_mul_ps(vb_f32, vb0);
        __m128 vb1_scaled = _mm_mul_ps(vb_f32, vb1);

        // Multiply and accumulate
        sum = _mm_add_ps(sum, _mm_mul_ps(va0, vb0_scaled));
        sum = _mm_add_ps(sum, _mm_mul_ps(va1, vb1_scaled));
    }

    // Horizontal sum of the vector to get dot product
    float result[4];
    _mm_storeu_ps(result, sum);

    float dot = 0.0;
    for(int i = 0; i < 4; ++i) {
        dot += result[i];
    }

    return dot;
}

float dot_product_f32_q8(int flags, const float* a, int aoffset, const float *bf, const char* b, int boffset, int length) {
      return dot_product_f32_q8_128(a, aoffset, bf, b, boffset, length);
}

#endif

void dot_product_f32_q8_chunked(int flags, float *r, int roffset, const float* a, int aoffset, const float *bf, const char* b, int boffset, int length, int bchunkstart, int bchunksize) {
    for (int c = bchunkstart; c < bchunkstart + bchunksize; c++) {
        int bo = boffset + (c * length);
        r[roffset++] = dot_product_f32_q8(flags, a, aoffset, bf, b, bo, length);
    }
}


#if !defined(__ARM_NEON__)
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

#else

float dot_product_f32_128(const float* a, int aoffset, const float* b, int boffset, int length) {
    __m128 sum = _mm_setzero_ps();

    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;

    for(; ao < alim && bo < blim; ao += 4, bo += 4) {
        // Load float32
        __m128 va = _mm_loadu_ps(a + ao);
        __m128 vb = _mm_loadu_ps(b + bo);

        // Multiply and accumulate
        sum = _mm_add_ps(sum, _mm_mul_ps(va, vb));
    }

    // Horizontal sum of the vector to get dot product
    float result[4];
    _mm_storeu_ps(result, sum);

    float dot = 0.0;
    for(int i = 0; i < 4; ++i) {
        dot += result[i];
    }

    return dot;
}

float dot_product_f32(int flags, const float* a, int aoffset, const float* b, int boffset, int length) {
   return dot_product_f32_128(a, aoffset, b, boffset, length);
}
#endif

void dot_product_f32_chunked(int flags, float *r, int roffset, const float* a, int aoffset, const float* b, int boffset, int length, int bchunkstart, int bchunksize) {
    for (int c = bchunkstart; c < bchunkstart + bchunksize; c++) {
        int bo = boffset + (c * length);
        r[roffset++] = dot_product_f32(flags, a, aoffset, b, bo, length);
    }
}


#if !defined(__ARM_NEON__)
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
        __m256 va0 = _mm256_loadu_ps(a + ao);
        __m256 va1 = _mm256_loadu_ps(a + ao + 8);
        __m256 va2 = _mm256_loadu_ps(a + ao + 8 + 8);
        __m256 va3 = _mm256_loadu_ps(a + ao + 8 + 8 + 8);

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
        __m512 va0 = _mm512_loadu_ps(a + ao);
        __m512 va1 = _mm512_loadu_ps(a + ao + 16);

        // Load 8 bytes into a 128-bit integer register
        __m128i int_vb0 = _mm_loadu_si128((__m128i const*)(b + bo)); // Load 128 bits

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

#else

float dot_product_f32_q4_128(const float* a, int aoffset, const float *bf, const char* b, int boffset, int length) {
    __m128 sum = _mm_setzero_ps();

    int ao = aoffset;
    int bo = boffset;
    int alim = aoffset + length;
    int blim = boffset + length;

    // perform a block at a time
    for(; ao < alim && bo < blim; ao += 32, bo += 16) {
        int bf_idx = (bo*2) / Q4_BLOCK_SIZE;
        // broadcast the float32 version of 'factor' to all elements
        __m128 vb_f32 = _mm_set1_ps(*(bf + bf_idx));

        // Load float32
        __m128 va0 = _mm_loadu_ps(a + ao);
        __m128 va1 = _mm_loadu_ps(a + ao + 4);
        __m128 va2 = _mm_loadu_ps(a + ao + 4 + 4);
        __m128 va3 = _mm_loadu_ps(a + ao + 4 + 4 + 4);
        __m128 va4 = _mm_loadu_ps(a + ao + 4 + 4 + 4 + 4);
        __m128 va5 = _mm_loadu_ps(a + ao + 4 + 4 + 4 + 4 + 4);
        __m128 va6 = _mm_loadu_ps(a + ao + 4 + 4 + 4 + 4 + 4 + 4);
        __m128 va7 = _mm_loadu_ps(a + ao + 4 + 4 + 4 + 4 + 4 + 4 + 4);

        // Load 8 bytes into a 128-bit integer register
        __m128i int_vb0 = _mm_loadu_si32((__m128i const*)(b + bo));
        __m128i int_vb1 = _mm_loadu_si32((__m128i const*)(b + bo + 4));
        __m128i int_vb2 = _mm_loadu_si32((__m128i const*)(b + bo + 4 + 4));
        __m128i int_vb3 = _mm_loadu_si32((__m128i const*)(b + bo + 4 + 4 + 4));

        // Mask to keep the first 4 bits of each byte
        __m128i mask_first_4bits = _mm_set1_epi8(0xF);

        // Masked values
        __m128i first_4bits0 = _mm_and_si128(int_vb0, mask_first_4bits);
        __m128i first_4bits1 = _mm_and_si128(int_vb1, mask_first_4bits);
        __m128i first_4bits2 = _mm_and_si128(int_vb2, mask_first_4bits);
        __m128i first_4bits3 = _mm_and_si128(int_vb3, mask_first_4bits);

        // Shift first 4 bits to rightmost positions
        __m128i last_4bits0 = _mm_srli_epi16(int_vb0, 4);
        __m128i last_4bits1 = _mm_srli_epi16(int_vb1, 4);
        __m128i last_4bits2 = _mm_srli_epi16(int_vb2, 4);
        __m128i last_4bits3 = _mm_srli_epi16(int_vb3, 4);

        last_4bits0 = _mm_and_si128(last_4bits0, mask_first_4bits);
        last_4bits1 = _mm_and_si128(last_4bits1, mask_first_4bits);
        last_4bits2 = _mm_and_si128(last_4bits2, mask_first_4bits);
        last_4bits3 = _mm_and_si128(last_4bits3, mask_first_4bits);

        //Subtract 8 from each int
        __m128i eight = _mm_set1_epi8(8);
        first_4bits0 = _mm_sub_epi8(first_4bits0, eight);
        first_4bits1 = _mm_sub_epi8(first_4bits1, eight);
        first_4bits2 = _mm_sub_epi8(first_4bits2, eight);
        first_4bits3 = _mm_sub_epi8(first_4bits3, eight);

        last_4bits0 = _mm_sub_epi8(last_4bits0, eight);
        last_4bits1 = _mm_sub_epi8(last_4bits1, eight);
        last_4bits2 = _mm_sub_epi8(last_4bits2, eight);
        last_4bits3 = _mm_sub_epi8(last_4bits3, eight);

        // Extend bytes to 32-bit integers
        __m128i int_vb_ext_lo0 = _mm_cvtepi8_epi32(first_4bits0);
        __m128i int_vb_ext_lo1 = _mm_cvtepi8_epi32(first_4bits1);
        __m128i int_vb_ext_lo2 = _mm_cvtepi8_epi32(first_4bits2);
        __m128i int_vb_ext_lo3 = _mm_cvtepi8_epi32(first_4bits3);

        __m128i int_vb_ext_hi0 = _mm_cvtepi8_epi32(last_4bits0);
        __m128i int_vb_ext_hi1 = _mm_cvtepi8_epi32(last_4bits1);
        __m128i int_vb_ext_hi2 = _mm_cvtepi8_epi32(last_4bits2);
        __m128i int_vb_ext_hi3 = _mm_cvtepi8_epi32(last_4bits3);


        // Convert these 32-bit integers to floats
        __m128 float_vb_lo0 = _mm_cvtepi32_ps(int_vb_ext_lo0);
        __m128 float_vb_lo1 = _mm_cvtepi32_ps(int_vb_ext_lo1);
        __m128 float_vb_lo2 = _mm_cvtepi32_ps(int_vb_ext_lo2);
        __m128 float_vb_lo3 = _mm_cvtepi32_ps(int_vb_ext_lo3);

        __m128 float_vb_hi0 = _mm_cvtepi32_ps(int_vb_ext_hi0);
        __m128 float_vb_hi1 = _mm_cvtepi32_ps(int_vb_ext_hi1);
        __m128 float_vb_hi2 = _mm_cvtepi32_ps(int_vb_ext_hi2);
        __m128 float_vb_hi3 = _mm_cvtepi32_ps(int_vb_ext_hi3);

        // Perform the scaling
        __m128 vb_scaled_lo0 = _mm_mul_ps(vb_f32, float_vb_lo0);
        __m128 vb_scaled_lo1 = _mm_mul_ps(vb_f32, float_vb_lo1);
        __m128 vb_scaled_lo2 = _mm_mul_ps(vb_f32, float_vb_lo2);
        __m128 vb_scaled_lo3 = _mm_mul_ps(vb_f32, float_vb_lo3);

        __m128 vb_scaled_hi0 = _mm_mul_ps(vb_f32, float_vb_hi0);
        __m128 vb_scaled_hi1 = _mm_mul_ps(vb_f32, float_vb_hi1);
        __m128 vb_scaled_hi2 = _mm_mul_ps(vb_f32, float_vb_hi2);
        __m128 vb_scaled_hi3 = _mm_mul_ps(vb_f32, float_vb_hi3);

        // Multiply and accumulate
        sum = _mm_add_ps(sum, _mm_mul_ps(va0, vb_scaled_lo0));
        sum = _mm_add_ps(sum, _mm_mul_ps(va1, vb_scaled_lo1));
        sum = _mm_add_ps(sum, _mm_mul_ps(va2, vb_scaled_lo2));
        sum = _mm_add_ps(sum, _mm_mul_ps(va3, vb_scaled_lo3));

        sum = _mm_add_ps(sum, _mm_mul_ps(va4, vb_scaled_hi0));
        sum = _mm_add_ps(sum, _mm_mul_ps(va5, vb_scaled_hi1));
        sum = _mm_add_ps(sum, _mm_mul_ps(va6, vb_scaled_hi2));
        sum = _mm_add_ps(sum, _mm_mul_ps(va7, vb_scaled_hi3));
    }

    // Horizontal sum of the vector to get dot product
    __attribute__((aligned(16))) float result[4];
    _mm_store_ps(result, sum);

    float dot = 0.0;
    for(int i = 0; i < 4; ++i) {
        dot += result[i];
    }

    return dot;
}

float dot_product_f32_q4(int flags, const float* a, int aoffset, const float *bf, const char* b, int boffset, int length) {
    return dot_product_f32_q4_128(a, aoffset, bf, b, boffset, length);
}
#endif

void dot_product_f32_q4_chunked(int flags, float *r, int roffset, const float* a, int aoffset, const float *bf, const char* b, int boffset, int length, int bchunkstart, int bchunksize) {
    for (int c = bchunkstart; c < bchunkstart + bchunksize; c++) {
        int bo = boffset + (c * (length/2)); // offset by chunk since q4 then divide by 2 since 4bits per element
        r[roffset++] = dot_product_f32_q4(flags, a, aoffset, bf, b, bo, length);
    }
}

#if !defined(__ARM_NEON__)
float dot_product_q8_q4_256(const float *af, const char* a, int aoffset, const float *bf, const char* b, int boffset, int length) {
    __m256 sum = _mm256_setzero_ps();

    int ao = aoffset;
    int bo = boffset;
    int numBlocks = length / Q4_BLOCK_SIZE;

    // Mask to keep the first 4 bits of each byte
    __m128i mask_first_4bits = _mm_set1_epi8(0xF);
    //Subtract 8 from each byte to get signed values
    __m128i eight = _mm_set1_epi8(8);

    __attribute__((aligned(16))) float scalef[8];

    //First take the scaling factors of both tensors and multiply them in SIMD
    for (int i = 0; i < numBlocks; i += 8) { //256bits == 8floats
        // Load float32
        __m256 ablock = _mm256_loadu_ps(af + (ao / Q4_BLOCK_SIZE));
        __m256 bblock = _mm256_loadu_ps(bf + ((bo*2) / Q4_BLOCK_SIZE));
        __m256 scaled = _mm256_mul_ps(ablock, bblock);
        _mm256_store_ps(scalef, scaled);

        // perform a block at a time
        for(int j = 0; j < 8; j++, ao += 32, bo += 16) {
            // broadcast the float32 version of 'factor' to all elements
            __m256 scale_f32 = _mm256_set1_ps(scalef[j]);

            // Load 8 bytes into a 128-bit integer register
            __m128i int_vb0 = _mm_loadl_epi64((__m128i const*)(b + bo)); // Load lower 64 bits
            __m128i int_vb1 = _mm_loadl_epi64((__m128i const*)(b + bo + 8)); // Load lower 64 bits

            // Masked values
            __m128i first_4bits0 = _mm_and_si128(int_vb0, mask_first_4bits);
            __m128i first_4bits1 = _mm_and_si128(int_vb1, mask_first_4bits);

            // Shift first 4 bits to rightmost positions
            __m128i last_4bits0 = _mm_srli_epi16(int_vb0, 4);
            __m128i last_4bits1 = _mm_srli_epi16(int_vb1, 4);

            last_4bits0 = _mm_and_si128(last_4bits0, mask_first_4bits);
            last_4bits1 = _mm_and_si128(last_4bits1, mask_first_4bits);

            //Subtract 8 from each int
            first_4bits0 = _mm_sub_epi8(first_4bits0, eight);
            first_4bits1 = _mm_sub_epi8(first_4bits1, eight);
            last_4bits0 = _mm_sub_epi8(last_4bits0, eight);
            last_4bits1 = _mm_sub_epi8(last_4bits1, eight);

            // Extend these bytes to 32-bit ints (low and high)
            __m256i int_vb_ext_lo0 = _mm256_cvtepi8_epi32(first_4bits0);
            __m256i int_vb_ext_lo1 = _mm256_cvtepi8_epi32(first_4bits1);
            __m256i int_vb_ext_hi0 = _mm256_cvtepi8_epi32(last_4bits0);
            __m256i int_vb_ext_hi1 = _mm256_cvtepi8_epi32(last_4bits1);

            // Load 8 bytes into 4 128-bit integer registers
            __m128i int_va0 = _mm_loadl_epi64((__m128i const*)(a + ao));
            __m128i int_va1 = _mm_loadl_epi64((__m128i const*)(a + ao + 8));
            __m128i int_va2 = _mm_loadl_epi64((__m128i const*)(a + ao + 16));
            __m128i int_va3 = _mm_loadl_epi64((__m128i const*)(a + ao + 24));

            //Extend to 32-bit ints
            __m256i int_va0_ext = _mm256_cvtepi8_epi32(int_va0);
            __m256i int_va1_ext = _mm256_cvtepi8_epi32(int_va1);
            __m256i int_va2_ext = _mm256_cvtepi8_epi32(int_va2);
            __m256i int_va3_ext = _mm256_cvtepi8_epi32(int_va3);

            // Multiply the 32-bit integers
            __m256i isum = _mm256_mullo_epi32(int_va0_ext, int_vb_ext_lo0);
            isum = _mm256_add_epi32(_mm256_mullo_epi32(int_va1_ext, int_vb_ext_lo1), isum);
            isum = _mm256_add_epi32(_mm256_mullo_epi32(int_va2_ext, int_vb_ext_hi0), isum);
            isum = _mm256_add_epi32(_mm256_mullo_epi32(int_va3_ext, int_vb_ext_hi1), isum);

            // Convert these 32-bit integers to floats
            __m256 fsum = _mm256_cvtepi32_ps(isum);

            // Multiply and accumulate
            sum = _mm256_fmadd_ps(scale_f32, fsum, sum);
        }
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

float dot_product_q8_q4_512(const float *af, const char* a, int aoffset, const float *bf, const char* b, int boffset, int length) {
#if defined(__AVX512F__)
    __m512 sum = _mm512_setzero_ps();

    int ao = aoffset;
    int bo = boffset;
    int numBlocks = length / Q4_BLOCK_SIZE;

    // Mask to keep the first 4 bits of each byte
    __m128i mask_first_4bits = _mm_set1_epi8(0xF);
    //Subtract 8 from each byte to get signed values
    __m128i eight = _mm_set1_epi8(8);

    __attribute__((aligned(16))) float scalef[16];

    //First take the scaling factors of both tensors and multiply them in SIMD
    for (int i = 0; i < numBlocks; i += 16) { //512bits == 16floats
        // Load float32
        __m512 ablock = _mm512_loadu_ps(af + (ao / Q4_BLOCK_SIZE));
        __m512 bblock = _mm512_loadu_ps(bf + ((bo*2) / Q4_BLOCK_SIZE));
        __m512 scaled = _mm512_mul_ps(ablock, bblock);
        _mm512_store_ps(scalef, scaled);

        // perform a block at a time
        for(int j = 0; j < 16; j++, ao += 32, bo += 16) {
            // broadcast the float32 version of 'factor' to all elements
            __m512 scale_f32 = _mm512_set1_ps(scalef[j]);

            // Load 8 bytes into a 128-bit integer register
            __m128i int_vb0 = _mm_loadu_si128((__m128i const*)(b + bo)); // Load 128 bits

            // Masked values
            __m128i first_4bits0 = _mm_and_si128(int_vb0, mask_first_4bits);

            // Shift first 4 bits to rightmost positions
            __m128i last_4bits0 = _mm_srli_epi16(int_vb0, 4);
            last_4bits0 = _mm_and_si128(last_4bits0, mask_first_4bits);

            //Subtract 8 from each int
            first_4bits0 = _mm_sub_epi8(first_4bits0, eight);
            last_4bits0 = _mm_sub_epi8(last_4bits0, eight);

            // Extend these bytes to 32-bit integers (low and high)
            __m512i int_vb_ext_lo0 = _mm512_cvtepi8_epi32(first_4bits0);
            __m512i int_vb_ext_hi0 = _mm512_cvtepi8_epi32(last_4bits0);

            // Load 16 bytes into 2 128-bit integer registers
            __m128i int_va0 = _mm_loadu_si128((__m128i const*)(a + ao));
            __m128i int_va1 = _mm_loadu_si128((__m128i const*)(a + ao + 16));

            //Extend to 32-bit ints
            __m512i int_va0_ext = _mm512_cvtepi8_epi32(int_va0);
            __m512i int_va1_ext = _mm512_cvtepi8_epi32(int_va1);

            // Multiply the 32-bit integers
            __m512i isum = _mm512_mullo_epi32(int_va0_ext, int_vb_ext_lo0);
            isum = _mm512_add_epi32(_mm512_mullo_epi32(int_va1_ext, int_vb_ext_hi0), isum);

            // Convert these 32-bit integers to floats
            __m512 fsum = _mm512_cvtepi32_ps(isum);

            // Multiply and accumulate
            sum = _mm512_fmadd_ps(scale_f32, fsum, sum);
        }
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
    return dot_product_q8_q4_256(af, a, aoffset, bf, b, boffset, length);
#endif
}

float dot_product_q8_q4(int flags, const float* af, const char* a, int aoffset, const float *bf, const char* b, int boffset, int length) {
    return ((flags & HAS_AVX2) != 0)
         ? dot_product_q8_q4_512(af, a, aoffset, bf, b, boffset, length)
         : dot_product_q8_q4_256(af, a, aoffset, bf, b, boffset, length);
}

#else

float dot_product_q8_q4_128(const float* af, const char* a, int aoffset, const float *bf, const char* b, int boffset, int length) {
    __m128 sum = _mm_setzero_ps();

    int ao = aoffset;
    int bo = boffset;
    int numBlocks = length / Q4_BLOCK_SIZE;

    // Mask to keep the first 4 bits of each byte
    __m128i mask_first_4bits = _mm_set1_epi8(0xF);
    //Subtract 8 from each byte to get signed values
    __m128i eight = _mm_set1_epi8(8);

    __attribute__((aligned(16))) float scalef[8];

    //First take the scaling factors of both tensors and multiply them in SIMD
    for (int i = 0; i < numBlocks; i += 4) { //256bits == 8floats
        // Load float32
        __m128 ablock = _mm_loadu_ps(af + (ao / Q4_BLOCK_SIZE));
        __m128 bblock = _mm_loadu_ps(bf + ((bo*2) / Q4_BLOCK_SIZE));
        __m128 scaled = _mm_mul_ps(ablock, bblock);
        _mm_store_ps(scalef, scaled);

        // perform a block at a time
        for(int i = 0; i < 4; i++, ao += 32, bo += 16) {

            // broadcast the float32 version of 'factor' to all elements
            __m128 vb_f32 = _mm_set1_ps(scalef[i]);

            // Load 4 bytes into a 128-bit integer register
            __m128i int_va0 = _mm_cvtepi8_epi32(_mm_loadu_si32((__m128i const*)(a + ao)));
            __m128i int_va1 = _mm_cvtepi8_epi32(_mm_loadu_si32((__m128i const*)(a + ao + 4)));
            __m128i int_va2 = _mm_cvtepi8_epi32(_mm_loadu_si32((__m128i const*)(a + ao + 4 + 4)));
            __m128i int_va3 = _mm_cvtepi8_epi32(_mm_loadu_si32((__m128i const*)(a + ao + 4 + 4 + 4)));
            __m128i int_va4 = _mm_cvtepi8_epi32(_mm_loadu_si32((__m128i const*)(a + ao + 4 + 4 + 4 + 4)));
            __m128i int_va5 = _mm_cvtepi8_epi32(_mm_loadu_si32((__m128i const*)(a + ao + 4 + 4 + 4 + 4 + 4)));
            __m128i int_va6 = _mm_cvtepi8_epi32(_mm_loadu_si32((__m128i const*)(a + ao + 4 + 4 + 4 + 4 + 4 + 4)));
            __m128i int_va7 = _mm_cvtepi8_epi32(_mm_loadu_si32((__m128i const*)(a + ao + 4 + 4 + 4 + 4 + 4 + 4 + 4)));

            // Load 8 bytes into a 128-bit integer register
            __m128i int_vb0 = _mm_loadu_si32((__m128i const*)(b + bo));
            __m128i int_vb1 = _mm_loadu_si32((__m128i const*)(b + bo + 4));
            __m128i int_vb2 = _mm_loadu_si32((__m128i const*)(b + bo + 4 + 4));
            __m128i int_vb3 = _mm_loadu_si32((__m128i const*)(b + bo + 4 + 4 + 4));

            // Masked values
            __m128i first_4bits0 = _mm_and_si128(int_vb0, mask_first_4bits);
            __m128i first_4bits1 = _mm_and_si128(int_vb1, mask_first_4bits);
            __m128i first_4bits2 = _mm_and_si128(int_vb2, mask_first_4bits);
            __m128i first_4bits3 = _mm_and_si128(int_vb3, mask_first_4bits);

            // Shift first 4 bits to rightmost positions
            __m128i last_4bits0 = _mm_srli_epi16(int_vb0, 4);
            __m128i last_4bits1 = _mm_srli_epi16(int_vb1, 4);
            __m128i last_4bits2 = _mm_srli_epi16(int_vb2, 4);
            __m128i last_4bits3 = _mm_srli_epi16(int_vb3, 4);

            last_4bits0 = _mm_and_si128(last_4bits0, mask_first_4bits);
            last_4bits1 = _mm_and_si128(last_4bits1, mask_first_4bits);
            last_4bits2 = _mm_and_si128(last_4bits2, mask_first_4bits);
            last_4bits3 = _mm_and_si128(last_4bits3, mask_first_4bits);

            //Subtract 8 from each int
            first_4bits0 = _mm_sub_epi8(first_4bits0, eight);
            first_4bits1 = _mm_sub_epi8(first_4bits1, eight);
            first_4bits2 = _mm_sub_epi8(first_4bits2, eight);
            first_4bits3 = _mm_sub_epi8(first_4bits3, eight);

            last_4bits0 = _mm_sub_epi8(last_4bits0, eight);
            last_4bits1 = _mm_sub_epi8(last_4bits1, eight);
            last_4bits2 = _mm_sub_epi8(last_4bits2, eight);
            last_4bits3 = _mm_sub_epi8(last_4bits3, eight);

            // Extend bytes to 32-bit integers
            __m128i int_vb_ext_lo0 = _mm_cvtepi8_epi32(first_4bits0);
            __m128i int_vb_ext_lo1 = _mm_cvtepi8_epi32(first_4bits1);
            __m128i int_vb_ext_lo2 = _mm_cvtepi8_epi32(first_4bits2);
            __m128i int_vb_ext_lo3 = _mm_cvtepi8_epi32(first_4bits3);

            __m128i int_vb_ext_hi0 = _mm_cvtepi8_epi32(last_4bits0);
            __m128i int_vb_ext_hi1 = _mm_cvtepi8_epi32(last_4bits1);
            __m128i int_vb_ext_hi2 = _mm_cvtepi8_epi32(last_4bits2);
            __m128i int_vb_ext_hi3 = _mm_cvtepi8_epi32(last_4bits3);

            __m128i isum = _mm_mullo_epi32(int_va0, int_vb_ext_lo0);
            isum = _mm_add_epi32(isum, _mm_mullo_epi32(int_va1, int_vb_ext_lo1));
            isum = _mm_add_epi32(isum, _mm_mullo_epi32(int_va2, int_vb_ext_lo2));
            isum = _mm_add_epi32(isum, _mm_mullo_epi32(int_va3, int_vb_ext_lo3));

            isum = _mm_add_epi32(isum, _mm_mullo_epi32(int_va4, int_vb_ext_hi0));
            isum = _mm_add_epi32(isum, _mm_mullo_epi32(int_va5, int_vb_ext_hi1));
            isum = _mm_add_epi32(isum, _mm_mullo_epi32(int_va6, int_vb_ext_hi2));
            isum = _mm_add_epi32(isum, _mm_mullo_epi32(int_va7, int_vb_ext_hi3));

            // Convert these 32-bit integers to floats
            __m128 fsum = _mm_cvtepi32_ps(isum);
            sum = _mm_add_ps(sum, _mm_mul_ps(fsum, vb_f32));
        }
    }

    // Horizontal sum of the vector to get dot product
    __attribute__((aligned(16))) float result[4];
    _mm_store_ps(result, sum);

    float dot = 0.0;
    for(int i = 0; i < 4; ++i) {
        dot += result[i];
    }

    return dot;
}

float dot_product_q8_q4(int flags, const float* af, const char* a, int aoffset, const float *bf, const char* b, int boffset, int length) {
    return dot_product_q8_q4_128(af, a, aoffset, bf, b, boffset, length);
}
#endif

void dot_product_q8_q4_chunked(int flags, float *r, int roffset, const float* af, const char *a, int aoffset, const float *bf, const char* b, int boffset, int length, int bchunkstart, int bchunksize) {
     for (int c = bchunkstart; c < bchunkstart + bchunksize; c++) {
        int bo = boffset + (c * (length/2));
        r[roffset++] = dot_product_q8_q4(flags, af, a, aoffset, bf, b, bo, length);
     }
}
