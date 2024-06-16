#include <stdio.h>
#if defined(__ARM_NEON__)
#include <arm_neon.h>
#else
#include <immintrin.h>
#endif
#include <inttypes.h>
#include <math.h>
#include "vector_simd.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

//All params
struct gemm_params {
    int flags;
    const float* restrict af;
    const char* restrict a;
    int aoffset;
    const float* restrict bf;
    const char* restrict b;
    int boffset;
    float * restrict r;
    int roffset;
    int m;
    int n;
    int k;
    int lda;
    int ldaf;
    int ldb;
    int ldbf;
    int ldc;
} gemm_params;

void __attribute__((noinline)) gemm(int m0, int m, int n0, int n, void (*gemmPtr)(int, int, int, int, int, int, struct gemm_params), struct gemm_params params) {
    int mc, nc, mp, np;
    switch ((MIN(m - m0, 5) << 4) | MIN(n - n0, 5)) {
            case 0x55:
                mc = 5;
                nc = 5;
                break;
            case 0x45:
                mc = 4;
                nc = 5;
                break;
            case 0x54:
                mc = 5;
                nc = 4;
                break;
            case 0x44:
                mc = 4;
                nc = 4;
                break;
            case 0x53:
                mc = 5;
                nc = 3;
                break;
            case 0x35:
                mc = 3;
                nc = 5;
                break;
            case 0x43:
                mc = 4;
                nc = 3;
                break;
            case 0x34:
                mc = 3;
                nc = 4;
                break;
            case 0x52:
                mc = 5;
                nc = 2;
                break;
            case 0x33:
                mc = 3;
                nc = 3;
                break;
            case 0x25:
                mc = 2;
                nc = 5;
                break;
            case 0x42:
                mc = 4;
                nc = 2;
                break;
            case 0x24:
                mc = 2;
                nc = 4;
                break;
            case 0x32:
                mc = 3;
                nc = 2;
                break;
            case 0x23:
                mc = 2;
                nc = 3;
                break;
            case 0x51:
                mc = 5;
                nc = 1;
                break;
            case 0x41:
                mc = 4;
                nc = 1;
                break;
            case 0x22:
                mc = 2;
                nc = 2;
                break;
            case 0x15:
                mc = 1;
                nc = 5;
                break;
            case 0x14:
                mc = 1;
                nc = 4;
                break;
            case 0x31:
                mc = 3;
                nc = 1;
                break;
            case 0x13:
                mc = 1;
                nc = 3;
                break;
            case 0x21:
                mc = 2;
                nc = 1;
                break;
            case 0x12:
                mc = 1;
                nc = 2;
                break;
            case 0x11:
                mc = 1;
                nc = 1;
                break;
            default:
                return;
    }

    // If AVX512 is not supported, we can't use > 4x4 blocks
    if (((params.flags & HAS_AVX2) == 0 || (params.flags & IS_M_SERIES_MAC) == 0) && mc >= 4 && nc >= 4) {
        mc = 4;
        nc = 4;
    }

    gemmPtr(m0, m, n0, n, mc, nc, params);

    mp = m0 + (m - m0) / mc * mc;
    np = n0 + (n - n0) / nc * nc;
    gemm(mp, m, n0, np, gemmPtr, params);
    gemm(m0, mp, np, n, gemmPtr, params);
}

#if defined(__ARM_NEON__)
void __attribute__((noinline)) gemm_q8_q4_128_arm(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;
    // Mask to keep the first 4 bits of each byte
    int8x16_t mask_first_4bits = vdupq_n_u8(0x0f);
    //Subtract 8 from each byte to get signed values
    int8x16_t eight = vdupq_n_s8(0x8);
    int numBlocks = params.k / Q4_BLOCK_SIZE;

    __attribute__((aligned(16))) float scalef[4];

    // This fits on the stack (max of 5x5)
    for (int job = 0; job < tiles; ++job) {

        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        float32x4_t sums[RM][RN] __attribute__((aligned(32)));

        //Reset the sums to zero for this tile
        for (int i = 0; i < RM; i++) {
            for (int j = 0; j < RN; j++) {
                sums[i][j] = vdupq_n_f32(0.0f);
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;

            for (int i = 0; i < numBlocks; i += 4) { //128bits == 4floats
                int aoo = ao;
                int boo = bo;

                for (int mi = 0; mi < RM; ++mi) {
                    ao = aoo;
                    bo = boo;

                    // Load float32
                    float32x4_t ablock = vld1q_f32(params.af + (params.ldaf * (ii + mi) + (ao / Q4_BLOCK_SIZE)));
                    float32x4_t bblock = vld1q_f32(params.bf + (params.ldbf * (jj + ni) + ((bo*2) / Q4_BLOCK_SIZE)));
                    float32x4_t scaled = vmulq_f32(ablock, bblock);
                    vst1q_f32(scalef, scaled);

                    for(int j = 0; j < 4; j++, ao += 32, bo += 16) {
                        // Load 4 bytes into a 128-bit integer register
                        int8x16_t int_va0 = vld1q_s8((const signed char *)(params.a + params.lda * (ii + mi) + ao));
                        int8x16_t int_va1 = vld1q_s8((const signed char *)(params.a + params.lda * (ii + mi) + ao + 16));

                        // Load 8 bytes into a 128-bit integer register
                        int8x16_t int_vb0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8((const unsigned char *)(params.b + params.ldb * (jj + ni) + bo)),
                                            mask_first_4bits)), eight);

                        int8x16_t int_vb1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8((const unsigned char *)(params.b + params.ldb * (jj + ni) + bo)), 4)), eight);

                        sums[mi][ni] = vmlaq_n_f32(sums[mi][ni],
                            vcvtq_f32_s32(
                                vdotq_s32(
                                    vdotq_s32(vdupq_n_s32(0), int_va0, int_vb0),
                                    int_va1, int_vb1)), scalef[j]);
                    }
                }
            }
        }

        for (int mi = 0; mi < RM; ++mi) {
            for (int ni = 0; ni < RN; ++ni) {
                params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = vaddvq_f32(sums[mi][ni]);
            }
        }
    }
}
#else
void __attribute__((noinline)) gemm_q8_q4_128(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;
    // Mask to keep the first 4 bits of each byte
    __m128i mask_first_4bits = _mm_set1_epi8(0xF);
    //Subtract 8 from each byte to get signed values
    __m128i eight = _mm_set1_epi8(8);
    int numBlocks = params.k / Q4_BLOCK_SIZE;

    // This fits on the stack (max of 5x5)
    __attribute__((aligned(16))) float scalef[4];
    for (int job = 0; job < tiles; ++job) {

        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        __m128 sums[RM][RN] __attribute__((aligned(32)));

        //Reset the sums to zero for this tile
        for (int i = 0; i < RM; i++) {
            for (int j = 0; j < RN; j++) {
                sums[i][j] = _mm_setzero_ps();
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;

            for (int i = 0; i < numBlocks; i += 4) { //128bits == 4floats
                int aoo = ao;
                int boo = bo;

                for (int mi = 0; mi < RM; ++mi) {
                    ao = aoo;
                    bo = boo;

                    // Load float32
                     __m128 ablock = _mm_loadu_ps(params.af + (params.ldaf * (ii + mi) + (ao / Q4_BLOCK_SIZE)));
                     __m128 bblock = _mm_loadu_ps(params.bf + (params.ldbf * (jj + ni) + ((bo*2) / Q4_BLOCK_SIZE)));
                     __m128 scaled = _mm_mul_ps(ablock, bblock);
                     _mm_store_ps(scalef, scaled);

                    for(int j = 0; j < 4; j++, ao += 32, bo += 16) {
                        // Load 4 bytes into a 128-bit integer register
                        __m128i int_va0 = _mm_cvtepi8_epi32(_mm_loadu_si32((__m128i const*)(params.a + params.lda * (ii + mi) + ao)));
                        __m128i int_va1 = _mm_cvtepi8_epi32(_mm_loadu_si32((__m128i const*)(params.a + params.lda * (ii + mi) + ao + 4)));
                        __m128i int_va2 = _mm_cvtepi8_epi32(_mm_loadu_si32((__m128i const*)(params.a + params.lda * (ii + mi) + ao + 4 + 4)));
                        __m128i int_va3 = _mm_cvtepi8_epi32(_mm_loadu_si32((__m128i const*)(params.a + params.lda * (ii + mi) + ao + 4 + 4 + 4)));
                        __m128i int_va4 = _mm_cvtepi8_epi32(_mm_loadu_si32((__m128i const*)(params.a + params.lda * (ii + mi) + ao + 4 + 4 + 4 + 4)));
                        __m128i int_va5 = _mm_cvtepi8_epi32(_mm_loadu_si32((__m128i const*)(params.a + params.lda * (ii + mi) + ao + 4 + 4 + 4 + 4 + 4)));
                        __m128i int_va6 = _mm_cvtepi8_epi32(_mm_loadu_si32((__m128i const*)(params.a + params.lda * (ii + mi) + ao + 4 + 4 + 4 + 4 + 4 + 4)));
                        __m128i int_va7 = _mm_cvtepi8_epi32(_mm_loadu_si32((__m128i const*)(params.a + params.lda * (ii + mi) + ao + 4 + 4 + 4 + 4 + 4 + 4 + 4)));

                        // Load 8 bytes into a 128-bit integer register
                        __m128i int_vb0 = _mm_loadu_si32((__m128i const*)(params.b + params.ldb * (jj + ni) + bo));
                        __m128i int_vb1 = _mm_loadu_si32((__m128i const*)(params.b + params.ldb * (jj + ni) + bo + 4));
                        __m128i int_vb2 = _mm_loadu_si32((__m128i const*)(params.b + params.ldb * (jj + ni) + bo + 4 + 4));
                        __m128i int_vb3 = _mm_loadu_si32((__m128i const*)(params.b + params.ldb * (jj + ni) + bo + 4 + 4 + 4));

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

                        // broadcast the float32 version of 'factor' to all elements
                        __m128 vb_f32 = _mm_set1_ps(scalef[j]);
                        // Convert these 32-bit integers to floats
                        __m128 fsum = _mm_cvtepi32_ps(isum);
                        sums[mi][ni] = _mm_add_ps(sums[mi][ni], _mm_mul_ps(fsum, vb_f32));
                    }
                }
            }
        }


        for (int mi = 0; mi < RM; ++mi) {
            for (int ni = 0; ni < RN; ++ni) {
                __attribute__((aligned(16))) float result[4];
                _mm_store_ps(result, sums[mi][ni]);

                float dot = 0.0;
                for(int i = 0; i < 4; ++i) {
                    dot += result[i];
                }
                //fprintf(stderr, "ii: %d, ni: %d, jj: %d, mi: %d, ldc: %d\n", ii, ni, jj, mi, params.ldc);
                params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = dot;
            }
        }
    }
}

void __attribute__((noinline)) gemm_q8_q4_256(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;
    // Mask to keep the first 4 bits of each byte
    __m256i mask_first_4bits = _mm256_set1_epi8(0xF);
    //Subtract 8 from each byte to get signed values
    __m256i eight = _mm256_set1_epi8(8);
    int numBlocks = params.k / Q4_BLOCK_SIZE;

    // This fits on the stack (max of 5x5)
    __attribute__((aligned(16))) float scalef[8];
    for (int job = 0; job < tiles; ++job) {

        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        __m256 sums[RM][RN] __attribute__((aligned(32)));

        //Reset the sums to zero for this tile
        for (int i = 0; i < RM; i++) {
            for (int j = 0; j < RN; j++) {
                sums[i][j] = _mm256_setzero_ps();
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;

            for (int i = 0; i < numBlocks; i += 8) { //256bits == 8floats
                int aoo = ao;
                int boo = bo;

                for (int mi = 0; mi < RM; ++mi) {
                    ao = aoo;
                    bo = boo;

                    // Load float32
                     __m256 ablock = _mm256_loadu_ps(params.af + (params.ldaf * (ii + mi) + (ao / Q4_BLOCK_SIZE)));
                     __m256 bblock = _mm256_loadu_ps(params.bf + (params.ldbf * (jj + ni) + ((bo*2) / Q4_BLOCK_SIZE)));
                     __m256 scaled = _mm256_mul_ps(ablock, bblock);
                     _mm256_store_ps(scalef, scaled);

                    for(int j = 0; j < 8; j++, ao += 32, bo += 16) {
                        // Load 16 bytes into 2 128-bit integer registers
                        __m256i int_va1 = _mm256_loadu_si256((__m256i const*)(params.a + params.lda * (ii + mi) + ao));
                        __m256i int_va0 = _mm256_sign_epi8(int_va1, int_va1);

                        // Load 8 bytes into a 128-bit integer register
                        __m128i int_vb0 = _mm_loadu_si128((__m128i const*)(params.b + params.ldb * (jj + ni) + bo)); // Load 128 bits

                        __m256i vb0 = _mm256_and_si256(mask_first_4bits,
                                                       _mm256_insertf128_si256(_mm256_castsi128_si256(int_vb0),
                                                                               _mm_srli_epi16(int_vb0, 4), 1));

                        vb0 = _mm256_sign_epi8(_mm256_sub_epi8(vb0, eight), int_va1);

                        __m256i res = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(int_va0, vb0));
                        __m256 resf = _mm256_cvtepi32_ps(res);

                        // broadcast the float32 version of 'factor' to all elements
                        __m256 scale_f32 = _mm256_set1_ps(scalef[j]);

                        sums[mi][ni] = _mm256_fmadd_ps(scale_f32, resf, sums[mi][ni]);
                    }
                }
            }
        }

        for (int mi = 0; mi < RM; ++mi) {
            for (int ni = 0; ni < RN; ++ni) {
                __attribute__((aligned(16))) float result[8];
                _mm256_store_ps(result, sums[mi][ni]);

                float dot = 0.0;
                for(int i = 0; i < 8; ++i) {
                    dot += result[i];
                }
                //int idx = (params.ldc * (ii + mi)) + (jj + ni);
                //if (idx > params.roffset)
                //    fprintf(stderr, "ii: %d, ni: %d, jj: %d, mi: %d, ldc: %d, idx: %d, lim: %d\n", ii, ni, jj, mi, params.ldc,  idx, params.roffset);
                params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = dot;
            }
        }
    }
}

void __attribute__((noinline)) gemm_q8_q4_512(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
#if defined(__AVX512F__)
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;
    // Mask to keep the first 4 bits of each byte
    __m256i mask_first_4bits = _mm256_set1_epi8(0xF);
    //Subtract 8 from each byte to get signed values
    __m256i eight = _mm256_set1_epi8(8);
    int numBlocks = params.k / Q4_BLOCK_SIZE;

    // This fits on the stack (max of 5x5)
    __attribute__((aligned(16))) float scalef[8];
    for (int job = 0; job < tiles; ++job) {
        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        __m256 sums[RM][RN] __attribute__((aligned(32)));

        //Reset the sums to zero for this tile
        for (int i = 0; i < RM; i++) {
            for (int j = 0; j < RN; j++) {
                sums[i][j] = _mm256_setzero_ps();
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;

            for (int i = 0; i < numBlocks; i += 8) { //256bits == 8floats
                int aoo = ao;
                int boo = bo;

                for (int mi = 0; mi < RM; ++mi) {
                    ao = aoo;
                    bo = boo;

                    // Load float32
                     __m256 ablock = _mm256_loadu_ps(params.af + (params.ldaf * (ii + mi) + (ao / Q4_BLOCK_SIZE)));
                     __m256 bblock = _mm256_loadu_ps(params.bf + (params.ldbf * (jj + ni) + ((bo*2) / Q4_BLOCK_SIZE)));
                     __m256 scaled = _mm256_mul_ps(ablock, bblock);
                     _mm256_store_ps(scalef, scaled);

                    for(int j = 0; j < 8; j++, ao += 32, bo += 16) {
                        // Load 16 bytes into 2 128-bit integer registers
                        __m256i int_va1 = _mm256_loadu_si256((__m256i const*)(params.a + params.lda * (ii + mi) + ao));
                        __m256i int_va0 = _mm256_sign_epi8(int_va1, int_va1);

                        // Load 8 bytes into a 128-bit integer register
                        __m128i int_vb0 = _mm_loadu_si128((__m128i const*)(params.b + params.ldb * (jj + ni) + bo)); // Load 128 bits

                        __m256i vb0 = _mm256_and_si256(mask_first_4bits,
                                                       _mm256_insertf128_si256(_mm256_castsi128_si256(int_vb0),
                                                                               _mm_srli_epi16(int_vb0, 4), 1));

                        vb0 = _mm256_sign_epi8(_mm256_sub_epi8(vb0, eight), int_va1);

                        __m256i res;
                        #if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
                                res = _mm256_dpbusd_epi32(_mm256_setzero_si256(), int_va0, vb0);
                        #else
                                res = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(int_va0, vb0));
                        #endif

                        __m256 resf = _mm256_cvtepi32_ps(res);

                        // broadcast the float32 version of 'factor' to all elements
                        __m256 scale_f32 = _mm256_set1_ps(scalef[j]);

                        sums[mi][ni] = _mm256_fmadd_ps(scale_f32, resf, sums[mi][ni]);
                    }
                }
            }
        }

        for (int mi = 0; mi < RM; ++mi) {
            for (int ni = 0; ni < RN; ++ni) {
                // Horizontal sum of the vector to get dot product
                float dot = _mm512_reduce_add_ps(_mm512_castps256_ps512(sums[mi][ni]));
                params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = dot;
            }
        }
    }
#else
    gemm_q8_q4_256(m0, m, n0, n, RM, RN, params);
#endif
}
#endif //!ARM_NEON


void gemm_q8_q4(int flags, const float * restrict af, const char * restrict a, int aoffset, const float * restrict bf, const char* restrict b, int boffset, float * restrict r, int roffset, int m, int n0, int n, int k, int lda, int ldaf, int ldb, int ldbf, int ldc) {

    struct gemm_params p = {
                        .flags = flags,
                        .af = af,
                        .a = a,
                        .aoffset = aoffset,
                        .bf = bf,
                        .b = b,
                        .boffset = boffset,
                        .r = r,
                        .roffset = roffset,
                        .m = m,
                        .n = n,
                        .k = k,
                        .ldaf = ldaf,
                        .ldbf = ldbf,
                        .lda = lda,
                        .ldb = ldb,
                        .ldc = ldc
    };

    //fprintf(stderr, "m: %d, n0: %d, n: %d, k: %d, lda: %d, ldaf: %d, ldb: %d, ldbf: %d, ldc: %d\n", m, n0, n, k, lda, ldaf, ldb, ldbf, ldc);

#if !defined(__ARM_NEON__)
    ((flags & HAS_AVX2) != 0)
           ? gemm(0, m, n0, n0 + n, gemm_q8_q4_512, p)
           : gemm(0, m, n0, n0 + n, gemm_q8_q4_256, p);
#else
    gemm(0, m, n0, n0 + n, gemm_q8_q4_128_arm, p);
#endif
}

void gemm_q8_q4_batch(int flags, int batch_num, const float *af, const char *a, int aoffset, const float **bf, const char **b, int boffset, float **r, int roffset, int m, int n0, int n, int k, int lda, int ldaf, int ldb, int ldbf, int ldc)
{
    for (int i = 0; i < batch_num; i++) {
        gemm_q8_q4(flags, af, a, aoffset, bf[i], b[i], boffset, r[i], roffset, m, n0, n, k, lda, ldaf, ldb, ldbf, ldc);
    }
}

#if defined(__ARM_NEON__)
void gemm_f32_128_arm(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;

    // This fits on the stack (max of 5x5)
    float32x4_t sums[RM][RN] __attribute__((aligned(32)));

    for (int job = 0; job < tiles; ++job) {
        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        //Reset the sums to zero for this tile
        for (int i = 0; i < RM; i++) {
            for (int j = 0; j < RN; j++) {
                sums[i][j] = vdupq_n_f32(0.0f);
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;
            for(int j = 0; j < params.k; j += 4, ao += 4, bo += 4) { // 128bits == 4floats
                // Load float32
                float32x4_t vb = vld1q_f32(params.bf + params.ldb * (jj + ni) + bo);

                for (int mi = 0; mi < RM; ++mi) {
                    float32x4_t va = vld1q_f32(params.af + params.lda * (ii + mi) + ao);

                    // Multiply and accumulate
                    sums[mi][ni] = vmlaq_f32(sums[mi][ni], va, vb);
                }
            }
        }

        for (int mi = 0; mi < RM; ++mi) {
            for (int ni = 0; ni < RN; ++ni) {
                // Horizontal sum of the vector to get dot product
                params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = vaddvq_f32(sums[mi][ni]);
            }
        }
    }
}

#else
void gemm_f32_256(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;

    // This fits on the stack (max of 5x5)
    __m256 sums[RM][RN] __attribute__((aligned(32)));

    for (int job = 0; job < tiles; ++job) {
        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        //Reset the sums to zero for this tile
        for (int i = 0; i < RM; i++) {
            for (int j = 0; j < RN; j++) {
                sums[i][j] = _mm256_setzero_ps();
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;
            for(int j = 0; j < params.k; j += 8, ao += 8, bo += 8) { // 256bits == 8floats
                // Load float32
                __m256 vb = _mm256_loadu_ps(params.bf + params.ldb * (jj + ni) + bo);

                for (int mi = 0; mi < RM; ++mi) {
                    __m256 va = _mm256_loadu_ps(params.af + params.lda * (ii + mi) + ao);

                    // Multiply and accumulate
                    sums[mi][ni] = _mm256_fmadd_ps(va, vb, sums[mi][ni]);
                }
            }
        }

        for (int mi = 0; mi < RM; ++mi) {
            for (int ni = 0; ni < RN; ++ni) {
                // Horizontal sum of the vector to get dot product
                __attribute__((aligned(16))) float result[8];
                _mm256_store_ps(result, sums[mi][ni]);

                float dot = 0.0;
                for(int i = 0; i < 8; ++i) {
                    dot += result[i];
                }
                params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = dot;
            }
        }
    }
}

void gemm_f32_512(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
#if defined(__AVX512F__)
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;

    // This fits on the stack (max of 5x5)
    __m512 sums[RM][RN] __attribute__((aligned(32)));

    for (int job = 0; job < tiles; ++job) {
        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        //Reset the sums to zero for this tile
        for (int i = 0; i < RM; i++) {
            for (int j = 0; j < RN; j++) {
                sums[i][j] = _mm512_setzero_ps();
            }
        }

        for (int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;
            for(int j = 0; j < params.k; j += 16, ao += 16, bo += 16) { // 512bits == 16floats
                // Load float32
                __m512 vb = _mm512_loadu_ps(params.bf + params.ldb * (jj + ni) + bo);

                for (int mi = 0; mi < RM; ++mi) {
                    __m512 va = _mm512_loadu_ps(params.af + params.lda * (ii + mi) + ao);

                    // Multiply and accumulate
                    sums[mi][ni] = _mm512_fmadd_ps(va, vb, sums[mi][ni]);
                }
            }
        }

        for (int mi = 0; mi < RM; ++mi) {
            for (int ni = 0; ni < RN; ++ni) {
                // Horizontal sum of the vector to get dot product
                float r = _mm512_reduce_add_ps(sums[mi][ni]);
                params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = r;
            }
        }
    }
#else
    gemm_f32_256(m0, m, n0, n, RM, RN, params);
#endif
}
#endif //!ARM_NEON

void gemm_f32(int flags, const float *a, int aoffset, const float *b, int boffset, float *r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc)
{
    struct gemm_params p = {
                        .flags = flags,
                        .af = a,
                        .a = NULL,
                        .aoffset = aoffset,
                        .bf = b,
                        .b = NULL,
                        .boffset = boffset,
                        .r = r,
                        .roffset = roffset,
                        .m = m,
                        .n = n,
                        .k = k,
                        .ldaf = 0,
                        .ldbf = 0,
                        .lda = lda,
                        .ldb = ldb,
                        .ldc = ldc
    };

#if !defined(__ARM_NEON__)
    ((flags & HAS_AVX2) != 0)
           ? gemm(0, m, n0, n0 + n, gemm_f32_512, p)
           : gemm(0, m, n0, n0 + n, gemm_f32_256, p);
#else
    gemm(0, m, n0, n0 + n, gemm_f32_128_arm, p);
#endif
}

void gemm_f32_batch(int flags, int batch_num, const float *a, int aoffset, const float **b, int boffset, float **r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc)
{
    for (int i = 0; i < batch_num; i++) {
        gemm_f32(flags, a, aoffset, b[i], boffset, r[i], roffset, m, n0, n, k, lda, ldb, ldc);
    }
}


#if !defined(__ARM_NEON__)
void gemm_f32_q4_256(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;
    // Mask to keep the first 4 bits of each byte
    __m128i mask_first_4bits = _mm_set1_epi8(0xF);
    //Subtract 8 from each byte to get signed values
    __m128i eight = _mm_set1_epi8(8);

    // This fits on the stack (max of 5x5)
    __m256 sums[RM][RN];

    for (int job = 0; job < tiles; ++job) {
        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        //Reset the sums to zero for this tile
        for (int i = 0; i < RM; i++) {
            for (int j = 0; j < RN; j++) {
                sums[i][j] = _mm256_setzero_ps();
            }
        }

        for(int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;

            for(int j = 0; j < params.k; j += 32, ao += 32, bo += 16) {
                for (int mi = 0; mi < RM; ++mi) {
                        // Load float32
                        __m256 va0 = _mm256_loadu_ps(params.af + params.lda * (ii + mi) + ao);
                        __m256 va1 = _mm256_loadu_ps(params.af + params.lda * (ii + mi) + ao + 8);
                        __m256 va2 = _mm256_loadu_ps(params.af + params.lda * (ii + mi) + ao + 8 + 8);
                        __m256 va3 = _mm256_loadu_ps(params.af + params.lda * (ii + mi) + ao + 8 + 8 + 8);

                        // Load float32
                        float bfactor = params.bf[params.ldbf * (jj + ni) + ((bo*2) / Q4_BLOCK_SIZE)];

                        // broadcast the float32 version of 'factor' to all elements
                        __m256 vb_f32 = _mm256_set1_ps(bfactor);

                        // Load 8 bytes into a 128-bit integer register
                        __m128i int_vb0 = _mm_loadl_epi64((__m128i const*)(params.b + params.ldb * (jj + ni) + bo)); // Load lower 64 bits
                        __m128i int_vb1 = _mm_loadl_epi64((__m128i const*)(params.b + params.ldb * (jj + ni) + bo + 8)); // Load lower 64 bits

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
                        sums[mi][ni] = _mm256_fmadd_ps(va0, vb_scaled_lo0, sums[mi][ni]);
                        sums[mi][ni] = _mm256_fmadd_ps(va1, vb_scaled_lo1, sums[mi][ni]);
                        sums[mi][ni] = _mm256_fmadd_ps(va2, vb_scaled_hi0, sums[mi][ni]);
                        sums[mi][ni] = _mm256_fmadd_ps(va3, vb_scaled_hi1, sums[mi][ni]);
                }
            }
        }

        for (int mi = 0; mi < RM; ++mi) {
            for (int ni = 0; ni < RN; ++ni) {
                // Horizontal sum of the vector to get dot product
                __attribute__((aligned(16))) float result[8];
                _mm256_store_ps(result, sums[mi][ni]);

                float dot = 0.0;
                for(int i = 0; i < 8; ++i) {
                    dot += result[i];
                }
                //if (params.roffset > 0)
                //    fprintf(stderr, "ii: %d, ni: %d, jj: %d, mi: %d, ldc: %d, roffset: %d\n", ii, ni, jj, mi, params.ldc, params.roffset);
                params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = dot;
            }
        }
    }
}

void gemm_f32_q4_512(int m0, int m, int n0, int n, int RM, int RN, struct gemm_params params) {
#if defined(__AVX512F__)
    int ytiles = (m - m0) / RM;
    int xtiles = (n - n0) / RN;
    int tiles = xtiles * ytiles;
    // Mask to keep the first 4 bits of each byte
    __m128i mask_first_4bits = _mm_set1_epi8(0xF);
    //Subtract 8 from each byte to get signed values
    __m128i eight = _mm_set1_epi8(8);
    //int numBlocks = params.k / Q4_BLOCK_SIZE;

    // This fits on the stack (max of 5x5)
    __m512 sums[RM][RN];

    for (int job = 0; job < tiles; ++job) {
        int ii = m0 + job / xtiles * RM;
        int jj = n0 + job % xtiles * RN;

        //Reset the sums to zero for this tile
        for (int i = 0; i < RM; i++) {
            for (int j = 0; j < RN; j++) {
                sums[i][j] = _mm512_setzero_ps();
            }
        }

        for(int ni = 0; ni < RN; ++ni) {
            int ao = params.aoffset;
            int bo = params.boffset;

            for(int j = 0; j < params.k; j += 32, ao += 32, bo += 16) {
                for (int mi = 0; mi < RM; ++mi) {
                        // Load float32
                        __m512 va0 = _mm512_loadu_ps(params.af + params.lda * (ii + mi) + ao);
                        __m512 va1 = _mm512_loadu_ps(params.af + params.lda * (ii + mi) + ao + 16);

                        // Load float32
                        float bfactor = params.bf[params.ldbf * (jj + ni) + ((bo*2) / Q4_BLOCK_SIZE)];

                        // broadcast the float32 version of 'factor' to all elements
                        __m512 vb_f32 = _mm512_set1_ps(bfactor);

                        // Load 8 bytes into a 128-bit integer register
                        __m128i int_vb0 = _mm_loadu_si128((__m128i const*)(params.b + params.ldb * (jj + ni) + bo)); // Load 128 bits

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

                        __m512 float_vb_lo0 = _mm512_cvtepi32_ps(int_vb_ext_lo0);
                        __m512 float_vb_hi0 = _mm512_cvtepi32_ps(int_vb_ext_hi0);

                        // Perform the scaling
                        __m512 vb_scaled_lo0 = _mm512_mul_ps(vb_f32, float_vb_lo0);
                        __m512 vb_scaled_hi0 = _mm512_mul_ps(vb_f32, float_vb_hi0);

                        // Multiply and accumulate
                        sums[mi][ni] = _mm512_fmadd_ps(va0, vb_scaled_lo0, sums[mi][ni]);
                        sums[mi][ni] = _mm512_fmadd_ps(va1, vb_scaled_hi0, sums[mi][ni]);
                }
            }
        }

        for (int mi = 0; mi < RM; ++mi) {
            for (int ni = 0; ni < RN; ++ni) {
                // Horizontal sum of the vector to get dot product
                float r = _mm512_reduce_add_ps(sums[mi][ni]);
                params.r[(params.ldc * (ii + mi)) + (jj + ni) - params.roffset] = r;
            }
        }
   }

#else
    gemm_f32_q4_256(m0, m, n0, n, RM, RN, params);
#endif
}
#endif //!ARM_NEON

void gemm_f32_q4(int flags, const float *a, int aoffset, const float *bf, const char* b, int boffset, float *r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldbf, int ldc)
{
#if !defined(__ARM_NEON__)
    struct gemm_params p = {
                        .flags = flags,
                        .af = a,
                        .a = NULL,
                        .aoffset = aoffset,
                        .bf = bf,
                        .b = b,
                        .boffset = boffset,
                        .r = r,
                        .roffset = roffset,
                        .m = m,
                        .n = n,
                        .k = k,
                        .ldaf = 0,
                        .ldbf = ldbf,
                        .lda = lda,
                        .ldb = ldb,
                        .ldc = ldc
    };

    ((flags & HAS_AVX2) != 0)
           ? gemm(0, m, n0, n0 + n, gemm_f32_q4_512, p)
           : gemm(0, m, n0, n0 + n, gemm_f32_q4_256, p);
#endif
}

void gemm_f32_q4_batch(int flags, int batch_num, const float *a, int aoffset, const float **bf, const char **b, int boffset, float **r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldbf, int ldc)
{
    for (int i = 0; i < batch_num; i++) {
        gemm_f32_q4(flags, a, aoffset, bf[i], b[i], boffset, r[i], roffset, m, n0, n, k, lda, ldb, ldbf, ldc);
    }
}


