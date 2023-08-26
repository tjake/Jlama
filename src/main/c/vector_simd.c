#include <stdio.h>
#include <immintrin.h>

float dot_product(const short* a, int aoffset, const short* b, int boffset, int length) {
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

    // Capture tail
    //for(; ao < alim && bo < blim; ao++, bo++) {
    //    dot += a[ao] * b[bo];
    //}

    return dot;
}

void accumulate(short* a, const short* b, int length) {
    int i = 0;
    for (; i < length; i += 8) {
        // Load and convert float16 data to float32 using F16C
        __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(a + i)));
        __m256 vb = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(b + i)));

        // Add the vectors
        __m256 sum = _mm256_add_ps(va, vb);

        // Convert the sum back to float16
        __m128i sum_f16 = _mm256_cvtps_ph(sum, 0);

        // Store the result back into array 'a'
        _mm_storeu_si128((__m128i*)(a + i), sum_f16);
    }

    //Tail
    for (; i < length; i++) {
        a[i] += b[i];
    }
}


void saxpy(float a, short* x, int xoffset,  short* y, int yoffset, int length) {
    // broadcast the float32 version of 'a' to all elements
    __m256 va_f32 = _mm256_set1_ps(a);

    int xoff = xoffset;
    int yoff = yoffset;

    int xlen = xoffset + length;
    int ylen = yoffset + length;

    for (; xoff < xlen && yoff < ylen; xoff += 8, yoff += 8) {
        // Load and convert float16 data of x and y to float32
        __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(x + xoff)));
        __m256 vy = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(y + yoff)));

        // Perform the saxpy operation: y = a * x + y
        __m256 result = _mm256_fmadd_ps(va_f32, vx, vy);

        // Convert the result back to float16
        __m128i result_f16 = _mm256_cvtps_ph(result, 0);

        // Store the result back into array 'y'
        _mm_storeu_si128((__m128i*)(y + yoff), result_f16);
    }

    //tail
    for (; xoff < xlen && yoff < ylen; xoff++, yoff++) {
        y[yoff] = a * x[xoff] + y[yoff];
    }

}

void scale(float factor, const short* t, int toffset, int length) {
     // broadcast the float32 version of 'factor' to all elements
     __m256 va_f32 = _mm256_set1_ps(factor);

     int toff = toffset;
     int tlen = toffset + length;

     for (; toff < tlen ; toff += 8) {
         // Load and convert float16 data of t to float32
         __m256 vt = _mm256_cvtph_ps(_mm_loadu_si128((__m128i const*)(t + toff)));

         // Perform the scaling
         __m256 result = _mm256_mul_ps(va_f32, vt);

         // Convert the result back to float16
         __m128i result_f16 = _mm256_cvtps_ph(result, 0);

         // Store the result back into array 't'
         _mm_storeu_si128((__m128i*)(t + toff), result_f16);
     }
}

void debug(const short* a, int length) {
    for (int i = 0; i < length; i++) {
        printf("%d %d\n", i, a[i]);
    }
}