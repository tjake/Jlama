#ifndef DOT_H
#define DOT_H

//Flags passes in at runtime
#define HAS_F16C 2
#define HAS_AVX2 4
#define IS_M_SERIES_MAC 8

// Info for quantization
#define Q8_BLOCK_SIZE 256
#define Q4_BLOCK_SIZE 32

//F16
float dot_product_f16(int flags, const short* a, int aoffset, const short* b, int boffset, int length);

void accumulate_f16(int flags, short* a, const short* b, int length);

//void saxpy_f16(int flags, float a, const short* x, int xoffset,  const short* y, int yoffset, int length);

//void sxpby_f16(int flags, float a, const short* x, int xoffset,  const short* y, int yoffset, int length);

//void scale_f16(int flags, float factor, const short* t, int offset, int length);


//F32
float dot_product_f32(int flags, const float* a, int aoffset, const float* b, int boffset, int length);

void accumulate_f32(int flags, float* a, const float* b, int length);

float dot_product_f16_q8(int flags, const short* a, int aoffset, const float *bf, const char* b, int boffset, int length);
float dot_product_f16_q4(int flags, const short* a, int aoffset, const float *bf, const char* b, int boffset, int length);

float dot_product_f32_q8(int flags, const float* a, int aoffset, const float *bf, const char* b, int boffset, int length);
float dot_product_f32_q4(int flags, const float* a, int aoffset, const float *bf, const char* b, int boffset, int length);


#endif