#ifndef DOT_H
#define DOT_H

//Flags passes in at runtime
#define HAS_F16C 2
#define HAS_AVX2 4
#define IS_M_SERIES_MAC 8

// Info for quantization
#define Q8_BLOCK_SIZE 32
#define Q4_BLOCK_SIZE 32

//F32
float dot_product_f32(int flags, const float* a, int aoffset, const float* b, int boffset, int length);
void dot_product_f32_chunked(int flags, float *r, const float* a, int aoffset, const float* b, int boffset, int length, int bchunkstart, int bchunksize);

float dot_product_f32_q8(int flags, const float* a, int aoffset, const float *bf, const char* b, int boffset, int length);
void dot_product_f32_q8_chunked(int flags, float *r, const float* a, int aoffset, const float *bf, const char* b, int boffset, int length, int bchunkstart, int bchunksize);

float dot_product_f32_q4(int flags, const float *a, int aoffset, const float *bf, const char* b, int boffset, int length);
void dot_product_f32_q4_chunked(int flags, float *r, const float* a, int aoffset, const float *bf, const char* b, int boffset, int length, int bchunkstart, int bchunksize);

//I8
float dot_product_q8(int flags, const float *af, const char* a, int aoffset, const float *bf, const char* b, int boffset, int length);
float dot_product_q8_q4(int flags, const float *af, const char* a, int aoffset, const float *bf, const char* b, int boffset, int length);
void dot_product_q8_q4_chunked(int flags, float *r, const float* af, const char *a, int aoffset, const float *bf, const char* b, int boffset, int length, int bchunkstart, int bchunksize);

#endif