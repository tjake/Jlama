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
void dot_product_f32_chunked(int flags, float *r, int roffset, const float* a, int aoffset, const float* b, int boffset, int length, int bchunkstart, int bchunksize);
void dot_product_f32_batch_chunked(int flags, int batch_num, void **r /*list of addresses*/, int roffset, const float* a, int aoffset, void **b /*list of addresses*/, int boffset, int length, int bchunkstart, int bchunksize);

float dot_product_f32_q8(int flags, const float* a, int aoffset, const float *bf, const char* b, int boffset, int length);
void dot_product_f32_q8_chunked(int flags, float *r, int roffset, const float* a, int aoffset, const float *bf, const char* b, int boffset, int length, int bchunkstart, int bchunksize);
void dot_product_f32_q8_batch_chunked(int flags, int batch_num, void **r, int roffset, const float* a, int aoffset, const void **bf, const void **b, int boffset, int length, int bchunkstart, int bchunksize);

float dot_product_f32_q4(int flags, const float *a, int aoffset, const float *bf, const char* b, int boffset, int length);
void dot_product_f32_q4_chunked(int flags, float *r, int roffset, const float* a, int aoffset, const float *bf, const char* b, int boffset, int length, int bchunkstart, int bchunksize);
void dot_product_f32_q4_batch_chunked(int flags, int batch_num, void **r, int roffset, const float* a, int aoffset, const void **bf, const void **b, int boffset, int length, int bchunkstart, int bchunksize);

//I8
float dot_product_q8(int flags, const float *af, const char* a, int aoffset, const float *bf, const char* b, int boffset, int length);
float dot_product_q8_q4(int flags, const float *af, const char* a, int aoffset, const float *bf, const char* b, int boffset, int length);
void dot_product_q8_q4_chunked(int flags, float *r, int roffset, const float* af, const char *a, int aoffset, const float *bf, const char* b, int boffset, int length, int bchunkstart, int bchunksize);
void dot_product_q8_q4_batch_chunked(int flags, int batch_num, void **r, int roffset, const float* af, const char *a, int aoffset, const void **bf, const void **b, int boffset, int length, int bchunkstart, int bchunksize);

//GEMM I8 Q4
void gemm_q8_q4(int flags, const float * restrict af, const char* restrict a, int aoffset, const float * restrict bf, const char* restrict b, int boffset, float * restrict r, int roffset, int m, int n0, int n, int k, int lda, int ldaf, int ldb, int ldbf, int ldc);
void gemm_q8_q4_batch(int flags, int batch_num, const float * restrict af, const char * restrict a, int aoffset, const float ** restrict bf, const char ** restrict b, int boffset, float ** restrict r, int roffset, int m, int n0, int n, int k, int lda, int ldaf, int ldb, int ldbf, int ldc);

//GEMM F32
void gemm_f32(int flags, const float *a, int aoffset, const float *b, int boffset, float *r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc);
void gemm_f32_batch(int flags, int batch_num, const float *a, int aoffset, const float **b, int boffset, float **r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc);

//GEMM F32 Q4
void gemm_f32_q4(int flags, const float *a, int aoffset, const float *bf, const char* b, int boffset, float *r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldbf, int ldc);
void gemm_f32_q4_batch(int flags, int batch_num, const float *a, int aoffset, const float **bf, const char **b, int boffset, float **r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldbf, int ldc);
#endif