#ifndef GPU_DOT_H
#define GPU_DOT_H

#include <stdint.h>

//Returns the memory free on the GPU and the max group size
void init_gpu(int64_t *results);

//Returns a unique identifier for the tensor
int64_t register_tensor(const char *data, int size);

int64_t register_scratch_buffers(int params_size, int input_size, int result_size);

//Returns a unique identifier for the shader
int64_t register_shader(const char *data, int size);

//GEMM F32/BF16/Q4
void gpu_gemm(int64_t scratch_id, int64_t shader, const void *a, const void *a2, int aoffset, int alimit, int64_t bid, int64_t bid2, int boffset, int blimit, float *r, int roffset, int rlimit, int m, int n0, int n, int k, int lda, int ldb, int ldc, int m1_optimized);
void gpu_gemm_batch(int64_t shader, int batch_num, const void *a, const void *a2, int aoffset, const int64_t *bid, int boffset, float **r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc);

#endif