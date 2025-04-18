#ifndef GPU_DOT_H
#define GPU_DOT_H

//Returns the memory free on the GPU and the max group size
void init_gpu(long *results);

//Returns a unique identifier for the tensor
long register_tensor(const char *data, int size);

long register_scratch_buffers(int params_size, int input_size, int result_size);

//Returns a unique identifier for the shader
long register_shader(const char *data, int size);

//GEMM F32/BF16/Q4
void gpu_gemm(long scratch_id, long shader, const void *a, const void *a2, int aoffset, int alimit, long bid, long bid2, int boffset, int blimit, float *r, int roffset, int rlimit, int m, int n0, int n, int k, int lda, int ldb, int ldc, int m1_optimized);
void gpu_gemm_batch(long shader, int batch_num, const void *a, const void *a2, int aoffset, const long *bid, int boffset, float **r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc);

#endif