#ifndef GPU_DOT_H
#define GPU_DOT_H

//Returns the memory free on the GPU and the max group size
void init_gpu(long *results);

//Returns a unique identifier for the tensor
long register_tensor(const char *data, int size);

long register_scratch_buffers(int params_size, int input_size, int result_size);

//Returns a unique identifier for the shader
long register_shader(const char *data, int size);

//GEMM F32
void gemm(long scratch_id, long shader, const float *a, int aoffset, int alimit, long bid, int boffset, int blimit, float *r, int roffset, int rlimit, int m, int n0, int n, int k, int lda, int ldb, int ldc);
void gemm_batch(long shader, int batch_num, const float *a, int aoffset, const long *bid, int boffset, float **r, int roffset, int m, int n0, int n, int k, int lda, int ldb, int ldc);

#endif