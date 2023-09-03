#ifndef DOT_H
#define DOT_H

float dot_product(const short* a, int aoffset, const short* b, int boffset, int length);

void accumulate(const short* a, const short* b, int length);

void saxpy(float a, const short* x, int xoffset,  const short* y, int yoffset, int length);

void scale(float factor, const short* t, int offset, int length);

void debug(const short* x, int length);


// CPU INFO

int cpu_has_avx(void);

int cpu_has_avx2(void);

int cpu_has_avx512(void);

int cpu_has_avx512_vbmi(void);

int cpu_has_avx512_vnni(void);

int cpu_has_fma(void);

int cpu_has_neon(void);

int cpu_has_arm_fma(void);

int cpu_has_f16c(void);

int cpu_has_fp16_va(void);

int cpu_has_wasm_simd(void);

int cpu_has_sse3(void);

int cpu_has_ssse3(void);

int cpu_has_vsx(void);

#endif