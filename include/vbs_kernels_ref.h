#ifndef VBS_KERNELS_REF_H
#define VBS_KERNELS_REF_H

#include "vbs_gguf.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- GGML Block Type Definitions ---
#pragma pack(push, 1)
typedef struct { uint16_t d; int8_t qs[32]; } block_q8_0_t;
typedef struct { uint16_t d; uint8_t qs[16]; } block_q4_0_t;
typedef struct { uint16_t d; uint16_t m; uint8_t qs[16]; } block_q4_1_t;
typedef struct { uint16_t d; uint8_t qh[4]; uint8_t qs[16]; } block_q5_0_t;
typedef struct { uint16_t d; uint16_t m; uint8_t qh[4]; uint8_t qs[16]; } block_q5_1_t;
typedef struct { uint8_t scales[16]; uint8_t qs[64]; uint16_t d; uint16_t dmin; } block_q2_K_t;
typedef struct { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; uint16_t d; } block_q3_K_t;
typedef struct { uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qs[128]; } block_q4_K_t;
typedef struct { uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; } block_q5_K_t;
typedef struct { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; } block_q6_K_t;
#pragma pack(pop)

// --- Matrix-Vector Multiplication (Dequantize + Multiply + Accumulate) ---
void vbs_cpu_matvec_f32(const float *w, const float *x, float *y, size_t cols, size_t rows);
void vbs_cpu_matvec_f16(const uint16_t *w, const float *x, float *y, size_t cols, size_t rows);
void vbs_cpu_matvec_q4_0(const block_q4_0_t *w, const float *x, float *y, size_t cols, size_t rows);
void vbs_cpu_matvec_q4_1(const block_q4_1_t *w, const float *x, float *y, size_t cols, size_t rows);
void vbs_cpu_matvec_q5_0(const block_q5_0_t *w, const float *x, float *y, size_t cols, size_t rows);
void vbs_cpu_matvec_q5_1(const block_q5_1_t *w, const float *x, float *y, size_t cols, size_t rows);
void vbs_cpu_matvec_q8_0(const block_q8_0_t *w, const float *x, float *y, size_t cols, size_t rows);
void vbs_cpu_matvec_q2_K(const block_q2_K_t *w, const float *x, float *y, size_t cols, size_t rows);
void vbs_cpu_matvec_q3_K(const block_q3_K_t *w, const float *x, float *y, size_t cols, size_t rows);
void vbs_cpu_matvec_q4_K(const block_q4_K_t *w, const float *x, float *y, size_t cols, size_t rows);
void vbs_cpu_matvec_q5_K(const block_q5_K_t *w, const float *x, float *y, size_t cols, size_t rows);
void vbs_cpu_matvec_q6_K(const block_q6_K_t *w, const float *x, float *y, size_t cols, size_t rows);

// --- Primitive Operations ---
void vbs_cpu_rms_norm(const float *x, const float *w, float *y, size_t n, float eps);
void vbs_cpu_silu_mul(const float *gate, const float *up, float *out, size_t n);
void vbs_cpu_rope(float *x, size_t pos, size_t head_dim, size_t n_heads, float theta);
void vbs_cpu_softmax(float *x, size_t n);
void vbs_cpu_attention_gqa(
    const float *q, const uint16_t *k_cache, const uint16_t *v_cache,
    float *out, size_t seq_len, size_t n_q_heads, size_t n_kv_heads, size_t head_dim, float scale
);
void vbs_cpu_argmax(const float *logits, uint32_t *out_token, size_t vocab_size);
void vbs_cpu_sample_min_p(float *logits, uint32_t *out_token, size_t vocab_size, float min_p, float temperature, uint32_t seed);
void vbs_cpu_embedding(const float *w, float *y, uint32_t token, size_t dim);
void vbs_cpu_elementwise_add(float *out, const float *a, const float *b, size_t n);

// --- Dispatch Table ---
typedef void (*vbs_matvec_fn_t)(const void *w, const float *x, float *y, size_t cols, size_t rows);
vbs_matvec_fn_t vbs_get_matvec_kernel(vbs_ggml_type_t type);

// --- F16/BF16 Conversion Helpers ---
float vbs_f16_to_f32(uint16_t h);
uint16_t vbs_f32_to_f16(float f);
float vbs_bf16_to_f32(uint16_t h);

#ifdef __cplusplus
}
#endif

#endif
