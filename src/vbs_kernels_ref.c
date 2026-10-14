#include "vbs_kernels_ref.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>

#if VBS_HAVE_AVX2
#include <immintrin.h>
#endif

#if VBS_HAVE_NEON
#include <arm_neon.h>
#endif

// =====================================================================
// F16 <-> F32 Conversion
// =====================================================================
float vbs_f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h & 0x7C00) >> 10;
    uint32_t mant = (h & 0x03FF);
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) f = sign;
        else {
            exp = 1;
            while (!(mant & 0x0400)) { mant <<= 1; exp--; }
            mant &= 0x03FF;
            f = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        f = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &f, sizeof(out));
    return out;
}

float vbs_bf16_to_f32(uint16_t h) {
    uint32_t f = ((uint32_t)h) << 16;
    float out;
    memcpy(&out, &f, sizeof(out));
    return out;
}

// =====================================================================
// Matrix-Vector Multiplication Kernels
// =====================================================================

void vbs_cpu_matvec_f32(const float *w, const float *x, float *y, size_t cols, size_t rows) {
    for (size_t r = 0; r < rows; r++) {
        float sum = 0.0f;
#if VBS_HAVE_AVX2
        __m256 acc = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 7 < cols; i += 8) {
            __m256 wv = _mm256_loadu_ps(w + r * cols + i);
            __m256 xv = _mm256_loadu_ps(x + i);
            acc = _mm256_fmadd_ps(wv, xv, acc);
        }
        float tmp[8];
        _mm256_storeu_ps(tmp, acc);
        for (int k = 0; k < 8; k++) sum += tmp[k];
        for (; i < cols; i++) sum += w[r * cols + i] * x[i];
#else
        for (size_t i = 0; i < cols; i++) sum += w[r * cols + i] * x[i];
#endif
        y[r] = sum;
    }
}

void vbs_cpu_matvec_f16(const uint16_t *w, const float *x, float *y, size_t cols, size_t rows) {
    for (size_t r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (size_t i = 0; i < cols; i++) {
            sum += vbs_f16_to_f32(w[r * cols + i]) * x[i];
        }
        y[r] = sum;
    }
}

void vbs_cpu_matvec_q4_0(const block_q4_0_t *w, const float *x, float *y, size_t cols, size_t rows) {
    size_t blocks_per_row = cols / 32;
    for (size_t r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (size_t bb = 0; bb < blocks_per_row; bb++) {
            const block_q4_0_t *b = &w[r * blocks_per_row + bb];
            float d = vbs_f16_to_f32(b->d);
            float acc = 0.0f;
            for (size_t i = 0; i < 16; i++) {
                int q0 = (b->qs[i] & 0x0F) - 8;
                int q1 = (b->qs[i] >> 4)   - 8;
                acc += (float)q0 * x[bb * 32 + i];
                acc += (float)q1 * x[bb * 32 + i + 16];
            }
            sum += d * acc;
        }
        y[r] = sum;
    }
}

void vbs_cpu_matvec_q4_1(const block_q4_1_t *w, const float *x, float *y, size_t cols, size_t rows) {
    size_t blocks_per_row = cols / 32;
    for (size_t r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (size_t bb = 0; bb < blocks_per_row; bb++) {
            const block_q4_1_t *b = &w[r * blocks_per_row + bb];
            float d = vbs_f16_to_f32(b->d);
            float m = vbs_f16_to_f32(b->m);
            float acc = 0.0f, xs = 0.0f;
            for (size_t i = 0; i < 16; i++) {
                int q0 = (b->qs[i] & 0x0F);
                int q1 = (b->qs[i] >> 4);
                acc += (float)q0 * x[bb * 32 + i];
                acc += (float)q1 * x[bb * 32 + i + 16];
                xs  += x[bb * 32 + i] + x[bb * 32 + i + 16];
            }
            sum += d * acc + m * xs;
        }
        y[r] = sum;
    }
}

void vbs_cpu_matvec_q5_0(const block_q5_0_t *w, const float *x, float *y, size_t cols, size_t rows) {
    size_t blocks_per_row = cols / 32;
    for (size_t r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (size_t bb = 0; bb < blocks_per_row; bb++) {
            const block_q5_0_t *b = &w[r * blocks_per_row + bb];
            float d = vbs_f16_to_f32(b->d);
            uint32_t qh = (uint32_t)b->qh[0] | ((uint32_t)b->qh[1] << 8) |
                          ((uint32_t)b->qh[2] << 16) | ((uint32_t)b->qh[3] << 24);
            float acc = 0.0f;
            for (size_t i = 0; i < 16; i++) {
                int h0 = (qh >> i) & 1;
                int h1 = (qh >> (i + 16)) & 1;
                int q0 = ((b->qs[i] & 0x0F) | (h0 << 4)) - 16;
                int q1 = ((b->qs[i] >> 4)   | (h1 << 4)) - 16;
                acc += (float)q0 * x[bb * 32 + i];
                acc += (float)q1 * x[bb * 32 + i + 16];
            }
            sum += d * acc;
        }
        y[r] = sum;
    }
}

void vbs_cpu_matvec_q5_1(const block_q5_1_t *w, const float *x, float *y, size_t cols, size_t rows) {
    size_t blocks_per_row = cols / 32;
    for (size_t r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (size_t bb = 0; bb < blocks_per_row; bb++) {
            const block_q5_1_t *b = &w[r * blocks_per_row + bb];
            float d = vbs_f16_to_f32(b->d);
            float m = vbs_f16_to_f32(b->m);
            uint32_t qh = (uint32_t)b->qh[0] | ((uint32_t)b->qh[1] << 8) |
                          ((uint32_t)b->qh[2] << 16) | ((uint32_t)b->qh[3] << 24);
            float acc = 0.0f, xs = 0.0f;
            for (size_t i = 0; i < 16; i++) {
                int h0 = (qh >> i) & 1;
                int h1 = (qh >> (i + 16)) & 1;
                int q0 = (b->qs[i] & 0x0F) | (h0 << 4);
                int q1 = (b->qs[i] >> 4)   | (h1 << 4);
                acc += (float)q0 * x[bb * 32 + i];
                acc += (float)q1 * x[bb * 32 + i + 16];
                xs  += x[bb * 32 + i] + x[bb * 32 + i + 16];
            }
            sum += d * acc + m * xs;
        }
        y[r] = sum;
    }
}

void vbs_cpu_matvec_q8_0(const block_q8_0_t *w, const float *x, float *y, size_t cols, size_t rows) {
    size_t blocks_per_row = cols / 32;
    for (size_t r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (size_t bb = 0; bb < blocks_per_row; bb++) {
            const block_q8_0_t *b = &w[r * blocks_per_row + bb];
            float d = vbs_f16_to_f32(b->d);
            int32_t acc = 0;
            for (size_t i = 0; i < 32; i++) {
                acc += (int32_t)b->qs[i] * (int32_t)(x[bb * 32 + i] * 127.0f);
            }
            sum += d * (float)acc / 127.0f;
        }
        y[r] = sum;
    }
}

// --- K-Quants Helper: Extract 6-bit scale/min ---
static inline void q_K_scale_min(const uint8_t *sc, int j, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = sc[j] & 63;
        *m = sc[j + 4] & 63;
    } else {
        *d = (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
        *m = (sc[j + 4] >> 4)   | ((sc[j] >> 6) << 4);
    }
}

void vbs_cpu_matvec_q2_K(const block_q2_K_t *w, const float *x, float *y, size_t cols, size_t rows) {
    size_t blocks_per_row = cols / 256;
    for (size_t r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (size_t bb = 0; bb < blocks_per_row; bb++) {
            const block_q2_K_t *b = &w[r * blocks_per_row + bb];
            float d = vbs_f16_to_f32(b->d);
            float dmin = vbs_f16_to_f32(b->dmin);
            for (int el = 0; el < 256; el++) {
                int h = el / 128, m = (el % 128) / 32, l = el % 32;
                int q = (b->qs[h * 32 + l] >> (2 * m)) & 3;
                int si = h * 8 + m * 2 + (l / 16);
                float sc = (float)(b->scales[si] & 0xF);
                float mn = (float)(b->scales[si] >> 4);
                sum += (d * sc * (float)q - dmin * mn) * x[bb * 256 + el];
            }
        }
        y[r] = sum;
    }
}

void vbs_cpu_matvec_q3_K(const block_q3_K_t *w, const float *x, float *y, size_t cols, size_t rows) {
    size_t bpr = cols / 256;
    for (size_t r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (size_t bb = 0; bb < bpr; bb++) {
            const block_q3_K_t *b = &w[r * bpr + bb];
            float d = vbs_f16_to_f32(b->d);
            for (int el = 0; el < 256; el++) {
                int ni = el / 128, j = (el / 32) % 4, l = el % 32;
                int lo = (b->qs[ni * 32 + l] >> (2 * j)) & 3;
                int hb = (b->hmask[l] >> (j + 4 * ni)) & 1;
                int sb = 8 * ni + 2 * j + (l / 16);
                int8_t scale = (int8_t)(((b->scales[sb >> 1] >> (4 * (sb & 1))) & 0xF) - 8);
                sum += d * (float)scale * (float)(lo - (hb ? 0 : 4)) * x[bb * 256 + el];
            }
        }
        y[r] = sum;
    }
}

void vbs_cpu_matvec_q4_K(const block_q4_K_t *w, const float *x, float *y, size_t cols, size_t rows) {
    size_t bpr = cols / 256;
    for (size_t r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (size_t bb = 0; bb < bpr; bb++) {
            const block_q4_K_t *b = &w[r * bpr + bb];
            float d = vbs_f16_to_f32(b->d);
            float dm = vbs_f16_to_f32(b->dmin);
            for (int sub = 0; sub < 8; sub++) {
                uint8_t sc, mn;
                q_K_scale_min(b->scales, sub, &sc, &mn);
                for (int l = 0; l < 32; l++) {
                    int qb = (sub / 2) * 32 + l;
                    int qs = (sub & 1) * 4;
                    float q = (float)((b->qs[qb] >> qs) & 0xF);
                    sum += (d * (float)sc * q - dm * (float)mn) * x[bb * 256 + sub * 32 + l];
                }
            }
        }
        y[r] = sum;
    }
}

void vbs_cpu_matvec_q5_K(const block_q5_K_t *w, const float *x, float *y, size_t cols, size_t rows) {
    size_t bpr = cols / 256;
    for (size_t r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (size_t bb = 0; bb < bpr; bb++) {
            const block_q5_K_t *b = &w[r * bpr + bb];
            float d = vbs_f16_to_f32(b->d);
            float dm = vbs_f16_to_f32(b->dmin);
            for (int sub = 0; sub < 8; sub++) {
                uint8_t sc, mn;
                q_K_scale_min(b->scales, sub, &sc, &mn);
                for (int l = 0; l < 32; l++) {
                    int qb = (sub / 2) * 32 + l;
                    int qs = (sub & 1) * 4;
                    float q4 = (float)((b->qs[qb] >> qs) & 0xF);
                    float qh = (float)((b->qh[l] >> sub) & 1) * 16.0f;
                    sum += (d * (float)sc * (q4 + qh) - dm * (float)mn) * x[bb * 256 + sub * 32 + l];
                }
            }
        }
        y[r] = sum;
    }
}

void vbs_cpu_matvec_q6_K(const block_q6_K_t *w, const float *x, float *y, size_t cols, size_t rows) {
    size_t bpr = cols / 256;
    for (size_t r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (size_t bb = 0; bb < bpr; bb++) {
            const block_q6_K_t *b = &w[r * bpr + bb];
            float d = vbs_f16_to_f32(b->d);
            for (int el = 0; el < 256; el++) {
                int h = el / 128, q = (el % 128) / 32, l = el % 32;
                int qi = (q < 2) ? (h * 64 + q * 32 + l) : (h * 64 + (q - 2) * 32 + l);
                int qs = (q < 2) ? 0 : 4;
                int lo = (b->ql[qi] >> qs) & 0xF;
                int hi = (b->qh[h * 32 + l] >> (q * 2)) & 3;
                int si = h * 8 + q * 2 + (l / 16);
                sum += d * (float)b->scales[si] * (float)((lo | (hi << 4)) - 32) * x[bb * 256 + el];
            }
        }
        y[r] = sum;
    }
}

// =====================================================================
// Primitive Operations
// =====================================================================

void vbs_cpu_rms_norm(const float *x, const float *w, float *y, size_t n, float eps) {
    float ss = 0.0f;
    for (size_t i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (size_t i = 0; i < n; i++) y[i] = x[i] * inv * w[i];
}

void vbs_cpu_silu_mul(const float *gate, const float *up, float *out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

void vbs_cpu_rope(float *x, size_t pos, size_t head_dim, size_t n_heads, float theta) {
    for (size_t h = 0; h < n_heads; h++) {
        for (size_t p = 0; p < head_dim / 2; p++) {
            size_t i0 = h * head_dim + p * 2;
            size_t i1 = i0 + 1;
            float freq = powf(theta, -(float)(p * 2) / (float)head_dim);
            float a = (float)pos * freq;
            float c = cosf(a), s = sinf(a);
            float v0 = x[i0], v1 = x[i1];
            x[i0] = v0 * c - v1 * s;
            x[i1] = v1 * c + v0 * s;
        }
    }
}

void vbs_cpu_softmax(float *x, size_t n) {
    float max_val = -FLT_MAX;
    for (size_t i = 0; i < n; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv = 1.0f / sum;
    for (size_t i = 0; i < n; i++) x[i] *= inv;
}

void vbs_cpu_attention_gqa(
    const float *q, const uint16_t *k_cache, const uint16_t *v_cache,
    float *out, size_t seq_len, size_t n_q_heads, size_t n_kv_heads, size_t head_dim, float scale
) {
    size_t ratio = n_q_heads / n_kv_heads;
    size_t kv_dim = n_kv_heads * head_dim;
    float *scores = (float *)malloc(seq_len * sizeof(float));

    for (size_t h = 0; h < n_q_heads; h++) {
        size_t kh = h / ratio;
        const float *qh = q + h * head_dim;

        for (size_t t = 0; t < seq_len; t++) {
            const uint16_t *kh_ptr = k_cache + t * kv_dim + kh * head_dim;
            float dot = 0.0f;
            for (size_t d = 0; d < head_dim; d++) dot += qh[d] * vbs_f16_to_f32(kh_ptr[d]);
            scores[t] = dot * scale;
        }

        vbs_cpu_softmax(scores, seq_len);

        float *oh = out + h * head_dim;
        for (size_t d = 0; d < head_dim; d++) oh[d] = 0.0f;
        for (size_t t = 0; t < seq_len; t++) {
            const uint16_t *vh = v_cache + t * kv_dim + kh * head_dim;
            for (size_t d = 0; d < head_dim; d++) oh[d] += scores[t] * vbs_f16_to_f32(vh[d]);
        }
    }
    free(scores);
}

void vbs_cpu_argmax(const float *logits, uint32_t *out_token, size_t vocab_size) {
    float best = -FLT_MAX;
    uint32_t idx = 0;
    for (size_t i = 0; i < vocab_size; i++) {
        if (logits[i] > best) { best = logits[i]; idx = (uint32_t)i; }
    }
    *out_token = idx;
}

static inline uint32_t rng_u32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *state = x;
    return x;
}

void vbs_cpu_sample_min_p(float *logits, uint32_t *out_token, size_t vocab_size, float min_p, float temperature, uint32_t seed) {
    float max_val = -FLT_MAX;
    for (size_t i = 0; i < vocab_size; i++) if (logits[i] > max_val) max_val = logits[i];
    float threshold = min_p * max_val;

    for (size_t i = 0; i < vocab_size; i++) {
        if (logits[i] < threshold) logits[i] = -FLT_MAX;
        else logits[i] /= temperature;
    }

    vbs_cpu_softmax(logits, vocab_size);

    uint32_t state = seed ? seed : 0xBADC0DEu;
    float r = (float)rng_u32(&state) / (float)UINT32_MAX;
    float cumulative = 0.0f;
    for (size_t i = 0; i < vocab_size; i++) {
        cumulative += logits[i];
        if (r <= cumulative) { *out_token = (uint32_t)i; return; }
    }
    *out_token = (uint32_t)(vocab_size - 1);
}

void vbs_cpu_embedding(const float *w, float *y, uint32_t token, size_t dim) {
    memcpy(y, w + (size_t)token * dim, dim * sizeof(float));
}

void vbs_cpu_elementwise_add(float *out, const float *a, const float *b, size_t n) {
    for (size_t i = 0; i < n; i++) out[i] = a[i] + b[i];
}

// =====================================================================
// Dispatch Table
// =====================================================================
vbs_matvec_fn_t vbs_get_matvec_kernel(vbs_ggml_type_t type) {
    switch (type) {
        case VBS_GGML_TYPE_F32:  return (vbs_matvec_fn_t)vbs_cpu_matvec_f32;
        case VBS_GGML_TYPE_F16:  return (vbs_matvec_fn_t)vbs_cpu_matvec_f16;
        case VBS_GGML_TYPE_Q4_0: return (vbs_matvec_fn_t)vbs_cpu_matvec_q4_0;
        case VBS_GGML_TYPE_Q4_1: return (vbs_matvec_fn_t)vbs_cpu_matvec_q4_1;
        case VBS_GGML_TYPE_Q5_0: return (vbs_matvec_fn_t)vbs_cpu_matvec_q5_0;
        case VBS_GGML_TYPE_Q5_1: return (vbs_matvec_fn_t)vbs_cpu_matvec_q5_1;
        case VBS_GGML_TYPE_Q8_0: return (vbs_matvec_fn_t)vbs_cpu_matvec_q8_0;
        case VBS_GGML_TYPE_Q2_K: return (vbs_matvec_fn_t)vbs_cpu_matvec_q2_K;
        case VBS_GGML_TYPE_Q3_K: return (vbs_matvec_fn_t)vbs_cpu_matvec_q3_K;
        case VBS_GGML_TYPE_Q4_K: return (vbs_matvec_fn_t)vbs_cpu_matvec_q4_K;
        case VBS_GGML_TYPE_Q5_K: return (vbs_matvec_fn_t)vbs_cpu_matvec_q5_K;
        case VBS_GGML_TYPE_Q6_K: return (vbs_matvec_fn_t)vbs_cpu_matvec_q6_K;
        default: return NULL;
    }
}

// =====================================================================
// FP8 (E4M3) Conversions
// =====================================================================
uint8_t vbs_float_to_fp8_e4m3(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 31) & 1;
    int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 7;
    uint32_t mant = (x >> 20) & 0x7;

    if (exp <= 0) return 0;
    if (exp > 15) return (uint8_t)((sign << 7) | 0x7E);
    return (uint8_t)((sign << 7) | ((uint32_t)exp << 3) | mant);
}

float vbs_fp8_e4m3_to_float(uint8_t b) {
    uint32_t sign = (b >> 7) & 1;
    uint32_t exp  = (b >> 3) & 0xF;
    uint32_t mant = b & 0x7;
    if (exp == 0 && mant == 0) return 0.0f;
    uint32_t f = (sign << 31) | ((exp - 7 + 127) << 23) | (mant << 20);
    float out;
    memcpy(&out, &f, 4);
    return out;
}

uint16_t vbs_f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x03FF;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

// =====================================================================
// Fused Normalization + Matvec Implementations
// =====================================================================
void vbs_fused_rmsnorm_matvec_q8_0(
    const float *x, const float *norm_w, const block_q8_0_t *mat_w,
    float *y, size_t cols, size_t rows, float eps
) {
    float ss = 0.0f;
    for (size_t i = 0; i < cols; i++) ss += x[i] * x[i];
    float inv_scale = 1.0f / sqrtf(ss / (float)cols + eps);

    size_t blocks_per_row = cols / 32;
    for (size_t r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (size_t bb = 0; bb < blocks_per_row; bb++) {
            const block_q8_0_t *b = &mat_w[r * blocks_per_row + bb];
            float d = vbs_f16_to_f32(b->d);
            int32_t acc = 0;
            for (size_t i = 0; i < 32; i++) {
                float norm_x = x[bb * 32 + i] * inv_scale * norm_w[bb * 32 + i];
                acc += (int32_t)b->qs[i] * (int32_t)(norm_x * 127.0f);
            }
            sum += d * (float)acc / 127.0f;
        }
        y[r] = sum;
    }
}

void vbs_fused_rmsnorm_matvec_q4_0(
    const float *x, const float *norm_w, const block_q4_0_t *mat_w,
    float *y, size_t cols, size_t rows, float eps
) {
    float ss = 0.0f;
    for (size_t i = 0; i < cols; i++) ss += x[i] * x[i];
    float inv_scale = 1.0f / sqrtf(ss / (float)cols + eps);

    size_t blocks_per_row = cols / 32;
    for (size_t r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (size_t bb = 0; bb < blocks_per_row; bb++) {
            const block_q4_0_t *b = &mat_w[r * blocks_per_row + bb];
            float d = vbs_f16_to_f32(b->d);
            float acc = 0.0f;
            for (size_t i = 0; i < 16; i++) {
                int q0 = (b->qs[i] & 0x0F) - 8;
                int q1 = (b->qs[i] >> 4)   - 8;
                float nx0 = x[bb * 32 + i] * inv_scale * norm_w[bb * 32 + i];
                float nx1 = x[bb * 32 + i + 16] * inv_scale * norm_w[bb * 32 + i + 16];
                acc += (float)q0 * nx0 + (float)q1 * nx1;
            }
            sum += d * acc;
        }
        y[r] = sum;
    }
}

void vbs_fused_rmsnorm_matvec_f16(
    const float *x, const float *norm_w, const uint16_t *mat_w,
    float *y, size_t cols, size_t rows, float eps
) {
    float ss = 0.0f;
    for (size_t i = 0; i < cols; i++) ss += x[i] * x[i];
    float inv_scale = 1.0f / sqrtf(ss / (float)cols + eps);

    for (size_t r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (size_t i = 0; i < cols; i++) {
            float norm_x = x[i] * inv_scale * norm_w[i];
            sum += vbs_f16_to_f32(mat_w[r * cols + i]) * norm_x;
        }
        y[r] = sum;
    }
}
