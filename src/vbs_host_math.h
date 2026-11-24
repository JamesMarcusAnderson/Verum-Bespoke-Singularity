/* vbs_host_math.h — shared host-side reference math for the CPU
 * fallback backend and the host-staged HIP backend.
 *
 * Every function here is the CORRECT reference implementation of
 * the op the GPU kernels perform (same formulas as the CUDA PTX
 * kernels in kernels/vendor_kernels.ptx.h and the GLSL kernels in
 * kernels/gpu/). The CPU backend calls these directly on host
 * memory; the HIP backend stages device buffers through them.
 *
 * Static functions in a header (same pattern as
 * include/vbs_sha256.h): each including translation unit gets its
 * own copy; no link-time duplicates. Depends on vbs_dequant.h for
 * block dequantization and vbs_model.h for block geometry.
 */

#ifndef VBS_HOST_MATH_H
#define VBS_HOST_MATH_H

#include "vbs_backend.h"
#include "vbs_dequant.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* RMS norm: out = in * (1/sqrt(mean(in^2) + eps)) * w.
 * Double accumulation, like vbs_cpu.c: the reference is the most
 * accurate form; GPU kernels use float sums and agree to ~1e-6. */
static inline void vbs_host_rms_norm(const float *in, const float *w,
                                     float *out, uint32_t n, float eps) {
    double ss = 0.0;
    for (uint32_t i = 0; i < n; i++) ss += (double)in[i] * in[i];
    float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (uint32_t i = 0; i < n; i++) out[i] = in[i] * scale * w[i];
}

/* Rotary position embedding, in place. Theta is honored (an old
 * adapter port used a fixed 10000 base). */
static inline void vbs_host_rope(float *x, uint32_t pos, uint32_t head_dim,
                                 uint32_t n_heads, float theta) {
    for (uint32_t h = 0; h < n_heads; h++) {
        float *head = x + (size_t)h * head_dim;
        for (uint32_t i = 0; i + 1 < head_dim; i += 2) {
            float angle = (float)pos * powf(theta, -(float)i / (float)head_dim);
            float c = cosf(angle), s = sinf(angle);
            float x0 = head[i], x1 = head[i + 1];
            head[i] = x0 * c - x1 * s;
            head[i + 1] = x0 * s + x1 * c;
        }
    }
}

/* SiLU(gate) * up, elementwise. */
static inline void vbs_host_silu_mul(const float *gate, const float *up,
                                     float *out, uint32_t n) {
    for (uint32_t i = 0; i < n; i++)
        out[i] = gate[i] / (1.0f + expf(-gate[i])) * up[i];
}

static inline void vbs_host_add(const float *a, const float *b,
                                float *out, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) out[i] = a[i] + b[i];
}

/* Weight types the host math supports for linear/embed. */
static const vbs_model_type_t k_host_supported_types[] = {
    VBS_MODEL_TYPE_F32, VBS_MODEL_TYPE_F16,
    VBS_MODEL_TYPE_Q4_0, VBS_MODEL_TYPE_Q4_1,
    VBS_MODEL_TYPE_Q5_0, VBS_MODEL_TYPE_Q5_1,
    VBS_MODEL_TYPE_Q8_0, VBS_MODEL_TYPE_Q8_1,
    VBS_MODEL_TYPE_Q2_K, VBS_MODEL_TYPE_Q3_K,
    VBS_MODEL_TYPE_Q4_K, VBS_MODEL_TYPE_Q5_K,
    VBS_MODEL_TYPE_Q6_K, VBS_MODEL_TYPE_LNS8,
};

static inline bool vbs_host_type_supported(vbs_model_type_t type) {
    for (size_t i = 0; i < sizeof(k_host_supported_types) / sizeof(k_host_supported_types[0]); i++)
        if (k_host_supported_types[i] == type) return true;
    return false;
}

/* Dequantize one packed row (cols elements) starting at `packed`
 * into `out` (floats). Row stride is computed from the type
 * geometry. Returns false for unsupported types. */
static inline bool vbs_host_dequant_row(const uint8_t *packed,
                                        vbs_model_type_t type,
                                        uint32_t cols, float *out) {
    vbs_block_geometry_t g;
    if (!vbs_get_geometry(type, &g)) return false;
    if (type == VBS_MODEL_TYPE_F32) {
        memcpy(out, packed, (size_t)cols * sizeof(float));
        return true;
    }
    if (type == VBS_MODEL_TYPE_F16) {
        for (uint32_t i = 0; i < cols; i++) {
            uint16_t h;
            memcpy(&h, packed + (size_t)i * 2, 2);
            out[i] = vbs_f16_to_f32(h);
        }
        return true;
    }
    if (type == VBS_MODEL_TYPE_LNS8) {
        /* Block (36 B): 32 x u8 log-values, f32 scale at byte 32;
         * value = 2^q * scale (same as the CUDA matvec_lns8 kernel). */
        for (uint32_t b = 0; b < cols / 32; b++) {
            const uint8_t *blk = packed + (size_t)b * 36;
            float scale;
            memcpy(&scale, blk + 32, sizeof(scale));
            for (uint32_t i = 0; i < 32; i++)
                out[b * 32 + i] = exp2f((float)blk[i]) * scale;
        }
        return true;
    }
    for (uint32_t b = 0; b < cols / g.elements_per_block; b++) {
        int n = vbs_dequant_block(type, packed + (size_t)b * g.bytes_per_block,
                                  out + (size_t)b * g.elements_per_block);
        if (n != (int)g.elements_per_block) return false;
    }
    return true;
}

/* out[0..dim) = dequantized row `token` of the packed table `w`. */
static inline bool vbs_host_embed_row(const uint8_t *w, vbs_model_type_t type,
                                      uint32_t token, float *out, uint32_t dim) {
    vbs_block_geometry_t g;
    if (!vbs_get_geometry(type, &g)) return false;
    uint64_t row_bytes = (uint64_t)(dim / g.elements_per_block) * g.bytes_per_block;
    return vbs_host_dequant_row(w + (uint64_t)token * row_bytes, type, dim, out);
}

/* Store f32 K/V into the f16 caches at (layer, pos). Layout
 * matches the CUDA kv_cache_store kernel: index
 * (layer*max_ctx + pos)*kv_dim + i. */
static inline void vbs_host_cache_store(const float *k, const float *v,
                                        uint16_t *kc, uint16_t *vc,
                                        uint32_t pos, uint32_t layer,
                                        uint32_t max_ctx, uint32_t kv_dim) {
    uint64_t base = ((uint64_t)layer * max_ctx + pos) * kv_dim;
    for (uint32_t i = 0; i < kv_dim; i++) {
        kc[base + i] = vbs_f32_to_f16(k[i]);
        vc[base + i] = vbs_f32_to_f16(v[i]);
    }
}

/* Dot product of an f32 query row with an f16 key row, accumulated
 * in double. */
static inline double dot_f32_f16(const float *q, const uint16_t *k, uint32_t n) {
    double sum = 0.0;
    for (uint32_t d = 0; d < n; d++)
        sum += (double)q[d] * (double)vbs_f16_to_f32(k[d]);
    return sum;
}

/* Grouped-query attention (reference).
 *
 * q: f32 [n_q_heads * head_dim]; kc/vc: f16 caches (full buffers,
 * the layer slice offset is folded here from layer/max_ctx); out:
 * f32. Causal by construction: only positions 0..seq_len-1 are
 * attended. Mirrors the CUDA attention_gqa kernel (online
 * softmax, per-head kv head = q_head / group size, scale applied
 * to the score). */

/* Attention scalars, unpacked from the canonical bridge arg order
 * (args 4..10): seq_len, layer, max_ctx, n_q_heads, n_kv_heads,
 * head_dim, scale. Every backend reads the same layout, so the
 * unpacking lives here instead of in each dispatch function.
 * Callers must check n_args >= 11 first. */
typedef struct {
    uint32_t seq_len;
    uint32_t layer;
    uint32_t max_ctx;
    uint32_t nqh;
    uint32_t nkh;
    uint32_t hd;
    float scale;
} vbs_attn_shape_t;

static inline vbs_attn_shape_t vbs_attention_shape(const vbs_arg_t *args) {
    vbs_attn_shape_t s;
    s.seq_len = *(const uint32_t *)args[4].value;
    s.layer = *(const uint32_t *)args[5].value;
    s.max_ctx = *(const uint32_t *)args[6].value;
    s.nqh = *(const uint32_t *)args[7].value;
    s.nkh = *(const uint32_t *)args[8].value;
    s.hd = *(const uint32_t *)args[9].value;
    s.scale = *(const float *)args[10].value;
    return s;
}

static inline void vbs_host_attention(const float *q, const uint16_t *kc,
                                      const uint16_t *vc, float *out,
                                      uint32_t seq_len, uint32_t layer,
                                      uint32_t max_ctx, uint32_t n_q_heads,
                                      uint32_t n_kv_heads, uint32_t head_dim,
                                      float scale) {
    if (n_kv_heads == 0 || n_q_heads == 0 || head_dim == 0) return;
    uint32_t group = n_q_heads / n_kv_heads;
    if (group == 0) group = 1;
    uint64_t layer_off = (uint64_t)layer * max_ctx * n_kv_heads * head_dim;
    for (uint32_t h = 0; h < n_q_heads; h++) {
        uint32_t kvh = h / group;
        const float *qrow = q + (size_t)h * head_dim;
        /* Online softmax with a running output accumulator that
         * lives in the output buffer itself. */
        float m = -3.4028235e38f; /* -FLT_MAX */
        float l = 0.0f;
        float *acc = out + (size_t)h * head_dim;
        for (uint32_t d = 0; d < head_dim; d++) acc[d] = 0.0f;
        for (uint32_t s = 0; s < seq_len; s++) {
            uint64_t kbase = layer_off +
                ((uint64_t)s * n_kv_heads + kvh) * head_dim;
            float score = (float)dot_f32_f16(qrow, kc + kbase, head_dim) * scale;
            float m_new = score > m ? score : m;
            float r = expf(m - m_new);
            float e = expf(score - m_new);
            l = l * r + e;
            m = m_new;
            for (uint32_t d = 0; d < head_dim; d++) {
                float vv = vbs_f16_to_f32(vc[kbase + d]);
                acc[d] = acc[d] * r + e * vv;
            }
        }
        float inv = 1.0f / l;
        for (uint32_t d = 0; d < head_dim; d++) acc[d] *= inv;
    }
}

/* Linear tile (matvec) — the reference every backend must match.
 *
 * y[row_off + r] (+= if accumulate) = dot(w_tile[r],
 * x[col_off..]). `packed_w` points at the tile start: either the
 * staging buffer's host copy (slice-executor path) or the resident
 * packed tensor (bridge path). The tile row stride is computed from
 * (type, cols), which is correct for both paths: a tile's row holds
 * exactly `cols` packed columns, and a full tensor's row holds its
 * full `cols`. Exact float math (dequantize, then dot in double);
 * no Q8-activation approximation like vbs_cpu.c's matrix(). */
static inline bool vbs_host_linear_tile(const uint8_t *packed_w,
                                        vbs_model_type_t type,
                                        const float *x, float *y,
                                        uint32_t cols, uint32_t rows,
                                        uint32_t col_off, uint32_t row_off,
                                        bool accumulate) {
    vbs_block_geometry_t g;
    if (!vbs_get_geometry(type, &g)) return false;
    if (cols % g.elements_per_block != 0) return false;
    uint64_t row_bytes = (uint64_t)(cols / g.elements_per_block) * g.bytes_per_block;
    float vals[256]; /* largest block dequantizes to 256 floats (K-quants) */
    uint32_t nblocks = cols / g.elements_per_block;
    for (uint32_t r = 0; r < rows; r++) {
        const uint8_t *row_ptr = packed_w + (uint64_t)r * row_bytes;
        double sum = 0.0;
        for (uint32_t b = 0; b < nblocks; b++) {
            const uint8_t *blk = row_ptr + (uint64_t)b * g.bytes_per_block;
            const float *xblk = x + col_off + b * g.elements_per_block;
            if (type == VBS_MODEL_TYPE_F32) {
                const float *wf = (const float *)(const void *)blk;
                for (uint32_t i = 0; i < g.elements_per_block; i++)
                    sum += (double)wf[i] * xblk[i];
            } else if (type == VBS_MODEL_TYPE_F16) {
                for (uint32_t i = 0; i < g.elements_per_block; i++) {
                    uint16_t h;
                    memcpy(&h, blk + (size_t)i * 2, 2);
                    sum += (double)vbs_f16_to_f32(h) * xblk[i];
                }
            } else if (type == VBS_MODEL_TYPE_LNS8) {
                float scale;
                memcpy(&scale, blk + 32, sizeof(scale));
                for (uint32_t i = 0; i < 32; i++)
                    sum += (double)(exp2f((float)blk[i]) * scale) * xblk[i];
            } else {
                int n = vbs_dequant_block(type, blk, vals);
                if (n != (int)g.elements_per_block) return false;
                for (int i = 0; i < n; i++)
                    sum += (double)vals[i] * xblk[i];
            }
        }
        float *dst = y + row_off + r;
        *dst = accumulate ? *dst + (float)sum : (float)sum;
    }
    return true;
}

/* Host-staged pipeline handles (shared by the CPU and HIP backends).
 *
 * Both backends used to carry verbatim copies of these two tiny
 * structs, their constructors, and the three compile_* wrappers
 * (differing only in the "CPU backend" / "HIP backend" label in
 * the error strings). One canonical copy lives here; each
 * backend's ops-table entry delegates to it with its own label.
 * The dispatch functions stay per-backend: CPU dereferences host
 * pointers directly, HIP stages device buffers through host
 * memory.
 */

typedef struct {
    vbs_model_type_t type;
} vbs_host_linear_pipeline_t;

typedef struct {
    vbs_prim_pipeline_t base; /* .native = this struct, .kind = op kind */
    vbs_model_type_t type;    /* EMBEDDING weight type (F32 default) */
} vbs_host_prim_pipeline_t;

static inline vbs_host_prim_pipeline_t *vbs_host_prim_pipeline(vbs_op_kind_t kind,
                                                               vbs_model_type_t type) {
    vbs_host_prim_pipeline_t *p =
        (vbs_host_prim_pipeline_t *)malloc(sizeof(*p));
    if (!p) return NULL;
    p->base.native = p;
    p->base.kind = kind;
    p->type = type;
    return p;
}

static inline void *vbs_host_compile_linear_pipeline(const char *backend_name,
                                                     vbs_model_type_t type,
                                                     char *err_buf, size_t err_cap) {
    if (!vbs_host_type_supported(type)) {
        snprintf(err_buf, err_cap,
                 "%s: no linear kernel for weight type %d", backend_name, (int)type);
        return NULL;
    }
    vbs_host_linear_pipeline_t *p =
        (vbs_host_linear_pipeline_t *)malloc(sizeof(*p));
    if (!p) return NULL;
    p->type = type;
    return p;
}

static inline void *vbs_host_compile_embed_pipeline(const char *backend_name,
                                                    vbs_model_type_t type,
                                                    char *err_buf, size_t err_cap) {
    if (!vbs_host_type_supported(type)) {
        snprintf(err_buf, err_cap,
                 "%s: no embedding kernel for weight type %d", backend_name, (int)type);
        return NULL;
    }
    return vbs_host_prim_pipeline(VBS_OP_EMBEDDING, type);
}

/* Primitive op kinds the host path implements. */
static const vbs_op_kind_t k_host_prim_kinds[] = {
    VBS_OP_EMBEDDING, /* F32 table (same convention as the GPU paths) */
    VBS_OP_NORMALIZE,
    VBS_OP_ROTARY,
    VBS_OP_CACHE,
    VBS_OP_ATTENTION,
    VBS_OP_ACTIVATION,
    VBS_OP_ELEMENTWISE,
};

static inline void *vbs_host_compile_primitive_pipeline(const char *backend_name,
                                                        vbs_op_kind_t kind,
                                                        char *err_buf, size_t err_cap) {
    for (size_t i = 0; i < sizeof(k_host_prim_kinds) / sizeof(k_host_prim_kinds[0]); i++)
        if (k_host_prim_kinds[i] == kind)
            return vbs_host_prim_pipeline(kind, VBS_MODEL_TYPE_F32);
    snprintf(err_buf, err_cap,
             "%s: no primitive kernel for op kind %d", backend_name, (int)kind);
    return NULL;
}

#endif /* VBS_HOST_MATH_H */
