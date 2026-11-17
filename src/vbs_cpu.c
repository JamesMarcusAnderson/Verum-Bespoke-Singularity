#include "vbs_cpu.h"
#include "vbs_gguf.h"
#include "vbs_dequant.h"
#include "vbs_hotpath.h"
#include "vbs_fuse.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* This CPU path intentionally implements one clearly checked architecture,
 * rather than guessing how arbitrary GGUF models should execute. It owns no
 * model files and never changes the mapped weights. */
typedef struct {
    const vbs_tensor_t *an, *q, *k, *v, *ao;
    const vbs_tensor_t *fn, *gate, *up, *down;
} cpu_layer_t;

struct vbs_cpu_model {
    vbs_gguf_model_t *gguf;
    const vbs_tensor_t *embed, *out_norm, *out;
    /* Shadow copies of weight tensors whose stored type the scalar kernels
       cannot read directly (K-quants, Q5_0/1, Q8_1, BF16): dequantized to
       Q8_0 once at load, so any dequantizable model runs. shadow[i] is the
       converted tensor descriptor, shadow_src[i] the original GGUF tensor
       it came from (for dedup when output is tied to the embedding),
       shadow_data[i] the converted bytes. */
    vbs_tensor_t *shadow;
    const vbs_tensor_t **shadow_src;
    void **shadow_data;
    size_t n_shadow, shadow_cap;
    cpu_layer_t *layers;
    size_t n_layers, n_heads, n_kv_heads, head_dim, emb, kv_dim, ffn, vocab;
    size_t context, position;
    uint32_t eos;
    float theta, eps;
    float *cache_k, *cache_v;
    float *x, *norm, *q, *k, *v, *att, *tmp, *gate, *up, *hidden, *scores, *logits;
    int8_t *q8_input;
    float *q8_d, *q8_sum;
};

static bool mul_size(size_t a, size_t b, size_t *out) {
    if (a && b > SIZE_MAX / a) return false;
    *out = a * b;
    return true;
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 31u;
    uint32_t frac = (uint32_t)h & 1023u;
    uint32_t bits;
    if (exp == 0) {
        if (frac == 0) bits = sign;
        else {
            int e = -14;
            while ((frac & 0x400u) == 0) { frac <<= 1; e--; }
            frac &= 0x3ffu;
            bits = sign | (uint32_t)(e + 127) << 23 | frac << 13;
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | frac << 13;
    } else {
        bits = sign | (exp + 112u) << 23 | frac << 13;
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

/* IEEE-754 round-to-nearest-even conversion for the Q8 activation scales. */
static uint16_t f32_to_f16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
    int exp = (int)((bits >> 23) & 255u) - 127;
    uint32_t mant = bits & 0x7fffffu;
    if (exp > 15) return (uint16_t)(sign | 0x7c00u);
    if (exp >= -14) {
        uint32_t h = (uint32_t)(exp + 15) << 10 | mant >> 13;
        uint32_t rem = mant & 0x1fffu;
        if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;
        return (uint16_t)(sign | h);
    }
    if (exp < -25) return sign;
    mant |= 0x800000u;
    int shift = -exp - 1;
    uint32_t h = mant >> shift, rem = mant & ((1u << shift) - 1u), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (h & 1u))) h++;
    return (uint16_t)(sign | h);
}

static uint16_t read_u16(const uint8_t *p) {
    uint16_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static bool supported_weight(vbs_ggml_type_t type) {
    return type == VBS_GGML_TYPE_F32 || type == VBS_GGML_TYPE_F16 ||
           type == VBS_GGML_TYPE_Q4_0 || type == VBS_GGML_TYPE_Q4_1 ||
           type == VBS_GGML_TYPE_Q8_0;
}

/* Base address of a tensor's bytes: the converted buffer for shadow
   tensors, the mmap'd file for native ones. */
static const uint8_t *tensor_base(const vbs_cpu_model_t *m, const vbs_tensor_t *t) {
    if (m->shadow && t >= m->shadow && t < m->shadow + m->n_shadow) {
        return (const uint8_t *)m->shadow_data[t - m->shadow];
    }
    return (const uint8_t *)m->gguf->map + t->data_offset;
}

static const uint8_t *row_data(const vbs_cpu_model_t *m, const vbs_tensor_t *t, size_t row) {
    return tensor_base(m, t) + row * t->packed_row_bytes;
}

/* GGML Q4 blocks pack elements 0..15 in low nibbles and 16..31 high. */
static float dot_row(const vbs_cpu_model_t *m, const vbs_tensor_t *t, size_t row, const float *x) {
    const uint8_t *p = row_data(m, t, row);
    size_t n = (size_t)t->columns;
    float sum = 0.0f;
    if (t->type == VBS_GGML_TYPE_F32) {
        const float *w = (const float *)(const void *)p;
        for (size_t i = 0; i < n; i++) sum += w[i] * x[i];
    } else if (t->type == VBS_GGML_TYPE_F16) {
        for (size_t i = 0; i < n; i++) sum += f16_to_f32(read_u16(p + i * 2)) * x[i];
    } else if (t->type == VBS_GGML_TYPE_Q8_0) {
        for (size_t b = 0; b < n / 32; b++) {
            const uint8_t *q = p + b * 34;
            float d = f16_to_f32(read_u16(q));
            const int8_t *vals = (const int8_t *)(const void *)(q + 2);
            float block = 0.0f;
            for (size_t i = 0; i < 32; i++) block += (float)vals[i] * x[b * 32 + i];
            sum += d * block;
        }
    } else if (t->type == VBS_GGML_TYPE_Q4_0 || t->type == VBS_GGML_TYPE_Q4_1) {
        bool q41 = t->type == VBS_GGML_TYPE_Q4_1;
        size_t stride = q41 ? 20 : 18;
        for (size_t b = 0; b < n / 32; b++) {
            const uint8_t *q = p + b * stride;
            float d = f16_to_f32(read_u16(q));
            float minval = q41 ? f16_to_f32(read_u16(q + 2)) : 0.0f;
            const uint8_t *vals = q + (q41 ? 4 : 2);
            float block = 0.0f, xsum = 0.0f;
            for (size_t i = 0; i < 16; i++) {
                float lo = (float)(vals[i] & 15u) - (q41 ? 0.0f : 8.0f);
                float hi = (float)(vals[i] >> 4) - (q41 ? 0.0f : 8.0f);
                float a = x[b * 32 + i], z = x[b * 32 + i + 16];
                block += lo * a + hi * z;
                if (q41) xsum += a + z;
            }
            sum += d * block + minval * xsum;
        }
    }
    return sum;
}

/* Like GGML's CPU matmul, quantize each activation block once to Q8. This
 * introduces small rounding error but avoids a costly per-element float
 * dequantization for every row and matches GGML's quantized dot semantics.
 * Factored out of matrix() so the fused FFN kernel can quantize once and
 * reuse the result for both the gate and up projections. */
static void quantize_q8(vbs_cpu_model_t *m, const float *x, size_t cols) {
    size_t nblocks = cols / 32;
    for (size_t b = 0; b < nblocks; b++) {
        float amax = 0.0f;
        for (size_t i = 0; i < 32; i++) if (fabsf(x[b * 32 + i]) > amax) amax = fabsf(x[b * 32 + i]);
        float d = amax / 127.0f, inv = d ? 1.0f / d : 0.0f;
        int sum = 0;
        for (size_t i = 0; i < 32; i++) {
            int q = (int)roundf(x[b * 32 + i] * inv);
            if (q < -127) q = -127;
            if (q > 127) q = 127;
            m->q8_input[b * 32 + i] = (int8_t)q; sum += q;
        }
        m->q8_d[b] = f16_to_f32(f32_to_f16(d));
        m->q8_sum[b] = f16_to_f32(f32_to_f16((float)sum * d));
    }
}

static void matrix(vbs_cpu_model_t *m, const vbs_tensor_t *t, const float *x, float *y) {
    for (size_t i = 0; i < (size_t)t->columns; i++) {
        if (!isfinite(x[i])) {
            for (size_t r = 0; r < (size_t)t->rows; r++) y[r] = NAN;
            return;
        }
    }
    /* The hot path runs through the versioned kernel table (vbs_hotpath.h):
     * builtin reference kernels by default, swappable between tokens by the
     * recursive loop. Semantics are identical to the old inline loops. */
    const uint8_t *w0 = row_data(m, t, 0);
    size_t rows = (size_t)t->rows, cols = (size_t)t->columns;
    switch (t->type) {
        case VBS_GGML_TYPE_F32:
            vbs_hot_matvec_f32((const float *)(const void *)w0, rows, cols, x, y);
            return;
        case VBS_GGML_TYPE_F16:
            vbs_hot_matvec_f16((const uint16_t *)(const void *)w0, rows, cols, x, y);
            return;
        case VBS_GGML_TYPE_Q8_0:
            quantize_q8(m, x, cols);
            vbs_hot_matvec_q8_0(w0, rows, cols, m->q8_input, m->q8_d, y);
            return;
        case VBS_GGML_TYPE_Q4_0:
            quantize_q8(m, x, cols);
            vbs_hot_matvec_q4_0(w0, rows, cols, m->q8_input, m->q8_d, y);
            return;
        case VBS_GGML_TYPE_Q4_1:
            quantize_q8(m, x, cols);
            vbs_hot_matvec_q4_1(w0, rows, cols, m->q8_input, m->q8_d, m->q8_sum, y);
            return;
        default:
            break; /* unreachable: vbs_cpu_open rejects other types */
    }
    for (size_t r = 0; r < rows; r++) y[r] = dot_row(m, t, r, x);
}

static void row_to_float(const vbs_cpu_model_t *m, const vbs_tensor_t *t, size_t row, float *out) {
    const uint8_t *p = row_data(m, t, row);
    size_t n = (size_t)t->columns;
    if (t->type == VBS_GGML_TYPE_F32) {
        memcpy(out, p, n * sizeof(float));
    } else if (t->type == VBS_GGML_TYPE_F16) {
        for (size_t i = 0; i < n; i++) out[i] = f16_to_f32(read_u16(p + i * 2));
    } else if (t->type == VBS_GGML_TYPE_Q8_0) {
        for (size_t b = 0; b < n / 32; b++) {
            const uint8_t *q = p + b * 34;
            float d = f16_to_f32(read_u16(q));
            const int8_t *vals = (const int8_t *)(const void *)(q + 2);
            for (size_t i = 0; i < 32; i++) out[b * 32 + i] = d * (float)vals[i];
        }
    } else if (t->type == VBS_GGML_TYPE_Q4_0 || t->type == VBS_GGML_TYPE_Q4_1) {
        bool q41 = t->type == VBS_GGML_TYPE_Q4_1;
        size_t stride = q41 ? 20 : 18;
        for (size_t b = 0; b < n / 32; b++) {
            const uint8_t *q = p + b * stride;
            float d = f16_to_f32(read_u16(q));
            float minval = q41 ? f16_to_f32(read_u16(q + 2)) : 0.0f;
            const uint8_t *vals = q + (q41 ? 4 : 2);
            for (size_t i = 0; i < 16; i++) {
                out[b * 32 + i] = d * ((float)(vals[i] & 15u) - (q41 ? 0.0f : 8.0f)) + minval;
                out[b * 32 + i + 16] = d * ((float)(vals[i] >> 4) - (q41 ? 0.0f : 8.0f)) + minval;
            }
        }
    }
}

static void rms_norm(vbs_cpu_model_t *m, const vbs_tensor_t *weight, const float *input, float *output) {
    double sum = 0.0;
    for (size_t i = 0; i < m->emb; i++) sum += (double)input[i] * input[i];
    float scale = 1.0f / sqrtf((float)(sum / (double)m->emb) + m->eps);
    const uint8_t *p = row_data(m, weight, 0);
    for (size_t i = 0; i < m->emb; i++) {
        float w;
        if (weight->type == VBS_GGML_TYPE_F32) memcpy(&w, p + i * 4, sizeof(w));
        else w = f16_to_f32(read_u16(p + i * 2));
        output[i] = input[i] * scale * w;
    }
}

static void rope(float *x, size_t n_heads, size_t head_dim, size_t pos, float theta) {
    for (size_t h = 0; h < n_heads; h++) {
        for (size_t i = 0; i < head_dim; i += 2) {
            float angle = (float)pos * powf(theta, -(float)i / (float)head_dim);
            float c = cosf(angle), s = sinf(angle);
            size_t j = h * head_dim + i;
            float a = x[j], b = x[j + 1];
            x[j] = a * c - b * s;
            x[j + 1] = a * s + b * c;
        }
    }
}

static bool read_u32(const vbs_gguf_model_t *g, const char *key, uint32_t *v) {
    return vbs_gguf_get_uint32_metadata(g, key, v);
}

static bool check_tensor(const vbs_tensor_t *t, size_t rows, size_t cols, bool is_norm,
                         const char *name, char *err, size_t cap) {
    if (!t) { snprintf(err, cap, "Missing required tensor: %s", name); return false; }
    if (t->rows != rows || t->columns != cols ||
        (is_norm ? t->n_dims != 1 : t->n_dims != 2) ||
        (t->type == VBS_GGML_TYPE_F32 && t->data_offset % sizeof(float) != 0)) {
        snprintf(err, cap, "Unexpected shape for tensor %s", name); return false;
    }
    if (is_norm ? (t->type != VBS_GGML_TYPE_F32 && t->type != VBS_GGML_TYPE_F16)
                : (!supported_weight(t->type) && !vbs_dequant_supported(t->type))) {
        snprintf(err, cap, "Unsupported tensor type %d for %s", (int)t->type, name); return false;
    }
    return true;
}

/* If t's stored type has no scalar kernel, dequantize it once to Q8_0 and
   record a shadow tensor describing the converted copy; otherwise do
   nothing. Returns false only on allocation/conversion failure. A tensor
   already converted (tied output/embedding) reuses its shadow.
   Target choice, measured on SmolLM-135M Q4_K_M (2026-11-17): Q8_0
   decodes at 11.7 tok/s vs 1.2 tok/s for an F16 target, because the F16
   scalar dot converts every element per use while Q8_0 runs the integer
   dot; Q8_0 also halves the shadow memory. Accuracy cost vs an exact
   float64 reference: mean |logit| error 0.52 (Q8_0) vs 0.11 (F16) — the
   same order as llama.cpp's own 0.28 on this file — with identical
   argmax, identical top-10 set, and identical generated text on the
   test prompt. F16 stays available by changing the target constant
   below if a future accuracy-sensitive use wants it.
   NOTE: this may realloc the shadow arrays, so callers must NOT hold
   pointers into them across calls; repoint fields afterwards with
   shadow_for(), once all conversions are done. */
static bool maybe_convert(vbs_cpu_model_t *m, const vbs_tensor_t *t,
                          char *err, size_t cap) {
    if (!t || supported_weight(t->type)) return true;
    for (size_t i = 0; i < m->n_shadow; i++) {
        if (m->shadow_src[i] == t) return true;
    }
    if (m->n_shadow == m->shadow_cap) {
        size_t ncap = m->shadow_cap ? m->shadow_cap * 2 : 16;
        vbs_tensor_t *ns = (vbs_tensor_t *)realloc(m->shadow, ncap * sizeof(*ns));
        if (!ns) { snprintf(err, cap, "Out of memory converting tensor %s", t->name ? t->name : "?"); return false; }
        m->shadow = ns;
        const vbs_tensor_t **nsrc = (const vbs_tensor_t **)realloc(m->shadow_src, ncap * sizeof(*nsrc));
        if (!nsrc) { snprintf(err, cap, "Out of memory converting tensor %s", t->name ? t->name : "?"); return false; }
        m->shadow_src = nsrc;
        void **ndata = (void **)realloc(m->shadow_data, ncap * sizeof(*ndata));
        if (!ndata) { snprintf(err, cap, "Out of memory converting tensor %s", t->name ? t->name : "?"); return false; }
        m->shadow_data = ndata;
        m->shadow_cap = ncap;
    }
    const void *src = (const uint8_t *)m->gguf->map + t->data_offset;
    vbs_ggml_type_t out_type = VBS_GGML_TYPE_F16;
    uint64_t row_bytes = 0, total_bytes = 0;
    void *buf = vbs_dequant_tensor(t, src, VBS_GGML_TYPE_Q8_0, 1.0f,
                                   &out_type, &row_bytes, &total_bytes, err, cap);
    if (!buf) return false;
    size_t i = m->n_shadow++;
    m->shadow[i] = *t;
    m->shadow[i].type = out_type;
    m->shadow[i].packed_row_bytes = row_bytes;
    m->shadow[i].byte_length = total_bytes;
    m->shadow[i].data_offset = 0; /* unused for shadows; tensor_base routes by pointer */
    m->shadow_src[i] = t;
    m->shadow_data[i] = buf;
    return true;
}

/* After all conversions, resolve a tensor to its shadow (if it has one)
   or itself. Only call once the shadow array will not grow again. */
static const vbs_tensor_t *shadow_for(const vbs_cpu_model_t *m, const vbs_tensor_t *t) {
    if (!t || supported_weight(t->type)) return t;
    for (size_t i = 0; i < m->n_shadow; i++) {
        if (m->shadow_src[i] == t) return &m->shadow[i];
    }
    return t;
}

static const vbs_tensor_t *layer_tensor(const vbs_gguf_model_t *g, size_t l, const char *suffix,
                                        size_t rows, size_t cols, bool norm, char *err, size_t cap) {
    char name[128];
    snprintf(name, sizeof(name), "blk.%zu.%s.weight", l, suffix);
    const vbs_tensor_t *t = vbs_gguf_get_tensor(g, name);
    return check_tensor(t, rows, cols, norm, name, err, cap) ? t : NULL;
}

static bool known_tensor(const vbs_cpu_model_t *m, const vbs_tensor_t *t) {
    if (t == m->embed || t == m->out || t == m->out_norm) return true;
    for (size_t i = 0; i < m->n_layers; i++) {
        const cpu_layer_t *l = &m->layers[i];
        if (t == l->an || t == l->q || t == l->k || t == l->v || t == l->ao ||
            t == l->fn || t == l->gate || t == l->up || t == l->down) return true;
    }
    return false;
}

static bool alloc_floats(float **ptr, size_t count) {
    size_t bytes;
    if (!mul_size(count, sizeof(float), &bytes)) return false;
    *ptr = (float *)calloc(1, bytes);
    return *ptr != NULL;
}

vbs_cpu_model_t *vbs_cpu_open(const char *path, size_t context, char *err, size_t cap) {
    if (!path || !err || cap == 0 || context == 0) return NULL;
    vbs_cpu_model_t *m = (vbs_cpu_model_t *)calloc(1, sizeof(*m));
    if (!m) { snprintf(err, cap, "Out of memory"); return NULL; }
    uint32_t endian_test = 1;
    if (*(const uint8_t *)&endian_test != 1) {
        snprintf(err, cap, "CPU reference currently requires a little-endian host"); goto fail;
    }
    m->gguf = vbs_gguf_open(path, err, cap);
    if (!m->gguf) goto fail;
    if (m->gguf->version != 3) {
        snprintf(err, cap, "CPU reference supports GGUF version 3 only"); goto fail;
    }
    static const char *supported_llama_keys[] = {
        "llama.block_count", "llama.context_length", "llama.embedding_length",
        "llama.feed_forward_length", "llama.attention.head_count", "llama.attention.head_count_kv",
        "llama.rope.freq_base", "llama.attention.layer_norm_rms_epsilon", "llama.vocab_size",
        "llama.rope.dimension_count", "llama.rope.scaling.type"
    };
    for (size_t i = 0; i < m->gguf->n_metadata; i++) {
        const char *key = m->gguf->metadata[i].key;
        for (size_t j = 0; j < i; j++) {
            if (strcmp(key, m->gguf->metadata[j].key) == 0) {
                snprintf(err, cap, "Duplicate GGUF metadata key: %.300s", key); goto fail;
            }
        }
        if (strncmp(key, "llama.", 6) == 0) {
            bool supported = false;
            for (size_t k = 0; k < sizeof(supported_llama_keys) / sizeof(supported_llama_keys[0]); k++)
                if (strcmp(key, supported_llama_keys[k]) == 0) supported = true;
            if (!supported) { snprintf(err, cap, "Unsupported llama metadata: %.300s", key); goto fail; }
            uint32_t typ = m->gguf->metadata[i].type;
            bool is_float = strcmp(key, "llama.rope.freq_base") == 0 ||
                            strcmp(key, "llama.attention.layer_norm_rms_epsilon") == 0;
            bool is_string = strcmp(key, "llama.rope.scaling.type") == 0;
            if ((is_float && typ != 6 && typ != 12) || (is_string && typ != 8) ||
                (!is_float && !is_string && typ != 4 && typ != 0)) {
                snprintf(err, cap, "Unexpected metadata type for %.300s", key); goto fail;
            }
        }
    }
    char *arch = (char *)vbs_gguf_get_string_metadata(m->gguf, "general.architecture");
    if (!arch || strcmp(arch, "llama") != 0) {
        snprintf(err, cap, "CPU reference only supports verified llama-style GGUF architecture");
        free(arch); goto fail;
    }
    free(arch);
    char *scaling = (char *)vbs_gguf_get_string_metadata(m->gguf, "llama.rope.scaling.type");
    if (scaling && strcmp(scaling, "none") != 0) {
        snprintf(err, cap, "Unsupported RoPE scaling type: %s", scaling);
        free(scaling); goto fail;
    }
    free(scaling);
    uint32_t layers = 0, heads = 0, kvheads = 0, emb = 0, ctx = 0;
    if (!read_u32(m->gguf, "llama.block_count", &layers) ||
        !read_u32(m->gguf, "llama.attention.head_count", &heads) ||
        !read_u32(m->gguf, "llama.embedding_length", &emb) ||
        !read_u32(m->gguf, "llama.context_length", &ctx)) {
        snprintf(err, cap, "Model lacks required llama metadata"); goto fail;
    }
    if (!read_u32(m->gguf, "llama.attention.head_count_kv", &kvheads)) kvheads = heads;
    if (!layers || layers > 256 || !heads || !kvheads || !emb || emb > 32768 ||
        heads > emb || emb % heads || heads % kvheads || (emb / heads) % 2 || !ctx || context > ctx) {
        snprintf(err, cap, "Invalid or unsupported model dimensions/context"); goto fail;
    }
    uint32_t rotary_dim = 0;
    if (read_u32(m->gguf, "llama.rope.dimension_count", &rotary_dim) && rotary_dim != emb / heads) {
        snprintf(err, cap, "Partial RoPE dimensions are not supported"); goto fail;
    }
    m->n_layers = layers; m->n_heads = heads; m->n_kv_heads = kvheads;
    m->emb = emb; m->head_dim = emb / heads; m->kv_dim = kvheads * m->head_dim;
    m->context = context; m->theta = 10000.0f; m->eps = 1e-5f;
    (void)vbs_gguf_get_float32_metadata(m->gguf, "llama.rope.freq_base", &m->theta);
    (void)vbs_gguf_get_float32_metadata(m->gguf, "llama.attention.layer_norm_rms_epsilon", &m->eps);
    if (!isfinite(m->theta) || m->theta <= 0.0f || !isfinite(m->eps) || m->eps <= 0.0f) {
        snprintf(err, cap, "Invalid RoPE or RMS metadata"); goto fail;
    }
    m->eos = UINT32_MAX;
    (void)read_u32(m->gguf, "tokenizer.ggml.eos_token_id", &m->eos);
    m->embed = vbs_gguf_get_tensor(m->gguf, "token_embd.weight");
    if (!m->embed || m->embed->rows == 0 || m->embed->rows > 1048576) {
        snprintf(err, cap, "Missing or invalid token_embd.weight"); goto fail;
    }
    if (!check_tensor(m->embed, (size_t)m->embed->rows, m->emb, false, "token_embd.weight", err, cap)) goto fail;
    m->vocab = (size_t)m->embed->rows;
    uint32_t metadata_vocab = 0;
    if (read_u32(m->gguf, "llama.vocab_size", &metadata_vocab) && metadata_vocab != m->vocab) {
        snprintf(err, cap, "Vocabulary metadata does not match token embeddings"); goto fail;
    }
    m->out_norm = vbs_gguf_get_tensor(m->gguf, "output_norm.weight");
    if (!check_tensor(m->out_norm, 1, m->emb, true, "output_norm.weight", err, cap)) goto fail;
    m->out = vbs_gguf_get_tensor(m->gguf, "output.weight");
    if (!m->out) m->out = m->embed;
    if (!check_tensor(m->out, m->vocab, m->emb, false, "output.weight", err, cap)) goto fail;
    m->layers = (cpu_layer_t *)calloc(m->n_layers, sizeof(*m->layers));
    if (!m->layers) { snprintf(err, cap, "Out of memory allocating layers"); goto fail; }
    char name[128];
    snprintf(name, sizeof(name), "blk.0.ffn_gate.weight");
    const vbs_tensor_t *gate0 = vbs_gguf_get_tensor(m->gguf, name);
    if (!gate0 || !gate0->rows || gate0->rows > 131072) { snprintf(err, cap, "Invalid feed-forward dimension"); goto fail; }
    m->ffn = (size_t)gate0->rows;
    uint32_t metadata_ffn = 0;
    if (read_u32(m->gguf, "llama.feed_forward_length", &metadata_ffn) && metadata_ffn != m->ffn) {
        snprintf(err, cap, "Feed-forward metadata does not match tensor shape"); goto fail;
    }
    for (size_t i = 0; i < m->n_layers; i++) {
        cpu_layer_t *l = &m->layers[i];
#define LOAD(field, suffix, rows, cols, norm) \
        do { l->field = layer_tensor(m->gguf, i, suffix, rows, cols, norm, err, cap); if (!l->field) goto fail; } while (0)
        LOAD(an, "attn_norm", 1, m->emb, true);
        LOAD(q, "attn_q", m->emb, m->emb, false);
        LOAD(k, "attn_k", m->kv_dim, m->emb, false);
        LOAD(v, "attn_v", m->kv_dim, m->emb, false);
        LOAD(ao, "attn_output", m->emb, m->emb, false);
        LOAD(fn, "ffn_norm", 1, m->emb, true);
        LOAD(gate, "ffn_gate", m->ffn, m->emb, false);
        LOAD(up, "ffn_up", m->ffn, m->emb, false);
        LOAD(down, "ffn_down", m->emb, m->ffn, false);
#undef LOAD
    }
    for (size_t i = 0; i < m->gguf->n_tensors; i++) {
        if (!known_tensor(m, &m->gguf->tensors[i])) {
            snprintf(err, cap, "Unsupported extra tensor: %s", m->gguf->tensors[i].name); goto fail;
        }
    }
    /* Convert weights the scalar kernels cannot read (K-quants, Q5_0/1,
       Q8_1, BF16) to Q8_0 shadows. Phase 1 builds every shadow; phase 2
       repoints the fields, once the shadow array can no longer move. */
    if (!maybe_convert(m, m->embed, err, cap) || !maybe_convert(m, m->out, err, cap)) goto fail;
    for (size_t i = 0; i < m->n_layers; i++) {
        cpu_layer_t *l = &m->layers[i];
        if (!maybe_convert(m, l->q, err, cap) || !maybe_convert(m, l->k, err, cap) ||
            !maybe_convert(m, l->v, err, cap) || !maybe_convert(m, l->ao, err, cap) ||
            !maybe_convert(m, l->gate, err, cap) || !maybe_convert(m, l->up, err, cap) ||
            !maybe_convert(m, l->down, err, cap)) goto fail;
    }
    m->embed = shadow_for(m, m->embed);
    m->out = shadow_for(m, m->out);
    for (size_t i = 0; i < m->n_layers; i++) {
        cpu_layer_t *l = &m->layers[i];
        l->q = shadow_for(m, l->q); l->k = shadow_for(m, l->k);
        l->v = shadow_for(m, l->v); l->ao = shadow_for(m, l->ao);
        l->gate = shadow_for(m, l->gate); l->up = shadow_for(m, l->up);
        l->down = shadow_for(m, l->down);
    }
    size_t cache_n, t;
    size_t max_quant = m->emb > m->ffn ? m->emb : m->ffn;
    m->q8_input = (int8_t *)calloc(max_quant, sizeof(*m->q8_input));
    if (!m->q8_input || !alloc_floats(&m->q8_d, max_quant / 32 + 1) ||
        !alloc_floats(&m->q8_sum, max_quant / 32 + 1)) {
        snprintf(err, cap, "Insufficient memory for quantization buffers"); goto fail;
    }
    if (!mul_size(m->n_layers, context, &t) || !mul_size(t, m->kv_dim, &cache_n) ||
        !alloc_floats(&m->cache_k, cache_n) || !alloc_floats(&m->cache_v, cache_n) ||
        !alloc_floats(&m->x, m->emb) || !alloc_floats(&m->norm, m->emb) ||
        !alloc_floats(&m->q, m->emb) || !alloc_floats(&m->k, m->kv_dim) ||
        !alloc_floats(&m->v, m->kv_dim) || !alloc_floats(&m->att, m->emb) ||
        !alloc_floats(&m->tmp, m->emb) || !alloc_floats(&m->gate, m->ffn) ||
        !alloc_floats(&m->up, m->ffn) || !alloc_floats(&m->hidden, m->ffn) ||
        !alloc_floats(&m->scores, context) || !alloc_floats(&m->logits, m->vocab)) {
        snprintf(err, cap, "Insufficient memory for CPU model buffers/context"); goto fail;
    }
    return m;
fail:
    vbs_cpu_close(m);
    return NULL;
}

void vbs_cpu_close(vbs_cpu_model_t *m) {
    if (!m) return;
    if (m->shadow_data) {
        for (size_t i = 0; i < m->n_shadow; i++) free(m->shadow_data[i]);
    }
    free(m->shadow); free(m->shadow_src); free(m->shadow_data);
    free(m->layers); free(m->cache_k); free(m->cache_v); free(m->x); free(m->norm);
    free(m->q); free(m->k); free(m->v); free(m->att); free(m->tmp); free(m->gate);
    free(m->up); free(m->hidden); free(m->scores); free(m->logits);
    free(m->q8_input); free(m->q8_d); free(m->q8_sum);
    vbs_gguf_close(m->gguf); free(m);
}
void vbs_cpu_reset(vbs_cpu_model_t *m) { if (m) m->position = 0; }
const float *vbs_cpu_logits(const vbs_cpu_model_t *m) { return m ? m->logits : NULL; }
size_t vbs_cpu_vocab_size(const vbs_cpu_model_t *m) { return m ? m->vocab : 0; }
uint32_t vbs_cpu_eos_id(const vbs_cpu_model_t *m) { return m ? m->eos : UINT32_MAX; }
size_t vbs_cpu_context_capacity(const vbs_cpu_model_t *m) { return m ? m->context : 0; }

bool vbs_cpu_step(vbs_cpu_model_t *m, uint32_t token, char *err, size_t cap) {
    return vbs_cpu_step_ex(m, token, true, err, cap);
}
bool vbs_cpu_step_ex(vbs_cpu_model_t *m, uint32_t token, bool compute_logits, char *err, size_t cap) {
    if (!m || token >= m->vocab || m->position >= m->context) {
        if (err && cap) snprintf(err, cap, "Token or CPU context position out of bounds");
        return false;
    }
    const size_t pos = m->position;
    row_to_float(m, m->embed, token, m->x);
    for (size_t layer = 0; layer < m->n_layers; layer++) {
        const cpu_layer_t *l = &m->layers[layer];
        rms_norm(m, l->an, m->x, m->norm);
        matrix(m, l->q, m->norm, m->q); matrix(m, l->k, m->norm, m->k); matrix(m, l->v, m->norm, m->v);
        rope(m->q, m->n_heads, m->head_dim, pos, m->theta);
        rope(m->k, m->n_kv_heads, m->head_dim, pos, m->theta);
        size_t here = (layer * m->context + pos) * m->kv_dim;
        memcpy(m->cache_k + here, m->k, m->kv_dim * sizeof(float));
        memcpy(m->cache_v + here, m->v, m->kv_dim * sizeof(float));
        size_t groups = m->n_heads / m->n_kv_heads;
        float att_scale = 1.0f / sqrtf((float)m->head_dim);
        for (size_t h = 0; h < m->n_heads; h++) {
            size_t kh = h / groups;
            const float *q = m->q + h * m->head_dim;
            float max_score = -INFINITY;
            for (size_t t = 0; t <= pos; t++) {
                const float *k = m->cache_k + (layer * m->context + t) * m->kv_dim + kh * m->head_dim;
                float s = 0.0f;
                for (size_t i = 0; i < m->head_dim; i++) s += q[i] * k[i];
                m->scores[t] = s * att_scale;
                if (m->scores[t] > max_score) max_score = m->scores[t];
            }
            float denom = 0.0f;
            for (size_t t = 0; t <= pos; t++) { m->scores[t] = expf(m->scores[t] - max_score); denom += m->scores[t]; }
            if (!isfinite(denom) || denom <= 0.0f) { snprintf(err, cap, "Invalid attention scores"); return false; }
            float *out = m->att + h * m->head_dim;
            memset(out, 0, m->head_dim * sizeof(float));
            for (size_t t = 0; t <= pos; t++) {
                const float *v = m->cache_v + (layer * m->context + t) * m->kv_dim + kh * m->head_dim;
                float p = m->scores[t] / denom;
                for (size_t i = 0; i < m->head_dim; i++) out[i] += p * v[i];
            }
        }
        matrix(m, l->ao, m->att, m->tmp);
        for (size_t i = 0; i < m->emb; i++) m->x[i] += m->tmp[i];
        rms_norm(m, l->fn, m->x, m->norm);
        /* Fused FFN: when a verified model-specialized kernel is loaded for
         * this model's dims and quant type, gate+up+SiLU run as one pass
         * over the shared Q8 activations. Otherwise the reference path. */
        vbs_fused_ffn_fn fused = (l->gate->type == l->up->type)
            ? vbs_fuse_lookup(l->gate->type, m->emb, m->ffn) : NULL;
        if (fused) {
            quantize_q8(m, m->norm, m->emb);
            fused(row_data(m, l->gate, 0), row_data(m, l->up, 0),
                  m->ffn, m->emb, m->q8_input, m->q8_d, m->q8_sum, m->hidden);
        } else {
            matrix(m, l->gate, m->norm, m->gate); matrix(m, l->up, m->norm, m->up);
            for (size_t i = 0; i < m->ffn; i++) m->hidden[i] = (m->gate[i] / (1.0f + expf(-m->gate[i]))) * m->up[i];
        }
        matrix(m, l->down, m->hidden, m->tmp);
        for (size_t i = 0; i < m->emb; i++) m->x[i] += m->tmp[i];
    }
    if (compute_logits) {
        rms_norm(m, m->out_norm, m->x, m->norm);
        matrix(m, m->out, m->norm, m->logits);
        for (size_t i = 0; i < m->vocab; i++) {
            if (!isfinite(m->logits[i])) { snprintf(err, cap, "Non-finite model logits"); return false; }
        }
    }
    m->position++;
    return true;
}
