/* vbs_dequant.c — universal model block dequantization.
 *
 * Purpose: the GPU kernels natively execute f32/f16/Q4_0/Q4_1/
 * Q6_K/Q8_0. Every other common model weight format (Q5_0, Q5_1,
 * Q2_K, Q3_K, Q4_K, Q5_K, Q8_K, BF16) can be converted at load
 * time into Q8_0 (compact, native kernel) or F16 (exact for
 * 16-bit sources), so any model runs on any backend. Block
 * layouts follow the model file format's public tensor-encoding
 * specification (super-blocks of 256 with nested scale
 * hierarchies).
 *
 * Half-precision conversion lives in vbs_f16.h as static inline
 * functions (the single canonical copy, shared with the
 * hot-kernel modules). Nothing to define here.
 */

#include "vbs_dequant.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* Capability tables: which types run natively, which convert. */
static const vbs_model_type_t k_native_types[] = {
    VBS_MODEL_TYPE_F32, VBS_MODEL_TYPE_F16,
    VBS_MODEL_TYPE_Q4_0, VBS_MODEL_TYPE_Q4_1,
    VBS_MODEL_TYPE_Q6_K, VBS_MODEL_TYPE_Q8_0,
    VBS_MODEL_TYPE_I8G, /* CPU: vbs_cpu_matvec_i8g (resident only) */
};

static const vbs_model_type_t k_convertible_types[] = {
    VBS_MODEL_TYPE_Q5_0, VBS_MODEL_TYPE_Q5_1,
    VBS_MODEL_TYPE_Q2_K, VBS_MODEL_TYPE_Q3_K,
    VBS_MODEL_TYPE_Q4_K, VBS_MODEL_TYPE_Q5_K,
    VBS_MODEL_TYPE_Q6_K, /* no *matvec* kernel path uses it via bridge on all backends */
    VBS_MODEL_TYPE_Q8_0, VBS_MODEL_TYPE_Q8_1,
    VBS_MODEL_TYPE_Q8_K, VBS_MODEL_TYPE_Q4_0,
    VBS_MODEL_TYPE_Q4_1, VBS_MODEL_TYPE_F16,
    VBS_MODEL_TYPE_F32, VBS_MODEL_TYPE_BF16,
};

static bool type_in_table(vbs_model_type_t t, const vbs_model_type_t *table,
                          size_t n) {
    for (size_t i = 0; i < n; i++)
        if (table[i] == t) return true;
    return false;
}

bool vbs_dequant_has_native_kernel(vbs_model_type_t t) {
    return type_in_table(t, k_native_types,
                         sizeof(k_native_types) / sizeof(k_native_types[0]));
}

bool vbs_dequant_supported(vbs_model_type_t t) {
    /* IQ* / TQ* / MXFP4: no reference layout implemented here. */
    return type_in_table(t, k_convertible_types,
                         sizeof(k_convertible_types) / sizeof(k_convertible_types[0]));
}

/* Read an f16 scale from a block at a byte offset. */
static float block_scale(const uint8_t *b, size_t off) {
    uint16_t h;
    memcpy(&h, b + off, 2);
    return vbs_f16_to_f32(h);
}

/* Legacy 32-element blocks. */

static int decode_q4_0(const uint8_t *b, float *y) {
    float d = block_scale(b, 0);
    for (int i = 0; i < 16; i++) {
        uint8_t byte = b[2 + i];
        y[i] = d * ((float)(byte & 0x0F) - 8.0f);
        y[i + 16] = d * ((float)(byte >> 4) - 8.0f);
    }
    return 32;
}

static int decode_q4_1(const uint8_t *b, float *y) {
    float d = block_scale(b, 0);
    float m = block_scale(b, 2);
    for (int i = 0; i < 16; i++) {
        uint8_t byte = b[4 + i];
        y[i] = d * (float)(byte & 0x0F) + m;
        y[i + 16] = d * (float)(byte >> 4) + m;
    }
    return 32;
}

static int decode_q5_0(const uint8_t *b, float *y) {
    float d = block_scale(b, 0);
    uint32_t qh;
    memcpy(&qh, b + 2, 4);
    for (int i = 0; i < 16; i++) {
        uint8_t byte = b[6 + i];
        uint32_t lo = (uint32_t)(byte & 0x0F) | (((qh >> i) & 1u) << 4);
        uint32_t hi = (uint32_t)(byte >> 4) | (((qh >> (i + 16)) & 1u) << 4);
        y[i] = d * ((float)lo - 16.0f);
        y[i + 16] = d * ((float)hi - 16.0f);
    }
    return 32;
}

static int decode_q5_1(const uint8_t *b, float *y) {
    float d = block_scale(b, 0);
    float m = block_scale(b, 2);
    uint32_t qh;
    memcpy(&qh, b + 4, 4);
    for (int i = 0; i < 16; i++) {
        uint8_t byte = b[8 + i];
        uint32_t lo = (uint32_t)(byte & 0x0F) | (((qh >> i) & 1u) << 4);
        uint32_t hi = (uint32_t)(byte >> 4) | (((qh >> (i + 16)) & 1u) << 4);
        y[i] = d * (float)lo + m;
        y[i + 16] = d * (float)hi + m;
    }
    return 32;
}

static int decode_q8_0(const uint8_t *b, float *y) {
    float d = block_scale(b, 0);
    const int8_t *q = (const int8_t *)(b + 2);
    for (int i = 0; i < 32; i++) y[i] = d * (float)q[i];
    return 32;
}

static int decode_q8_1(const uint8_t *b, float *y) {
    /* Q8_1 = { f16 d; f16 m; i8 qs[32] }: quants carry d; m is a
     * dot-product helper (block sum), value = q * d. Same 40-byte
     * block as the spec. */
    float d = block_scale(b, 0);
    const int8_t *q = (const int8_t *)(b + 4);
    for (int i = 0; i < 32; i++) y[i] = d * (float)q[i];
    return 32;
}

/* K-quants: 256-element super-blocks. */

/* Extract the 2-bit quant for element `e` from a 64-byte qs array
 * split into two 32-byte halves (Q2_K/Q3_K layout). */
static int quant_2bit(const uint8_t *qs, int e) {
    int half = e >> 7;
    int w = e & 127;
    return (qs[half * 32 + (w & 31)] >> (2 * (w >> 5))) & 3;
}

static int decode_q2_k(const uint8_t *b, float *y) {
    /* { u8 scales[16]; u8 qs[64]; f16 d; f16 dmin } */
    const uint8_t *scales = b;
    const uint8_t *qs = b + 16;
    float d = block_scale(b, 80);
    float dmn = block_scale(b, 82);
    for (int e = 0; e < 256; e++) {
        uint8_t sc = scales[e >> 4];
        int q = quant_2bit(qs, e);
        y[e] = d * (float)(sc & 0x0F) * (float)q - dmn * (float)(sc >> 4);
    }
    return 256;
}

/* Q3_K/Q4_K/Q5_K share the 6-bit scale packing. 16 values (Q3_K:
 * signed, offset by 32) live in 12 bytes:
 *   low nibble  : bytes 0..7 hold scales 0..7 in low nibbles,
 *                 8..15 in high nibbles
 *   high 2 bits : bytes 8..11, scale j -> byte 8 + j%4, bits 2*(j/4) */
static void unpack_scales6(const uint8_t s[12], int out[16]) {
    for (int j = 0; j < 16; j++) {
        int low = (j < 8) ? (s[j] & 0x0F) : (s[j - 8] >> 4);
        int hi = (s[8 + (j & 3)] >> (2 * (j >> 2))) & 3;
        out[j] = (low | (hi << 4)) - 32;
    }
}

/* Q4_K/Q5_K scale/min pair: 8 sub-blocks of 32, 6-bit scale +
 * 6-bit min packed into 12 bytes. */
static void unpack_scale_min_k4(const uint8_t *q, int j, int *sc, int *m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static int decode_q3_k(const uint8_t *b, float *y) {
    /* { u8 hmask[32]; u8 qs[64]; u8 scales[12]; f16 d } */
    const uint8_t *hm = b;
    const uint8_t *qs = b + 32;
    float d = block_scale(b, 108);
    int sc[16];
    unpack_scales6(b + 96, sc);
    for (int e = 0; e < 256; e++) {
        /* 3-bit value: low 2 bits from qs; the high bit lives in
         * hmask. hmask SET -> q in [0,3] stored directly, value
         * = q; hmask CLEAR -> q in [0,3] representing (q - 4),
         * value = q - 4. */
        int q = quant_2bit(qs, e);
        if (!((hm[e & 31] >> (e >> 5)) & 1)) q -= 4;
        y[e] = d * (float)sc[e >> 4] * (float)q;
    }
    return 256;
}

/* Apply one K4 scale/min pair to a nibble value: the shared tail
 * of the Q4_K and Q5_K decoders (same 12-byte scale packing).
 * d/dmn are the block's super-block scale and minimum. */
static float k4_apply(float d, float dmn, const uint8_t *scales, int e, int nib) {
    int sc, m;
    unpack_scale_min_k4(scales, e >> 5, &sc, &m);
    return d * (float)sc * (float)nib - dmn * (float)m;
}

static int decode_q4_k(const uint8_t *b, float *y) {
    /* { f16 d; f16 dmin; u8 scales[12]; u8 qs[128] } — the
     * super-block scale/min come FIRST in the Q4_K block layout
     * (unlike Q2_K/Q3_K/Q6_K, where they come last); reading them
     * from the tail decodes garbage (values in the tens of
     * millions). */
    float d = block_scale(b, 0);
    float dmn = block_scale(b, 2);
    const uint8_t *scales = b + 4;
    const uint8_t *qs = b + 16;
    for (int e = 0; e < 256; e++) {
        int g = e >> 6; /* 64-element group */
        int w = e & 63;
        uint8_t byte = qs[g * 32 + (w & 31)];
        int nib = (w < 32) ? (byte & 0x0F) : (byte >> 4);
        y[e] = k4_apply(d, dmn, scales, e, nib);
    }
    return 256;
}

static int decode_q5_k(const uint8_t *b, float *y) {
    /* { f16 d; f16 dmin; u8 scales[12]; u8 qh[32]; u8 qs[128] } —
     * same d/dmin-first order as Q4_K; the high bit of each
     * 5-bit quant lives in qh. */
    float d = block_scale(b, 0);
    float dmn = block_scale(b, 2);
    const uint8_t *scales = b + 4;
    const uint8_t *qh = b + 16;
    const uint8_t *ql = b + 48;
    for (int e = 0; e < 256; e++) {
        int g = e >> 6;
        int w = e & 63;
        uint8_t byte = ql[g * 32 + (w & 31)];
        int nib = (w < 32) ? (byte & 0x0F) : (byte >> 4);
        /* High bit: group g, second half -> odd bit position. */
        int bit = 2 * g + ((w >= 32) ? 1 : 0);
        if ((qh[w & 31] >> bit) & 1) nib += 16;
        y[e] = k4_apply(d, dmn, scales, e, nib);
    }
    return 256;
}

static int decode_q8_k(const uint8_t *b, float *y) {
    /* { f32 d; i8 qs[256]; i16 bsums[16] } = 292 bytes. Unlike
     * the other K-quants the super-block scale is a full f32,
     * there are no per-block scales, and the trailing bsums are
     * dot-product helpers (sums of the quants in each group of
     * 16) that dequantization does not need. Value is d * q. */
    float d;
    memcpy(&d, b, 4);
    const int8_t *q = (const int8_t *)(b + 4);
    for (int i = 0; i < 256; i++) y[i] = d * (float)q[i];
    return 256;
}

static int decode_q6_k(const uint8_t *b, float *y) {
    /* { u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d } */
    const uint8_t *ql = b;
    const uint8_t *qh = b + 128;
    const int8_t *sc = (const int8_t *)(b + 192);
    float d = block_scale(b, 208);
    for (int e = 0; e < 256; e++) {
        int h = e >> 7;         /* 128-element half */
        int q = (e & 127) >> 5; /* 32-element quarter */
        int l = e & 31;
        int qi = (q < 2) ? (h * 64 + q * 32 + l) : (h * 64 + (q - 2) * 32 + l);
        int shift = (q < 2) ? 0 : 4;
        int lo = (ql[qi] >> shift) & 0x0F;
        int hi = (qh[h * 32 + l] >> (q * 2)) & 3;
        y[e] = d * (float)sc[h * 8 + q * 2 + (l >> 4)] *
               (float)((lo | (hi << 4)) - 32);
    }
    return 256;
}

/* One block -> floats. */
int vbs_dequant_block(vbs_model_type_t type, const void *block, float *out_values) {
    if (!block || !out_values) return 0;
    const uint8_t *b = (const uint8_t *)block;
    switch (type) {
        case VBS_MODEL_TYPE_Q4_0: return decode_q4_0(b, out_values);
        case VBS_MODEL_TYPE_Q4_1: return decode_q4_1(b, out_values);
        case VBS_MODEL_TYPE_Q5_0: return decode_q5_0(b, out_values);
        case VBS_MODEL_TYPE_Q5_1: return decode_q5_1(b, out_values);
        case VBS_MODEL_TYPE_Q8_0: return decode_q8_0(b, out_values);
        case VBS_MODEL_TYPE_Q8_1: return decode_q8_1(b, out_values);
        case VBS_MODEL_TYPE_Q2_K: return decode_q2_k(b, out_values);
        case VBS_MODEL_TYPE_Q3_K: return decode_q3_k(b, out_values);
        case VBS_MODEL_TYPE_Q4_K: return decode_q4_k(b, out_values);
        case VBS_MODEL_TYPE_Q5_K: return decode_q5_k(b, out_values);
        case VBS_MODEL_TYPE_Q6_K: return decode_q6_k(b, out_values);
        case VBS_MODEL_TYPE_Q8_K: return decode_q8_k(b, out_values);
        default: return 0;
    }
}

/* Requantize 32 floats into a Q8_0 block: f16 scale, then 32 int8
 * quants rounded to nearest (lrintf), clamped to +/-127. */
void vbs_requant_q8_0_block(const float *v, void *out_block) {
    uint8_t *blk = (uint8_t *)out_block;
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        float a = fabsf(v[i]);
        if (a > amax) amax = a;
    }
    float d = amax > 0.0f ? amax / 127.0f : 0.0f;
    uint16_t h = vbs_f32_to_f16(d);
    memcpy(blk, &h, 2);
    int8_t *q = (int8_t *)(blk + 2);
    if (d == 0.0f) {
        memset(q, 0, 32);
        return;
    }
    float inv = 1.0f / d;
    for (int i = 0; i < 32; i++) {
        int qi = (int)lrintf(v[i] * inv);
        if (qi > 127) qi = 127;
        if (qi < -127) qi = -127;
        q[i] = (int8_t)qi;
    }
}

/* I8G: internal 1-byte-per-element target with one f32 scale per
 * ROW. Per-row layout: [f32 scale][int8 quants x cols].
 * Self-contained rows make the format sliceable by the thread
 * pool AND streamable through executor windows (a staged row tile
 * carries its own scales).
 *
 * Quality metric: the per-element quantization error bound is
 * amax_row/127 (rounding to nearest), vs Q8_0's
 * amax_32block/127. The ratio below is the element-mean of
 * amax_row / amax_32block: 1.0 means rows are flat and I8G is
 * quality-equivalent to Q8_0; larger means outlier rows inflate
 * the error of their small-amplitude elements. The engine
 * refuses to AUTO-select I8G when this ratio is poor; an
 * explicit pin overrides with the number printed.
 */

/* Dequantize one packed row into rowbuf (floats). Returns false
 * on decoder gaps. On success *out_n is the element count. */
static bool decode_row_floats(const vbs_tensor_t *t, const uint8_t *row_src,
                              float *rowbuf, uint64_t *out_n) {
    vbs_block_geometry_t g;
    if (!vbs_get_geometry(t->type, &g)) return false;
    uint64_t blocks = t->columns / g.elements_per_block;
    uint64_t o = 0;
    for (uint64_t b = 0; b < blocks; b++) {
        int n = vbs_dequant_block(t->type,
                                  row_src + (size_t)(b * g.bytes_per_block),
                                  rowbuf + o);
        if (n <= 0) return false;
        o += (uint64_t)n;
    }
    *out_n = o;
    return o == t->columns;
}

/* Decode an elementwise source row (f32/f16/bf16) directly.
 * Returns false when the type has no elementwise decoder. */
static bool decode_row_elementwise(vbs_model_type_t type, const uint8_t *row_src,
                                   float *rowbuf, uint64_t cols) {
    if (type == VBS_MODEL_TYPE_F32) {
        memcpy(rowbuf, row_src, (size_t)cols * 4);
        return true;
    }
    if (type == VBS_MODEL_TYPE_F16) {
        const uint16_t *h = (const uint16_t *)row_src;
        for (uint64_t i = 0; i < cols; i++) rowbuf[i] = vbs_f16_to_f32(h[i]);
        return true;
    }
    if (type == VBS_MODEL_TYPE_BF16) {
        const uint16_t *h = (const uint16_t *)row_src;
        for (uint64_t i = 0; i < cols; i++) {
            uint32_t bits = (uint32_t)h[i] << 16;
            memcpy(&rowbuf[i], &bits, 4);
        }
        return true;
    }
    return false;
}

void *vbs_dequant_tensor_i8g(
    const vbs_tensor_t *t, const void *src, float scale,
    vbs_model_type_t *out_type, uint64_t *out_packed_row_bytes,
    uint64_t *out_total_bytes, char *err, size_t err_cap
) {
    vbs_block_geometry_t srcg;
    if (!vbs_get_geometry(t->type, &srcg)) {
        snprintf(err, err_cap, "i8g: unknown source geometry (%d)", (int)t->type);
        return NULL;
    }
    if (t->columns % 32 != 0) {
        snprintf(err, err_cap, "i8g: columns (%llu) not a multiple of 32",
                 (unsigned long long)t->columns);
        return NULL;
    }
    uint64_t rows = t->rows, cols = t->columns;
    uint64_t row_bytes = cols + 4;
    uint64_t total = rows * row_bytes;
    uint8_t *out = (uint8_t *)malloc(total ? (size_t)total : 1);
    float *rowbuf = (float *)malloc((size_t)cols * sizeof(float));
    if (!out || !rowbuf) {
        free(out);
        free(rowbuf);
        snprintf(err, err_cap, "i8g: out of memory (%llu bytes)", (unsigned long long)total);
        return NULL;
    }

    uint64_t src_row_bytes = (cols / srcg.elements_per_block) * srcg.bytes_per_block;
    for (uint64_t r = 0; r < rows; r++) {
        const uint8_t *row_src = (const uint8_t *)src + (size_t)(r * src_row_bytes);
        uint64_t n = 0;
        if (!decode_row_floats(t, row_src, rowbuf, &n) &&
            !decode_row_elementwise(t->type, row_src, rowbuf, cols)) {
            free(out);
            free(rowbuf);
            snprintf(err, err_cap, "i8g: no decoder for source type %d (tensor %s)",
                     (int)t->type, t->name ? t->name : "?");
            return NULL;
        }
        if (scale != 1.0f) {
            for (uint64_t i = 0; i < cols; i++) rowbuf[i] *= scale;
        }
        float amax = 0.0f;
        for (uint64_t i = 0; i < cols; i++) {
            float a = fabsf(rowbuf[i]);
            if (a > amax) amax = a;
        }
        float d = amax > 0.0f ? amax / 127.0f : 1.0f;
        float inv = 1.0f / d;
        uint8_t *orow = out + (size_t)(r * row_bytes);
        memcpy(orow, &d, 4); /* row scale */
        int8_t *qrow = (int8_t *)(orow + 4);
        for (uint64_t i = 0; i < cols; i++) {
            float v = rowbuf[i] * inv;
            int qi = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));
            if (qi > 127) qi = 127;
            else if (qi < -127) qi = -127;
            qrow[i] = (int8_t)qi;
        }
    }
    free(rowbuf);
    if (out_type) *out_type = VBS_MODEL_TYPE_I8G;
    if (out_packed_row_bytes) *out_packed_row_bytes = row_bytes;
    if (out_total_bytes) *out_total_bytes = total;
    return out;
}

double vbs_dequant_i8g_quality_ratio(const vbs_tensor_t *t, const void *src) {
    if (!t || !src || !t->columns || !t->rows) return 1e30;
    vbs_block_geometry_t srcg;
    if (!vbs_get_geometry(t->type, &srcg)) return 1e30;
    uint64_t cols = t->columns, rows = t->rows;
    float *rowbuf = (float *)malloc((size_t)cols * sizeof(float));
    if (!rowbuf) return 1e30;
    uint64_t src_row_bytes = (cols / srcg.elements_per_block) * srcg.bytes_per_block;
    double ratio_sum = 0.0;
    for (uint64_t r = 0; r < rows; r++) {
        const uint8_t *row_src = (const uint8_t *)src + (size_t)(r * src_row_bytes);
        uint64_t n = 0;
        if (!decode_row_floats(t, row_src, rowbuf, &n) &&
            !decode_row_elementwise(t->type, row_src, rowbuf, cols)) {
            free(rowbuf);
            return 1e30; /* no decoder: treat as worst quality */
        }
        float amax_row = 0.0f;
        for (uint64_t i = 0; i < cols; i++) {
            float a = fabsf(rowbuf[i]);
            if (a > amax_row) amax_row = a;
        }
        if (amax_row <= 0.0f) continue; /* flat zero row: ratio irrelevant */
        for (uint64_t b = 0; b < cols; b += 32) {
            float amax_blk = 0.0f;
            for (int i = 0; i < 32; i++) {
                float a = fabsf(rowbuf[b + i]);
                if (a > amax_blk) amax_blk = a;
            }
            /* Elements in this 32-block pay amax_row/amax_blk vs Q8_0. */
            double ratio = amax_blk > 0.0f ? (double)amax_row / (double)amax_blk : 1.0;
            ratio_sum += ratio * 32.0;
        }
    }
    free(rowbuf);
    double elems = (double)rows * (double)cols;
    return elems > 0.0 ? ratio_sum / elems : 1.0;
}

/* Decode an elementwise source chunk the way the tensor
 * converter does: F32 and F16 by their own decoders, anything
 * else with 1 element per block read as BF16 (the original
 * fallthrough — I8/I16/I32/I64/F64 sources decode as BF16 here,
 * matching the old behavior exactly). */
static void decode_elementwise_chunk(vbs_model_type_t type, const uint8_t *blk,
                                     float *vals, int n) {
    if (type == VBS_MODEL_TYPE_F32) {
        memcpy(vals, blk, (size_t)n * 4);
    } else if (type == VBS_MODEL_TYPE_F16) {
        const uint16_t *h = (const uint16_t *)blk;
        for (int i = 0; i < n; i++) vals[i] = vbs_f16_to_f32(h[i]);
    } else { /* BF16 (and any other 1-element-per-block type) */
        const uint16_t *h = (const uint16_t *)blk;
        for (int i = 0; i < n; i++) {
            uint32_t bits = (uint32_t)h[i] << 16;
            memcpy(&vals[i], &bits, 4);
        }
    }
}

/* Emit one 32-float chunk (or a ragged tail) into the destination
 * buffer in the target format, advancing *dp. The Q8_0 case needs
 * full 32-element chunks; callers guarantee that. */
static void emit_chunk(const float *vals, int n, vbs_model_type_t target,
                       uint8_t **dp) {
    if (target == VBS_MODEL_TYPE_F32) {
        memcpy(*dp, vals, (size_t)n * 4);
        *dp += (size_t)n * 4;
    } else if (target == VBS_MODEL_TYPE_F16) {
        uint16_t *h = (uint16_t *)*dp;
        for (int i = 0; i < n; i++) h[i] = vbs_f32_to_f16(vals[i]);
        *dp += (size_t)n * 2;
    } else { /* Q8_0 */
        for (int i = 0; i < n; i += 32) {
            vbs_requant_q8_0_block(vals + i, *dp);
            *dp += 34;
        }
    }
}

void *vbs_dequant_tensor(const vbs_dequant_request_t *req) {
    const vbs_tensor_t *t = req->t;
    const void *src = req->src;
    vbs_model_type_t target = req->target;
    float scale = req->scale;
    vbs_model_type_t *out_type = req->out_type;
    uint64_t *out_packed_row_bytes = req->out_packed_row_bytes;
    uint64_t *out_total_bytes = req->out_total_bytes;
    char *err = req->err;
    size_t err_cap = req->err_cap;
    if (!t || !src) {
        snprintf(err, err_cap, "dequant: null tensor or source");
        return NULL;
    }
    if (target != VBS_MODEL_TYPE_Q8_0 && target != VBS_MODEL_TYPE_F16 &&
        target != VBS_MODEL_TYPE_F32 && target != VBS_MODEL_TYPE_I8G) {
        snprintf(err, err_cap, "dequant: invalid target type %d", (int)target);
        return NULL;
    }
    if (target == VBS_MODEL_TYPE_I8G) {
        return vbs_dequant_tensor_i8g(t, src, scale, out_type,
                                      out_packed_row_bytes, out_total_bytes,
                                      err, err_cap);
    }

    vbs_block_geometry_t srcg, dstg;
    if (!vbs_get_geometry(t->type, &srcg) || !vbs_get_geometry(target, &dstg)) {
        snprintf(err, err_cap, "dequant: unknown geometry (src %d, dst %d)",
                 (int)t->type, (int)target);
        return NULL;
    }
    if (t->columns % srcg.elements_per_block != 0) {
        snprintf(err, err_cap, "dequant: columns (%llu) not aligned to source block (%u)",
                 (unsigned long long)t->columns, srcg.elements_per_block);
        return NULL;
    }

    uint64_t src_blocks_per_row = t->columns / srcg.elements_per_block;
    uint64_t dst_blocks_per_row =
        (target == VBS_MODEL_TYPE_F32) ? t->columns : (t->columns / dstg.elements_per_block);
    if (target != VBS_MODEL_TYPE_F32 && t->columns % dstg.elements_per_block != 0) {
        snprintf(err, err_cap, "dequant: columns (%llu) not aligned to target block (%u)",
                 (unsigned long long)t->columns, dstg.elements_per_block);
        return NULL;
    }
    uint64_t dst_row_bytes =
        (target == VBS_MODEL_TYPE_F32) ? t->columns * 4 : dst_blocks_per_row * dstg.bytes_per_block;
    uint64_t total = dst_row_bytes * t->rows;

    uint8_t *out = (uint8_t *)malloc(total ? (size_t)total : 1);
    if (!out) {
        snprintf(err, err_cap, "dequant: out of memory for %llu bytes", (unsigned long long)total);
        return NULL;
    }

    float vals[256];
    const uint8_t *sp = (const uint8_t *)src;
    uint8_t *dp = out;

    /* Element-wise formats (f32/f16/bf16) have 1 element per source
     * block; walk those 32 elements at a time (with a ragged tail)
     * so every path emits whole 32-float chunks, or whole
     * 256-element k-quant super-blocks. */
    bool elementwise = (srcg.elements_per_block == 1);

    for (uint64_t r = 0; r < t->rows; r++) {
        const uint8_t *row = sp + (size_t)(r * src_blocks_per_row * srcg.bytes_per_block);
        if (elementwise) {
            for (uint64_t e0 = 0; e0 < t->columns; e0 += 32) {
                uint64_t left = t->columns - e0;
                int n = (int)(left < 32 ? left : 32);
                const uint8_t *blk = row + (size_t)(e0 * srcg.bytes_per_block);
                decode_elementwise_chunk(t->type, blk, vals, n);
                if (scale != 1.0f) {
                    for (int i = 0; i < n; i++) vals[i] *= scale;
                }
                /* Q8_0 target: the alignment check above
                 * guarantees full 32-element chunks here.
                 * (Rewrite fix: the old code wrote F16 bytes for
                 * any non-F32 target, which corrupted and
                 * overflowed the buffer for a Q8_0 target.) */
                emit_chunk(vals, n, target, &dp);
            }
        } else {
            for (uint64_t cb = 0; cb < src_blocks_per_row; cb++) {
                const uint8_t *blk = row + (size_t)(cb * srcg.bytes_per_block);
                int n = vbs_dequant_block(t->type, blk, vals);
                if (n <= 0) {
                    snprintf(err, err_cap, "dequant: no decoder for weight type %d (tensor %s)",
                             (int)t->type, t->name ? t->name : "?");
                    free(out);
                    return NULL;
                }
                if (scale != 1.0f) {
                    for (int i = 0; i < n; i++) vals[i] *= scale;
                }
                emit_chunk(vals, n, target, &dp);
            }
        }
    }

    if (out_type) *out_type = target;
    if (out_packed_row_bytes) *out_packed_row_bytes = dst_row_bytes;
    if (out_total_bytes) *out_total_bytes = total;
    return out;
}
