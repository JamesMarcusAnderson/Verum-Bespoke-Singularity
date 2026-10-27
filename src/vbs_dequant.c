// vbs_dequant.c — universal GGUF block dequantization, written from scratch.
//
// Purpose: the GPU kernels natively execute f32/f16/Q4_0/Q4_1/Q6_K/Q8_0.
// Every other common GGUF weight format (Q5_0, Q5_1, Q2_K, Q3_K, Q4_K,
// Q5_K, BF16) can be converted at load time into Q8_0 (compact, native
// kernel) or F16 (exact for 16-bit sources), so any model runs on any
// backend. Block layouts follow the public GGUF tensor-encoding spec
// (super-blocks of 256 with nested scale hierarchies); all code here is
// original.

#include "vbs_dequant.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// half-precision conversion (round-to-nearest-even)
// ---------------------------------------------------------------------------

float vbs_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t man  = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign; // +-0
        } else {
            // subnormal half: value = man * 2^-24; normalize
            int e = -1;
            uint32_t m = man;
            do { m <<= 1; e++; } while (!(m & 0x400u));
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (man << 13); // inf / nan
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

uint16_t vbs_f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t biased = (x >> 23) & 0xFFu;
    uint32_t man = x & 0x7FFFFFu;
    if (biased == 0xFFu) {
        return (uint16_t)(sign | 0x7C00u | (man ? 1u : 0u)); // inf/nan
    }
    int32_t exp = (int32_t)biased - 127 + 15;
    if (exp >= 0x1F) {
        return (uint16_t)(sign | 0x7C00u); // overflow -> inf
    }
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign; // underflow -> 0
        // subnormal half: 1.man * 2^(exp-15) = m * 2^-24
        uint32_t m = man | 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = 1u << (shift - 1);
        uint32_t q = (m + half) >> shift;
        if (q >= 0x400u) return (uint16_t)(sign | 0x0400u); // -> min normal
        return (uint16_t)(sign | q);
    }
    // normal, round to nearest even
    uint32_t rounded = man + 0xFFFu + ((man >> 13) & 1u);
    if (rounded & 0x800000u) { rounded = 0; exp += 1; }
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (rounded >> 13));
}

// ---------------------------------------------------------------------------
// capability tables
// ---------------------------------------------------------------------------

bool vbs_dequant_has_native_kernel(vbs_ggml_type_t t) {
    switch (t) {
        case VBS_GGML_TYPE_F32:
        case VBS_GGML_TYPE_F16:
        case VBS_GGML_TYPE_Q4_0:
        case VBS_GGML_TYPE_Q4_1:
        case VBS_GGML_TYPE_Q6_K:
        case VBS_GGML_TYPE_Q8_0:
        case VBS_GGML_TYPE_I8G: // CPU: vbs_cpu_matvec_i8g (resident only)
            return true;
        default:
            return false;
    }
}

bool vbs_dequant_supported(vbs_ggml_type_t t) {
    switch (t) {
        case VBS_GGML_TYPE_Q5_0:
        case VBS_GGML_TYPE_Q5_1:
        case VBS_GGML_TYPE_Q2_K:
        case VBS_GGML_TYPE_Q3_K:
        case VBS_GGML_TYPE_Q4_K:
        case VBS_GGML_TYPE_Q5_K:
        case VBS_GGML_TYPE_Q6_K: // no *matvec* kernel path uses it via bridge on all backends
        case VBS_GGML_TYPE_Q8_0:
        case VBS_GGML_TYPE_Q8_1:
        case VBS_GGML_TYPE_Q4_0:
        case VBS_GGML_TYPE_Q4_1:
        case VBS_GGML_TYPE_F16:
        case VBS_GGML_TYPE_F32:
        case VBS_GGML_TYPE_BF16:
            return true;
        default:
            // IQ* / TQ* / MXFP4 / Q8_K: no reference layout implemented here.
            return false;
    }
}

// ---------------------------------------------------------------------------
// legacy 32-element blocks
// ---------------------------------------------------------------------------

static int deq_q4_0(const uint8_t *b, float *y) {
    float d = vbs_f16_to_f32(((uint16_t *)b)[0]);
    for (int i = 0; i < 16; i++) {
        uint8_t byte = b[2 + i];
        y[i]      = (float)(byte & 0x0F) - 8.0f;
        y[i + 16] = (float)(byte >> 4)   - 8.0f;
    }
    for (int i = 0; i < 32; i++) y[i] *= d;
    return 32;
}

static int deq_q4_1(const uint8_t *b, float *y) {
    float d = vbs_f16_to_f32(((uint16_t *)b)[0]);
    float m = vbs_f16_to_f32(((uint16_t *)b)[1]);
    for (int i = 0; i < 16; i++) {
        uint8_t byte = b[4 + i];
        y[i]      = d * (float)(byte & 0x0F) + m;
        y[i + 16] = d * (float)(byte >> 4)   + m;
    }
    return 32;
}

static int deq_q5_0(const uint8_t *b, float *y) {
    float d = vbs_f16_to_f32(((uint16_t *)b)[0]);
    uint32_t qh;
    memcpy(&qh, b + 2, 4);
    for (int i = 0; i < 16; i++) {
        uint8_t byte = b[6 + i];
        uint32_t lo = (uint32_t)(byte & 0x0F) | ((qh >> (i     )) & 1u) << 4;
        uint32_t hi = (uint32_t)(byte >> 4)   | ((qh >> (i + 16)) & 1u) << 4;
        y[i]      = d * ((float)lo - 16.0f);
        y[i + 16] = d * ((float)hi - 16.0f);
    }
    return 32;
}

static int deq_q5_1(const uint8_t *b, float *y) {
    float d = vbs_f16_to_f32(((uint16_t *)b)[0]);
    float m = vbs_f16_to_f32(((uint16_t *)b)[1]);
    uint32_t qh;
    memcpy(&qh, b + 4, 4);
    for (int i = 0; i < 16; i++) {
        uint8_t byte = b[8 + i];
        uint32_t lo = (uint32_t)(byte & 0x0F) | ((qh >> (i     )) & 1u) << 4;
        uint32_t hi = (uint32_t)(byte >> 4)   | ((qh >> (i + 16)) & 1u) << 4;
        y[i]      = d * (float)lo + m;
        y[i + 16] = d * (float)hi + m;
    }
    return 32;
}

static int deq_q8_0(const uint8_t *b, float *y) {
    float d = vbs_f16_to_f32(((uint16_t *)b)[0]);
    const int8_t *q = (const int8_t *)(b + 2);
    for (int i = 0; i < 32; i++) y[i] = d * (float)q[i];
    return 32;
}

static int deq_q8_1(const uint8_t *b, float *y) {
    // Q8_1 = { f16 d; f16 m; i8 qs[32] }: quants carry d; m is a dot-product
    // helper (block sum), value = q * d. Same 40-byte block as the spec.
    float d = vbs_f16_to_f32(((uint16_t *)b)[0]);
    const int8_t *q = (const int8_t *)(b + 4);
    for (int i = 0; i < 32; i++) y[i] = d * (float)q[i];
    return 32;
}

// ---------------------------------------------------------------------------
// k-quants: 256-element super-blocks
// ---------------------------------------------------------------------------

static int deq_q2_k(const uint8_t *b, float *y) {
    // { u8 scales[16]; u8 qs[64]; f16 d; f16 dmin }
    const uint8_t *scales = b;
    const uint8_t *qs     = b + 16;
    float d   = vbs_f16_to_f32(((uint16_t *)(b + 80))[0]);
    float dmn = vbs_f16_to_f32(((uint16_t *)(b + 80))[1]);
    for (int e = 0; e < 256; e++) {
        int sb   = e >> 4;              // 16-element sub-block
        int n128 = e >> 7;              // which 128 half
        int w    = e & 127;
        uint8_t qbyte = qs[n128 * 32 + (w & 31)];
        int q = (qbyte >> (2 * (w >> 5))) & 3;
        uint8_t sc = scales[sb];
        y[e] = d * (float)(sc & 0x0F) * (float)q - dmn * (float)(sc >> 4);
    }
    return 256;
}

// Q3_K/Q4_K/Q5_K share the 6-bit scale packing.
// 16 values (Q3_K: signed, offset by 32) live in 12 bytes:
//   low nibble  : bytes 0..7 hold scales 0..7 in low nibbles, 8..15 in high
//   high 2 bits : bytes 8..11, scale j -> byte 8 + j%4, bits 2*(j/4)
static void unpack_scales6(const uint8_t s[12], int out[16]) {
    for (int j = 0; j < 16; j++) {
        int low = (j < 8) ? (s[j] & 0x0F) : (s[j - 8] >> 4);
        int hi  = (s[8 + (j & 3)] >> (2 * (j >> 2))) & 3;
        out[j] = (low | (hi << 4)) - 32;
    }
}

// Q4_K/Q5_K scale/min pair: 8 sub-blocks of 32, 6-bit scale + 6-bit min
static void unpack_scale_min_k4(const uint8_t *q, int j, int *sc, int *m) {
    if (j < 4) {
        *sc = q[j]     & 63;
        *m  = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        *m  = (q[j + 4] >>  4) | ((q[j    ] >> 6) << 4);
    }
}

static int deq_q3_k(const uint8_t *b, float *y) {
    // { u8 hmask[32]; u8 qs[64]; u8 scales[12]; f16 d }
    const uint8_t *hm     = b;
    const uint8_t *qs     = b + 32;
    const uint8_t *scales = b + 96;
    float d = vbs_f16_to_f32(((uint16_t *)(b + 108))[0]);
    int sc[16];
    unpack_scales6(scales, sc);
    for (int e = 0; e < 256; e++) {
        int sb   = e >> 4;
        int n128 = e >> 7;
        int w    = e & 127;
        uint8_t qbyte = qs[n128 * 32 + (w & 31)];
        // 3-bit value: low 2 bits from qs; the high bit lives in hmask.
        // hmask SET  -> q in [0,3] stored directly      -> value = q
        // hmask CLEAR-> q in [0,3] representing (q - 4) -> value = q - 4
        int q = (qbyte >> (2 * (w >> 5))) & 3;
        if (!((hm[e & 31] >> (e >> 5)) & 1)) q -= 4;
        y[e] = d * (float)sc[sb] * (float)q;
    }
    return 256;
}

static int deq_q4_k(const uint8_t *b, float *y) {
    // { u8 qs[128]; u8 scales[12]; f16 d; f16 dmin }
    const uint8_t *qs     = b;
    const uint8_t *scales = b + 128;
    float d   = vbs_f16_to_f32(((uint16_t *)(b + 140))[0]);
    float dmn = vbs_f16_to_f32(((uint16_t *)(b + 140))[1]);
    for (int e = 0; e < 256; e++) {
        int g = e >> 6;            // 64-element group
        int w = e & 63;
        uint8_t byte = qs[g * 32 + (w & 31)];
        int nib = (w < 32) ? (byte & 0x0F) : (byte >> 4);
        int sc, m;
        unpack_scale_min_k4(scales, e >> 5, &sc, &m);
        y[e] = d * (float)sc * (float)nib - dmn * (float)m;
    }
    return 256;
}

static int deq_q5_k(const uint8_t *b, float *y) {
    // { u8 ql[128]; u8 qh[32]; u8 scales[12]; f16 d; f16 dmin }
    const uint8_t *ql     = b;
    const uint8_t *qh     = b + 128;
    const uint8_t *scales = b + 160;
    float d   = vbs_f16_to_f32(((uint16_t *)(b + 172))[0]);
    float dmn = vbs_f16_to_f32(((uint16_t *)(b + 172))[1]);
    for (int e = 0; e < 256; e++) {
        int g = e >> 6;
        int w = e & 63;
        uint8_t byte = ql[g * 32 + (w & 31)];
        int nib = (w < 32) ? (byte & 0x0F) : (byte >> 4);
        // high bit: group g, second half -> odd bit position
        int bit = 2 * g + ((w >= 32) ? 1 : 0);
        if ((qh[w & 31] >> bit) & 1) nib += 16;
        int sc, m;
        unpack_scale_min_k4(scales, e >> 5, &sc, &m);
        y[e] = d * (float)sc * (float)nib - dmn * (float)m;
    }
    return 256;
}

static int deq_q6_k(const uint8_t *b, float *y) {
    // { u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d }
    const uint8_t *ql = b;
    const uint8_t *qh = b + 128;
    const int8_t *sc  = (const int8_t *)(b + 192);
    float d = vbs_f16_to_f32(((uint16_t *)(b + 208))[0]);
    for (int e = 0; e < 256; e++) {
        int h = e >> 7;          // 128-element half
        int q = (e & 127) >> 5;  // 32-element quarter
        int l = e & 31;
        int qi    = (q < 2) ? (h * 64 + q * 32 + l) : (h * 64 + (q - 2) * 32 + l);
        int shift = (q < 2) ? 0 : 4;
        int lo = (ql[qi] >> shift) & 0x0F;
        int hi = (qh[h * 32 + l] >> (q * 2)) & 3;
        y[e] = d * (float)sc[h * 8 + q * 2 + (l >> 4)] * (float)((lo | (hi << 4)) - 32);
    }
    return 256;
}

// ---------------------------------------------------------------------------
// public: one block -> floats
// ---------------------------------------------------------------------------

int vbs_dequant_block(vbs_ggml_type_t type, const void *block, float *out_values) {
    if (!block || !out_values) return 0;
    switch (type) {
        case VBS_GGML_TYPE_Q4_0: return deq_q4_0((const uint8_t *)block, out_values);
        case VBS_GGML_TYPE_Q4_1: return deq_q4_1((const uint8_t *)block, out_values);
        case VBS_GGML_TYPE_Q5_0: return deq_q5_0((const uint8_t *)block, out_values);
        case VBS_GGML_TYPE_Q5_1: return deq_q5_1((const uint8_t *)block, out_values);
        case VBS_GGML_TYPE_Q8_0: return deq_q8_0((const uint8_t *)block, out_values);
        case VBS_GGML_TYPE_Q8_1: return deq_q8_1((const uint8_t *)block, out_values);
        case VBS_GGML_TYPE_Q2_K: return deq_q2_k((const uint8_t *)block, out_values);
        case VBS_GGML_TYPE_Q3_K: return deq_q3_k((const uint8_t *)block, out_values);
        case VBS_GGML_TYPE_Q4_K: return deq_q4_k((const uint8_t *)block, out_values);
        case VBS_GGML_TYPE_Q5_K: return deq_q5_k((const uint8_t *)block, out_values);
        case VBS_GGML_TYPE_Q6_K: return deq_q6_k((const uint8_t *)block, out_values);
        default: return 0;
    }
}

// ---------------------------------------------------------------------------
// requantization to Q8_0
// ---------------------------------------------------------------------------

void vbs_requant_q8_0_block(const float *v, void *out_block) {
    uint8_t *blk = (uint8_t *)out_block;
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        float a = fabsf(v[i]);
        if (a > amax) amax = a;
    }
    float d = amax > 0.0f ? amax / 127.0f : 0.0f;
    ((uint16_t *)blk)[0] = vbs_f32_to_f16(d);
    int8_t *q = (int8_t *)(blk + 2);
    if (d == 0.0f) {
        memset(q, 0, 32);
        return;
    }
    float inv = 1.0f / d;
    for (int i = 0; i < 32; i++) {
        int qi = (int)lrintf(v[i] * inv);
        if (qi >  127) qi =  127;
        if (qi < -127) qi = -127;
        q[i] = (int8_t)qi;
    }
}

// ---------------------------------------------------------------------------
// public: whole-tensor conversion
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// I8G: internal 1-byte-per-element target with one f32 scale per ROW.
// Buffer layout: [scales: rows * f32][quants: rows * cols * int8].
// Quality metric: the per-element quantization error bound is amax_row/127
// (rounding to nearest), vs Q8_0's amax_32block/127. The ratio below is the
// element-mean of amax_row / amax_32block: 1.0 means rows are flat and I8G
// is quality-equivalent to Q8_0; larger means outlier rows inflate the
// error of their small-amplitude elements. The engine refuses to AUTO-select
// I8G when this ratio is poor; an explicit pin overrides with the number
// printed.
// ---------------------------------------------------------------------------

static bool i8g_dequant_row(const vbs_tensor_t *t, const uint8_t *row_src,
                            float *rowbuf, uint64_t *src_off_inout) {
    // Dequantize one packed row into rowbuf. Returns false on decoder gaps.
    vbs_block_geometry_t g;
    if (!vbs_get_geometry(t->type, &g)) return false;
    uint64_t blocks = t->columns / g.elements_per_block;
    uint64_t o = 0;
    for (uint64_t b = 0; b < blocks; b++) {
        int n = vbs_dequant_block(t->type, row_src + (size_t)(b * g.bytes_per_block),
                                  rowbuf + o);
        if (n <= 0) return false;
        o += (uint64_t)n;
    }
    *src_off_inout = o;
    return o == t->columns;
}

void *vbs_dequant_tensor_i8g(
    const vbs_tensor_t *t, const void *src, float scale,
    vbs_ggml_type_t *out_type, uint64_t *out_packed_row_bytes,
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
    // Per-row layout: [f32 scale][int8 quants x cols]. Self-contained rows
    // make the format sliceable by the thread pool AND streamable through
    // executor windows (a staged row tile carries its own scales).
    uint64_t row_bytes = cols + 4;
    uint64_t total = rows * row_bytes;
    uint8_t *out = (uint8_t *)malloc(total ? (size_t)total : 1);
    float *rowbuf = (float *)malloc((size_t)cols * sizeof(float));
    if (!out || !rowbuf) {
        free(out); free(rowbuf);
        snprintf(err, err_cap, "i8g: out of memory (%llu bytes)", (unsigned long long)total);
        return NULL;
    }

    uint64_t src_row_bytes = (cols / srcg.elements_per_block) * srcg.bytes_per_block;
    for (uint64_t r = 0; r < rows; r++) {
        const uint8_t *row_src = (const uint8_t *)src + (size_t)(r * src_row_bytes);
        uint64_t n = 0;
        if (!i8g_dequant_row(t, row_src, rowbuf, &n)) {
            // elementwise sources (f32/f16/bf16): decode directly
            if (t->type == VBS_GGML_TYPE_F32) {
                memcpy(rowbuf, row_src, (size_t)cols * 4);
            } else if (t->type == VBS_GGML_TYPE_F16) {
                const uint16_t *h = (const uint16_t *)row_src;
                for (uint64_t i = 0; i < cols; i++) rowbuf[i] = vbs_f16_to_f32(h[i]);
            } else if (t->type == VBS_GGML_TYPE_BF16) {
                const uint16_t *h = (const uint16_t *)row_src;
                for (uint64_t i = 0; i < cols; i++) {
                    uint32_t bits = (uint32_t)h[i] << 16;
                    memcpy(&rowbuf[i], &bits, 4);
                }
            } else {
                free(out); free(rowbuf);
                snprintf(err, err_cap, "i8g: no decoder for source type %d (tensor %s)",
                         (int)t->type, t->name ? t->name : "?");
                return NULL;
            }
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
        memcpy(orow, &d, 4); // row scale
        int8_t *qrow = (int8_t *)(orow + 4);
        for (uint64_t i = 0; i < cols; i++) {
            float v = rowbuf[i] * inv;
            int qi = (int)(v + (v >= 0.0f ? 0.5f : -0.5f));
            if (qi > 127) qi = 127; else if (qi < -127) qi = -127;
            qrow[i] = (int8_t)qi;
        }
    }
    free(rowbuf);
    if (out_type) *out_type = VBS_GGML_TYPE_I8G;
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
        if (!i8g_dequant_row(t, row_src, rowbuf, &n)) {
            if (t->type == VBS_GGML_TYPE_F32) {
                memcpy(rowbuf, row_src, (size_t)cols * 4);
            } else if (t->type == VBS_GGML_TYPE_F16) {
                const uint16_t *h = (const uint16_t *)row_src;
                for (uint64_t i = 0; i < cols; i++) rowbuf[i] = vbs_f16_to_f32(h[i]);
            } else if (t->type == VBS_GGML_TYPE_BF16) {
                const uint16_t *h = (const uint16_t *)row_src;
                for (uint64_t i = 0; i < cols; i++) {
                    uint32_t bits = (uint32_t)h[i] << 16;
                    memcpy(&rowbuf[i], &bits, 4);
                }
            } else {
                free(rowbuf);
                return 1e30; // no decoder: treat as worst quality
            }
        }
        float amax_row = 0.0f;
        for (uint64_t i = 0; i < cols; i++) {
            float a = fabsf(rowbuf[i]);
            if (a > amax_row) amax_row = a;
        }
        if (amax_row <= 0.0f) { continue; } // flat zero row: ratio irrelevant
        for (uint64_t b = 0; b < cols; b += 32) {
            float amax_blk = 0.0f;
            for (int i = 0; i < 32; i++) {
                float a = fabsf(rowbuf[b + i]);
                if (a > amax_blk) amax_blk = a;
            }
            // elements in this 32-block pay amax_row/amax_blk vs Q8_0
            double ratio = amax_blk > 0.0f ? (double)amax_row / (double)amax_blk : 1.0;
            ratio_sum += ratio * 32.0;
        }
    }
    free(rowbuf);
    double elems = (double)rows * (double)cols;
    return elems > 0.0 ? ratio_sum / elems : 1.0;
}

void *vbs_dequant_tensor(
    const vbs_tensor_t *t,
    const void *src,
    vbs_ggml_type_t target,
    float scale,
    vbs_ggml_type_t *out_type,
    uint64_t *out_packed_row_bytes,
    uint64_t *out_total_bytes,
    char *err, size_t err_cap
) {
    if (!t || !src) {
        snprintf(err, err_cap, "dequant: null tensor or source");
        return NULL;
    }
    if (target != VBS_GGML_TYPE_Q8_0 && target != VBS_GGML_TYPE_F16 &&
        target != VBS_GGML_TYPE_F32 && target != VBS_GGML_TYPE_I8G) {
        snprintf(err, err_cap, "dequant: invalid target type %d", (int)target);
        return NULL;
    }
    if (target == VBS_GGML_TYPE_I8G) {
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
        (target == VBS_GGML_TYPE_F32) ? t->columns : (t->columns / dstg.elements_per_block);
    if (target != VBS_GGML_TYPE_F32 && t->columns % dstg.elements_per_block != 0) {
        snprintf(err, err_cap, "dequant: columns (%llu) not aligned to target block (%u)",
                 (unsigned long long)t->columns, dstg.elements_per_block);
        return NULL;
    }
    uint64_t dst_row_bytes =
        (target == VBS_GGML_TYPE_F32) ? t->columns * 4 : dst_blocks_per_row * dstg.bytes_per_block;
    uint64_t total = dst_row_bytes * t->rows;

    uint8_t *out = (uint8_t *)malloc(total ? (size_t)total : 1);
    if (!out) {
        snprintf(err, err_cap, "dequant: out of memory for %llu bytes", (unsigned long long)total);
        return NULL;
    }

    float vals[256];
    const uint8_t *sp = (const uint8_t *)src;
    uint8_t *dp = out;

    // Element-wise formats (f32/f16/bf16) have 1 element per source block;
    // walk those 32 elements at a time (with a ragged tail) so every path
    // emits whole 32-float chunks, or whole 256-element k-quant super-blocks.
    bool elementwise = (srcg.elements_per_block == 1);

    for (uint64_t r = 0; r < t->rows; r++) {
        const uint8_t *row = sp + (size_t)(r * src_blocks_per_row * srcg.bytes_per_block);
        if (elementwise) {
            for (uint64_t e0 = 0; e0 < t->columns; e0 += 32) {
                uint64_t left = t->columns - e0;
                int n = (int)(left < 32 ? left : 32);
                const uint8_t *blk = row + (size_t)(e0 * srcg.bytes_per_block);
                if (t->type == VBS_GGML_TYPE_F32) {
                    const float *f = (const float *)blk;
                    for (int i = 0; i < n; i++) vals[i] = f[i];
                } else if (t->type == VBS_GGML_TYPE_F16) {
                    const uint16_t *h = (const uint16_t *)blk;
                    for (int i = 0; i < n; i++) vals[i] = vbs_f16_to_f32(h[i]);
                } else { // BF16
                    const uint16_t *h = (const uint16_t *)blk;
                    for (int i = 0; i < n; i++) {
                        uint32_t bits = (uint32_t)h[i] << 16;
                        memcpy(&vals[i], &bits, 4);
                    }
                }
                if (scale != 1.0f) {
                    for (int i = 0; i < n; i++) vals[i] *= scale;
                }
                if (target == VBS_GGML_TYPE_F32) {
                    memcpy(dp, vals, (size_t)n * 4);
                    dp += (size_t)n * 4;
                } else { // F16 (Q8_0 target requires column % 32 == 0, already checked)
                    uint16_t *h = (uint16_t *)dp;
                    for (int i = 0; i < n; i++) h[i] = vbs_f32_to_f16(vals[i]);
                    dp += (size_t)n * 2;
                }
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
                if (target == VBS_GGML_TYPE_F32) {
                    memcpy(dp, vals, (size_t)n * 4);
                    dp += (size_t)n * 4;
                } else if (target == VBS_GGML_TYPE_F16) {
                    uint16_t *h = (uint16_t *)dp;
                    for (int i = 0; i < n; i++) h[i] = vbs_f32_to_f16(vals[i]);
                    dp += (size_t)n * 2;
                } else { // Q8_0, 32 per block
                    for (int i = 0; i < n; i += 32) {
                        vbs_requant_q8_0_block(vals + i, dp);
                        dp += 34;
                    }
                }
            }
        }
    }

    if (out_type) *out_type = target;
    if (out_packed_row_bytes) *out_packed_row_bytes = dst_row_bytes;
    if (out_total_bytes) *out_total_bytes = total;
    return out;
}
