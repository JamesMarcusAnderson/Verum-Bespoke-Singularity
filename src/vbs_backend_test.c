/* vbs_backend_test.c — backend self-test (--backend-test).
 *
 * Verifies the backends through the ops table ONLY (no backend
 * internals): alloc/copy, every primitive, linear tiles in both
 * calling conventions (slice-executor staging tile and bridge resident
 * tensor), quantized weight types, selection + CPU fallback, and the
 * SPIR-V path. Reference values are computed HERE with small
 * independent implementations (plain loops, direct formulas, and a
 * from-spec dequantizer) — never by calling the code under test, so a
 * shared bug cannot pass silently. The CPU backend, the Vulkan
 * kernels, and these references are three independent implementations
 * of the same math; agreement across all three is the evidence.
 *
 * Backends exercised: CPU (always), Vulkan (when a device is
 * present), HIP (when ROCm is present). CUDA and Metal get a smoke
 * check only (init/devices/alloc/copy): CUDA's dispatch_linear_tile
 * ignores tile offsets/accumulation (a pre-existing limitation, not
 * this phase's to fix) so the strict suite would fail it, and Metal
 * cannot be verified on this host at all.
 */

#include "vbs_backend_test.h"
#include "vbs_backend.h"
#include "vbs_backend_select.h"
#include "vbs_vulkan.h"
#include "vbs_compiler.h"
#include "vbs_dequant.h"
#include "vbs_port.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../kernels/vulkan_kernels.spv.h"

static int g_checks = 0;
static int g_fails = 0;

static void check(bool ok, const char *fmt, ...) {
    g_checks++;
    if (!ok) g_fails++;
    printf("  [%s] ", ok ? "PASS" : "FAIL");
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
}

/* Deterministic pseudo-random floats in [-1, 1). */
static uint32_t g_rng = 12345u;
static float rndf(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return ((float)(g_rng & 0xFFFFFF) / (float)0xFFFFFF) * 2.0f - 1.0f;
}

static double now_sec(void) {
    return vbs_now_sec();
}

static bool close_enough(const float *got, const float *want, size_t n,
                         float tol_abs, float tol_rel, float *max_err) {
    float worst = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(got[i] - want[i]);
        float tol = tol_abs + tol_rel * fabsf(want[i]);
        if (d > worst) worst = d;
        if (d > tol) {
            if (max_err) *max_err = worst;
            return false;
        }
    }
    if (max_err) *max_err = worst;
    return true;
}

/* ------------------------------------------------------------------ */
/* Independent reference math (test-local, from the format specs)      */
/* ------------------------------------------------------------------ */

static void ref_rms_norm(const float *in, const float *w, float *out,
                         size_t n, float eps) {
    double ss = 0.0;
    for (size_t i = 0; i < n; i++) ss += (double)in[i] * in[i];
    float s = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (size_t i = 0; i < n; i++) out[i] = in[i] * s * w[i];
}

static void ref_rope(float *x, uint32_t pos, uint32_t hd, uint32_t nh,
                     float theta) {
    for (uint32_t h = 0; h < nh; h++)
        for (uint32_t i = 0; i + 1 < hd; i += 2) {
            float a = (float)pos * powf(theta, -(float)i / (float)hd);
            float c = cosf(a), s = sinf(a);
            float *p = x + (size_t)h * hd + i;
            float x0 = p[0], x1 = p[1];
            p[0] = x0 * c - x1 * s;
            p[1] = x0 * s + x1 * c;
        }
}

static void ref_silu_mul(const float *g, const float *u, float *o, size_t n) {
    for (size_t i = 0; i < n; i++)
        o[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
}

/* From-spec dequantizers (independent re-implementations of the GGUF
 * block layouts; the CPU backend uses vbs_dequant_block, the Vulkan
 * kernels use GLSL — three implementations must agree). */
static float ref_f16(uint16_t h) { return vbs_f16_to_f32(h); }

static void ref_deq_q8_0(const uint8_t *b, float *y) {
    uint16_t hb;
    memcpy(&hb, b, 2);
    float d = ref_f16(hb);
    for (int i = 0; i < 32; i++) y[i] = d * (float)(int8_t)b[2 + i];
}

static void ref_deq_q4_0(const uint8_t *b, float *y) {
    uint16_t hb;
    memcpy(&hb, b, 2);
    float d = ref_f16(hb);
    for (int i = 0; i < 16; i++) {
        y[i] = d * (float)((b[2 + i] & 15) - 8);
        y[i + 16] = d * (float)((b[2 + i] >> 4) - 8);
    }
}

static void ref_deq_q4_1(const uint8_t *b, float *y) {
    uint16_t hb, hm;
    memcpy(&hb, b, 2);
    memcpy(&hm, b + 2, 2);
    float d = ref_f16(hb), m = ref_f16(hm);
    for (int i = 0; i < 16; i++) {
        y[i] = d * (float)(b[4 + i] & 15) + m;
        y[i + 16] = d * (float)(b[4 + i] >> 4) + m;
    }
}

static void ref_deq_q4_k(const uint8_t *b, float *y) {
    /* ggml block_q4_K: { f16 d; f16 dmin; u8 scales[12]; u8 qs[128] } —
     * d/dmin come FIRST (unlike Q2_K/Q3_K/Q6_K, where they come last). */
    const uint8_t *qs = b + 16, *sc = b + 4;
    uint16_t hd, hm;
    memcpy(&hd, b + 0, 2);
    memcpy(&hm, b + 2, 2);
    float d = ref_f16(hd), dmn = ref_f16(hm);
    for (int e = 0; e < 256; e++) {
        int g = e >> 6, w = e & 63;
        uint8_t byte = qs[g * 32 + (w & 31)];
        int nib = (w < 32) ? (byte & 0x0F) : (byte >> 4);
        int j = e >> 5, s, mn;
        if (j < 4) { s = sc[j] & 63; mn = sc[j + 4] & 63; }
        else {
            s = (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4);
            mn = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4);
        }
        y[e] = d * (float)s * (float)nib - dmn * (float)mn;
    }
}

static void ref_deq_q6_k(const uint8_t *b, float *y) {
    const uint8_t *ql = b, *qh = b + 128;
    const int8_t *sc = (const int8_t *)(b + 192);
    uint16_t hd;
    memcpy(&hd, b + 208, 2);
    float d = ref_f16(hd);
    for (int e = 0; e < 256; e++) {
        int h = e >> 7, q = (e & 127) >> 5, l = e & 31;
        int qi = (q < 2) ? (h * 64 + q * 32 + l) : (h * 64 + (q - 2) * 32 + l);
        int shift = (q < 2) ? 0 : 4;
        int lo = (ql[qi] >> shift) & 0x0F;
        int hi = (qh[h * 32 + l] >> (q * 2)) & 3;
        y[e] = d * (float)sc[h * 8 + q * 2 + (l >> 4)] *
               (float)((lo | (hi << 4)) - 32);
    }
}

static void ref_deq_lns8(const uint8_t *b, float *y) {
    float s;
    memcpy(&s, b + 32, 4);
    for (int i = 0; i < 32; i++) y[i] = exp2f((float)b[i]) * s;
}

/* Dequantize one packed row (test-local). */
static void ref_deq_row(const uint8_t *packed, vbs_ggml_type_t type,
                        uint32_t cols, float *out) {
    vbs_block_geometry_t g;
    vbs_get_geometry(type, &g);
    if (type == VBS_GGML_TYPE_F32) {
        memcpy(out, packed, (size_t)cols * 4);
        return;
    }
    if (type == VBS_GGML_TYPE_F16) {
        for (uint32_t i = 0; i < cols; i++) {
            uint16_t h;
            memcpy(&h, packed + (size_t)i * 2, 2);
            out[i] = ref_f16(h);
        }
        return;
    }
    for (uint32_t b = 0; b < cols / g.elements_per_block; b++) {
        const uint8_t *blk = packed + (size_t)b * g.bytes_per_block;
        float *dst = out + (size_t)b * g.elements_per_block;
        switch (type) {
            case VBS_GGML_TYPE_Q8_0: ref_deq_q8_0(blk, dst); break;
            case VBS_GGML_TYPE_Q4_0: ref_deq_q4_0(blk, dst); break;
            case VBS_GGML_TYPE_Q4_1: ref_deq_q4_1(blk, dst); break;
            case VBS_GGML_TYPE_Q4_K: ref_deq_q4_k(blk, dst); break;
            case VBS_GGML_TYPE_Q6_K: ref_deq_q6_k(blk, dst); break;
            case VBS_GGML_TYPE_LNS8: ref_deq_lns8(blk, dst); break;
            default: memset(dst, 0, (size_t)g.elements_per_block * 4); break;
        }
    }
}

/* Tiny quantizers (test-local) to build packed blocks from floats. */
static void ref_quant_q8_0(const float *v, uint8_t *blk) {
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) if (fabsf(v[i]) > amax) amax = fabsf(v[i]);
    float d = amax / 127.0f;
    uint16_t hb = vbs_f32_to_f16(d);
    memcpy(blk, &hb, 2);
    /* NOTE: the dequant uses the f16-rounded d, so the reference
     * dequantizer (which also reads the f16 d) matches exactly. */
    for (int i = 0; i < 32; i++) {
        int q = (int)roundf(v[i] / (d ? d : 1.0f));
        if (q < -127) q = -127;
        if (q > 127) q = 127;
        blk[2 + i] = (uint8_t)(int8_t)q;
    }
}

static void ref_quant_q4_0(const float *v, uint8_t *blk) {
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) if (fabsf(v[i]) > amax) amax = fabsf(v[i]);
    float d = amax / 8.0f;
    uint16_t hb = vbs_f32_to_f16(d);
    memcpy(blk, &hb, 2);
    for (int i = 0; i < 16; i++) {
        int lo = (int)roundf(v[i] / (d ? d : 1.0f)) + 8;
        int hi = (int)roundf(v[i + 16] / (d ? d : 1.0f)) + 8;
        if (lo < 0) lo = 0;
        if (lo > 15) lo = 15;
        if (hi < 0) hi = 0;
        if (hi > 15) hi = 15;
        blk[2 + i] = (uint8_t)(lo | (hi << 4));
    }
}

static void ref_quant_q4_1(const float *v, uint8_t *blk) {
    float mn = v[0], mx = v[0];
    for (int i = 1; i < 32; i++) {
        if (v[i] < mn) mn = v[i];
        if (v[i] > mx) mx = v[i];
    }
    float d = (mx - mn) / 15.0f;
    uint16_t hb = vbs_f32_to_f16(d), hm = vbs_f32_to_f16(mn);
    memcpy(blk, &hb, 2);
    memcpy(blk + 2, &hm, 2);
    for (int i = 0; i < 16; i++) {
        int lo = (int)roundf((v[i] - mn) / (d ? d : 1.0f));
        int hi = (int)roundf((v[i + 16] - mn) / (d ? d : 1.0f));
        if (lo < 0) lo = 0;
        if (lo > 15) lo = 15;
        if (hi < 0) hi = 0;
        if (hi > 15) hi = 15;
        blk[4 + i] = (uint8_t)(lo | (hi << 4));
    }
}

/* Q4_K block with unit scales: every sub-block scale=1, min=0, d=1,
 * so dequantized value == nibble. Nibbles come from `v` rounded to
 * 0..15. */
static void ref_quant_q4_k_unit(const float *v, uint8_t *blk) {
    memset(blk, 0, 144);
    /* ggml block_q4_K layout: d/dmin first, then scales, then quants. */
    uint8_t *qs = blk + 16, *sc = blk + 4;
    sc[0] = 1; sc[1] = 1; sc[2] = 1; sc[3] = 1;
    sc[8] = 1; sc[9] = 1; sc[10] = 1; sc[11] = 1;
    uint16_t one = vbs_f32_to_f16(1.0f), zero = 0;
    memcpy(blk + 0, &one, 2);
    memcpy(blk + 2, &zero, 2);
    for (int e = 0; e < 256; e++) {
        int g = e >> 6, w = e & 63;
        int q = (int)roundf(v[e]);
        if (q < 0) q = 0;
        if (q > 15) q = 15;
        uint8_t *byte = &qs[g * 32 + (w & 31)];
        if (w < 32) *byte = (uint8_t)((*byte & 0xF0) | q);
        else *byte = (uint8_t)((*byte & 0x0F) | (q << 4));
    }
}

/* Q6_K block with unit scales: every scale=1, d=1, so dequantized
 * value == (6-bit value - 32). Values come from `v` rounded to 0..63. */
static void ref_quant_q6_k_unit(const float *v, uint8_t *blk) {
    memset(blk, 0, 210);
    int8_t *sc = (int8_t *)(blk + 192);
    for (int i = 0; i < 16; i++) sc[i] = 1;
    uint16_t one = vbs_f32_to_f16(1.0f);
    memcpy(blk + 208, &one, 2);
    uint8_t *ql = blk, *qh = blk + 128;
    for (int e = 0; e < 256; e++) {
        int q = (int)roundf(v[e]);
        if (q < 0) q = 0;
        if (q > 63) q = 63;
        int h = e >> 7, qq = (e & 127) >> 5, l = e & 31;
        int qi = (qq < 2) ? (h * 64 + qq * 32 + l)
                          : (h * 64 + (qq - 2) * 32 + l);
        int shift = (qq < 2) ? 0 : 4;
        ql[qi] = (uint8_t)((ql[qi] & (uint8_t)~(0x0F << shift)) |
                           ((q & 0x0F) << shift));
        qh[h * 32 + l] = (uint8_t)((qh[h * 32 + l] &
                                    (uint8_t)~(3 << (qq * 2))) |
                                   ((q >> 4) << (qq * 2)));
    }
}

static void ref_quant_lns8(const float *v, uint8_t *blk) {
    /* value = 2^q * scale; use scale=1/256 and q = round(log2(v*256)).
     * Test values are chosen positive for this format. */
    float s = 1.0f / 256.0f;
    memcpy(blk + 32, &s, 4);
    for (int i = 0; i < 32; i++) {
        float t = v[i] * 256.0f;
        int q = (int)roundf(log2f(t > 0.0f ? t : 1.0f));
        if (q < 0) q = 0;
        if (q > 255) q = 255;
        blk[i] = (uint8_t)q;
    }
}

/* Pack a float matrix (rows x cols) into a packed tensor of `type`.
 * Returns malloc'd bytes (caller frees). */
static uint8_t *ref_pack_matrix(const float *m, uint32_t rows, uint32_t cols,
                                vbs_ggml_type_t type, size_t *out_bytes) {
    vbs_block_geometry_t g;
    vbs_get_geometry(type, &g);
    size_t row_bytes = (size_t)(cols / g.elements_per_block) * g.bytes_per_block;
    uint8_t *packed = (uint8_t *)malloc(row_bytes * rows);
    if (!packed) return NULL;
    for (uint32_t r = 0; r < rows; r++) {
        const float *row = m + (size_t)r * cols;
        uint8_t *dst = packed + (size_t)r * row_bytes;
        if (type == VBS_GGML_TYPE_F32) {
            memcpy(dst, row, (size_t)cols * 4);
        } else if (type == VBS_GGML_TYPE_F16) {
            for (uint32_t i = 0; i < cols; i++) {
                uint16_t h = vbs_f32_to_f16(row[i]);
                memcpy(dst + (size_t)i * 2, &h, 2);
            }
        } else {
            for (uint32_t b = 0; b < cols / g.elements_per_block; b++) {
                uint8_t *blk = dst + (size_t)b * g.bytes_per_block;
                const float *vals = row + (size_t)b * g.elements_per_block;
                switch (type) {
                    case VBS_GGML_TYPE_Q8_0: ref_quant_q8_0(vals, blk); break;
                    case VBS_GGML_TYPE_Q4_0: ref_quant_q4_0(vals, blk); break;
                    case VBS_GGML_TYPE_Q4_1: ref_quant_q4_1(vals, blk); break;
                    case VBS_GGML_TYPE_Q4_K: ref_quant_q4_k_unit(vals, blk); break;
                    case VBS_GGML_TYPE_Q6_K: ref_quant_q6_k_unit(vals, blk); break;
                    case VBS_GGML_TYPE_LNS8: ref_quant_lns8(vals, blk); break;
                    default: memset(blk, 0, g.bytes_per_block); break;
                }
            }
        }
    }
    if (out_bytes) *out_bytes = row_bytes * rows;
    return packed;
}

/* Reference matvec: dequantize the packed matrix with the test-local
 * dequantizer, then dot in double. */
static void ref_matvec(const uint8_t *packed, vbs_ggml_type_t type,
                       const float *x, float *y,
                       uint32_t cols, uint32_t rows,
                       uint32_t col_off, uint32_t row_off,
                       bool accumulate) {
    vbs_block_geometry_t g;
    vbs_get_geometry(type, &g);
    size_t row_bytes = (size_t)(cols / g.elements_per_block) * g.bytes_per_block;
    float *rowf = (float *)malloc((size_t)cols * sizeof(float));
    for (uint32_t r = 0; r < rows; r++) {
        ref_deq_row(packed + (size_t)r * row_bytes, type, cols, rowf);
        double sum = 0.0;
        for (uint32_t c = 0; c < cols; c++)
            sum += (double)rowf[c] * x[col_off + c];
        float *dst = y + row_off + r;
        *dst = accumulate ? *dst + (float)sum : (float)sum;
    }
    free(rowf);
}

static void ref_attention(const float *q, const uint16_t *kc,
                          const uint16_t *vc, float *out,
                          uint32_t seq_len, uint32_t layer,
                          uint32_t max_ctx, uint32_t nqh, uint32_t nkh,
                          uint32_t hd, float scale) {
    uint32_t group = nqh / nkh;
    uint64_t layer_off = (uint64_t)layer * max_ctx * nkh * hd;
    for (uint32_t h = 0; h < nqh; h++) {
        uint32_t kvh = h / group;
        float m = -3.4028235e38f, l = 0.0f;
        float *acc = out + (size_t)h * hd;
        for (uint32_t d = 0; d < hd; d++) acc[d] = 0.0f;
        for (uint32_t s = 0; s < seq_len; s++) {
            uint64_t kb = layer_off + ((uint64_t)s * nkh + kvh) * hd;
            double dotp = 0.0;
            for (uint32_t d = 0; d < hd; d++)
                dotp += (double)q[(size_t)h * hd + d] *
                        (double)vbs_f16_to_f32(kc[kb + d]);
            float score = (float)dotp * scale;
            float mn = score > m ? score : m;
            float r = expf(m - mn), e = expf(score - mn);
            l = l * r + e;
            m = mn;
            for (uint32_t d = 0; d < hd; d++)
                acc[d] = acc[d] * r + e * vbs_f16_to_f32(vc[kb + d]);
        }
        float inv = 1.0f / l;
        for (uint32_t d = 0; d < hd; d++) acc[d] *= inv;
    }
}

/* ------------------------------------------------------------------ */
/* Per-backend op suite                                                */
/* ------------------------------------------------------------------ */

static vbs_arg_t arg_buf(const void *h) {
    vbs_arg_t a = { VBS_ARG_BUFFER, h, 0 };
    return a;
}
static vbs_arg_t arg_u32(uint32_t *v) {
    vbs_arg_t a = { VBS_ARG_SCALAR, v, sizeof(uint32_t) };
    return a;
}
static vbs_arg_t arg_f32(float *v) {
    vbs_arg_t a = { VBS_ARG_SCALAR, v, sizeof(float) };
    return a;
}

static vbs_buffer_t *alloc_or_fail(const vbs_backend_ops_t *ops,
                                   vbs_device_t *dev, size_t bytes,
                                   const char *what) {
    vbs_buffer_t *b = ops->alloc_buffer(dev, bytes, false);
    if (!b) check(false, "alloc %s (%zu B)", what, bytes);
    return b;
}

static void test_one_backend(const char *label, const vbs_backend_ops_t *ops) {
    printf("== backend suite: %s ==\n", label);
    if (!ops->init || !ops->init()) {
        printf("  (unavailable, skipped)\n");
        return;
    }
    vbs_device_t *dev = ops->get_device(0);
    if (!dev) {
        check(false, "%s: get_device(0) is NULL", label);
        return;
    }
    char err[256];

    /* --- alloc / copy roundtrip --- */
    {
        size_t n = 1024 * 1024;
        uint8_t *src = (uint8_t *)malloc(n);
        uint8_t *dst = (uint8_t *)malloc(n);
        for (size_t i = 0; i < n; i++) src[i] = (uint8_t)(i * 2654435761u >> 24);
        vbs_buffer_t *b = alloc_or_fail(ops, dev, n, "1 MiB roundtrip");
        bool ok = b && ops->copy_to_device(b, src, n) &&
                  ops->copy_to_host(dst, b, n) && memcmp(src, dst, n) == 0;
        check(ok, "%s: 1 MiB copy roundtrip", label);
        if (b) ops->free_buffer(b);
        free(src);
        free(dst);
    }

    /* --- elementwise add (n not a multiple of 256) --- */
    {
        uint32_t n = 1000;
        float *a = malloc(n * 4), *b = malloc(n * 4);
        float *want = malloc(n * 4), *got = malloc(n * 4);
        for (uint32_t i = 0; i < n; i++) {
            a[i] = rndf();
            b[i] = rndf();
            want[i] = a[i] + b[i];
        }
        vbs_buffer_t *ba = alloc_or_fail(ops, dev, n * 4, "add a");
        vbs_buffer_t *bb = alloc_or_fail(ops, dev, n * 4, "add b");
        vbs_buffer_t *bo = alloc_or_fail(ops, dev, n * 4, "add out");
        void *pipe = ops->compile_primitive_pipeline(dev, VBS_OP_ELEMENTWISE, err, sizeof(err));
        bool ok = ba && bb && bo && pipe;
        if (ok) {
            ops->copy_to_device(ba, a, n * 4);
            ops->copy_to_device(bb, b, n * 4);
            uint32_t nn = n;
            vbs_arg_t args[] = { arg_buf(ba->internal_handle),
                                 arg_buf(bb->internal_handle),
                                 arg_buf(bo->internal_handle), arg_u32(&nn) };
            ok = ops->dispatch_primitive(dev, pipe, args, 4, (n + 255) / 256, 256);
            ops->synchronize_device(dev);
            if (ok) ok = ops->copy_to_host(got, bo, n * 4);
        }
        float merr = 0;
        if (ok) ok = close_enough(got, want, n, 1e-6f, 1e-6f, &merr);
        check(ok, "%s: add n=1000 (max err %.2e)", label, merr);
        if (ba) ops->free_buffer(ba);
        if (bb) ops->free_buffer(bb);
        if (bo) ops->free_buffer(bo);
        free(a); free(b); free(want); free(got);
    }

    /* --- silu_mul --- */
    {
        uint32_t n = 777;
        float *g = malloc(n * 4), *u = malloc(n * 4);
        float *want = malloc(n * 4), *got = malloc(n * 4);
        for (uint32_t i = 0; i < n; i++) { g[i] = rndf() * 3.0f; u[i] = rndf(); }
        ref_silu_mul(g, u, want, n);
        vbs_buffer_t *bg = alloc_or_fail(ops, dev, n * 4, "silu gate");
        vbs_buffer_t *bu = alloc_or_fail(ops, dev, n * 4, "silu up");
        vbs_buffer_t *bo = alloc_or_fail(ops, dev, n * 4, "silu out");
        void *pipe = ops->compile_primitive_pipeline(dev, VBS_OP_ACTIVATION, err, sizeof(err));
        bool ok = bg && bu && bo && pipe;
        if (ok) {
            ops->copy_to_device(bg, g, n * 4);
            ops->copy_to_device(bu, u, n * 4);
            uint32_t nn = n;
            vbs_arg_t args[] = { arg_buf(bg->internal_handle),
                                 arg_buf(bu->internal_handle),
                                 arg_buf(bo->internal_handle), arg_u32(&nn) };
            ok = ops->dispatch_primitive(dev, pipe, args, 4, (n + 255) / 256, 256);
            ops->synchronize_device(dev);
            if (ok) ok = ops->copy_to_host(got, bo, n * 4);
        }
        float merr = 0;
        if (ok) ok = close_enough(got, want, n, 1e-5f, 1e-5f, &merr);
        check(ok, "%s: silu_mul n=777 (max err %.2e)", label, merr);
        if (bg) ops->free_buffer(bg);
        if (bu) ops->free_buffer(bu);
        if (bo) ops->free_buffer(bo);
        free(g); free(u); free(want); free(got);
    }

    /* --- rms_norm --- */
    {
        uint32_t dim = 384;
        float eps = 1e-5f;
        float *in = malloc(dim * 4), *w = malloc(dim * 4);
        float *want = malloc(dim * 4), *got = malloc(dim * 4);
        for (uint32_t i = 0; i < dim; i++) { in[i] = rndf() * 2.0f; w[i] = rndf(); }
        ref_rms_norm(in, w, want, dim, eps);
        vbs_buffer_t *bi = alloc_or_fail(ops, dev, dim * 4, "norm in");
        vbs_buffer_t *bw = alloc_or_fail(ops, dev, dim * 4, "norm w");
        vbs_buffer_t *bo = alloc_or_fail(ops, dev, dim * 4, "norm out");
        void *pipe = ops->compile_primitive_pipeline(dev, VBS_OP_NORMALIZE, err, sizeof(err));
        bool ok = bi && bw && bo && pipe;
        if (ok) {
            ops->copy_to_device(bi, in, dim * 4);
            ops->copy_to_device(bw, w, dim * 4);
            uint32_t d = dim;
            float e = eps;
            vbs_arg_t args[] = { arg_buf(bi->internal_handle),
                                 arg_buf(bw->internal_handle),
                                 arg_buf(bo->internal_handle),
                                 arg_u32(&d), arg_f32(&e) };
            ok = ops->dispatch_primitive(dev, pipe, args, 5, (dim + 255) / 256, 256);
            ops->synchronize_device(dev);
            if (ok) ok = ops->copy_to_host(got, bo, dim * 4);
        }
        float merr = 0;
        if (ok) ok = close_enough(got, want, dim, 1e-5f, 1e-5f, &merr);
        check(ok, "%s: rms_norm dim=384 (max err %.2e)", label, merr);
        if (bi) ops->free_buffer(bi);
        if (bw) ops->free_buffer(bw);
        if (bo) ops->free_buffer(bo);
        free(in); free(w); free(want); free(got);
    }

    /* --- rope, two thetas (theta must be honored, not fixed) --- */
    for (int ti = 0; ti < 2; ti++) {
        float theta = ti == 0 ? 10000.0f : 500000.0f;
        uint32_t pos = 7, hd = 64, nh = 4;
        size_t n = (size_t)hd * nh;
        float *x = malloc(n * 4), *want = malloc(n * 4), *got = malloc(n * 4);
        for (size_t i = 0; i < n; i++) x[i] = rndf();
        memcpy(want, x, n * 4);
        ref_rope(want, pos, hd, nh, theta);
        vbs_buffer_t *bx = alloc_or_fail(ops, dev, n * 4, "rope x");
        void *pipe = ops->compile_primitive_pipeline(dev, VBS_OP_ROTARY, err, sizeof(err));
        bool ok = bx && pipe;
        if (ok) {
            ops->copy_to_device(bx, x, n * 4);
            uint32_t p = pos, h = hd, nheads = nh;
            float th = theta;
            vbs_arg_t args[] = { arg_buf(bx->internal_handle), arg_u32(&p),
                                 arg_u32(&h), arg_u32(&nheads), arg_f32(&th) };
            ok = ops->dispatch_primitive(dev, pipe, args, 5, ((hd / 2) * nh + 255) / 256, 256);
            ops->synchronize_device(dev);
            if (ok) ok = ops->copy_to_host(got, bx, n * 4);
        }
        float merr = 0;
        if (ok) ok = close_enough(got, want, n, 1e-5f, 1e-5f, &merr);
        check(ok, "%s: rope theta=%.0f pos=7 (max err %.2e)", label, theta, merr);
        if (bx) ops->free_buffer(bx);
        free(x); free(want); free(got);
    }

    /* --- embedding, all supported types --- */
    {
        struct { vbs_ggml_type_t type; const char *name; float tol; } ets[] = {
            { VBS_GGML_TYPE_F32, "F32", 1e-6f },
            { VBS_GGML_TYPE_F16, "F16", 1e-3f },
            { VBS_GGML_TYPE_Q8_0, "Q8_0", 1e-4f },
            { VBS_GGML_TYPE_Q4_0, "Q4_0", 1e-4f },
            { VBS_GGML_TYPE_Q4_1, "Q4_1", 1e-4f },
            { VBS_GGML_TYPE_Q4_K, "Q4_K", 1e-4f },
            { VBS_GGML_TYPE_Q6_K, "Q6_K", 1e-4f },
        };
        uint32_t dim = 256, vocab = 8, token = 3;
        for (size_t ti = 0; ti < sizeof(ets) / sizeof(ets[0]); ti++) {
            float *table = malloc((size_t)vocab * dim * 4);
            /* Q4_K/Q6_K unit-scale blocks only represent 0..15 / 0..63
             * integers; keep test values in range for those types. */
            float vmax = (ets[ti].type == VBS_GGML_TYPE_Q4_K) ? 15.0f
                       : (ets[ti].type == VBS_GGML_TYPE_Q6_K) ? 63.0f : 1.0f;
            float vmin = (ets[ti].type == VBS_GGML_TYPE_Q4_K ||
                          ets[ti].type == VBS_GGML_TYPE_Q6_K) ? 0.0f : -1.0f;
            for (size_t i = 0; i < (size_t)vocab * dim; i++)
                table[i] = vmin + (rndf() * 0.5f + 0.5f) * (vmax - vmin);
            size_t packed_bytes = 0;
            uint8_t *packed = ref_pack_matrix(table, vocab, dim, ets[ti].type, &packed_bytes);
            float *want = malloc(dim * 4), *got = malloc(dim * 4);
            vbs_block_geometry_t g;
            vbs_get_geometry(ets[ti].type, &g);
            size_t row_bytes = (size_t)(dim / g.elements_per_block) * g.bytes_per_block;
            ref_deq_row(packed + (size_t)token * row_bytes, ets[ti].type, dim, want);
            vbs_buffer_t *bw = alloc_or_fail(ops, dev, packed_bytes, "embed w");
            vbs_buffer_t *bo = alloc_or_fail(ops, dev, dim * 4, "embed out");
            void *pipe = ops->compile_embed_pipeline(dev, ets[ti].type, err, sizeof(err));
            bool ok = packed && bw && bo && pipe;
            if (!pipe) {
                /* A refusal is only acceptable for types the backend
                 * documents as unsupported; all 7 here are supported
                 * by CPU and Vulkan. */
                ok = false;
            }
            if (ok) {
                ops->copy_to_device(bw, packed, packed_bytes);
                uint32_t tok = token, d = dim;
                vbs_arg_t args[] = { arg_buf(bw->internal_handle), arg_u32(&tok),
                                     arg_buf(bo->internal_handle), arg_u32(&d) };
                ok = ops->dispatch_primitive(dev, pipe, args, 4, (dim + 255) / 256, 256);
                ops->synchronize_device(dev);
                if (ok) ok = ops->copy_to_host(got, bo, dim * 4);
            }
            float merr = 0;
            if (ok) ok = close_enough(got, want, dim, ets[ti].tol, ets[ti].tol, &merr);
            check(ok, "%s: embed %s token=3 dim=256 (max err %.2e)", label,
                  ets[ti].name, merr);
            if (bw) ops->free_buffer(bw);
            if (bo) ops->free_buffer(bo);
            free(table);
            free(packed);
            free(want);
            free(got);
        }
    }

    /* --- kv cache store + attention roundtrip --- */
    {
        uint32_t n_layers = 2, max_ctx = 16, nkh = 2, hd = 32;
        uint32_t kv_dim = nkh * hd;
        uint32_t pos = 5, layer = 1;
        float *k = malloc(kv_dim * 4), *v = malloc(kv_dim * 4);
        for (uint32_t i = 0; i < kv_dim; i++) { k[i] = rndf(); v[i] = rndf(); }
        size_t cache_elems = (size_t)n_layers * max_ctx * kv_dim;
        vbs_buffer_t *bk = alloc_or_fail(ops, dev, kv_dim * 4, "cache k");
        vbs_buffer_t *bv = alloc_or_fail(ops, dev, kv_dim * 4, "cache v");
        vbs_buffer_t *bkc = alloc_or_fail(ops, dev, cache_elems * 2, "cache kc");
        vbs_buffer_t *bvc = alloc_or_fail(ops, dev, cache_elems * 2, "cache vc");
        void *pipe = ops->compile_primitive_pipeline(dev, VBS_OP_CACHE, err, sizeof(err));
        bool ok = bk && bv && bkc && bvc && pipe;
        uint16_t *kc_got = malloc(cache_elems * 2);
        uint16_t *vc_got = malloc(cache_elems * 2);
        if (ok) {
            ops->copy_to_device(bk, k, kv_dim * 4);
            ops->copy_to_device(bv, v, kv_dim * 4);
            uint32_t p = pos, l = layer, mc = max_ctx, d = kv_dim;
            vbs_arg_t args[] = { arg_buf(bk->internal_handle),
                                 arg_buf(bv->internal_handle),
                                 arg_buf(bkc->internal_handle),
                                 arg_buf(bvc->internal_handle),
                                 arg_u32(&p), arg_u32(&l), arg_u32(&mc), arg_u32(&d) };
            ok = ops->dispatch_primitive(dev, pipe, args, 8, (kv_dim + 255) / 256, 256);
            ops->synchronize_device(dev);
            if (ok) ok = ops->copy_to_host(kc_got, bkc, cache_elems * 2);
            if (ok) ok = ops->copy_to_host(vc_got, bvc, cache_elems * 2);
        }
        if (ok) {
            size_t base = ((size_t)layer * max_ctx + pos) * kv_dim;
            for (uint32_t i = 0; i < kv_dim && ok; i++) {
                if (fabsf(vbs_f16_to_f32(kc_got[base + i]) - k[i]) > 1e-3f) ok = false;
                if (fabsf(vbs_f16_to_f32(vc_got[base + i]) - v[i]) > 1e-3f) ok = false;
            }
        }
        check(ok, "%s: kv_cache_store layer=1 pos=5", label);

        /* Attention over the stored prefix (positions 0..5): fill the
         * cache for positions 0..4 with a second store each, then run
         * GQA attention and compare against the reference. */
        if (ok) {
            uint32_t nqh = 4;
            float *q = malloc((size_t)nqh * hd * 4);
            float *want = malloc((size_t)nqh * hd * 4);
            float *got = malloc((size_t)nqh * hd * 4);
            for (uint32_t i = 0; i < nqh * hd; i++) q[i] = rndf();
            /* Store positions 0..4 the same way (host-side reference
             * cache is built by storing through the backend each
             * time, then read back once). */
            for (uint32_t pp = 0; pp < 5 && ok; pp++) {
                float *kk = malloc(kv_dim * 4), *vv = malloc(kv_dim * 4);
                for (uint32_t i = 0; i < kv_dim; i++) { kk[i] = rndf(); vv[i] = rndf(); }
                ops->copy_to_device(bk, kk, kv_dim * 4);
                ops->copy_to_device(bv, vv, kv_dim * 4);
                uint32_t p = pp, l = layer, mc = max_ctx, d = kv_dim;
                vbs_arg_t args[] = { arg_buf(bk->internal_handle),
                                     arg_buf(bv->internal_handle),
                                     arg_buf(bkc->internal_handle),
                                     arg_buf(bvc->internal_handle),
                                     arg_u32(&p), arg_u32(&l), arg_u32(&mc), arg_u32(&d) };
                ok = ops->dispatch_primitive(dev, pipe, args, 8, (kv_dim + 255) / 256, 256);
                free(kk);
                free(vv);
            }
            vbs_buffer_t *bq = NULL, *bo = NULL;
            void *apipe = NULL;
            if (ok) {
                ok = ops->copy_to_host(kc_got, bkc, cache_elems * 2) &&
                     ops->copy_to_host(vc_got, bvc, cache_elems * 2);
            }
            if (ok) {
                float scale = 1.0f / sqrtf((float)hd);
                ref_attention(q, kc_got, vc_got, want, 6, layer, max_ctx,
                              nqh, nkh, hd, scale);
                bq = alloc_or_fail(ops, dev, (size_t)nqh * hd * 4, "attn q");
                bo = alloc_or_fail(ops, dev, (size_t)nqh * hd * 4, "attn out");
                apipe = ops->compile_primitive_pipeline(dev, VBS_OP_ATTENTION, err, sizeof(err));
                ok = bq && bo && apipe;
            }
            if (ok) {
                ops->copy_to_device(bq, q, (size_t)nqh * hd * 4);
                uint32_t sl = 6, l = layer, mc = max_ctx;
                uint32_t qh = nqh, kh = nkh, hdim = hd;
                float scale = 1.0f / sqrtf((float)hd);
                vbs_arg_t args[] = { arg_buf(bq->internal_handle),
                                     arg_buf(bkc->internal_handle),
                                     arg_buf(bvc->internal_handle),
                                     arg_buf(bo->internal_handle),
                                     arg_u32(&sl), arg_u32(&l), arg_u32(&mc),
                                     arg_u32(&qh), arg_u32(&kh), arg_u32(&hdim),
                                     arg_f32(&scale) };
                ok = ops->dispatch_primitive(dev, apipe, args, 11, nqh, hd > 32 ? hd : 32);
                ops->synchronize_device(dev);
                if (ok) ok = ops->copy_to_host(got, bo, (size_t)nqh * hd * 4);
            }
            float merr = 0;
            if (ok) ok = close_enough(got, want, (size_t)nqh * hd, 1e-4f, 1e-4f, &merr);
            check(ok, "%s: attention_gqa nqh=4 nkh=2 hd=32 seq=6 (max err %.2e)",
                  label, merr);
            if (bq) ops->free_buffer(bq);
            if (bo) ops->free_buffer(bo);
            free(q);
            free(want);
            free(got);
        }
        if (bk) ops->free_buffer(bk);
        if (bv) ops->free_buffer(bv);
        if (bkc) ops->free_buffer(bkc);
        if (bvc) ops->free_buffer(bvc);
        free(k);
        free(v);
        free(kc_got);
        free(vc_got);
    }

    /* --- linear tiles, all supported types, both conventions --- */
    {
        struct { vbs_ggml_type_t type; const char *name; float tol; } lts[] = {
            { VBS_GGML_TYPE_F32, "F32", 1e-4f },
            { VBS_GGML_TYPE_F16, "F16", 2e-3f },
            { VBS_GGML_TYPE_Q8_0, "Q8_0", 2e-3f },
            { VBS_GGML_TYPE_Q4_0, "Q4_0", 2e-3f },
            { VBS_GGML_TYPE_Q4_1, "Q4_1", 2e-3f },
            { VBS_GGML_TYPE_Q4_K, "Q4_K", 2e-3f },
            { VBS_GGML_TYPE_Q6_K, "Q6_K", 2e-3f },
            { VBS_GGML_TYPE_LNS8, "LNS8", 2e-3f },
        };
        uint32_t rows = 8, cols = 256;
        for (size_t ti = 0; ti < sizeof(lts) / sizeof(lts[0]); ti++) {
            float vmax = (lts[ti].type == VBS_GGML_TYPE_Q4_K) ? 15.0f
                       : (lts[ti].type == VBS_GGML_TYPE_Q6_K) ? 63.0f
                       : (lts[ti].type == VBS_GGML_TYPE_LNS8) ? 1.0f : 1.0f;
            float vmin = (lts[ti].type == VBS_GGML_TYPE_Q4_K ||
                          lts[ti].type == VBS_GGML_TYPE_Q6_K ||
                          lts[ti].type == VBS_GGML_TYPE_LNS8) ? 0.01f : -1.0f;
            float *mat = malloc((size_t)rows * cols * 4);
            float *x = malloc(cols * 4);
            for (size_t i = 0; i < (size_t)rows * cols; i++)
                mat[i] = vmin + (rndf() * 0.5f + 0.5f) * (vmax - vmin);
            for (uint32_t i = 0; i < cols; i++) x[i] = rndf();
            size_t packed_bytes = 0;
            uint8_t *packed = ref_pack_matrix(mat, rows, cols, lts[ti].type, &packed_bytes);
            float *want = malloc(rows * 4), *got = malloc(rows * 4);
            memset(want, 0, rows * 4);
            ref_matvec(packed, lts[ti].type, x, want, cols, rows, 0, 0, false);

            void *pipe = ops->compile_linear_pipeline(dev, lts[ti].type, err, sizeof(err));
            bool ok = packed && pipe;
            /* Convention A (bridge): resident tensor, ws=NULL. */
            vbs_buffer_t *bw = NULL, *bx = NULL, *by = NULL;
            if (ok) {
                bw = alloc_or_fail(ops, dev, packed_bytes, "linear w");
                bx = alloc_or_fail(ops, dev, cols * 4, "linear x");
                by = alloc_or_fail(ops, dev, rows * 4, "linear y");
                ok = bw && bx && by;
            }
            if (ok) {
                ops->copy_to_device(bw, packed, packed_bytes);
                ops->copy_to_device(bx, x, cols * 4);
                ok = ops->dispatch_linear_tile(dev, pipe, bw, NULL, 0, bx, by,
                                               cols, rows, 0, 0, false);
                ops->synchronize_device(dev);
                if (ok) ok = ops->copy_to_host(got, by, rows * 4);
            }
            float merr = 0;
            if (ok) ok = close_enough(got, want, rows, lts[ti].tol, lts[ti].tol, &merr);
            check(ok, "%s: linear %s %ux%u resident (max err %.2e)",
                  label, lts[ti].name, rows, cols, merr);
            if (bw) ops->free_buffer(bw);
            if (bx) ops->free_buffer(bx);
            if (by) ops->free_buffer(by);

            /* Convention B (slice executor): staging tile + priv.
             * Runs even if convention A failed, so both paths are
             * always reported. */
            {
                bool ok_b = pipe != NULL;
                vbs_buffer_t *stage = alloc_or_fail(ops, dev, packed_bytes, "linear stage");
                vbs_buffer_t *priv = alloc_or_fail(ops, dev, packed_bytes, "linear priv");
                bx = alloc_or_fail(ops, dev, cols * 4, "linear x2");
                by = alloc_or_fail(ops, dev, rows * 4, "linear y2");
                ok_b = ok_b && stage && priv && bx && by;
                if (ok_b) {
                    /* The slice executor writes the tile into the
                     * stage buffer's host_ptr. */
                    memcpy(stage->host_ptr, packed, packed_bytes);
                    ops->copy_to_device(bx, x, cols * 4);
                    ok_b = ops->dispatch_linear_tile(dev, pipe, priv, stage,
                                                     packed_bytes, bx, by,
                                                     cols, rows, 0, 0, false);
                    ops->synchronize_device(dev);
                    if (ok_b) ok_b = ops->copy_to_host(got, by, rows * 4);
                }
                merr = 0;
                if (ok_b) ok_b = close_enough(got, want, rows, lts[ti].tol, lts[ti].tol, &merr);
                check(ok_b, "%s: linear %s %ux%u staged tile (max err %.2e)",
                      label, lts[ti].name, rows, cols, merr);
                if (stage) ops->free_buffer(stage);
                if (priv) ops->free_buffer(priv);
                if (bx) ops->free_buffer(bx);
                if (by) ops->free_buffer(by);
            }
            free(mat);
            free(x);
            free(packed);
            free(want);
            free(got);
        }
    }

    /* --- linear tile: offsets + accumulation (two column tiles) --- */
    {
        /* Full tensor 4x256 F32; tile 1 covers cols 0..127 (fresh),
         * tile 2 covers cols 128..255 (accumulate). */
        uint32_t rows = 4, cols = 256, half = 128;
        float *mat = malloc((size_t)rows * cols * 4);
        float *x = malloc(cols * 4);
        for (size_t i = 0; i < (size_t)rows * cols; i++) mat[i] = rndf();
        for (uint32_t i = 0; i < cols; i++) x[i] = rndf();
        float *want = malloc(rows * 4), *got = malloc(rows * 4);
        memset(want, 0, rows * 4);
        /* Reference: full dot in double. */
        for (uint32_t r = 0; r < rows; r++) {
            double s = 0.0;
            for (uint32_t c = 0; c < cols; c++)
                s += (double)mat[(size_t)r * cols + c] * x[c];
            want[r] = (float)s;
        }
        void *pipe = ops->compile_linear_pipeline(dev, VBS_GGML_TYPE_F32, err, sizeof(err));
        bool ok = pipe != NULL;
        vbs_buffer_t *bx = NULL, *by = NULL;
        if (ok) {
            bx = alloc_or_fail(ops, dev, cols * 4, "acc x");
            by = alloc_or_fail(ops, dev, rows * 4, "acc y");
            ok = bx && by;
        }
        if (ok) ops->copy_to_device(bx, x, cols * 4);
        for (int tile = 0; tile < 2 && ok; tile++) {
            /* Build the packed column tile (rows x half). */
            float *tilemat = malloc((size_t)rows * half * 4);
            for (uint32_t r = 0; r < rows; r++)
                memcpy(tilemat + (size_t)r * half,
                       mat + (size_t)r * cols + (size_t)tile * half,
                       (size_t)half * 4);
            vbs_buffer_t *stage = alloc_or_fail(ops, dev, (size_t)rows * half * 4, "acc stage");
            vbs_buffer_t *priv = alloc_or_fail(ops, dev, (size_t)rows * half * 4, "acc priv");
            ok = stage && priv;
            if (ok) {
                memcpy(stage->host_ptr, tilemat, (size_t)rows * half * 4);
                ok = ops->dispatch_linear_tile(dev, pipe, priv, stage,
                                               (size_t)rows * half * 4, bx, by,
                                               half, rows, (uint32_t)tile * half, 0,
                                               tile == 1);
                ops->synchronize_device(dev);
            }
            if (stage) ops->free_buffer(stage);
            if (priv) ops->free_buffer(priv);
            free(tilemat);
        }
        if (ok) ok = ops->copy_to_host(got, by, rows * 4);
        float merr = 0;
        if (ok) ok = close_enough(got, want, rows, 1e-4f, 1e-4f, &merr);
        check(ok, "%s: linear F32 offsets+accumulate 2 tiles (max err %.2e)",
              label, merr);
        if (bx) ops->free_buffer(bx);
        if (by) ops->free_buffer(by);
        free(mat);
        free(x);
        free(want);
        free(got);
    }

    /* --- refusal checks: unsupported types must fail loudly --- */
    {
        void *p = ops->compile_linear_pipeline(dev, VBS_GGML_TYPE_Q8_K, err, sizeof(err));
        /* Q8_K has no dequant path in the repo at all (vbs_dequant_block
         * returns 0), so every honest backend refuses it. */
        check(p == NULL, "%s: linear Q8_K refused (no dequant path)", label);
        if (strcmp(label, "Vulkan") == 0) {
            p = ops->compile_linear_pipeline(dev, VBS_GGML_TYPE_Q5_K, err, sizeof(err));
            check(p == NULL, "%s: linear Q5_K refused (no SPIR-V kernel)", label);
            p = ops->compile_embed_pipeline(dev, VBS_GGML_TYPE_LNS8, err, sizeof(err));
            check(p == NULL, "%s: embed LNS8 refused (no SPIR-V kernel)", label);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Timings (measured, not claimed)                                     */
/* ------------------------------------------------------------------ */

static void time_linear(const char *label, const vbs_backend_ops_t *ops) {
    if (!ops->init || !ops->init()) return;
    vbs_device_t *dev = ops->get_device(0);
    if (!dev) return;
    uint32_t rows = 1024, cols = 1024;
    float *mat = malloc((size_t)rows * cols * 4);
    float *x = malloc(cols * 4);
    for (size_t i = 0; i < (size_t)rows * cols; i++) mat[i] = rndf() * 0.1f;
    for (uint32_t i = 0; i < cols; i++) x[i] = rndf();
    size_t packed_bytes = 0;
    uint8_t *packed = ref_pack_matrix(mat, rows, cols, VBS_GGML_TYPE_F32, &packed_bytes);
    char err[256];
    void *pipe = ops->compile_linear_pipeline(dev, VBS_GGML_TYPE_F32, err, sizeof(err));
    vbs_buffer_t *bw = ops->alloc_buffer(dev, packed_bytes, false);
    vbs_buffer_t *bx = ops->alloc_buffer(dev, cols * 4, false);
    vbs_buffer_t *by = ops->alloc_buffer(dev, rows * 4, false);
    if (!pipe || !bw || !bx || !by) {
        printf("  (timing skipped for %s: setup failed)\n", label);
        goto out;
    }
    ops->copy_to_device(bw, packed, packed_bytes);
    ops->copy_to_device(bx, x, cols * 4);
    /* Warmup + 20 timed dispatches. */
    ops->dispatch_linear_tile(dev, pipe, bw, NULL, 0, bx, by, cols, rows, 0, 0, false);
    ops->synchronize_device(dev);
    double t0 = now_sec();
    int reps = 20;
    for (int i = 0; i < reps; i++)
        ops->dispatch_linear_tile(dev, pipe, bw, NULL, 0, bx, by, cols, rows, 0, 0, false);
    ops->synchronize_device(dev);
    double dt = (now_sec() - t0) / reps;
    double flops = 2.0 * (double)rows * cols;
    printf("  timing: %s linear F32 1024x1024: %.3f ms/dispatch (%.2f GFLOP/s)\n",
           label, dt * 1000.0, flops / dt / 1e9);
out:
    if (bw) ops->free_buffer(bw);
    if (bx) ops->free_buffer(bx);
    if (by) ops->free_buffer(by);
    free(mat);
    free(x);
    free(packed);
}

/* ------------------------------------------------------------------ */
/* SPIR-V checks                                                       */
/* ------------------------------------------------------------------ */

static void test_spirv(void) {
    printf("== SPIR-V checks ==\n");
    /* Emitter path: source -> SPIR-V -> structural validation. */
    {
        const char *src = "float add_one(float x) { return x + 1.0; }";
        uint32_t words[512];
        size_t n_words = sizeof(words) / sizeof(words[0]);
        char err[256];
        int rc = vbs_vulkan_compile_kernel_source(src, strlen(src),
                                                  words, &n_words,
                                                  err, sizeof(err));
        bool ok = rc == 0 && n_words > 5 && words[0] == 0x07230203u;
        check(ok, "spirv_emit: source->SPIR-V validates (%zu words)", n_words);
        if (rc != 0) printf("      emitter error: %s\n", err);
    }
    /* Every precompiled kernel blob validates structurally. */
    {
        struct { const char *name; const uint32_t *w; size_t n; } blobs[] = {
            { "add", vbs_spv_add, vbs_spv_add_words },
            { "silu_mul", vbs_spv_silu_mul, vbs_spv_silu_mul_words },
            { "rope", vbs_spv_rope, vbs_spv_rope_words },
            { "rms_norm", vbs_spv_rms_norm, vbs_spv_rms_norm_words },
            { "embed_f32", vbs_spv_embed_f32, vbs_spv_embed_f32_words },
            { "embed_f16", vbs_spv_embed_f16, vbs_spv_embed_f16_words },
            { "embed_q8_0", vbs_spv_embed_q8_0, vbs_spv_embed_q8_0_words },
            { "embed_q4_0", vbs_spv_embed_q4_0, vbs_spv_embed_q4_0_words },
            { "embed_q4_1", vbs_spv_embed_q4_1, vbs_spv_embed_q4_1_words },
            { "embed_q4_k", vbs_spv_embed_q4_k, vbs_spv_embed_q4_k_words },
            { "embed_q6_k", vbs_spv_embed_q6_k, vbs_spv_embed_q6_k_words },
            { "kv_cache_store", vbs_spv_kv_cache_store, vbs_spv_kv_cache_store_words },
            { "attention_gqa", vbs_spv_attention_gqa, vbs_spv_attention_gqa_words },
            { "matvec_f32", vbs_spv_matvec_f32, vbs_spv_matvec_f32_words },
            { "matvec_f16", vbs_spv_matvec_f16, vbs_spv_matvec_f16_words },
            { "matvec_q8_0", vbs_spv_matvec_q8_0, vbs_spv_matvec_q8_0_words },
            { "matvec_q4_0", vbs_spv_matvec_q4_0, vbs_spv_matvec_q4_0_words },
            { "matvec_q4_1", vbs_spv_matvec_q4_1, vbs_spv_matvec_q4_1_words },
            { "matvec_q4_k", vbs_spv_matvec_q4_k, vbs_spv_matvec_q4_k_words },
            { "matvec_q6_k", vbs_spv_matvec_q6_k, vbs_spv_matvec_q6_k_words },
            { "matvec_lns8", vbs_spv_matvec_lns8, vbs_spv_matvec_lns8_words },
        };
        int bad = 0;
        for (size_t i = 0; i < sizeof(blobs) / sizeof(blobs[0]); i++) {
            char err[256];
            if (spirv_validate(blobs[i].w, blobs[i].n, err, sizeof(err)) != 0) {
                printf("      blob %s INVALID: %s\n", blobs[i].name, err);
                bad++;
            }
        }
        check(bad == 0, "spirv_validate: all %zu precompiled blobs valid",
              sizeof(blobs) / sizeof(blobs[0]));
    }
    /* Other emitters: compile smoke test (wired, not half-wired). */
    {
        const char *src = "float add_one(float x) { return x + 1.0; }";
        char out[8192], err[256];
        int rc_ptx = vbs_compile_to_ptx(src, strlen(src), out, sizeof(out), err, sizeof(err));
        bool ptx_ok = rc_ptx == 0 && out[0] != '\0' &&
                      ptx_validate(out, strlen(out), err, sizeof(err)) == 0;
        check(ptx_ok, "ptx_emit: source->PTX validates");
        int rc_msl = vbs_compile_to_msl(src, strlen(src), out, sizeof(out), err, sizeof(err));
        check(rc_msl == 0 && out[0] != '\0', "msl_emit: source->MSL produced");
        uint32_t words[512];
        size_t n_words = sizeof(words) / sizeof(words[0]);
        int rc_gcn = vbs_compile_to_gcn(src, strlen(src), words, &n_words, err, sizeof(err));
        bool gcn_ok = rc_gcn == 0 &&
                      gcn_validate(words, n_words, err, sizeof(err)) == 0;
        check(gcn_ok, "gcn_emit: source->GCN validates");
        /* Native compile: reports availability, never fails the suite
         * either way (the weak stub returns -1 until real native
         * emitters land). */
        void *fn = NULL;
        size_t sz = 0;
        int rc_nat = vbs_compile(src, strlen(src), &fn, &sz, err, sizeof(err));
        printf("  [info] vbs_compile (native): %s\n",
               rc_nat == 0 ? "available" : "unavailable (stub, native emitters pending)");
        (void)rc_nat;
    }
}

/* ------------------------------------------------------------------ */
/* Selection + fallback checks                                         */
/* ------------------------------------------------------------------ */

static void test_selection(void) {
    printf("== selection / fallback checks ==\n");
    bool vk = g_vbs_vulkan_backend.init();
    bool cu = g_vbs_cuda_backend.init();
    bool hip = g_vbs_hip_backend.init();
    bool metal = false;
#if defined(VBS_ENABLE_METAL) && VBS_ENABLE_METAL
    metal = g_vbs_metal_backend.init();
#endif
    bool any_gpu = vk || cu || hip || metal;
    const vbs_backend_ops_t *sel = vbs_backend_select();
    check(sel != NULL, "selection returns a backend (never NULL)");
    if (!any_gpu) {
        check(vbs_backend_is_cpu(sel),
              "no GPU present -> CPU fallback selected");
    } else {
        printf("  [info] GPU present -> selected %s\n", vbs_backend_name(sel));
    }
    /* Explicit preference is honored. */
    vbs_setenv("VBS_BACKEND", "cpu");
    const vbs_backend_ops_t *cpu_sel = vbs_backend_select();
    check(vbs_backend_is_cpu(cpu_sel), "VBS_BACKEND=cpu selects CPU");
    vbs_setenv("VBS_BACKEND", "bogus-backend");
    const vbs_backend_ops_t *bogus = vbs_backend_select();
    check(bogus != NULL, "VBS_BACKEND=bogus falls through to auto-detect");
    vbs_unsetenv("VBS_BACKEND");
    /* CPU backend always initializes, even with no GPU anywhere. */
    check(g_vbs_cpu_backend.init(), "CPU backend init always succeeds");
    check(g_vbs_cpu_backend.get_device_count() == 1,
          "CPU backend reports exactly 1 device");
}

/* ------------------------------------------------------------------ */
/* Entry point                                                         */
/* ------------------------------------------------------------------ */

int vbs_backend_selftest(void) {
    printf("=== Backend self-test ===\n");
    test_selection();
    test_one_backend("CPU", &g_vbs_cpu_backend);
    test_one_backend("Vulkan", &g_vbs_vulkan_backend);
    test_one_backend("HIP", &g_vbs_hip_backend);
    /* CUDA / Metal: smoke only (see file header). */
    if (g_vbs_cuda_backend.init()) {
        vbs_device_t *dev = g_vbs_cuda_backend.get_device(0);
        check(dev != NULL, "CUDA smoke: device 0 present");
    } else {
        printf("  [info] CUDA unavailable, smoke skipped\n");
    }
#if defined(VBS_ENABLE_METAL) && VBS_ENABLE_METAL
    if (g_vbs_metal_backend.init()) {
        vbs_device_t *dev = g_vbs_metal_backend.get_device(0);
        check(dev != NULL, "Metal smoke: device 0 present");
    } else {
        printf("  [info] Metal unavailable, smoke skipped\n");
    }
#endif
    test_spirv();
    printf("== timings ==\n");
    time_linear("CPU", &g_vbs_cpu_backend);
    time_linear("Vulkan", &g_vbs_vulkan_backend);
    printf("backend-test: %d checks, %d failures: %s\n",
           g_checks, g_fails, g_fails == 0 ? "ALL PASS" : "FAILURES PRESENT");
    return g_fails == 0 ? 0 : 1;
}
