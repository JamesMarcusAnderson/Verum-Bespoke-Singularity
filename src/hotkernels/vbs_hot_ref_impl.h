/* vbs_hot_ref_impl.h — reference hot-kernel implementations.
 *
 * Single source of truth for reference semantics. Included by both
 * src/vbs_hotpath.c (the built-in fallback table) and the kernel modules
 * under src/hotkernels/ (which use the reference for the types they do not
 * optimize). The code is the vbs_cpu.c matrix()/dot_row() inner loops,
 * factored into whole-matvec functions without changing any arithmetic:
 * quantized dots stay integer-exact and float accumulation stays
 * per-block, in row order.
 *
 * Inclusion contract: the including file must have included vbs_hotpath.h
 * first (for vbs_hot_kernels_t) and <stdint.h>/<stddef.h>/<string.h>.
 * Everything here is static, so each translation unit gets its own copy.
 */
#ifndef VBS_HOT_REF_IMPL_H
#define VBS_HOT_REF_IMPL_H

#include <string.h>

static inline uint16_t vbs_hot_read_u16(const uint8_t *p) {
    uint16_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static inline float vbs_hot_f16_to_f32(uint16_t h) {
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

static void vbs_hot_ref_matvec_q8_0(const uint8_t *w, size_t rows, size_t cols,
                                    const int8_t *xq, const float *qd, float *y) {
    size_t nblocks = cols / 32;
    for (size_t r = 0; r < rows; r++) {
        const uint8_t *p = w + r * nblocks * 34;
        float sum = 0.0f;
        for (size_t b = 0; b < nblocks; b++) {
            const uint8_t *blk = p + b * 34;
            const int8_t *wq = (const int8_t *)(const void *)(blk + 2);
            int dot = 0;
            for (size_t i = 0; i < 32; i++) dot += (int)wq[i] * (int)xq[b * 32 + i];
            sum += vbs_hot_f16_to_f32(vbs_hot_read_u16(blk)) * qd[b] * (float)dot;
        }
        y[r] = sum;
    }
}

static void vbs_hot_ref_matvec_q4_0(const uint8_t *w, size_t rows, size_t cols,
                                    const int8_t *xq, const float *qd, float *y) {
    size_t nblocks = cols / 32;
    for (size_t r = 0; r < rows; r++) {
        const uint8_t *p = w + r * nblocks * 18;
        float sum = 0.0f;
        for (size_t b = 0; b < nblocks; b++) {
            const uint8_t *blk = p + b * 18;
            const uint8_t *wq = blk + 2;
            int dot = 0;
            for (size_t i = 0; i < 16; i++) {
                int lo = (int)(wq[i] & 15u) - 8;
                int hi = (int)(wq[i] >> 4) - 8;
                dot += lo * (int)xq[b * 32 + i] + hi * (int)xq[b * 32 + i + 16];
            }
            sum += vbs_hot_f16_to_f32(vbs_hot_read_u16(blk)) * qd[b] * (float)dot;
        }
        y[r] = sum;
    }
}

static void vbs_hot_ref_matvec_q4_1(const uint8_t *w, size_t rows, size_t cols,
                                    const int8_t *xq, const float *qd,
                                    const float *qsum, float *y) {
    size_t nblocks = cols / 32;
    for (size_t r = 0; r < rows; r++) {
        const uint8_t *p = w + r * nblocks * 20;
        float sum = 0.0f;
        for (size_t b = 0; b < nblocks; b++) {
            const uint8_t *blk = p + b * 20;
            const uint8_t *wq = blk + 4;
            int dot = 0;
            for (size_t i = 0; i < 16; i++) {
                int lo = (int)(wq[i] & 15u);
                int hi = (int)(wq[i] >> 4);
                dot += lo * (int)xq[b * 32 + i] + hi * (int)xq[b * 32 + i + 16];
            }
            sum += vbs_hot_f16_to_f32(vbs_hot_read_u16(blk)) * qd[b] * (float)dot;
            sum += vbs_hot_f16_to_f32(vbs_hot_read_u16(blk + 2)) * qsum[b];
        }
        y[r] = sum;
    }
}

static void vbs_hot_ref_matvec_f32(const float *w, size_t rows, size_t cols,
                                   const float *x, float *y) {
    for (size_t r = 0; r < rows; r++) {
        const float *wr = w + r * cols;
        float sum = 0.0f;
        for (size_t i = 0; i < cols; i++) sum += wr[i] * x[i];
        y[r] = sum;
    }
}

static void vbs_hot_ref_matvec_f16(const uint16_t *w, size_t rows, size_t cols,
                                   const float *x, float *y) {
    for (size_t r = 0; r < rows; r++) {
        const uint16_t *wr = w + r * cols;
        float sum = 0.0f;
        for (size_t i = 0; i < cols; i++) sum += vbs_hot_f16_to_f32(wr[i]) * x[i];
        y[r] = sum;
    }
}

/* Build a full reference table with the given display name. */
#define VBS_HOT_REF_TABLE(var, display_name) \
    static const vbs_hot_kernels_t var = { \
        VBS_HOT_ABI_VERSION, (display_name), \
        vbs_hot_ref_matvec_q8_0, vbs_hot_ref_matvec_q4_0, \
        vbs_hot_ref_matvec_q4_1, vbs_hot_ref_matvec_f32, \
        vbs_hot_ref_matvec_f16 \
    }

#endif /* VBS_HOT_REF_IMPL_H */
