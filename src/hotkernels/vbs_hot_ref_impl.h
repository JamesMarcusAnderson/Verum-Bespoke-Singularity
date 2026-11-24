/* vbs_hot_ref_impl.h — reference hot-kernel implementations.
 *
 * Single source of truth for reference semantics. Included by both
 * src/vbs_hotpath.c (the built-in fallback table) and the kernel
 * modules under src/hotkernels/ (which use the reference for the
 * types they do not optimize). The arithmetic is the vbs_cpu.c
 * inner-loop math, factored into whole-matvec functions without
 * changing any operation: quantized dots stay integer-exact and
 * float accumulation stays per-block, in row order.
 *
 * Inclusion contract: the including file must have included
 * vbs_hotpath.h first (for vbs_hot_kernels_t) and
 * <stdint.h>/<stddef.h>/<string.h>. Everything here is static, so
 * each translation unit gets its own copy. Half-precision
 * conversion uses the canonical vbs_f16_to_f32 from vbs_f16.h
 * (static inline, dependency-free, so the standalone .so modules
 * share it with the engine instead of carrying a duplicate).
 */
#ifndef VBS_HOT_REF_IMPL_H
#define VBS_HOT_REF_IMPL_H

#include <stdbool.h>
#include <string.h>
#include "vbs_f16.h"

/* Read an unaligned little-endian u16 (the scale field of a block). */
static inline uint16_t vbs_hot_read_u16(const uint8_t *p) {
    uint16_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

/* 32-element int8 dot for one Q8_0 block: weights are 32 int8s at
 * blk+2, activations are 32 int8s at xq. Integer-exact. */
static inline int vbs_hot_dot_q8_0(const uint8_t *blk, const int8_t *xq) {
    const int8_t *wq = (const int8_t *)(const void *)(blk + 2);
    int dot = 0;
    for (size_t i = 0; i < 32; i++) dot += (int)wq[i] * (int)xq[i];
    return dot;
}

/* 32-element dot for one Q4 block: 16 packed bytes at wq, each
 * holding two nibbles (low -> elements 0..15, high -> 16..31).
 * `bias` is subtracted from every nibble (8 for Q4_0, 0 for Q4_1). */
static inline int vbs_hot_dot_q4(const uint8_t *wq, const int8_t *xq,
                                 int bias) {
    int dot = 0;
    for (size_t i = 0; i < 16; i++) {
        int lo = (int)(wq[i] & 15u) - bias;
        int hi = (int)(wq[i] >> 4) - bias;
        dot += lo * (int)xq[i] + hi * (int)xq[i + 16];
    }
    return dot;
}

/* One block's dot, by block start: weights at `blk`, activations
 * at `xq` (32 int8s). The three quantized types differ only in
 * how the weights are laid out and decoded. */
typedef int (*vbs_hot_block_dot_fn)(const uint8_t *blk, const int8_t *xq);

static inline int vbs_hot_dot_q8_0_blk(const uint8_t *blk, const int8_t *xq) {
    return vbs_hot_dot_q8_0(blk, xq);
}

static inline int vbs_hot_dot_q4_0_blk(const uint8_t *blk, const int8_t *xq) {
    return vbs_hot_dot_q4(blk + 2, xq, 8);
}

static inline int vbs_hot_dot_q4_1_blk(const uint8_t *blk, const int8_t *xq) {
    return vbs_hot_dot_q4(blk + 4, xq, 0);
}

/* Shared quantized matvec skeleton: one row at a time, one block
 * at a time, one float add per block in row order. `dot_fn`
 * decodes one block's dot; `has_min` adds the Q4_1 minimum term
 * as a second add per block, after the scale term — merging them
 * would change the rounding. `qsum` may be NULL when has_min is
 * false. */
static void vbs_hot_matvec_quant(const uint8_t *w, size_t rows, size_t cols,
                                 const int8_t *xq, const float *qd,
                                 const float *qsum, float *y,
                                 size_t block_bytes, vbs_hot_block_dot_fn dot_fn,
                                 bool has_min) {
    size_t nblocks = cols / 32;
    for (size_t r = 0; r < rows; r++) {
        const uint8_t *row = w + r * nblocks * block_bytes;
        float sum = 0.0f;
        for (size_t b = 0; b < nblocks; b++) {
            const uint8_t *blk = row + b * block_bytes;
            int dot = dot_fn(blk, xq + b * 32);
            sum += vbs_f16_to_f32(vbs_hot_read_u16(blk)) * qd[b] * (float)dot;
            if (has_min)
                sum += vbs_f16_to_f32(vbs_hot_read_u16(blk + 2)) * qsum[b];
        }
        y[r] = sum;
    }
}

/* Four rows' dots for one block: `blk` holds the four rows' block
 * starts, `xq_b` the block's 32 activations. The callback fills
 * `dot[4]` (it may add into the zeroed array or assign it outright).
 * `ctx` is the caller's own per-kind data, read-only. */
typedef void (*vbs_hot_dots4_fn)(const void *ctx, const uint8_t *blk[4],
                                 const int8_t *xq_b, int dot[4]);

/* Shared four-row quantized matvec skeleton: groups of four rows
 * share one pass over the input vector, and the per-block scale
 * (and Q4_1 minimum) accumulation is one add per block per row,
 * in order — bit-identical to the single-row skeleton above.
 * Returns the number of rows covered, so the caller can run its
 * own tail over the leftover rows. `qsum` may be NULL when
 * has_min is false. Inline so translation units that include
 * this header without using the four-row path stay warning-free. */
static inline size_t vbs_hot_matvec_quant4(const uint8_t *w, size_t rows,
                                           size_t cols, const int8_t *xq,
                                           const float *qd, const float *qsum,
                                           float *y, size_t block_bytes,
                                           bool has_min, vbs_hot_dots4_fn dots,
                                           const void *ctx) {
    size_t nblocks = cols / 32;
    size_t row_bytes = nblocks * block_bytes;
    size_t r = 0;
    for (; r + 4 <= rows; r += 4) {
        const uint8_t *row[4];
        float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        for (int k = 0; k < 4; k++) row[k] = w + (r + (size_t)k) * row_bytes;
        for (size_t b = 0; b < nblocks; b++) {
            const uint8_t *blk[4];
            for (int k = 0; k < 4; k++) blk[k] = row[k] + b * block_bytes;
            int dot[4] = { 0, 0, 0, 0 };
            dots(ctx, blk, xq + b * 32, dot);
            float qdb = qd[b];
            float qsb = has_min ? qsum[b] : 0.0f;
            for (int k = 0; k < 4; k++) {
                acc[k] += vbs_f16_to_f32(vbs_hot_read_u16(blk[k])) *
                          qdb * (float)dot[k];
                /* Two separate adds per block for Q4_1: the scale
                 * term, then the min term. Merging them would
                 * change the rounding. */
                if (has_min)
                    acc[k] += vbs_f16_to_f32(vbs_hot_read_u16(blk[k] + 2)) * qsb;
            }
        }
        for (int k = 0; k < 4; k++) y[r + (size_t)k] = acc[k];
    }
    return r;
}

static void vbs_hot_ref_matvec_q8_0(const uint8_t *w, size_t rows, size_t cols,
                                    const int8_t *xq, const float *qd, float *y) {
    vbs_hot_matvec_quant(w, rows, cols, xq, qd, NULL, y, 34,
                         vbs_hot_dot_q8_0_blk, false);
}

static void vbs_hot_ref_matvec_q4_0(const uint8_t *w, size_t rows, size_t cols,
                                    const int8_t *xq, const float *qd, float *y) {
    vbs_hot_matvec_quant(w, rows, cols, xq, qd, NULL, y, 18,
                         vbs_hot_dot_q4_0_blk, false);
}

static void vbs_hot_ref_matvec_q4_1(const uint8_t *w, size_t rows, size_t cols,
                                    const int8_t *xq, const float *qd,
                                    const float *qsum, float *y) {
    vbs_hot_matvec_quant(w, rows, cols, xq, qd, qsum, y, 20,
                         vbs_hot_dot_q4_1_blk, true);
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
        for (size_t i = 0; i < cols; i++) sum += vbs_f16_to_f32(wr[i]) * x[i];
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

/* Export the module ABI: version + kernel table pointer. Every
 * loadable module ends with this; `table` is the module's own
 * table variable. */
#define VBS_HOT_EXPORT_ABI(table) \
    uint32_t vbs_hot_abi_version(void) { \
        return VBS_HOT_ABI_VERSION; \
    } \
    const vbs_hot_kernels_t *vbs_hot_kernels(void) { \
        return &(table); \
    }

#endif /* VBS_HOT_REF_IMPL_H */
