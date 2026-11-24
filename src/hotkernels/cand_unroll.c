/* cand_unroll.c — row-unrolled candidate (ABI v1).
 * Processes 4 rows per outer iteration so the Q8 activation vector
 * (xq) and its scales are loaded once per 4 rows instead of once
 * per row. Each row keeps its own integer dot accumulator, and the
 * float accumulation order per row is unchanged (one add per
 * block, in order), so results are bit-identical to the reference.
 * F32/F16 fall back to the reference implementations.
 */
#include "vbs_hotpath.h"
#include "vbs_hot_ref_impl.h"

/* Compute four rows' dots for one block: wq[k] is row k's
 * packed weights, xq_b is the block's 32 activations. `bias` is
 * subtracted from every nibble (Q4_0: 8, Q4_1: 0); the Q8 dot
 * ignores it. */
typedef void (*unroll_dots_fn)(const void *wq[4], const int8_t *xq_b,
                               int dot[4], int bias);

static void unroll_dots_q8(const void *wq_in[4], const int8_t *xq_b,
                           int dot[4], int bias) {
    (void)bias;
    const int8_t *wq[4];
    for (int k = 0; k < 4; k++) wq[k] = (const int8_t *)wq_in[k];
    for (size_t i = 0; i < 32; i++) {
        int x = (int)xq_b[i];
        for (int k = 0; k < 4; k++) dot[k] += (int)wq[k][i] * x;
    }
}

static void unroll_dots_q4(const void *wq_in[4], const int8_t *xq_b,
                           int dot[4], int bias) {
    const uint8_t *wq[4];
    for (int k = 0; k < 4; k++) wq[k] = (const uint8_t *)wq_in[k];
    for (size_t i = 0; i < 16; i++) {
        int xa = (int)xq_b[i], xb = (int)xq_b[i + 16];
        for (int k = 0; k < 4; k++)
            dot[k] += ((int)(wq[k][i] & 15u) - bias) * xa +
                      ((int)(wq[k][i] >> 4) - bias) * xb;
    }
}

/* What differs between the three quantized types: block size,
 * weight offset within the block, the 4-row dot function and its
 * nibble bias, the scalar dot for the tail rows, and whether a
 * per-block minimum term exists. */
typedef struct {
    size_t block_bytes;
    size_t wq_off;
    unroll_dots_fn dots;
    int bias;
    vbs_hot_block_dot_fn tail_dot;
    bool has_min;
} unroll_kind_t;

static const unroll_kind_t k_unroll_q8_0 =
    { 34, 2, unroll_dots_q8, 0, vbs_hot_dot_q8_0_blk, false };
static const unroll_kind_t k_unroll_q4_0 =
    { 18, 2, unroll_dots_q4, 8, vbs_hot_dot_q4_0_blk, false };
static const unroll_kind_t k_unroll_q4_1 =
    { 20, 4, unroll_dots_q4, 0, vbs_hot_dot_q4_1_blk, true };

/* Four rows' dots for one scalar block: the packed weights sit
 * `wq_off` bytes into each block. */
static void unroll_dots4(const void *ctx, const uint8_t *blk[4],
                         const int8_t *xq_b, int dot[4]) {
    const unroll_kind_t *kind = ctx;
    const void *wq[4];
    for (int k = 0; k < 4; k++) wq[k] = blk[k] + kind->wq_off;
    kind->dots(wq, xq_b, dot, kind->bias);
}

/* One row-unrolled quantized matvec: 4 rows per iteration through
 * the shared skeleton, so the activation vector and its scales
 * load once per 4 rows; the leftover rows run through the shared
 * reference skeleton. `qsum` may be NULL when the kind has no
 * minimum term. Each row keeps its own integer dot accumulator,
 * and the float accumulation order per row is unchanged (one add
 * per block, in order), so results are bit-identical to the
 * reference. */
static void unroll_matvec_quant(const uint8_t *w, size_t rows, size_t cols,
                                const int8_t *xq, const float *qd,
                                const float *qsum, float *y,
                                const unroll_kind_t *kind) {
    size_t r = vbs_hot_matvec_quant4(w, rows, cols, xq, qd, qsum, y,
                                     kind->block_bytes, kind->has_min,
                                     unroll_dots4, kind);
    if (r < rows) {
        size_t row_bytes = (cols / 32) * kind->block_bytes;
        vbs_hot_matvec_quant(w + r * row_bytes, rows - r, cols, xq, qd,
                             qsum, y + r, kind->block_bytes,
                             kind->tail_dot, kind->has_min);
    }
}

static void unroll_q8_0(const uint8_t *w, size_t rows, size_t cols,
                        const int8_t *xq, const float *qd, float *y) {
    unroll_matvec_quant(w, rows, cols, xq, qd, NULL, y, &k_unroll_q8_0);
}

static void unroll_q4_0(const uint8_t *w, size_t rows, size_t cols,
                        const int8_t *xq, const float *qd, float *y) {
    unroll_matvec_quant(w, rows, cols, xq, qd, NULL, y, &k_unroll_q4_0);
}

static void unroll_q4_1(const uint8_t *w, size_t rows, size_t cols,
                        const int8_t *xq, const float *qd, const float *qsum,
                        float *y) {
    unroll_matvec_quant(w, rows, cols, xq, qd, qsum, y, &k_unroll_q4_1);
}

static const vbs_hot_kernels_t g_unroll_table = {
    VBS_HOT_ABI_VERSION, "unroll4",
    unroll_q8_0, unroll_q4_0, unroll_q4_1,
    vbs_hot_ref_matvec_f32, vbs_hot_ref_matvec_f16
};

VBS_HOT_EXPORT_ABI(g_unroll_table);
