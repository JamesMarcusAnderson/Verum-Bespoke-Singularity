/* cand_slow.c — deliberately wasteful candidate (ABI v1).
 * Numerically CORRECT, but it computes every dot twice and
 * discards the second result. It exists to prove the benchmark
 * gate rejects a candidate that passes correctness but regresses
 * performance. The loop compiles it with -O0 as well, so it is
 * unambiguously slower than the reference.
 */
#include "vbs_hotpath.h"
#include "vbs_hot_ref_impl.h"

/* Wasteful by design: recompute the same dot and throw it away.
 * The impossible comparison keeps the second computation live so
 * the compiler cannot delete it. (A 32-element int8 dot is at
 * most 32*127*127 = 516,128 in magnitude, so it can never equal
 * -123456789 — the branch never fires and the result is the
 * reference dot, exactly.) */
static int slow_dot_q8_0(const uint8_t *blk, const int8_t *xq) {
    int dot = vbs_hot_dot_q8_0(blk, xq);
    int dot_again = vbs_hot_dot_q8_0(blk, xq);
    if (dot_again == -123456789) dot += 1;
    return dot;
}

static void slow_q8_0(const uint8_t *w, size_t rows, size_t cols,
                      const int8_t *xq, const float *qd, float *y) {
    vbs_hot_matvec_quant(w, rows, cols, xq, qd, NULL, y, 34,
                         slow_dot_q8_0, false);
}

static const vbs_hot_kernels_t g_slow_table = {
    VBS_HOT_ABI_VERSION, "slow-wasteful",
    slow_q8_0, vbs_hot_ref_matvec_q4_0, vbs_hot_ref_matvec_q4_1,
    vbs_hot_ref_matvec_f32, vbs_hot_ref_matvec_f16
};

VBS_HOT_EXPORT_ABI(g_slow_table);
