/* cand_wrong.c — deliberately incorrect candidate (ABI v1).
 * The Q8_0 dot loop stops one element early (i < 31 instead of
 * 32), so every result is wrong. It exists to prove the
 * correctness gate rejects a candidate that compiles fine and
 * runs fast but computes the wrong answer. It must NEVER be
 * loaded into the live process.
 */
#include "vbs_hotpath.h"
#include "vbs_hot_ref_impl.h"

/* Defect: 31 elements instead of 32. */
static int wrong_dot_q8_0(const uint8_t *blk, const int8_t *xq) {
    const int8_t *wq = (const int8_t *)(const void *)(blk + 2);
    int dot = 0;
    for (size_t i = 0; i < 31; i++)
        dot += (int)wq[i] * (int)xq[i];
    return dot;
}

static void wrong_q8_0(const uint8_t *w, size_t rows, size_t cols,
                       const int8_t *xq, const float *qd, float *y) {
    vbs_hot_matvec_quant(w, rows, cols, xq, qd, NULL, y, 34,
                         wrong_dot_q8_0, false);
}

static const vbs_hot_kernels_t g_wrong_table = {
    VBS_HOT_ABI_VERSION, "wrong-result",
    wrong_q8_0, vbs_hot_ref_matvec_q4_0, vbs_hot_ref_matvec_q4_1,
    vbs_hot_ref_matvec_f32, vbs_hot_ref_matvec_f16
};

VBS_HOT_EXPORT_ABI(g_wrong_table);
