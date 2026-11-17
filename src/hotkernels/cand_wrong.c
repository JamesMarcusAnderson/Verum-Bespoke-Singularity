/* cand_wrong.c — deliberately incorrect candidate (ABI v1).
 * The Q8_0 dot loop stops one element early (i < 31 instead of 32), so
 * every result is wrong. It exists to prove the correctness gate rejects
 * a candidate that compiles fine and runs fast but computes the wrong
 * answer. It must NEVER be loaded into the live process.
 */
#include "vbs_hotpath.h"
#include "vbs_hot_ref_impl.h"

static void wrong_q8_0(const uint8_t *w, size_t rows, size_t cols,
                       const int8_t *xq, const float *qd, float *y) {
    size_t nblocks = cols / 32;
    for (size_t r = 0; r < rows; r++) {
        const uint8_t *p = w + r * nblocks * 34;
        float sum = 0.0f;
        for (size_t b = 0; b < nblocks; b++) {
            const uint8_t *blk = p + b * 34;
            const int8_t *wq = (const int8_t *)(const void *)(blk + 2);
            int dot = 0;
            for (size_t i = 0; i < 31; i++) dot += (int)wq[i] * (int)xq[b * 32 + i];
            sum += vbs_hot_f16_to_f32(vbs_hot_read_u16(blk)) * qd[b] * (float)dot;
        }
        y[r] = sum;
    }
}

static const vbs_hot_kernels_t g_wrong_table = {
    VBS_HOT_ABI_VERSION, "wrong-result",
    wrong_q8_0, vbs_hot_ref_matvec_q4_0, vbs_hot_ref_matvec_q4_1,
    vbs_hot_ref_matvec_f32, vbs_hot_ref_matvec_f16
};

uint32_t vbs_hot_abi_version(void) { return VBS_HOT_ABI_VERSION; }
const vbs_hot_kernels_t *vbs_hot_kernels(void) { return &g_wrong_table; }
