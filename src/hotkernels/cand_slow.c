/* cand_slow.c — deliberately wasteful candidate (ABI v1).
 * Numerically CORRECT, but it computes every dot twice and discards the
 * second result. It exists to prove the benchmark gate rejects a candidate
 * that passes correctness but regresses performance. The loop compiles it
 * with -O0 as well, so it is unambiguously slower than the reference.
 */
#include "vbs_hotpath.h"
#include "vbs_hot_ref_impl.h"

static void slow_q8_0(const uint8_t *w, size_t rows, size_t cols,
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
            /* Wasteful by design: recompute the same dot and throw it away. */
            int dot2 = 0;
            for (size_t i = 0; i < 32; i++) dot2 += (int)wq[i] * (int)xq[b * 32 + i];
            if (dot2 == -123456789) sum += 1.0f; /* never true; keeps dot2 live */
            sum += vbs_hot_f16_to_f32(vbs_hot_read_u16(blk)) * qd[b] * (float)dot;
        }
        y[r] = sum;
    }
}

static const vbs_hot_kernels_t g_slow_table = {
    VBS_HOT_ABI_VERSION, "slow-wasteful",
    slow_q8_0, vbs_hot_ref_matvec_q4_0, vbs_hot_ref_matvec_q4_1,
    vbs_hot_ref_matvec_f32, vbs_hot_ref_matvec_f16
};

uint32_t vbs_hot_abi_version(void) { return VBS_HOT_ABI_VERSION; }
const vbs_hot_kernels_t *vbs_hot_kernels(void) { return &g_slow_table; }
