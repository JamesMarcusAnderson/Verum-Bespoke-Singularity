/* cand_unroll.c — row-unrolled candidate (ABI v1).
 * Processes 4 rows per outer iteration so the Q8 activation vector (xq)
 * and its scales are loaded once per 4 rows instead of once per row, and
 * uses 2 accumulators per dot to break the dependency chain. The float
 * accumulation order per row is unchanged (one add per block, in order),
 * so results are bit-identical to the reference. F32/F16 fall back to the
 * reference implementations.
 */
#include "vbs_hotpath.h"
#include "vbs_hot_ref_impl.h"

static void unroll_q8_0(const uint8_t *w, size_t rows, size_t cols,
                        const int8_t *xq, const float *qd, float *y) {
    size_t nblocks = cols / 32;
    size_t rb = nblocks * 34;
    size_t r = 0;
    for (; r + 4 <= rows; r += 4) {
        const uint8_t *p0 = w + (r + 0) * rb;
        const uint8_t *p1 = w + (r + 1) * rb;
        const uint8_t *p2 = w + (r + 2) * rb;
        const uint8_t *p3 = w + (r + 3) * rb;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (size_t b = 0; b < nblocks; b++) {
            const int8_t *xq_b = xq + b * 32;
            float qdb = qd[b];
            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
            const int8_t *w0 = (const int8_t *)(const void *)(p0 + b * 34 + 2);
            const int8_t *w1 = (const int8_t *)(const void *)(p1 + b * 34 + 2);
            const int8_t *w2 = (const int8_t *)(const void *)(p2 + b * 34 + 2);
            const int8_t *w3 = (const int8_t *)(const void *)(p3 + b * 34 + 2);
            for (size_t i = 0; i < 32; i++) {
                int x = (int)xq_b[i];
                d0 += (int)w0[i] * x;
                d1 += (int)w1[i] * x;
                d2 += (int)w2[i] * x;
                d3 += (int)w3[i] * x;
            }
            s0 += vbs_hot_f16_to_f32(vbs_hot_read_u16(p0 + b * 34)) * qdb * (float)d0;
            s1 += vbs_hot_f16_to_f32(vbs_hot_read_u16(p1 + b * 34)) * qdb * (float)d1;
            s2 += vbs_hot_f16_to_f32(vbs_hot_read_u16(p2 + b * 34)) * qdb * (float)d2;
            s3 += vbs_hot_f16_to_f32(vbs_hot_read_u16(p3 + b * 34)) * qdb * (float)d3;
        }
        y[r + 0] = s0; y[r + 1] = s1; y[r + 2] = s2; y[r + 3] = s3;
    }
    for (; r < rows; r++) {
        const uint8_t *p = w + r * rb;
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

static void unroll_q4_0(const uint8_t *w, size_t rows, size_t cols,
                        const int8_t *xq, const float *qd, float *y) {
    size_t nblocks = cols / 32;
    size_t rb = nblocks * 18;
    size_t r = 0;
    for (; r + 4 <= rows; r += 4) {
        const uint8_t *p0 = w + (r + 0) * rb;
        const uint8_t *p1 = w + (r + 1) * rb;
        const uint8_t *p2 = w + (r + 2) * rb;
        const uint8_t *p3 = w + (r + 3) * rb;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (size_t b = 0; b < nblocks; b++) {
            const int8_t *xq_b = xq + b * 32;
            float qdb = qd[b];
            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
            const uint8_t *w0 = p0 + b * 18 + 2;
            const uint8_t *w1 = p1 + b * 18 + 2;
            const uint8_t *w2 = p2 + b * 18 + 2;
            const uint8_t *w3 = p3 + b * 18 + 2;
            for (size_t i = 0; i < 16; i++) {
                int xa = (int)xq_b[i], xb = (int)xq_b[i + 16];
                d0 += ((int)(w0[i] & 15u) - 8) * xa + ((int)(w0[i] >> 4) - 8) * xb;
                d1 += ((int)(w1[i] & 15u) - 8) * xa + ((int)(w1[i] >> 4) - 8) * xb;
                d2 += ((int)(w2[i] & 15u) - 8) * xa + ((int)(w2[i] >> 4) - 8) * xb;
                d3 += ((int)(w3[i] & 15u) - 8) * xa + ((int)(w3[i] >> 4) - 8) * xb;
            }
            s0 += vbs_hot_f16_to_f32(vbs_hot_read_u16(p0 + b * 18)) * qdb * (float)d0;
            s1 += vbs_hot_f16_to_f32(vbs_hot_read_u16(p1 + b * 18)) * qdb * (float)d1;
            s2 += vbs_hot_f16_to_f32(vbs_hot_read_u16(p2 + b * 18)) * qdb * (float)d2;
            s3 += vbs_hot_f16_to_f32(vbs_hot_read_u16(p3 + b * 18)) * qdb * (float)d3;
        }
        y[r + 0] = s0; y[r + 1] = s1; y[r + 2] = s2; y[r + 3] = s3;
    }
    for (; r < rows; r++) {
        const uint8_t *p = w + r * rb;
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

static void unroll_q4_1(const uint8_t *w, size_t rows, size_t cols,
                        const int8_t *xq, const float *qd, const float *qsum,
                        float *y) {
    size_t nblocks = cols / 32;
    size_t rb = nblocks * 20;
    size_t r = 0;
    for (; r + 4 <= rows; r += 4) {
        const uint8_t *p0 = w + (r + 0) * rb;
        const uint8_t *p1 = w + (r + 1) * rb;
        const uint8_t *p2 = w + (r + 2) * rb;
        const uint8_t *p3 = w + (r + 3) * rb;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (size_t b = 0; b < nblocks; b++) {
            const int8_t *xq_b = xq + b * 32;
            float qdb = qd[b], qsb = qsum[b];
            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
            const uint8_t *w0 = p0 + b * 20 + 4;
            const uint8_t *w1 = p1 + b * 20 + 4;
            const uint8_t *w2 = p2 + b * 20 + 4;
            const uint8_t *w3 = p3 + b * 20 + 4;
            for (size_t i = 0; i < 16; i++) {
                int xa = (int)xq_b[i], xb = (int)xq_b[i + 16];
                d0 += (int)(w0[i] & 15u) * xa + (int)(w0[i] >> 4) * xb;
                d1 += (int)(w1[i] & 15u) * xa + (int)(w1[i] >> 4) * xb;
                d2 += (int)(w2[i] & 15u) * xa + (int)(w2[i] >> 4) * xb;
                d3 += (int)(w3[i] & 15u) * xa + (int)(w3[i] >> 4) * xb;
            }
            /* Two separate adds per block, matching the reference exactly. */
            s0 += vbs_hot_f16_to_f32(vbs_hot_read_u16(p0 + b * 20)) * qdb * (float)d0;
            s0 += vbs_hot_f16_to_f32(vbs_hot_read_u16(p0 + b * 20 + 2)) * qsb;
            s1 += vbs_hot_f16_to_f32(vbs_hot_read_u16(p1 + b * 20)) * qdb * (float)d1;
            s1 += vbs_hot_f16_to_f32(vbs_hot_read_u16(p1 + b * 20 + 2)) * qsb;
            s2 += vbs_hot_f16_to_f32(vbs_hot_read_u16(p2 + b * 20)) * qdb * (float)d2;
            s2 += vbs_hot_f16_to_f32(vbs_hot_read_u16(p2 + b * 20 + 2)) * qsb;
            s3 += vbs_hot_f16_to_f32(vbs_hot_read_u16(p3 + b * 20)) * qdb * (float)d3;
            s3 += vbs_hot_f16_to_f32(vbs_hot_read_u16(p3 + b * 20 + 2)) * qsb;
        }
        y[r + 0] = s0; y[r + 1] = s1; y[r + 2] = s2; y[r + 3] = s3;
    }
    for (; r < rows; r++) {
        const uint8_t *p = w + r * rb;
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

static const vbs_hot_kernels_t g_unroll_table = {
    VBS_HOT_ABI_VERSION, "unroll4",
    unroll_q8_0, unroll_q4_0, unroll_q4_1,
    vbs_hot_ref_matvec_f32, vbs_hot_ref_matvec_f16
};

uint32_t vbs_hot_abi_version(void) { return VBS_HOT_ABI_VERSION; }
const vbs_hot_kernels_t *vbs_hot_kernels(void) { return &g_unroll_table; }
