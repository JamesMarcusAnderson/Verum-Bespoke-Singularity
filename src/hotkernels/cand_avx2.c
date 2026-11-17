/* cand_avx2.c — AVX2 candidate (ABI v1). Requires -mavx2 and a host with
 * AVX2 (the loop checks vbs_hot_cpu_has_avx2() before trying it).
 *
 * The quantized dots are integer-exact, so vectorizing them cannot change
 * the result: int8 values are widened to int16, multiplied pairwise with
 * _mm256_madd_epi16 (which sums adjacent products into int32), and the
 * eight int32 lanes are reduced. Each 32-element block still produces one
 * int dot, and the float accumulation per row stays one add per block, in
 * order — so results are bit-identical to the reference.
 *
 * Two extra wins over a naive vector dot:
 *  - 4 rows are processed per iteration, and the widened activation vector
 *    (x) is computed once per block and shared by all 4 rows.
 *  - Q4 nibbles are unpacked with SSE shuffles (and + shift) instead of
 *    scalar per-byte work.
 * F32/F16 fall back to the reference implementations.
 */
#include "vbs_hotpath.h"
#include "vbs_hot_ref_impl.h"
#include <immintrin.h>

#if !defined(__AVX2__)
#error "cand_avx2.c must be compiled with -mavx2"
#endif

/* Widen a 32-byte int8 vector to two int16 vectors (low/high halves). */
static inline void widen32(__m256i v, __m256i *lo, __m256i *hi) {
    *lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v));
    *hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1));
}

/* Horizontal sum of the eight int32 lanes. */
static inline int hsum256_epi32(__m256i s) {
    __m128i lo = _mm256_castsi256_si128(s);
    __m128i hi = _mm256_extracti128_si256(s, 1);
    __m128i t = _mm_add_epi32(lo, hi);
    t = _mm_hadd_epi32(t, t);
    t = _mm_hadd_epi32(t, t);
    return _mm_cvtsi128_si32(t);
}

/* 32-element int8 dot, given the widened x halves (shared across rows). */
static inline int dot32_shared(__m256i wv, __m256i xlo, __m256i xhi) {
    __m256i wlo, whi;
    widen32(wv, &wlo, &whi);
    __m256i p0 = _mm256_madd_epi16(wlo, xlo);
    __m256i p1 = _mm256_madd_epi16(whi, xhi);
    return hsum256_epi32(_mm256_add_epi32(p0, p1));
}

static void avx2_q8_0(const uint8_t *w, size_t rows, size_t cols,
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
            __m256i xv = _mm256_loadu_si256((const __m256i *)(xq + b * 32));
            __m256i xlo, xhi;
            widen32(xv, &xlo, &xhi);
            float qdb = qd[b];
            int d0 = dot32_shared(_mm256_loadu_si256((const __m256i *)(p0 + b * 34 + 2)), xlo, xhi);
            int d1 = dot32_shared(_mm256_loadu_si256((const __m256i *)(p1 + b * 34 + 2)), xlo, xhi);
            int d2 = dot32_shared(_mm256_loadu_si256((const __m256i *)(p2 + b * 34 + 2)), xlo, xhi);
            int d3 = dot32_shared(_mm256_loadu_si256((const __m256i *)(p3 + b * 34 + 2)), xlo, xhi);
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
            __m256i xv = _mm256_loadu_si256((const __m256i *)(xq + b * 32));
            __m256i xlo, xhi;
            widen32(xv, &xlo, &xhi);
            int dot = dot32_shared(_mm256_loadu_si256((const __m256i *)(p + b * 34 + 2)), xlo, xhi);
            sum += vbs_hot_f16_to_f32(vbs_hot_read_u16(p + b * 34)) * qd[b] * (float)dot;
        }
        y[r] = sum;
    }
}

/* Unpack 16 packed Q4 bytes into a 256-bit int8 vector: elements 0..15 are
 * the low nibbles, 16..31 the high nibbles. sub is 8 for Q4_0, 0 for Q4_1. */
static inline __m256i unpack_q4(const uint8_t *wq, int sub) {
    __m128i v = _mm_loadu_si128((const __m128i *)wq);
    __m128i mask = _mm_set1_epi8(0x0f);
    __m128i lo = _mm_and_si128(v, mask);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(v, 4), mask);
    if (sub) {
        __m128i s = _mm_set1_epi8((char)sub);
        lo = _mm_sub_epi8(lo, s);
        hi = _mm_sub_epi8(hi, s);
    }
    return _mm256_set_m128i(hi, lo);
}

static void avx2_q4_0(const uint8_t *w, size_t rows, size_t cols,
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
            __m256i xv = _mm256_loadu_si256((const __m256i *)(xq + b * 32));
            __m256i xlo, xhi;
            widen32(xv, &xlo, &xhi);
            float qdb = qd[b];
            int d0 = dot32_shared(unpack_q4(p0 + b * 18 + 2, 8), xlo, xhi);
            int d1 = dot32_shared(unpack_q4(p1 + b * 18 + 2, 8), xlo, xhi);
            int d2 = dot32_shared(unpack_q4(p2 + b * 18 + 2, 8), xlo, xhi);
            int d3 = dot32_shared(unpack_q4(p3 + b * 18 + 2, 8), xlo, xhi);
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
            __m256i xv = _mm256_loadu_si256((const __m256i *)(xq + b * 32));
            __m256i xlo, xhi;
            widen32(xv, &xlo, &xhi);
            int dot = dot32_shared(unpack_q4(p + b * 18 + 2, 8), xlo, xhi);
            sum += vbs_hot_f16_to_f32(vbs_hot_read_u16(p + b * 18)) * qd[b] * (float)dot;
        }
        y[r] = sum;
    }
}

static void avx2_q4_1(const uint8_t *w, size_t rows, size_t cols,
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
            __m256i xv = _mm256_loadu_si256((const __m256i *)(xq + b * 32));
            __m256i xlo, xhi;
            widen32(xv, &xlo, &xhi);
            float qdb = qd[b], qsb = qsum[b];
            int d0 = dot32_shared(unpack_q4(p0 + b * 20 + 4, 0), xlo, xhi);
            int d1 = dot32_shared(unpack_q4(p1 + b * 20 + 4, 0), xlo, xhi);
            int d2 = dot32_shared(unpack_q4(p2 + b * 20 + 4, 0), xlo, xhi);
            int d3 = dot32_shared(unpack_q4(p3 + b * 20 + 4, 0), xlo, xhi);
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
            __m256i xv = _mm256_loadu_si256((const __m256i *)(xq + b * 32));
            __m256i xlo, xhi;
            widen32(xv, &xlo, &xhi);
            int dot = dot32_shared(unpack_q4(p + b * 20 + 4, 0), xlo, xhi);
            sum += vbs_hot_f16_to_f32(vbs_hot_read_u16(p + b * 20)) * qd[b] * (float)dot;
            sum += vbs_hot_f16_to_f32(vbs_hot_read_u16(p + b * 20 + 2)) * qsum[b];
        }
        y[r] = sum;
    }
}

static const vbs_hot_kernels_t g_avx2_table = {
    VBS_HOT_ABI_VERSION, "avx2",
    avx2_q8_0, avx2_q4_0, avx2_q4_1,
    vbs_hot_ref_matvec_f32, vbs_hot_ref_matvec_f16
};

uint32_t vbs_hot_abi_version(void) { return VBS_HOT_ABI_VERSION; }
const vbs_hot_kernels_t *vbs_hot_kernels(void) { return &g_avx2_table; }
