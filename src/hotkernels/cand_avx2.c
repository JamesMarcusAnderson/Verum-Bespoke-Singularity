/* cand_avx2.c — AVX2 candidate (ABI v1). Requires -mavx2 and a
 * host with AVX2 (the loop checks vbs_hot_cpu_has_avx2() before
 * trying it).
 *
 * The quantized dots are integer-exact, so vectorizing them cannot
 * change the result: int8 values are widened to int16, multiplied
 * pairwise with _mm256_madd_epi16 (which sums adjacent products
 * into int32), and the eight int32 lanes are reduced. Each
 * 32-element block still produces one int dot, and the float
 * accumulation per row stays one add per block, in order — so
 * results are bit-identical to the reference.
 *
 * Two extra wins over a naive vector dot:
 *  - 4 rows are processed per iteration, and the widened
 *    activation vector (x) is computed once per block and shared
 *    by all 4 rows.
 *  - Q4 nibbles are unpacked with SSE shuffles (and + shift)
 *    instead of scalar per-byte work.
 * F32/F16 fall back to the reference implementations.
 */
#include "vbs_hotpath.h"
#include "vbs_hot_ref_impl.h"
#include <immintrin.h>

#if !defined(__AVX2__)
#error "cand_avx2.c must be compiled with -mavx2"
#endif

/* Widen a 32-byte int8 vector to two int16 vectors (low/high halves). */
static inline void split_i8_to_i16(__m256i v, __m256i *lo, __m256i *hi) {
    *lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v));
    *hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1));
}

/* Horizontal sum of the eight int32 lanes. */
static inline int reduce_i32x8(__m256i s) {
    __m128i lo = _mm256_castsi256_si128(s);
    __m128i hi = _mm256_extracti128_si256(s, 1);
    __m128i t = _mm_add_epi32(lo, hi);
    t = _mm_hadd_epi32(t, t);
    t = _mm_hadd_epi32(t, t);
    return _mm_cvtsi128_si32(t);
}

/* 32-element int8 dot, given the widened x halves (shared across rows). */
static inline int dot_with_shared_x(__m256i wv, __m256i xlo, __m256i xhi) {
    __m256i wlo, whi;
    split_i8_to_i16(wv, &wlo, &whi);
    __m256i p0 = _mm256_madd_epi16(wlo, xlo);
    __m256i p1 = _mm256_madd_epi16(whi, xhi);
    return reduce_i32x8(_mm256_add_epi32(p0, p1));
}

/* Unpack 16 packed Q4 bytes into a 256-bit int8 vector: elements
 * 0..15 are the low nibbles, 16..31 the high nibbles. `bias` is 8
 * for Q4_0, 0 for Q4_1. */
static inline __m256i unpack_nibbles(const uint8_t *wq, int bias) {
    __m128i v = _mm_loadu_si128((const __m128i *)wq);
    __m128i mask = _mm_set1_epi8(0x0f);
    __m128i lo = _mm_and_si128(v, mask);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(v, 4), mask);
    if (bias) {
        __m128i s = _mm_set1_epi8((char)bias);
        lo = _mm_sub_epi8(lo, s);
        hi = _mm_sub_epi8(hi, s);
    }
    return _mm256_set_m128i(hi, lo);
}

/* Load one block's weights as a 256-bit int8 vector (32 values):
 * raw int8s for Q8_0, unpacked nibbles for Q4. */
typedef __m256i (*avx2_wload_fn)(const uint8_t *blk);

static __m256i avx2_wload_q8_0(const uint8_t *blk) {
    return _mm256_loadu_si256((const __m256i *)(blk + 2));
}

static __m256i avx2_wload_q4_0(const uint8_t *blk) {
    return unpack_nibbles(blk + 2, 8);
}

static __m256i avx2_wload_q4_1(const uint8_t *blk) {
    return unpack_nibbles(blk + 4, 0);
}

/* What differs between the three quantized types: block size,
 * weight loading, and whether a per-block minimum term exists. */
typedef struct {
    size_t block_bytes;
    avx2_wload_fn wload;
    bool has_min;
} avx2_kind_t;

static const avx2_kind_t k_avx2_q8_0 = { 34, avx2_wload_q8_0, false };
static const avx2_kind_t k_avx2_q4_0 = { 18, avx2_wload_q4_0, false };
static const avx2_kind_t k_avx2_q4_1 = { 20, avx2_wload_q4_1, true };

/* Four rows' dots for one AVX2 block: the activation vector is
 * loaded and widened once, then shared by all four rows. */
static void avx2_dots4(const void *ctx, const uint8_t *blk[4],
                       const int8_t *xq_b, int dot[4]) {
    const avx2_kind_t *kind = ctx;
    __m256i xv = _mm256_loadu_si256((const __m256i *)xq_b);
    __m256i xlo, xhi;
    split_i8_to_i16(xv, &xlo, &xhi);
    for (int k = 0; k < 4; k++)
        dot[k] = dot_with_shared_x(kind->wload(blk[k]), xlo, xhi);
}

/* One AVX2 quantized matvec: 4 rows per iteration through the
 * shared skeleton, the widened activation vector shared by all 4
 * rows, a scalar tail for the leftover rows. `qsum` may be NULL
 * when the kind has no minimum term. Float accumulation per row
 * stays one add per block, in order — results are bit-identical
 * to the reference. */
static void avx2_matvec_quant(const uint8_t *w, size_t rows, size_t cols,
                              const int8_t *xq, const float *qd,
                              const float *qsum, float *y,
                              const avx2_kind_t *kind) {
    size_t r = vbs_hot_matvec_quant4(w, rows, cols, xq, qd, qsum, y,
                                     kind->block_bytes, kind->has_min,
                                     avx2_dots4, kind);
    size_t nblocks = cols / 32;
    size_t row_bytes = nblocks * kind->block_bytes;
    for (; r < rows; r++) {
        const uint8_t *row = w + r * row_bytes;
        float sum = 0.0f;
        for (size_t b = 0; b < nblocks; b++) {
            __m256i xv = _mm256_loadu_si256((const __m256i *)(xq + b * 32));
            __m256i xlo, xhi;
            split_i8_to_i16(xv, &xlo, &xhi);
            const uint8_t *blk = row + b * kind->block_bytes;
            int dot = dot_with_shared_x(kind->wload(blk), xlo, xhi);
            sum += vbs_f16_to_f32(vbs_hot_read_u16(blk)) *
                   qd[b] * (float)dot;
            if (kind->has_min)
                sum += vbs_f16_to_f32(vbs_hot_read_u16(blk + 2)) * qsum[b];
        }
        y[r] = sum;
    }
}

static void avx2_q8_0(const uint8_t *w, size_t rows, size_t cols,
                      const int8_t *xq, const float *qd, float *y) {
    avx2_matvec_quant(w, rows, cols, xq, qd, NULL, y, &k_avx2_q8_0);
}

static void avx2_q4_0(const uint8_t *w, size_t rows, size_t cols,
                      const int8_t *xq, const float *qd, float *y) {
    avx2_matvec_quant(w, rows, cols, xq, qd, NULL, y, &k_avx2_q4_0);
}

static void avx2_q4_1(const uint8_t *w, size_t rows, size_t cols,
                      const int8_t *xq, const float *qd, const float *qsum,
                      float *y) {
    avx2_matvec_quant(w, rows, cols, xq, qd, qsum, y, &k_avx2_q4_1);
}

static const vbs_hot_kernels_t g_avx2_table = {
    VBS_HOT_ABI_VERSION, "avx2",
    avx2_q8_0, avx2_q4_0, avx2_q4_1,
    vbs_hot_ref_matvec_f32, vbs_hot_ref_matvec_f16
};

VBS_HOT_EXPORT_ABI(g_avx2_table);
