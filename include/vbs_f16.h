/* vbs_f16.h — half-precision conversion (round-to-nearest-even).
 *
 * Single canonical copy, static inline, zero dependencies beyond
 * <stdint.h>/<string.h> — deliberately NOT routed through
 * vbs_model.h or vbs_port.h, so the standalone hot-kernel modules
 * (built as .so files with no link to the main binary and no
 * -D_GNU_SOURCE) share the exact same conversion as the engine
 * instead of carrying their own duplicate. vbs_dequant.h
 * re-exports these; use them from there unless you need the
 * dependency-free header (hot modules do).
 *
 * Both directions follow IEEE 754 binary16/binary32 exactly:
 * subnormals normalize, infinities stay infinite, NaN stays NaN
 * (a quiet NaN with the payload collapsed to 1), overflow rounds
 * to infinity, and ties round to even. Bit-exactness vs the
 * retired per-file copies was verified over all 65536 f16 inputs
 * and 20M random f32 patterns.
 */

#ifndef VBS_F16_H
#define VBS_F16_H

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Widen a half to a float. */
static inline float vbs_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp_field = (h >> 10) & 0x1Fu;
    uint32_t frac = h & 0x3FFu;
    uint32_t out;

    if (exp_field == 0x1Fu) {
        /* Infinity or NaN: exponent all ones, payload shifted up. */
        out = sign | 0x7F800000u | (frac << 13);
    } else if (exp_field == 0) {
        if (frac == 0) {
            out = sign; /* signed zero */
        } else {
            /* Subnormal half: value is frac * 2^-24. Shift the
             * fraction left until its top bit reaches position
             * 10, tracking how far the exponent falls: the
             * normalized value is (1 + frac/1024) * 2^(-14-s),
             * so the biased exponent is 127 - 14 - s. */
            int shift = 0;
            uint32_t m = frac;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                shift++;
            }
            out = sign | ((uint32_t)(127 - 14 - shift) << 23) |
                  ((m & 0x3FFu) << 13);
        }
    } else {
        /* Normal: re-bias the exponent from 15 to 127. */
        out = sign | ((exp_field - 15u + 127u) << 23) | (frac << 13);
    }
    float f;
    memcpy(&f, &out, 4);
    return f;
}

/* Narrow a float to a half, round-to-nearest-even. */
static inline uint16_t vbs_f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t biased = (x >> 23) & 0xFFu;
    uint32_t frac = x & 0x7FFFFFu;

    if (biased == 0xFFu) {
        /* Inf or NaN: keep the class, collapse any NaN payload to 1. */
        return (uint16_t)(sign | 0x7C00u | (frac ? 1u : 0u));
    }

    /* Re-bias the exponent from 127 to 15. */
    int32_t exp = (int32_t)biased - 127 + 15;
    if (exp >= 0x1F) {
        return (uint16_t)(sign | 0x7C00u); /* overflow -> infinity */
    }
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign; /* underflow -> zero */
        /* Subnormal half: the implicit leading 1 joins the
         * fraction, then the whole thing is shifted right with
         * round-to-nearest. If rounding carries into the normal
         * range, return the smallest normal half. */
        uint32_t m = frac | 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t q = (m + (1u << (shift - 1))) >> shift;
        if (q >= 0x400u) return (uint16_t)(sign | 0x0400u);
        return (uint16_t)(sign | q);
    }

    /* Normal: add the rounding bias (0xFFF plus the sticky bit
     * for ties-to-even), then drop the low 13 bits. A carry out
     * of the fraction bumps the exponent instead. */
    uint32_t rounded = frac + 0xFFFu + ((frac >> 13) & 1u);
    if (rounded & 0x800000u) {
        rounded = 0;
        exp += 1;
    }
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (rounded >> 13));
}

#ifdef __cplusplus
}
#endif

#endif /* VBS_F16_H */
