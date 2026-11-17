/* =============================================================================
 * spirv_validate.c
 * Verum-Bespoke-Singularity — SPIR-V binary validator
 *
 * Structural validation of SPIR-V produced by spirv_emit.c.
 * No spirv-tools, no Vulkan SDK — pure C11, offline.
 *
 * Checks:
 *   1. Magic word 0x07230203
 *   2. Word count >= 5 (minimum valid header)
 *   3. Version field is recognised (1.0 – 1.6)
 *   4. Bound is nonzero
 *   5. Every instruction word-count field is nonzero and in range
 *   6. OpFunctionEnd present
 * =========================================================================== */

#include "../../include/vbs_compiler.h"
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#define SPIRV_MAGIC  0x07230203u
#define OP_FN_END    56u

int spirv_validate(const uint32_t *words, size_t word_count,
                   char *err, size_t err_sz)
{
    if (!words || word_count < 5) {
        snprintf(err, err_sz, "spirv_validate: too short (%zu words)", word_count);
        return -1;
    }

    /* 1. Magic */
    if (words[0] != SPIRV_MAGIC) {
        snprintf(err, err_sz,
            "spirv_validate: bad magic 0x%08X (expected 0x%08X)",
            words[0], SPIRV_MAGIC);
        return -1;
    }

    /* 2. Version */
    uint32_t ver = words[1];
    if (ver < 0x00010000u || ver > 0x00010600u) {
        snprintf(err, err_sz,
            "spirv_validate: unrecognised version 0x%08X", ver);
        return -1;
    }

    /* 3. Bound nonzero */
    if (words[3] == 0) {
        snprintf(err, err_sz, "spirv_validate: ID bound is zero");
        return -1;
    }

    /* 4. Walk instructions */
    size_t  pos      = 5;
    int     fn_end   = 0;
    while (pos < word_count) {
        uint32_t wc     = words[pos] >> 16;
        uint32_t opcode = words[pos] & 0xFFFFu;
        if (wc == 0) {
            snprintf(err, err_sz,
                "spirv_validate: zero word-count at word %zu", pos);
            return -1;
        }
        if (pos + wc > word_count) {
            snprintf(err, err_sz,
                "spirv_validate: instruction at word %zu overruns buffer", pos);
            return -1;
        }
        if (opcode == OP_FN_END) fn_end = 1;
        pos += wc;
    }

    /* 5. OpFunctionEnd found */
    if (!fn_end) {
        snprintf(err, err_sz, "spirv_validate: OpFunctionEnd not found");
        return -1;
    }

    return 0;
}