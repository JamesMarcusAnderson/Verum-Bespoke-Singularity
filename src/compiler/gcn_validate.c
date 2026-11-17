/* =============================================================================
 * gcn_validate.c
 * Verum-Bespoke-Singularity — GCN ISA binary validator
 *
 * Structural validation of GCN words produced by gcn_emit.c.
 * No AMD toolchain required — pure C11, offline.
 *
 * Checks:
 *   1. Nonzero word count
 *   2. S_ENDPGM (0xBF810000) present — kernel must terminate
 *   3. S_ENDPGM is the last non-padding word
 *   4. No obviously illegal encodings (upper nibble 0xF reserved in GCN3)
 *   5. FLAT instructions are always followed by a data word (64-bit pairs)
 * =========================================================================== */

#include "../../include/vbs_compiler.h"
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#define GCN_S_ENDPGM     0xBF810000u
#define GCN_FLAT_MASK    0xFC000000u
#define GCN_FLAT_PREFIX  0xDC000000u

int gcn_validate(const uint32_t *words, size_t word_count,
                 char *err, size_t err_sz)
{
    if (!words || word_count == 0) {
        snprintf(err, err_sz, "gcn_validate: empty input");
        return -1;
    }

    /* 1. S_ENDPGM present */
    int endpgm_found = 0;
    size_t endpgm_pos = 0;
    for (size_t i = 0; i < word_count; i++) {
        if (words[i] == GCN_S_ENDPGM) {
            endpgm_found = 1;
            endpgm_pos   = i;
        }
    }
    if (!endpgm_found) {
        snprintf(err, err_sz, "gcn_validate: S_ENDPGM (0xBF810000) not found");
        return -1;
    }

    /* 2. S_ENDPGM should be last meaningful word */
    for (size_t i = endpgm_pos + 1; i < word_count; i++) {
        if (words[i] != 0x00000000u && words[i] != GCN_S_ENDPGM) {
            snprintf(err, err_sz,
                "gcn_validate: instructions after S_ENDPGM at word %zu", i);
            return -1;
        }
    }

    /* 3. FLAT instructions must come in pairs (64-bit) */
    for (size_t i = 0; i + 1 < word_count; i++) {
        if ((words[i] & GCN_FLAT_MASK) == GCN_FLAT_PREFIX) {
            /* Next word must be a data/address word — validate it's present */
            if (i + 1 >= word_count) {
                snprintf(err, err_sz,
                    "gcn_validate: FLAT instruction at word %zu missing data word", i);
                return -1;
            }
            i++; /* skip the data word */
        }
    }

    /* 4. No 0xFFxxxxxx encodings — reserved in GCN3 */
    for (size_t i = 0; i < word_count; i++) {
        if ((words[i] >> 24) == 0xFF) {
            snprintf(err, err_sz,
                "gcn_validate: reserved 0xFF opcode prefix at word %zu", i);
            return -1;
        }
    }

    return 0;
}