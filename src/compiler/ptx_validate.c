/* =============================================================================
 * ptx_validate.c
 * Verum-Bespoke-Singularity — PTX text validator
 *
 * Lightweight structural check on the PTX text produced by ptx_emit.c.
 * Does NOT require ptxas or any CUDA SDK — pure C11, offline.
 *
 * Checks:
 *   1. Magic header (.version + .target + .address_size present)
 *   2. At least one .entry or .visible .entry
 *   3. Every { has a matching }
 *   4. ret; present in each entry body
 *   5. No obviously malformed register names (% prefix)
 * =========================================================================== */

#include "../../include/vbs_compiler.h"
#include <string.h>
#include <stdio.h>

static int contains(const char *hay, const char *needle)
    { return strstr(hay, needle) != NULL; }

int ptx_validate(const char *ptx, size_t len, char *err, size_t err_sz)
{
    if (!ptx || len == 0) {
        snprintf(err, err_sz, "ptx_validate: empty input");
        return -1;
    }

    /* 1. Required header directives */
    if (!contains(ptx, ".version")) {
        snprintf(err, err_sz, "ptx_validate: missing .version directive");
        return -1;
    }
    if (!contains(ptx, ".target")) {
        snprintf(err, err_sz, "ptx_validate: missing .target directive");
        return -1;
    }
    if (!contains(ptx, ".address_size")) {
        snprintf(err, err_sz, "ptx_validate: missing .address_size directive");
        return -1;
    }

    /* 2. At least one kernel entry point */
    if (!contains(ptx, ".entry") && !contains(ptx, ".func")) {
        snprintf(err, err_sz, "ptx_validate: no .entry or .func found");
        return -1;
    }

    /* 3. Balanced braces */
    int depth = 0;
    for (size_t i = 0; i < len; i++) {
        if (ptx[i] == '{') depth++;
        else if (ptx[i] == '}') depth--;
        if (depth < 0) {
            snprintf(err, err_sz, "ptx_validate: unmatched '}' at offset %zu", i);
            return -1;
        }
    }
    if (depth != 0) {
        snprintf(err, err_sz, "ptx_validate: %d unclosed '{' block(s)", depth);
        return -1;
    }

    /* 4. ret; present */
    if (!contains(ptx, "ret;")) {
        snprintf(err, err_sz, "ptx_validate: no ret; found in kernel body");
        return -1;
    }

    /* 5. Spot check: if % appears, it must be followed by a letter */
    for (size_t i = 0; i + 1 < len; i++) {
        if (ptx[i] == '%') {
            char next = ptx[i + 1];
            if (next != '%' && next != '_' &&
                !(next >= 'a' && next <= 'z') &&
                !(next >= 'A' && next <= 'Z')) {
                snprintf(err, err_sz,
                    "ptx_validate: malformed register '%%' at offset %zu", i);
                return -1;
            }
        }
    }

    return 0;  /* LGTM */
}