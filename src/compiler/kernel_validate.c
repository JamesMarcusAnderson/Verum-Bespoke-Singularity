/* kernel_validate.c — structural validator for binary GPU kernel
 * modules (SPIR-V format).
 *
 * Checks only what can be checked without executing anything:
 * the header is well-formed, the version is one the engine knows,
 * every instruction's declared length fits inside the buffer, and
 * the module contains a function end. It does not typecheck the
 * module or verify the driver would accept it — that is the driver's
 * job, and the engine's precompiled blobs additionally passed the
 * external validator at generation time.
 *
 * No external tools, no SDK — pure C11, offline. The magic word and
 * version range are the format's constants, kept literal.
 */

#include "../../include/vbs_compiler.h"
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#define KERNEL_MAGIC 0x07230203u
#define KERNEL_VER_MIN 0x00010000u   /* 1.0 */
#define KERNEL_VER_MAX 0x00010600u   /* 1.6 */
#define OPCODE_FUNCTION_END 56u

int vbs_kernel_validate(const uint32_t *words, size_t word_count,
                        char *err, size_t err_sz) {
    /* A module shorter than its five-word header cannot be valid. */
    if (!words || word_count < 5) {
        snprintf(err, err_sz, "vbs_kernel_validate: too short (%zu words)",
                 word_count);
        return -1;
    }

    if (words[0] != KERNEL_MAGIC) {
        snprintf(err, err_sz,
                 "vbs_kernel_validate: bad magic 0x%08X (expected 0x%08X)",
                 words[0], KERNEL_MAGIC);
        return -1;
    }

    uint32_t version = words[1];
    if (version < KERNEL_VER_MIN || version > KERNEL_VER_MAX) {
        snprintf(err, err_sz,
                 "vbs_kernel_validate: unrecognised version 0x%08X", version);
        return -1;
    }

    if (words[3] == 0) {
        snprintf(err, err_sz, "vbs_kernel_validate: ID bound is zero");
        return -1;
    }

    /* Walk the instruction stream past the header. Each instruction's
     * first word packs its length in the high half and its opcode in
     * the low half; a zero length would loop forever, and a length
     * past the buffer end means the module is truncated. */
    bool saw_function_end = false;
    for (size_t pos = 5; pos < word_count;) {
        uint32_t first = words[pos];
        uint32_t length = first >> 16;
        uint32_t opcode = first & 0xFFFFu;
        if (length == 0) {
            snprintf(err, err_sz,
                     "vbs_kernel_validate: zero word-count at word %zu", pos);
            return -1;
        }
        if (pos + length > word_count) {
            snprintf(err, err_sz,
                     "vbs_kernel_validate: instruction at word %zu overruns buffer",
                     pos);
            return -1;
        }
        if (opcode == OPCODE_FUNCTION_END) saw_function_end = true;
        pos += length;
    }

    if (!saw_function_end) {
        snprintf(err, err_sz, "vbs_kernel_validate: OpFunctionEnd not found");
        return -1;
    }

    return 0;
}
