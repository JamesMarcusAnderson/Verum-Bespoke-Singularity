/* cand_nocompile.c — deliberately uncompilable candidate.
 * A syntax error (missing semicolon) proves the compile gate
 * rejects a candidate before it is ever benchmarked or loaded.
 * This file must never compile; if it does, the test is broken.
 */
#include "vbs_hotpath.h"
#include "vbs_hot_ref_impl.h"

static const vbs_hot_kernels_t g_broken_table = {
    VBS_HOT_ABI_VERSION, "broken-syntax",
    vbs_hot_ref_matvec_q8_0, vbs_hot_ref_matvec_q4_0,
    vbs_hot_ref_matvec_q4_1, vbs_hot_ref_matvec_f32,
    vbs_hot_ref_matvec_f16
} /* missing semicolon on purpose */

uint32_t vbs_hot_abi_version(void) {
    return VBS_HOT_ABI_VERSION;
}

const vbs_hot_kernels_t *vbs_hot_kernels(void) {
    return &g_broken_table;
}
