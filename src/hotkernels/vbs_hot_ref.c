/* vbs_hot_ref.c — reference hot-kernel module (ABI v1).
 * The same arithmetic as the built-in table, packaged as a loadable .so so
 * the loop has a known-good module to swap in and roll back to. Built by
 * the Makefile as libvbs_hot_ref.so.
 */
#include "vbs_hotpath.h"
#include "vbs_hot_ref_impl.h"

VBS_HOT_REF_TABLE(g_ref_table, "ref-module");

uint32_t vbs_hot_abi_version(void) { return VBS_HOT_ABI_VERSION; }
const vbs_hot_kernels_t *vbs_hot_kernels(void) { return &g_ref_table; }
