/* vbs_hot_ref.c — reference hot-kernel module (ABI v1).
 * The same arithmetic as the built-in table, packaged as a loadable
 * .so so the loop has a known-good module to swap in and roll back
 * to. Built by the Makefile as libvbs_hot_ref.so.
 */
#include "vbs_hotpath.h"
#include "vbs_hot_ref_impl.h"

VBS_HOT_REF_TABLE(g_ref_table, "ref-module");

VBS_HOT_EXPORT_ABI(g_ref_table);
