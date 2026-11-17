#ifndef VBS_VULKAN_H
#define VBS_VULKAN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Last Vulkan init failure reason (human-readable, for the backend
 * selection log). "ok" when init succeeded. */
const char *vbs_vulkan_last_error(void);

/* Compile C source to SPIR-V through the engine's own compiler
 * (vbs_compile_to_spirv) and structurally validate the result with
 * spirv_validate. Returns 0 on success, -1 on error (err_out filled).
 *
 * HONEST LIMIT: the spirv_emit emitter currently produces a valid
 * structural shell (header + entry point, constant body) with no AST
 * body codegen, so the returned module is a valid no-op, NOT a working
 * kernel. Real dispatch uses the precompiled blobs in
 * kernels/vulkan_kernels.spv.h until the emitter grows body codegen.
 * This function exists so the source->SPIR-V path is wired and tested
 * (see --backend-test), and so a future emitter upgrade flows straight
 * into the backend. */
int vbs_vulkan_compile_kernel_source(const char *src, size_t src_len,
                                     uint32_t *spirv_out, size_t *out_words,
                                     char *err_out, size_t err_sz);

#ifdef __cplusplus
}
#endif

#endif /* VBS_VULKAN_H */
