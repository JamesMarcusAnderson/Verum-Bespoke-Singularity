#ifndef VBS_GPU_H
#define VBS_GPU_H

/* vbs_gpu.h — public surface of the universal GPU backend.
 *
 * The backend behind this header drives whatever GPU driver the host
 * provides through its compute API (loaded at runtime, no link-time
 * dependency). One backend, every vendor.
 *
 * Kept external interface names in this file's implementation:
 *   - The driver's own entry-point strings ("vkCreateInstance", ...) and
 *     loader filenames ("libvulkan.so.1", ...): the code must speak the
 *     driver's names to load it. See src/vbs_backend_gpu.c.
 *   - vbs_vulkan_last_error: kept per the originality sweep's deferred
 *     item 9 (the backend-registry family stays for James's decision).
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Reason the GPU backend's last init attempt failed, in plain words,
 * for the backend-selection log. Reads "ok" after a successful init. */
const char *vbs_vulkan_last_error(void);

/* Compile C source to a GPU kernel module through the engine's own
 * compiler (vbs_compile_to_kernel) and run the in-repo structural
 * validator (vbs_kernel_validate) over the result. Returns 0 on
 * success, -1 on error (err_out filled).
 *
 * SCOPE: the kernel emitter compiles a straight-line subset of C
 * (int/float arithmetic, locals, returns) into a real module and
 * refuses control flow, calls, subscripts and other types with -1,
 * so a source outside the subset never yields a module. Real
 * dispatch uses the precompiled blobs in kernels/gpu_kernels.spv.h.
 * This function exists so the source-to-kernel path is wired and
 * tested (see --backend-test), and so a future emitter upgrade flows
 * straight into the backend. */
int vbs_gpu_compile_kernel_source(const char *src, size_t src_len,
                                  uint32_t *kernel_out, size_t *out_words,
                                  char *err_out, size_t err_sz);

#ifdef __cplusplus
}
#endif

#endif /* VBS_GPU_H */
