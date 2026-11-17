#ifndef VBS_BACKEND_SELECT_H
#define VBS_BACKEND_SELECT_H

#include "vbs_backend.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Backend selection — ONE place decides the order, and it logs what it
 * decided and why. Every caller (engine, autotune, --backends) goes
 * through here so the order can never drift between call sites.
 *
 * Order: VBS_BACKEND env preference first (vulkan|cuda|hip|metal|cpu),
 * then Vulkan (universal path) -> CUDA -> HIP -> Metal -> CPU.
 * The CPU backend always succeeds, so vbs_backend_select() never
 * returns NULL: a GPU request with no GPU present runs correctly on
 * CPU instead of failing.
 */
const vbs_backend_ops_t *vbs_backend_select(void);

/* Short name for a backend ops table ("Vulkan", "CUDA", "HIP",
 * "Metal", "CPU", "?"). */
const char *vbs_backend_name(const vbs_backend_ops_t *ops);

/* Last init failure reason for a backend ("ok" if it initialized).
 * Backends without a reason string return a generic message. */
const char *vbs_backend_last_error(const vbs_backend_ops_t *ops);

/* True if ops is the CPU fallback backend. */
bool vbs_backend_is_cpu(const vbs_backend_ops_t *ops);

/* Print the backend capability report (--backends): compiled/available/
 * devices per backend, plus the selected backend. Always exits 0. */
void vbs_backend_probe_all(void);

/* Per-backend last-error accessors (defined in the backend files). */
const char *vbs_vulkan_last_error(void);
const char *vbs_cuda_last_error(void);
const char *vbs_hip_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* VBS_BACKEND_SELECT_H */
