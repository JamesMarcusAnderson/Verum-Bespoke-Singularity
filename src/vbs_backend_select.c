/* vbs_backend_select.c — single backend-selection point for the engine.
 *
 * Order (documented in the header): VBS_BACKEND preference, then
 * Vulkan -> CUDA -> HIP -> Metal -> CPU fallback. Native is
 * explicit-only (VBS_BACKEND=native) until SASS kernels exist and
 * is never auto-detected. Selection is explicit and logged: every
 * attempt prints its outcome and reason to stderr, and the final
 * choice is printed on one line. The CPU backend always
 * initializes, so selection never fails — a GPU request with no
 * GPU present runs on CPU and produces correct results instead of
 * an error.
 */

#include "vbs_backend_select.h"
#include "vbs_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* One backend in the selection order: the env-preference value, the
 * display label, the ops table, and whether it is explicit-only
 * (reachable through VBS_BACKEND, never auto-detected). Native is
 * explicit-only until SASS kernels exist: its compile_* fails
 * cleanly today, and auto-selecting it on a machine with a
 * supported GPU would break inference instead of falling back. */
typedef struct {
    const char *env_value;
    const char *label;
    const vbs_backend_ops_t *ops;
    bool explicit_only;
} backend_entry_t;

static const backend_entry_t k_backends[] = {
    { "vulkan", "Vulkan", &g_vbs_vulkan_backend, false },
    { "native", "Native", &g_vbs_native_backend, true },
    { "cuda", "CUDA", &g_vbs_cuda_backend, false },
    { "hip", "HIP", &g_vbs_hip_backend, false },
#if defined(VBS_ENABLE_METAL) && VBS_ENABLE_METAL
    { "metal", "Metal", &g_vbs_metal_backend, false },
#endif
    { "cpu", "CPU", &g_vbs_cpu_backend, false },
};

#define N_BACKENDS (sizeof(k_backends) / sizeof(k_backends[0]))

const char *vbs_backend_name(const vbs_backend_ops_t *ops) {
    for (size_t i = 0; i < N_BACKENDS; i++) {
        if (ops == k_backends[i].ops) return k_backends[i].label;
    }
    return "?";
}

bool vbs_backend_is_cpu(const vbs_backend_ops_t *ops) {
    return ops == &g_vbs_cpu_backend;
}

const char *vbs_backend_last_error(const vbs_backend_ops_t *ops) {
    if (ops == &g_vbs_vulkan_backend) return vbs_vulkan_last_error();
    if (ops == &g_vbs_native_backend) return vbs_native_last_error();
    if (ops == &g_vbs_cuda_backend) return vbs_cuda_last_error();
    if (ops == &g_vbs_hip_backend) return vbs_hip_last_error();
    if (ops == &g_vbs_cpu_backend) return "ok";
    return "not available";
}

/* Try one backend, logging the outcome. Returns the ops on success,
 * NULL on failure. */
static const vbs_backend_ops_t *probe_backend(const char *label,
                                              const vbs_backend_ops_t *ops) {
    bool ok = ops && ops->init && ops->init();
    if (!ok) {
        fprintf(stderr, "[VBS] backend probe: %s: unavailable (%s)\n",
                label, vbs_backend_last_error(ops));
        return NULL;
    }
    size_t n = ops->get_device_count ? ops->get_device_count() : 0;
    const char *dev0 = "";
    if (n > 0 && ops->get_device && ops->get_device(0))
        dev0 = ops->get_device(0)->name;
    fprintf(stderr, "[VBS] backend probe: %s: available (%zu device%s%s%s)\n",
            label, n, n == 1 ? "" : "s",
            dev0[0] ? ", " : "", dev0);
    return ops;
}

const vbs_backend_ops_t *vbs_backend_select(void) {
    const char *pref = getenv("VBS_BACKEND");
    if (pref) {
        const backend_entry_t *wanted = NULL;
        for (size_t i = 0; i < N_BACKENDS; i++) {
            if (strcmp(pref, k_backends[i].env_value) == 0) {
                wanted = &k_backends[i];
                break;
            }
        }
        if (wanted) {
            const vbs_backend_ops_t *got =
                probe_backend(wanted->label, wanted->ops);
            if (got) {
                fprintf(stderr, "[VBS] backend selected: %s (VBS_BACKEND=%s)\n",
                        wanted->label, pref);
                return got;
            }
            fprintf(stderr, "[VBS] backend: VBS_BACKEND=%s unavailable, "
                    "falling through to auto-detect\n", pref);
        } else {
            fprintf(stderr, "[VBS] backend: unknown VBS_BACKEND=%s, "
                    "falling through to auto-detect\n", pref);
        }
    }

    /* Auto-detect: every GPU backend in order, then the CPU fallback.
     * Explicit-only backends (Native, until SASS kernels exist) are
     * skipped here. A preferred backend that failed above is probed
     * again here — its init is cached (probed once), so the second
     * probe is cheap and logs the same failure. */
    for (size_t i = 0; i < N_BACKENDS - 1; i++) {
        if (k_backends[i].explicit_only) continue;
        const vbs_backend_ops_t *got =
            probe_backend(k_backends[i].label, k_backends[i].ops);
        if (got) {
            if (k_backends[i].ops == &g_vbs_vulkan_backend)
                fprintf(stderr, "[VBS] backend selected: Vulkan (universal path)\n");
            else
                fprintf(stderr, "[VBS] backend selected: %s\n", k_backends[i].label);
            return got;
        }
    }
    /* CPU always initializes: the fallback cannot fail. */
    (void)probe_backend("CPU", &g_vbs_cpu_backend);
    fprintf(stderr, "[VBS] backend selected: CPU (fallback — no GPU available)\n");
    return &g_vbs_cpu_backend;
}

static void report_one_backend(const char *label,
                               const vbs_backend_ops_t *ops) {
    bool available = false;
    size_t count = 0;
    if (ops && ops->init) {
        available = ops->init();
        if (available && ops->get_device_count)
            count = ops->get_device_count();
    }
    printf("  %s: available: %s, devices: %zu", label,
           available ? "yes" : "no", count);
    if (!available)
        printf(" (%s)", vbs_backend_last_error(ops));
    printf("\n");
    if (available && ops && ops->get_device) {
        for (size_t i = 0; i < count; i++) {
            vbs_device_t *dev = ops->get_device(i);
            if (dev) printf("    [%zu] %s\n", i, dev->name);
        }
    }
}

void vbs_backend_probe_all(void) {
    printf("=== Backend capability report ===\n");
    printf("  Vulkan: compiled: yes\n");
    printf("  Native: compiled: yes\n");
#if defined(VBS_ENABLE_CUDA) && VBS_ENABLE_CUDA
    printf("  CUDA: compiled: yes\n");
#else
    printf("  CUDA: compiled: no\n");
#endif
#if defined(VBS_ENABLE_HIP) && VBS_ENABLE_HIP
    printf("  HIP: compiled: yes\n");
#else
    printf("  HIP: compiled: no\n");
#endif
#if defined(VBS_ENABLE_METAL) && VBS_ENABLE_METAL
    printf("  Metal: compiled: yes\n");
#else
    printf("  Metal: compiled: no\n");
#endif
    printf("  CPU: compiled: yes\n");
    for (size_t i = 0; i < N_BACKENDS; i++)
        report_one_backend(k_backends[i].label, k_backends[i].ops);
    const vbs_backend_ops_t *sel = vbs_backend_select();
    printf("  selected: %s\n", vbs_backend_name(sel));
    printf("===================================\n");
}
