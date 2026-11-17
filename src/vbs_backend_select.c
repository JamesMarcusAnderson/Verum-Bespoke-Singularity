/* vbs_backend_select.c — single backend-selection point for the engine.
 *
 * Order (documented in the header): VBS_BACKEND preference, then
 * Vulkan -> CUDA -> HIP -> Metal -> CPU fallback. Selection is
 * explicit and logged: every attempt prints its outcome and reason
 * to stderr, and the final choice is printed on one line. The CPU
 * backend always initializes, so selection never fails — a GPU
 * request with no GPU present runs on CPU and produces correct
 * results instead of an error.
 */

#include "vbs_backend_select.h"
#include "vbs_vulkan.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *vbs_backend_name(const vbs_backend_ops_t *ops) {
    if (ops == &g_vbs_vulkan_backend) return "Vulkan";
    if (ops == &g_vbs_cuda_backend) return "CUDA";
    if (ops == &g_vbs_hip_backend) return "HIP";
#if defined(VBS_ENABLE_METAL) && VBS_ENABLE_METAL
    if (ops == &g_vbs_metal_backend) return "Metal";
#endif
    if (ops == &g_vbs_cpu_backend) return "CPU";
    return "?";
}

bool vbs_backend_is_cpu(const vbs_backend_ops_t *ops) {
    return ops == &g_vbs_cpu_backend;
}

const char *vbs_backend_last_error(const vbs_backend_ops_t *ops) {
    if (ops == &g_vbs_vulkan_backend) return vbs_vulkan_last_error();
    if (ops == &g_vbs_cuda_backend) return vbs_cuda_last_error();
    if (ops == &g_vbs_hip_backend) return vbs_hip_last_error();
    if (ops == &g_vbs_cpu_backend) return "ok";
    return "not available";
}

/* Try one backend, logging the outcome. Returns the ops on success. */
static const vbs_backend_ops_t *try_backend(const char *label,
                                            const vbs_backend_ops_t *ops) {
    bool ok = ops && ops->init && ops->init();
    if (ok) {
        size_t n = ops->get_device_count ? ops->get_device_count() : 0;
        const char *dev0 = "";
        if (n > 0 && ops->get_device && ops->get_device(0))
            dev0 = ops->get_device(0)->name;
        fprintf(stderr, "[VBS] backend probe: %s: available (%zu device%s%s%s)\n",
                label, n, n == 1 ? "" : "s",
                dev0[0] ? ", " : "", dev0);
        return ops;
    }
    fprintf(stderr, "[VBS] backend probe: %s: unavailable (%s)\n",
            label, vbs_backend_last_error(ops));
    return NULL;
}

const vbs_backend_ops_t *vbs_backend_select(void) {
    const char *pref = getenv("VBS_BACKEND");
    if (pref) {
        const vbs_backend_ops_t *ops = NULL;
        const char *label = NULL;
        if (strcmp(pref, "vulkan") == 0) { ops = &g_vbs_vulkan_backend; label = "Vulkan"; }
        else if (strcmp(pref, "cuda") == 0) { ops = &g_vbs_cuda_backend; label = "CUDA"; }
        else if (strcmp(pref, "hip") == 0) { ops = &g_vbs_hip_backend; label = "HIP"; }
#if defined(VBS_ENABLE_METAL) && VBS_ENABLE_METAL
        else if (strcmp(pref, "metal") == 0) { ops = &g_vbs_metal_backend; label = "Metal"; }
#endif
        else if (strcmp(pref, "cpu") == 0) { ops = &g_vbs_cpu_backend; label = "CPU"; }
        if (ops) {
            const vbs_backend_ops_t *got = try_backend(label, ops);
            if (got) {
                fprintf(stderr, "[VBS] backend selected: %s (VBS_BACKEND=%s)\n",
                        label, pref);
                return got;
            }
            fprintf(stderr, "[VBS] backend: VBS_BACKEND=%s unavailable, "
                    "falling through to auto-detect\n", pref);
        } else {
            fprintf(stderr, "[VBS] backend: unknown VBS_BACKEND=%s, "
                    "falling through to auto-detect\n", pref);
        }
    }

    const vbs_backend_ops_t *got = NULL;
    if ((got = try_backend("Vulkan", &g_vbs_vulkan_backend))) {
        fprintf(stderr, "[VBS] backend selected: Vulkan (universal path)\n");
        return got;
    }
    if ((got = try_backend("CUDA", &g_vbs_cuda_backend))) {
        fprintf(stderr, "[VBS] backend selected: CUDA\n");
        return got;
    }
    if ((got = try_backend("HIP", &g_vbs_hip_backend))) {
        fprintf(stderr, "[VBS] backend selected: HIP\n");
        return got;
    }
#if defined(VBS_ENABLE_METAL) && VBS_ENABLE_METAL
    if ((got = try_backend("Metal", &g_vbs_metal_backend))) {
        fprintf(stderr, "[VBS] backend selected: Metal\n");
        return got;
    }
#endif
    /* CPU always initializes: the fallback cannot fail. */
    (void)try_backend("CPU", &g_vbs_cpu_backend);
    fprintf(stderr, "[VBS] backend selected: CPU (fallback — no GPU available)\n");
    return &g_vbs_cpu_backend;
}

static void print_one_backend(const char *label, const vbs_backend_ops_t *ops) {
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
    print_one_backend("Vulkan", &g_vbs_vulkan_backend);
    print_one_backend("CUDA", &g_vbs_cuda_backend);
    print_one_backend("HIP", &g_vbs_hip_backend);
#if defined(VBS_ENABLE_METAL) && VBS_ENABLE_METAL
    print_one_backend("Metal", &g_vbs_metal_backend);
#endif
    print_one_backend("CPU", &g_vbs_cpu_backend);
    const vbs_backend_ops_t *sel = vbs_backend_select();
    printf("  selected: %s\n", vbs_backend_name(sel));
    printf("===================================\n");
}
