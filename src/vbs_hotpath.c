/* vbs_hotpath.c — hot-kernel table, dlopen loader, swap/rollback, counters.
 *
 * The current table is swapped only between tokens (the CPU inference path
 * is single-threaded, so no call can be in flight across a swap). The
 * previous generation is kept so a live regression can roll back to it.
 * dlclose of the older generation happens only when a newer one is loaded,
 * never while it could still be referenced.
 */
#include "vbs_hotpath.h"
#include "hotkernels/vbs_hot_ref_impl.h"
#include "vbs_port.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

VBS_HOT_REF_TABLE(g_vbs_hot_builtin, "builtin-ref");

static const vbs_hot_kernels_t *g_current = &g_vbs_hot_builtin;
static void *g_current_handle = NULL; /* NULL == builtin table */
static char g_current_path[1024] = "";

static const vbs_hot_kernels_t *g_prev = NULL;
static void *g_prev_handle = NULL;
static char g_prev_path[1024] = "";

/* Atomic table swap. GCC/Clang use __atomic_*; MSVC has no __atomic
 * builtins, so plain load/store there — the CPU inference path is
 * single-threaded, so no call can be in flight across a swap. */
static const vbs_hot_kernels_t *hot_table_load(void) {
#if defined(_MSC_VER)
    return g_current;
#else
    const vbs_hot_kernels_t *t;
    __atomic_load(&g_current, &t, __ATOMIC_SEQ_CST);
    return t;
#endif
}

static void hot_table_store(const vbs_hot_kernels_t *t) {
#if defined(_MSC_VER)
    g_current = t;
#else
    __atomic_store(&g_current, &t, __ATOMIC_SEQ_CST);
#endif
}

static vbs_hot_counter_t g_counters[VBS_HOT_NTYPES];

static double now_ns(void) {
    return vbs_now_sec() * 1e9;
}

const vbs_hot_kernels_t *vbs_hot_builtin(void) { return &g_vbs_hot_builtin; }

const vbs_hot_kernels_t *vbs_hot_current(void) {
    return hot_table_load();
}

const char *vbs_hot_current_name(void) { return vbs_hot_current()->name; }
const char *vbs_hot_current_path(void) { return g_current_path; }
bool vbs_hot_has_rollback(void) { return g_prev != NULL; }

const char *vbs_hot_type_name(int idx) {
    static const char *names[VBS_HOT_NTYPES] = {"q8_0", "q4_0", "q4_1", "f32", "f16"};
    return (idx >= 0 && idx < VBS_HOT_NTYPES) ? names[idx] : "?";
}

void vbs_hot_counters(vbs_hot_counter_t out[VBS_HOT_NTYPES]) {
    memcpy(out, g_counters, sizeof(g_counters));
}

void vbs_hot_counters_reset(void) { memset(g_counters, 0, sizeof(g_counters)); }

bool vbs_hot_load(const char *path, char *err, size_t cap) {
    if (!path || !*path) {
        snprintf(err, cap, "hotpath: empty module path");
        return false;
    }
    void *h = vbs_dlopen(path, 1);
    if (!h) {
        snprintf(err, cap, "hotpath: dlopen failed: %s", vbs_dlerror());
        return false;
    }
    uint32_t (*abi_fn)(void) = (uint32_t (*)(void))vbs_dlsym(h, "vbs_hot_abi_version");
    const vbs_hot_kernels_t *(*table_fn)(void) =
        (const vbs_hot_kernels_t *(*)(void))vbs_dlsym(h, "vbs_hot_kernels");
    if (!abi_fn || !table_fn) {
        snprintf(err, cap, "hotpath: module exports missing (needs vbs_hot_abi_version + vbs_hot_kernels)");
        vbs_dlclose(h);
        return false;
    }
    if (abi_fn() != VBS_HOT_ABI_VERSION) {
        snprintf(err, cap, "hotpath: ABI version mismatch (module %u, host %u)",
                 abi_fn(), VBS_HOT_ABI_VERSION);
        vbs_dlclose(h);
        return false;
    }
    const vbs_hot_kernels_t *t = table_fn();
    if (!t || t->abi_version != VBS_HOT_ABI_VERSION || !t->name ||
        !t->matvec_q8_0 || !t->matvec_q4_0 || !t->matvec_q4_1 ||
        !t->matvec_f32 || !t->matvec_f16) {
        snprintf(err, cap, "hotpath: module table incomplete (null entry or bad version)");
        vbs_dlclose(h);
        return false;
    }
    /* Validation passed: retire the generation before previous, shift the
     * current generation into the rollback slot, then swap in the new one. */
    if (g_prev_handle) vbs_dlclose(g_prev_handle);
    g_prev = g_current;
    g_prev_handle = g_current_handle;
    snprintf(g_prev_path, sizeof(g_prev_path), "%s", g_current_path);
    hot_table_store(t);
    g_current_handle = h;
    snprintf(g_current_path, sizeof(g_current_path), "%s", path);
    return true;
}

bool vbs_hot_rollback(char *err, size_t cap) {
    if (!g_prev) {
        snprintf(err, cap, "hotpath: no previous module to roll back to");
        return false;
    }
    /* Swap current <-> previous; the bad module stays loaded (harmless,
     * unreachable) until the next load retires it. */
    const vbs_hot_kernels_t *bad = g_current;
    void *bad_handle = g_current_handle;
    char bad_path[1024];
    snprintf(bad_path, sizeof(bad_path), "%s", g_current_path);
    hot_table_store(g_prev);
    g_current_handle = g_prev_handle;
    snprintf(g_current_path, sizeof(g_current_path), "%s", g_prev_path);
    g_prev = bad;
    g_prev_handle = bad_handle;
    snprintf(g_prev_path, sizeof(g_prev_path), "%s", bad_path);
    return true;
}

/* Dispatch wrappers: timed and counted so the loop's hotspot analysis has
 * real feedback from the live session. */

#define VBS_HOT_DISPATCH(idx, fn, ...) do { \
        double t0 = now_ns(); \
        vbs_hot_current()->fn(__VA_ARGS__); \
        double dt = now_ns() - t0; \
        g_counters[idx].calls++; \
        g_counters[idx].ns += (uint64_t)dt; \
    } while (0)

void vbs_hot_matvec_q8_0(const uint8_t *w, size_t rows, size_t cols,
                         const int8_t *xq, const float *qd, float *y) {
    VBS_HOT_DISPATCH(0, matvec_q8_0, w, rows, cols, xq, qd, y);
}

void vbs_hot_matvec_q4_0(const uint8_t *w, size_t rows, size_t cols,
                         const int8_t *xq, const float *qd, float *y) {
    VBS_HOT_DISPATCH(1, matvec_q4_0, w, rows, cols, xq, qd, y);
}

void vbs_hot_matvec_q4_1(const uint8_t *w, size_t rows, size_t cols,
                         const int8_t *xq, const float *qd, const float *qsum,
                         float *y) {
    VBS_HOT_DISPATCH(2, matvec_q4_1, w, rows, cols, xq, qd, qsum, y);
}

void vbs_hot_matvec_f32(const float *w, size_t rows, size_t cols,
                        const float *x, float *y) {
    VBS_HOT_DISPATCH(3, matvec_f32, w, rows, cols, x, y);
}

void vbs_hot_matvec_f16(const uint16_t *w, size_t rows, size_t cols,
                        const float *x, float *y) {
    VBS_HOT_DISPATCH(4, matvec_f16, w, rows, cols, x, y);
}

bool vbs_hot_cpu_has_avx2(void) {
#if defined(__x86_64__) || defined(__i386__)
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;
    if (!(ecx & bit_OSXSAVE) || !(ecx & bit_AVX)) return false;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
    return (ebx & bit_AVX2) != 0;
#else
    return false;
#endif
}

bool vbs_hot_cpu_has_avx512f(void) {
#if defined(__x86_64__) || defined(__i386__)
    unsigned int eax, ebx, ecx, edx;
    if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;
    if (!(ecx & bit_OSXSAVE) || !(ecx & bit_AVX)) return false;
    if (!__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) return false;
    return (ebx & bit_AVX512F) != 0;
#else
    return false;
#endif
}
