/* vbs_hotpath.h — versioned hot-kernel ABI + loader + profiling counters.
 *
 * The inference hot path (quantized matvec in the CPU engine) runs through
 * this table instead of being called directly. The table starts out pointing
 * at built-in reference kernels (byte-identical semantics to the original
 * vbs_cpu.c inner loops), and the recursive loop may swap in a compiled
 * candidate module between tokens. A swap never touches model state: the
 * KV cache, position, and weights live in vbs_cpu_model_t, which the
 * kernels never see — they only get raw pointers and shapes.
 *
 * ABI contract (version 1):
 *  - A module exports vbs_hot_abi_version() == VBS_HOT_ABI_VERSION and
 *    vbs_hot_kernels() returning a pointer to a vbs_hot_kernels_t whose
 *    five function pointers are all non-NULL.
 *  - Kernels are pure functions of their arguments: no global state, no
 *    I/O, no allocation. They may read only the buffers passed in.
 *  - Semantics must match the reference exactly (see
 *    src/hotkernels/vbs_hot_ref_impl.h). Quantized dots are integer-exact;
 *    float accumulation order is per-block, so a candidate that keeps the
 *    block structure is bit-identical to the reference.
 *  - A module that fails any of this is rejected by the loader and never
 *    runs inside the live process. Candidates are additionally verified in
 *    a forked child before they are ever loaded (see vbs_loop.c).
 */
#ifndef VBS_HOTPATH_H
#define VBS_HOTPATH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VBS_HOT_ABI_VERSION 1

/* One row of packed quantized weights is (cols/32) blocks; block sizes are
 * Q8_0: 34 bytes, Q4_0: 18 bytes, Q4_1: 20 bytes. xq holds the Q8-quantized
 * activations (cols int8), qd the per-block activation scales (cols/32
 * floats), qsum the per-block activation sums (Q4_1 only, else NULL).
 * y holds rows outputs. All pointers are host memory, little-endian. */
typedef struct {
    uint32_t abi_version;
    const char *name;
    void (*matvec_q8_0)(const uint8_t *w, size_t rows, size_t cols,
                        const int8_t *xq, const float *qd, float *y);
    void (*matvec_q4_0)(const uint8_t *w, size_t rows, size_t cols,
                        const int8_t *xq, const float *qd, float *y);
    void (*matvec_q4_1)(const uint8_t *w, size_t rows, size_t cols,
                        const int8_t *xq, const float *qd, const float *qsum,
                        float *y);
    void (*matvec_f32)(const float *w, size_t rows, size_t cols,
                       const float *x, float *y);
    void (*matvec_f16)(const uint16_t *w, size_t rows, size_t cols,
                       const float *x, float *y);
} vbs_hot_kernels_t;

/* The built-in reference table. Always valid, never unloaded. */
const vbs_hot_kernels_t *vbs_hot_builtin(void);

/* The currently active table (builtin until a module is loaded). */
const vbs_hot_kernels_t *vbs_hot_current(void);

/* Validate a module and swap it in as the current table. The previous
 * module (or the builtin table) is kept for one rollback step. Returns
 * false with a message in err on any validation failure; the current
 * table is untouched in that case. Safe to call only between tokens. */
bool vbs_hot_load(const char *path, char *err, size_t err_cap);

/* Swap back to the previous table (the one active before the last load).
 * Returns false if there is nothing to roll back to. */
bool vbs_hot_rollback(char *err, size_t err_cap);

const char *vbs_hot_current_name(void);
const char *vbs_hot_current_path(void); /* "" for the builtin table */
bool vbs_hot_has_rollback(void);

/* Per-type profiling counters, cumulative since process start (or reset).
 * Index order: 0=q8_0, 1=q4_0, 2=q4_1, 3=f32, 4=f16. The loop's hotspot
 * analysis reads these to decide which kernel type to optimize next. */
#define VBS_HOT_NTYPES 5
typedef struct { uint64_t calls; uint64_t ns; } vbs_hot_counter_t;
void vbs_hot_counters(vbs_hot_counter_t out[VBS_HOT_NTYPES]);
void vbs_hot_counters_reset(void);
const char *vbs_hot_type_name(int idx);

/* Timed dispatch wrappers — the only path vbs_cpu.c uses to reach kernels. */
void vbs_hot_matvec_q8_0(const uint8_t *w, size_t rows, size_t cols,
                         const int8_t *xq, const float *qd, float *y);
void vbs_hot_matvec_q4_0(const uint8_t *w, size_t rows, size_t cols,
                         const int8_t *xq, const float *qd, float *y);
void vbs_hot_matvec_q4_1(const uint8_t *w, size_t rows, size_t cols,
                         const int8_t *xq, const float *qd, const float *qsum,
                         float *y);
void vbs_hot_matvec_f32(const float *w, size_t rows, size_t cols,
                        const float *x, float *y);
void vbs_hot_matvec_f16(const uint16_t *w, size_t rows, size_t cols,
                        const float *x, float *y);

/* Runtime CPU capability probes (used to skip SIMD candidates the host
 * cannot run). */
bool vbs_hot_cpu_has_avx2(void);
bool vbs_hot_cpu_has_avx512f(void);

#ifdef __cplusplus
}
#endif

#endif /* VBS_HOTPATH_H */
