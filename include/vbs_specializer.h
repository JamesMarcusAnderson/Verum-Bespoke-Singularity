#ifndef VBS_SPECIALIZER_H
#define VBS_SPECIALIZER_H

#include "vbs_planner.h"
#include "vbs_profiler.h"
#include "vbs_graph.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    // Identity: deterministic fingerprint of (model, hardware, plan)
    char specialization_hash[65];
    char cache_dir[512];
    char source_path[640];
    char binary_path[640];

    // The compiled artifact
    void *loaded_handle;              // dlopen handle
    size_t binary_size;
    bool  was_cached;                 // true if we hit the cache
    double compile_time_sec;
    double load_time_sec;

    // Entry points into the specialized binary
    void (*init_fn)(void *fabric, void *backend);
    void (*forward_fn)(const uint32_t *tokens, size_t n_tokens, float *logits_out);
    void (*reset_cache_fn)(void);
    void (*shutdown_fn)(void);

    // Statistics
    size_t n_kernels_emitted;
    size_t n_constants_inlined;
    size_t n_branches_eliminated;
    size_t n_loops_fully_unrolled;
    size_t estimated_cycles_per_token;
} vbs_specialized_binary_t;

// Compute deterministic hash of (model + hardware + plan) to key cache
void vbs_specializer_compute_hash(
    const vbs_graph_t *graph,
    const vbs_system_profile_t *sys,
    const vbs_execution_plan_t *plan,
    char *out_hash_hex, size_t out_cap
);

// Try to load pre-compiled specialized binary. Returns NULL on cache miss.
vbs_specialized_binary_t *vbs_specializer_load_cached(
    const char *specialization_hash,
    const char *cache_dir
);

// Generate specialized source for (model, hardware, plan)
bool vbs_specializer_emit_source(
    const vbs_graph_t *graph,
    const vbs_system_profile_t *sys,
    const vbs_execution_plan_t *plan,
    const vbs_model_profile_t *mp,
    const char *out_source_path
);

// Compile the emitted source into a .so/.dylib with the system compiler
bool vbs_specializer_compile(
    const char *source_path,
    const char *binary_path,
    const vbs_system_profile_t *sys,
    double *out_compile_time_sec
);

// Load compiled binary and resolve entry points
vbs_specialized_binary_t *vbs_specializer_load(
    const char *binary_path,
    const char *specialization_hash
);

void vbs_specializer_free(vbs_specialized_binary_t *bin);
void vbs_specializer_print_stats(const vbs_specialized_binary_t *bin);

#ifdef __cplusplus
}
#endif
#endif
