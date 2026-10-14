#include "vbs_engine.h"
#include "vbs_specializer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

extern const vbs_backend_ops_t g_vbs_cpu_backend;
extern const vbs_backend_ops_t g_vbs_cuda_backend;
extern const vbs_backend_ops_t g_vbs_hip_backend;
#if VBS_ENABLE_METAL
extern const vbs_backend_ops_t g_vbs_metal_backend;
#endif

static const vbs_backend_ops_t *select_backend(const vbs_system_profile_t *sys) {
    const char *pref = getenv("VBS_BACKEND");
    if (pref) {
        if (strcmp(pref, "cuda") == 0 && g_vbs_cuda_backend.init()) return &g_vbs_cuda_backend;
        if (strcmp(pref, "hip")  == 0 && g_vbs_hip_backend.init())  return &g_vbs_hip_backend;
#if VBS_ENABLE_METAL
        if (strcmp(pref, "metal") == 0 && g_vbs_metal_backend.init()) return &g_vbs_metal_backend;
#endif
        if (strcmp(pref, "cpu")  == 0 && g_vbs_cpu_backend.init())  return &g_vbs_cpu_backend;
    }

    for (size_t i = 0; i < sys->n_gpus; i++) {
        if (sys->gpus[i].type == VBS_BACKEND_CUDA)  return &g_vbs_cuda_backend;
#if VBS_ENABLE_METAL
        if (sys->gpus[i].type == VBS_BACKEND_METAL) return &g_vbs_metal_backend;
#endif
        if (sys->gpus[i].type == VBS_BACKEND_HIP)   return &g_vbs_hip_backend;
    }
    return &g_vbs_cpu_backend;
}

vbs_engine_t *vbs_engine_create(const char *path, size_t window_mib, char *err, size_t err_cap) {
    vbs_gguf_model_t *model = vbs_gguf_open(path, err, err_cap);
    if (!model) return NULL;

    vbs_engine_t *e = (vbs_engine_t *)calloc(1, sizeof(vbs_engine_t));
    e->model = model;

    e->sys_profile = vbs_profiler_scan_system();
    vbs_profiler_benchmark_devices(e->sys_profile);
    e->model_profile = vbs_profiler_scan_model(model);

    e->backend = select_backend(e->sys_profile);

    const char *swap_path = getenv("VBS_SWAP_PATH");
    size_t swap_gib = 0;
    const char *swap_env = getenv("VBS_SWAP_GIB");
    if (swap_env) swap_gib = (size_t)atoi(swap_env);

    e->fabric = vbs_fabric_create(
        e->sys_profile, e->backend,
        512, // 512 MiB pinned host pool
        swap_gib,
        swap_path ? swap_path : "/tmp/verum_swap.bin"
    );

    e->window_bytes = (window_mib ? window_mib : 256) * 1024 * 1024;
    e->n_devices = e->sys_profile->n_gpus ? e->sys_profile->n_gpus : 1;
    e->devices = (vbs_device_t **)calloc(e->n_devices, sizeof(vbs_device_t *));
    e->executors = (vbs_slice_executor_t **)calloc(e->n_devices, sizeof(vbs_slice_executor_t *));

    for (size_t i = 0; i < e->n_devices; i++) {
        e->devices[i] = e->sys_profile->n_gpus
            ? e->sys_profile->gpus[i].device
            : e->backend->get_device(0);
        e->executors[i] = vbs_slice_executor_create(
            model, e->backend, e->devices[i], e->window_bytes, err, err_cap);
    }
    return e;
}

bool vbs_engine_compile(vbs_engine_t *e, char *err, size_t err_cap) {
    e->graph = vbs_graph_compile(e->model, err, err_cap);
    if (!e->graph) return false;

    e->plan = vbs_planner_build(e->graph, e->sys_profile, e->model_profile, e->fabric);
    if (!e->plan) {
        snprintf(err, err_cap, "Planner failed to generate execution plan");
        return false;
    }

    // ===== SPECIALIZATION STAGE =====
    char hash[65] = {0};
    vbs_specializer_compute_hash(e->graph, e->sys_profile, e->plan, hash, sizeof(hash));

    const char *cache_dir = getenv("VBS_CACHE_DIR");
    if (!cache_dir) cache_dir = "/tmp/verum_cache";
    mkdir(cache_dir, 0755);

    e->specialized = vbs_specializer_load_cached(hash, cache_dir);
    if (!e->specialized) {
        char src_path[640], bin_path[640];
        snprintf(src_path, sizeof(src_path), "%s/verum_%s.c", cache_dir, hash);
        snprintf(bin_path, sizeof(bin_path), "%s/verum_%s.so", cache_dir, hash);

        if (!vbs_specializer_emit_source(e->graph, e->sys_profile, e->plan, e->model_profile, src_path)) {
            snprintf(err, err_cap, "Failed emitting bespoke C source for specialization");
            return false;
        }

        double compile_sec = 0;
        if (!vbs_specializer_compile(src_path, bin_path, e->sys_profile, &compile_sec)) {
            snprintf(err, err_cap, "Specializer compilation failed (clang check)");
            return false;
        }

        e->specialized = vbs_specializer_load(bin_path, hash);
        if (!e->specialized) {
            snprintf(err, err_cap, "Failed loading specialized binary module");
            return false;
        }
        e->specialized->compile_time_sec = compile_sec;
        e->specialized->was_cached = false;
    } else {
        e->specialized->was_cached = true;
    }

    e->specialized->n_kernels_emitted = e->plan->n_plans;
    e->specialized->init_fn(e->fabric, (void *)e->backend);

    return true;
}

bool vbs_engine_forward(vbs_engine_t *e, const uint32_t *tokens, size_t n_tokens, float *logits_out) {
    if (!e || !e->specialized || !e->specialized->forward_fn) return false;
    e->specialized->forward_fn(tokens, n_tokens, logits_out);
    return true;
}

void vbs_engine_destroy(vbs_engine_t *e) {
    if (!e) return;
    if (e->specialized) vbs_specializer_free(e->specialized);
    if (e->plan) vbs_planner_free(e->plan);
    if (e->fabric) vbs_fabric_free(e->fabric);
    if (e->graph) vbs_graph_free(e->graph);
    for (size_t i = 0; i < e->n_devices; i++) {
        if (e->executors && e->executors[i]) vbs_slice_executor_free(e->executors[i]);
    }
    free(e->executors);
    free(e->devices);
    if (e->model_profile) vbs_profiler_free_model(e->model_profile);
    if (e->sys_profile) vbs_profiler_free_system(e->sys_profile);
    if (e->tokenizer) tokenizer_free(e->tokenizer);
    if (e->model) vbs_gguf_close(e->model);
    free(e);
}

void vbs_engine_print_inspection(const vbs_engine_t *e) {
    if (!e) return;
    vbs_profiler_print_system(e->sys_profile);
    vbs_profiler_print_model(e->model_profile);
    if (e->plan) vbs_planner_print(e->plan);
    if (e->fabric) vbs_fabric_print_stats(e->fabric);
    if (e->specialized) vbs_specializer_print_stats(e->specialized);
}
