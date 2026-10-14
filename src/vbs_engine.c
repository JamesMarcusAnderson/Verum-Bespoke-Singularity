#include "vbs_engine.h"
#include "vbs_profiler.h"
#include "vbs_fabric.h"
#include "vbs_planner.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern const vbs_backend_ops_t g_vbs_cpu_backend;
extern const vbs_backend_ops_t g_vbs_cuda_backend;
extern const vbs_backend_ops_t g_vbs_hip_backend;
#if VBS_ENABLE_METAL
extern const vbs_backend_ops_t g_vbs_metal_backend;
#endif

static const vbs_backend_ops_t *select_backend(const vbs_system_profile_t *sys) {
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

    vbs_profiler_print_system(e->sys_profile);
    vbs_profiler_print_model(e->model_profile);

    e->backend = select_backend(e->sys_profile);

    const char *swap_path = getenv("VBS_SWAP_PATH");
    size_t swap_gib = 0;
    const char *swap_env = getenv("VBS_SWAP_GIB");
    if (swap_env) swap_gib = (size_t)atoi(swap_env);

    e->fabric = vbs_fabric_create(
        e->sys_profile, e->backend,
        /* pinned pool */ 512,
        /* nvme swap */ swap_gib,
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
        snprintf(err, err_cap, "Planner failed to build execution plan");
        return false;
    }

    vbs_planner_print(e->plan);
    vbs_fabric_print_stats(e->fabric);

    printf("graph %s | %zu operations planned\n", e->graph->fingerprint, e->graph->n_nodes);
    return true;
}

void vbs_engine_destroy(vbs_engine_t *e) {
    if (!e) return;
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
    if (e->model) vbs_gguf_close(e->model);
    free(e);
}

void vbs_engine_print_inspection(const vbs_engine_t *e) {
    (void)e;
}
