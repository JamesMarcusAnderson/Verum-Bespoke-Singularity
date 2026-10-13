#include "vbs_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern const vbs_backend_ops_t g_vbs_cpu_backend;
extern const vbs_backend_ops_t g_vbs_cuda_backend;
extern const vbs_backend_ops_t g_vbs_hip_backend;
#if VBS_ENABLE_METAL
extern const vbs_backend_ops_t g_vbs_metal_backend;
#endif

static const vbs_backend_ops_t *select_backend(const char *preferred) {
    if (preferred) {
        if (strcmp(preferred, "cuda") == 0 && g_vbs_cuda_backend.init()) return &g_vbs_cuda_backend;
        if (strcmp(preferred, "hip")  == 0 && g_vbs_hip_backend.init())  return &g_vbs_hip_backend;
#if VBS_ENABLE_METAL
        if (strcmp(preferred, "metal") == 0 && g_vbs_metal_backend.init()) return &g_vbs_metal_backend;
#endif
        if (strcmp(preferred, "cpu")  == 0 && g_vbs_cpu_backend.init())  return &g_vbs_cpu_backend;
    }
    // Auto-detect best available
    if (g_vbs_cuda_backend.init()) { fprintf(stderr, "backend: NVIDIA CUDA\n"); return &g_vbs_cuda_backend; }
#if VBS_ENABLE_METAL
    if (g_vbs_metal_backend.init()) { fprintf(stderr, "backend: Apple Metal\n"); return &g_vbs_metal_backend; }
#endif
    if (g_vbs_hip_backend.init())  { fprintf(stderr, "backend: AMD HIP\n"); return &g_vbs_hip_backend; }
    g_vbs_cpu_backend.init();
    fprintf(stderr, "backend: CPU (portable fallback)\n");
    return &g_vbs_cpu_backend;
}

vbs_engine_t *vbs_engine_create(const char *path, size_t window_mib, char *err_buf, size_t err_cap) {
    vbs_gguf_model_t *model = vbs_gguf_open(path, err_buf, err_cap);
    if (!model) return NULL;

    const vbs_backend_ops_t *backend = select_backend(getenv("VBS_BACKEND"));

    vbs_engine_t *e = (vbs_engine_t *)calloc(1, sizeof(vbs_engine_t));
    e->model = model;
    e->backend = backend;
    e->window_bytes = (window_mib ? window_mib : 256) * 1024 * 1024;
    e->n_devices = backend->get_device_count();
    e->devices = (vbs_device_t **)calloc(e->n_devices, sizeof(vbs_device_t *));
    e->executors = (vbs_slice_executor_t **)calloc(e->n_devices, sizeof(vbs_slice_executor_t *));

    for (size_t i = 0; i < e->n_devices; i++) {
        e->devices[i] = backend->get_device(i);
        e->executors[i] = vbs_slice_executor_create(model, backend, e->devices[i], e->window_bytes, err_buf, err_cap);
        if (!e->executors[i]) {
            vbs_engine_destroy(e);
            return NULL;
        }
    }
    return e;
}

void vbs_engine_destroy(vbs_engine_t *e) {
    if (!e) return;
    if (e->graph) vbs_graph_free(e->graph);
    for (size_t i = 0; i < e->n_devices; i++) {
        if (e->executors && e->executors[i]) vbs_slice_executor_free(e->executors[i]);
    }
    free(e->executors);
    free(e->devices);
    if (e->model) vbs_gguf_close(e->model);
    free(e);
}

void vbs_engine_print_inspection(const vbs_engine_t *e) {
    printf("GGUF v%u | %zu tensors | %zu metadata | window %zu MiB | %zu device(s)\n",
           e->model->version, e->model->n_tensors, e->model->n_metadata,
           e->window_bytes / (1024 * 1024), e->n_devices);
    const char *arch = vbs_gguf_get_string_metadata(e->model, "general.architecture");
    printf("declared architecture: %s\n", arch ? arch : "(not declared)");
    for (size_t i = 0; i < e->n_devices; i++) {
        printf("device[%zu]: %s\n", i, e->devices[i]->name);
    }
}

bool vbs_engine_compile(vbs_engine_t *e, char *err_buf, size_t err_cap) {
    e->graph = vbs_graph_compile(e->model, err_buf, err_cap);
    if (!e->graph) return false;
    printf("graph %s | %zu inferred operations\n", e->graph->fingerprint, e->graph->n_nodes);
    return true;
}
