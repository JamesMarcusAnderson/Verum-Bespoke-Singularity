#include "vbs_engine.h"
#include <stdio.h>
#include <stdlib.h>

vbs_engine_t *vbs_engine_create(const char *path, size_t window_mib, char *err_buf, size_t err_cap) {
    vbs_gguf_model_t *model = vbs_gguf_open(path, err_buf, err_cap);
    if (!model) return NULL;

    const vbs_backend_ops_t *backend = &g_vbs_metal_backend;
    if (!backend->init()) {
        snprintf(err_buf, err_cap, "Failed initializing GPU compute backend");
        vbs_gguf_close(model);
        return NULL;
    }

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
    if (e->executors) {
        for (size_t i = 0; i < e->n_devices; i++) {
            if (e->executors[i]) vbs_slice_executor_free(e->executors[i]);
        }
        free(e->executors);
    }
    free(e->devices);
    if (e->model) vbs_gguf_close(e->model);
    free(e);
}

void vbs_engine_print_inspection(const vbs_engine_t *e) {
    printf("GGUF v%u | %zu tensors | %zu metadata | window %zu MiB | %zu GPU(s)\n",
           e->model->version, e->model->n_tensors, e->model->n_metadata,
           e->window_bytes / (1024 * 1024), e->n_devices);

    const char *arch = vbs_gguf_get_string_metadata(e->model, "general.architecture");
    printf("declared architecture: %s\n", arch ? arch : "(not declared)");

    for (size_t i = 0; i < e->n_devices; i++) {
        printf("GPU: %s | max working set %llu MiB | max buffer %llu MiB\n",
               e->devices[i]->name,
               e->devices[i]->max_working_set_bytes / (1024 * 1024),
               e->devices[i]->max_buffer_bytes / (1024 * 1024));
    }
}

bool vbs_engine_compile(vbs_engine_t *e, char *err_buf, size_t err_cap) {
    e->graph = vbs_graph_compile(e->model, err_buf, err_cap);
    if (!e->graph) return false;

    printf("graph %s | %zu inferred operations | static graph validated\n",
           e->graph->fingerprint, e->graph->n_nodes);
    return true;
}
