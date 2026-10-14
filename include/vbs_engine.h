#ifndef VBS_ENGINE_H
#define VBS_ENGINE_H

#include "vbs_gguf.h"
#include "vbs_graph.h"
#include "vbs_backend.h"
#include "vbs_slice_executor.h"
#include "vbs_profiler.h"
#include "vbs_fabric.h"
#include "vbs_planner.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    vbs_gguf_model_t *model;
    vbs_graph_t *graph;
    const vbs_backend_ops_t *backend;

    vbs_system_profile_t *sys_profile;
    vbs_model_profile_t *model_profile;
    vbs_fabric_t *fabric;
    vbs_execution_plan_t *plan;

    size_t n_devices;
    vbs_device_t **devices;
    vbs_slice_executor_t **executors;
    size_t window_bytes;
} vbs_engine_t;

vbs_engine_t *vbs_engine_create(const char *model_path, size_t window_mib, char *err, size_t err_cap);
void          vbs_engine_destroy(vbs_engine_t *e);
void          vbs_engine_print_inspection(const vbs_engine_t *e);
bool          vbs_engine_compile(vbs_engine_t *e, char *err, size_t err_cap);

#ifdef __cplusplus
}
#endif
#endif
