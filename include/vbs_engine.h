#ifndef VBS_ENGINE_H
#define VBS_ENGINE_H

#include "vbs_gguf.h"
#include "vbs_graph.h"
#include "vbs_backend.h"
#include "vbs_slice_executor.h"
#include "vbs_profiler.h"
#include "vbs_fabric.h"
#include "vbs_planner.h"
#include "vbs_specializer.h"
#include "tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vbs_engine {
    vbs_gguf_model_t *model;
    vbs_graph_t *graph;
    const vbs_backend_ops_t *backend;

    vbs_system_profile_t *sys_profile;
    vbs_model_profile_t *model_profile;
    vbs_fabric_t *fabric;
    vbs_execution_plan_t *plan;

    vbs_specialized_binary_t *specialized;

    vbs_buffer_t *act_a;
    vbs_buffer_t *act_b;
    vbs_buffer_t *logits;
    vbs_buffer_t *kv_cache_k;
    vbs_buffer_t *kv_cache_v;
    size_t kv_ctx_capacity; // positions the KV caches were actually allocated for; kernels must use this as the cache stride, not the model's context_length

    size_t n_devices;
    vbs_device_t **devices;
    vbs_slice_executor_t **executors;
    size_t window_bytes;

    tokenizer_t *tokenizer;
} vbs_engine_t;

vbs_engine_t *vbs_engine_create(const char *model_path, size_t window_mib, char *err_buf, size_t err_cap);
void          vbs_engine_destroy(vbs_engine_t *engine);
void          vbs_engine_print_inspection(const vbs_engine_t *engine);
bool          vbs_engine_compile(vbs_engine_t *engine, char *err_buf, size_t err_cap);
bool          vbs_engine_forward(vbs_engine_t *engine, const uint32_t *tokens, size_t n_tokens, float *logits_out);

#ifdef __cplusplus
}
#endif

#endif // VBS_ENGINE_H
