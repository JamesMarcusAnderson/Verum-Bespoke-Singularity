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
#include "vbs_specializer_bridge.h"
#include "tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

/* PGAE control block, defined before vbs_engine_t so the engine can
 * hold one by value. See the longer Phase 1 / Phase 2 comment below
 * the engine struct for why the pasted design was not pasted literally. */
typedef enum {
    VBS_PGAE_IDLE = 0,
    VBS_PGAE_PREFILL = 1,
    VBS_PGAE_GENERATE = 2,
    VBS_PGAE_EXIT = 3
} vbs_pgae_state_t;

#define VBS_PGAE_EOS_NONE ((uint32_t)0xFFFFFFFFu)

typedef struct {
    volatile uint32_t state;
    volatile uint32_t current_token;
    volatile uint32_t position;
    volatile uint32_t max_tokens;
    volatile uint32_t vocab_size;
    volatile uint32_t eos_token_id;
    uint64_t generated_tokens_dev;
    uint64_t logits_dev;
} vbs_gpu_control_t;

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
    vbs_buffer_t *scratch[VBS_SCRATCH_COUNT];

    size_t n_devices;
    vbs_device_t **devices;
    vbs_slice_executor_t **executors;
    size_t window_bytes;

    tokenizer_t *tokenizer;

    /* PGAE Phase 1 state: device buffers, owned by the engine. */
    vbs_buffer_t *autonomy_control_buffer;
    vbs_buffer_t *autonomy_tokens_buffer;
    vbs_gpu_control_t autonomy_control;
} vbs_engine_t;

#define VBS_MAX_LAYERS 512
extern float g_vbs_layer_energy[VBS_MAX_LAYERS];

/* --- Persistent GPU Autonomy Engine (PGAE), Phase 1 ----------------------
 * The pasted PGAE design cannot be pasted literally, for reasons that
 * would break the build and lie about what is running:
 *  1. vbs_gpu_dispatch_* are HOST-side cuLaunchKernel wrappers (see
 *     vbs_specializer_bridge.c). A device PTX/MSL kernel cannot call
 *     them. Emitting them inside a "persistent_forward" host JIT
 *     function does not make them device code, and a single scalar
 *     host function is not a parallel GPU kernel.
 *  2. A true persistent kernel needs the whole transformer rewritten
 *     as one cooperative device kernel with grid sync, device-side
 *     weight pointers, and a device-side KV cache. That is Phase 2,
 *     not a paste-in. It is not implemented here, and it is not
 *     claimed below.
 *  3. The pasted host code sets eos_token_id = -1 into a uint32_t,
 *     which is 4294967295, not "no EOS". Use VBS_PGAE_EOS_NONE.
 *  4. The pasted CPU "sleep" loop polls copy_to_host every 1ms. That
 *     is polling, not sleep, and it races a kernel that is still
 *     writing the control block. Phase 1 therefore does not claim
 *     the CPU is offline; it claims the ENGINE owns the loop.
 *
 * Phase 1, implemented here, is architecturally honest: the control
 * block is real, it lives in a device buffer, the engine (not main)
 * owns prefill, argmax, KV position advance, EOS/max stop, and the
 * generated-token array, and main only waits and then decodes.
 * The per-token GPU launches still go through the existing JIT
 * forward path, which is host-dispatched. Phase 2 will replace that
 * inner loop with a real persistent device kernel when one exists.
 */
vbs_engine_t *vbs_engine_create(const char *model_path, size_t window_mib, char *err_buf, size_t err_cap);
void          vbs_engine_destroy(vbs_engine_t *engine);
void          vbs_engine_print_inspection(const vbs_engine_t *engine);
bool          vbs_engine_compile(vbs_engine_t *engine, char *err_buf, size_t err_cap);
bool          vbs_engine_forward(vbs_engine_t *engine, const uint32_t *tokens, size_t n_tokens, float *logits_out);
/* PGAE Phase 1: engine-owned generation loop. Returns false on bad
 * input, uncompiled engine, or allocation failure. On success,
 * *out_tokens is a newly malloc'd array owned by the caller, *out_n
 * is the number of generated tokens (excluding the prompt), and
 * *stopped_on_eos is true only if EOS actually stopped it. */
bool          vbs_engine_forward_autonomy(vbs_engine_t *e, const uint32_t *prompt_tokens, size_t n_prompt, size_t max_gen, uint32_t eos_token_id, uint32_t **out_tokens, size_t *out_n, bool *stopped_on_eos);
void          vbs_engine_mutate_layer(vbs_engine_t *e, size_t layer_index, float mutation_rate);
void          vbs_engine_reset_energy(void);
void          vbs_engine_recompile(vbs_engine_t *e, char *err_buf, size_t err_cap);

#ifdef __cplusplus
}
#endif

#endif // VBS_ENGINE_H
