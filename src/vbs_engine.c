#include "vbs_engine.h"
#include "vbs_specializer.h"
#include "vbs_specializer_bridge.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>

extern const vbs_backend_ops_t g_vbs_cuda_backend;
extern const vbs_backend_ops_t g_vbs_hip_backend;
#if defined(VBS_ENABLE_METAL) && VBS_ENABLE_METAL
extern const vbs_backend_ops_t g_vbs_metal_backend;
#endif

static const vbs_backend_ops_t *select_gpu_backend(const vbs_system_profile_t *sys) {
    const char *pref = getenv("VBS_BACKEND");
    if (pref) {
        if (strcmp(pref, "cuda") == 0 && g_vbs_cuda_backend.init()) return &g_vbs_cuda_backend;
        if (strcmp(pref, "hip")  == 0 && g_vbs_hip_backend.init())  return &g_vbs_hip_backend;
#if defined(VBS_ENABLE_METAL) && VBS_ENABLE_METAL
        if (strcmp(pref, "metal") == 0 && g_vbs_metal_backend.init()) return &g_vbs_metal_backend;
#endif
    }

    for (size_t i = 0; i < sys->n_gpus; i++) {
        if (sys->gpus[i].type == VBS_BACKEND_CUDA && g_vbs_cuda_backend.init())  return &g_vbs_cuda_backend;
#if defined(VBS_ENABLE_METAL) && VBS_ENABLE_METAL
        if (sys->gpus[i].type == VBS_BACKEND_METAL && g_vbs_metal_backend.init()) return &g_vbs_metal_backend;
#endif
        if (sys->gpus[i].type == VBS_BACKEND_HIP && g_vbs_hip_backend.init())   return &g_vbs_hip_backend;
    }
    return NULL;
}

vbs_engine_t *vbs_engine_create(const char *path, size_t window_mib, char *err, size_t err_cap) {
    vbs_gguf_model_t *model = vbs_gguf_open(path, err, err_cap);
    if (!model) return NULL;

    vbs_system_profile_t *sys = vbs_profiler_scan_system();
    vbs_profiler_benchmark_devices(sys);

    const vbs_backend_ops_t *backend = select_gpu_backend(sys);
    if (!backend) {
        snprintf(err, err_cap, "Fatal: No supported GPU device detected (CUDA, HIP, or Metal).");
        vbs_profiler_free_system(sys);
        vbs_gguf_close(model);
        return NULL;
    }

    vbs_engine_t *e = (vbs_engine_t *)calloc(1, sizeof(vbs_engine_t));
    e->model = model;
    e->sys_profile = sys;
    e->model_profile = vbs_profiler_scan_model(model);
    if (!e->model_profile) {
        snprintf(err, err_cap, "Model profile refused (see profiler message above)");
        free(e);
        vbs_gguf_close(model);
        return NULL;
    }
    e->backend = backend;

    const char *swap_path = getenv("VBS_SWAP_PATH");
    size_t swap_gib = 0;
    const char *swap_env = getenv("VBS_SWAP_GIB");
    if (swap_env) swap_gib = (size_t)atoi(swap_env);

    e->fabric = vbs_fabric_create(
        e->sys_profile, e->backend,
        512,
        swap_gib,
        swap_path ? swap_path : "/tmp/verum_swap.bin"
    );

    // Register all tensors into the fabric
    vbs_device_t *primary_dev = e->backend->get_device(0);
    for (size_t i = 0; i < model->n_tensors; i++) {
        const vbs_tensor_t *t = &model->tensors[i];
        const void *src_ptr = (const uint8_t *)model->map + t->data_offset;
        vbs_fabric_register_tensor(e->fabric, t, src_ptr, VBS_TIER_VRAM, primary_dev);
    }

    e->window_bytes = (window_mib ? window_mib : 256) * 1024 * 1024;
    e->n_devices = e->sys_profile->n_gpus ? e->sys_profile->n_gpus : 1;
    e->devices = (vbs_device_t **)calloc(e->n_devices, sizeof(vbs_device_t *));
    e->executors = (vbs_slice_executor_t **)calloc(e->n_devices, sizeof(vbs_slice_executor_t *));

    for (size_t i = 0; i < e->n_devices; i++) {
        e->devices[i] = e->backend->get_device(i);
        e->executors[i] = vbs_slice_executor_create(
            model, e->backend, e->devices[i], e->window_bytes, err, err_cap);
        if (!e->executors[i]) {
            vbs_engine_destroy(e);
            return NULL;
        }
    }

    // Allocate runtime activation and KV-cache buffers
    // The profiler refuses incomplete models, so these are all proven
    // nonzero here -- no fallback shapes.
    size_t embed_dim = e->model_profile->embedding_dim;
    size_t vocab_size = e->model_profile->vocab_size;
    size_t max_ctx = e->model_profile->context_length;
    size_t n_layers = e->model_profile->n_layers;
    size_t n_kv_h = e->model_profile->n_kv_heads;
    size_t head_d = e->model_profile->head_dim;

    // Scratch sizes first: the KV budget below must account for them.
    // (Q/K/V, attention out, FFN gate/up/hidden, normed activation.)
    size_t q_dim = (e->model_profile->n_heads && head_d) ? (size_t)e->model_profile->n_heads * head_d : embed_dim;
    size_t kv_dim = n_kv_h * head_d;
    size_t ffn_dim = embed_dim;
    for (size_t i = 0; i < model->n_tensors; i++) {
        if (strstr(model->tensors[i].name, "ffn_gate") && model->tensors[i].rows > ffn_dim) {
            ffn_dim = model->tensors[i].rows;
        }
    }
    size_t scratch_sizes[VBS_SCRATCH_COUNT] = {
        q_dim, kv_dim, kv_dim, q_dim, ffn_dim, ffn_dim, ffn_dim, embed_dim
    };
    size_t scratch_bytes = 0;
    for (size_t i = 0; i < VBS_SCRATCH_COUNT; i++) scratch_bytes += scratch_sizes[i] * sizeof(float);

    // KV cache capacity, computed from the two profiles -- no fixed caps.
    // Ceiling: the model's own context length. Budget: the GPU's measured
    // free VRAM (profiled before the weights are resident, so the model
    // bytes are subtracted; falls back to total VRAM) minus the other
    // runtime buffers this function allocates, all sized from the model's
    // dimensions. Positions = budget / bytes per position across both
    // caches. Kernels must use e->kv_ctx_capacity as the cache stride.
    size_t kv_pos_bytes = n_layers * n_kv_h * head_d * sizeof(uint16_t); // per position, per cache
    uint64_t total_vram = (e->sys_profile && e->sys_profile->n_gpus > 0) ? e->sys_profile->gpus[0].total_vram_bytes : 0;
    uint64_t free_vram  = (e->sys_profile && e->sys_profile->n_gpus > 0) ? e->sys_profile->gpus[0].free_vram_bytes : 0;
    uint64_t base_vram  = free_vram ? free_vram : total_vram;
    uint64_t model_bytes = e->model_profile->total_model_bytes;
    uint64_t runtime_bytes = (uint64_t)(2 * embed_dim * sizeof(float)) + (uint64_t)(vocab_size * sizeof(float)) + (uint64_t)scratch_bytes;
    size_t kv_cap = 0;
    if (kv_pos_bytes > 0 && base_vram > model_bytes + runtime_bytes) {
        size_t fit = (size_t)((base_vram - model_bytes - runtime_bytes) / (2 * (uint64_t)kv_pos_bytes));
        kv_cap = fit < max_ctx ? fit : max_ctx;
    }
    if (kv_cap == 0) {
        snprintf(err, err_cap, "No VRAM budget for a KV cache: free/total %llu B, model %llu B, runtime buffers %llu B, %zu B per position",
                 (unsigned long long)base_vram, (unsigned long long)model_bytes,
                 (unsigned long long)runtime_bytes, kv_pos_bytes);
        vbs_engine_destroy(e);
        return NULL;
    }
    e->kv_ctx_capacity = kv_cap;
    e->model_profile->kv_ctx_capacity = (uint32_t)kv_cap;

    e->act_a  = backend->alloc_buffer(primary_dev, embed_dim * sizeof(float), false);
    e->act_b  = backend->alloc_buffer(primary_dev, embed_dim * sizeof(float), false);
    e->logits = backend->alloc_buffer(primary_dev, vocab_size * sizeof(float), false);

    size_t kv_bytes = kv_pos_bytes * kv_cap;
    e->kv_cache_k = backend->alloc_buffer(primary_dev, kv_bytes, false);
    e->kv_cache_v = backend->alloc_buffer(primary_dev, kv_bytes, false);

    if (!e->act_a || !e->act_b || !e->logits || !e->kv_cache_k || !e->kv_cache_v) {
        snprintf(err, err_cap, "Failed allocating runtime buffers (act %zu B, logits %zu B, KV %zu B x2 for %zu positions)",
                 embed_dim * sizeof(float), vocab_size * sizeof(float), kv_bytes, kv_cap);
        vbs_engine_destroy(e);
        return NULL;
    }

    for (size_t i = 0; i < VBS_SCRATCH_COUNT; i++) {
        e->scratch[i] = backend->alloc_buffer(primary_dev, scratch_sizes[i] * sizeof(float), false);
        if (!e->scratch[i]) {
            snprintf(err, err_cap, "Failed allocating scratch buffer %zu (%zu B)", i, scratch_sizes[i] * sizeof(float));
            vbs_engine_destroy(e);
            return NULL;
        }
    }

    // Bind bridge pointers
    vbs_bridge_bind_runtime(primary_dev, e->backend, e->fabric);
    vbs_bridge_set_activations(e->act_a, e->act_b, e->kv_cache_k, e->kv_cache_v, e->logits);
    vbs_bridge_set_scratch(e->scratch, VBS_SCRATCH_COUNT);

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
            snprintf(err, err_cap, "Specializer compilation failed");
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

/* PGAE Phase 1: engine-owned generation loop.
 * Boot phase (host): allocate control + token buffers on the device,
 * write the control block, prefill the prompt through the existing
 * JIT forward path. Generation phase (engine loop): argmax the
 * logits readback, write the token to the generated array, feed it
 * back as the next input, advance position, stop on EOS or max.
 * This is NOT a persistent device kernel yet -- see the header
 * comment. The loop lives in the engine, not in main, so main does
 * not decode, re-feed, or decide when to stop. */
bool vbs_engine_forward_autonomy(vbs_engine_t *e, const uint32_t *prompt_tokens, size_t n_prompt, size_t max_gen, uint32_t eos_token_id, uint32_t **out_tokens, size_t *out_n, bool *stopped_on_eos) {
    if (!e || !prompt_tokens || n_prompt == 0 || max_gen == 0 || !out_tokens || !out_n) return false;
    if (!e->specialized || !e->specialized->forward_fn) return false;
    if (!e->backend || !e->devices || !e->devices[0]) return false;
    size_t vocab_size = e->model_profile ? e->model_profile->vocab_size : 0;
    if (vocab_size == 0) return false;

    /* Free any previous autonomy buffers before allocating new ones. */
    if (e->autonomy_control_buffer) { e->backend->free_buffer(e->autonomy_control_buffer); e->autonomy_control_buffer = NULL; }
    if (e->autonomy_tokens_buffer) { e->backend->free_buffer(e->autonomy_tokens_buffer); e->autonomy_tokens_buffer = NULL; }

    e->autonomy_control_buffer = e->backend->alloc_buffer(e->devices[0], sizeof(vbs_gpu_control_t), false);
    e->autonomy_tokens_buffer = e->backend->alloc_buffer(e->devices[0], max_gen * sizeof(uint32_t), false);
    if (!e->autonomy_control_buffer || !e->autonomy_tokens_buffer) {
        if (e->autonomy_control_buffer) { e->backend->free_buffer(e->autonomy_control_buffer); e->autonomy_control_buffer = NULL; }
        if (e->autonomy_tokens_buffer) { e->backend->free_buffer(e->autonomy_tokens_buffer); e->autonomy_tokens_buffer = NULL; }
        return false;
    }

    uint32_t *generated = (uint32_t *)calloc(max_gen, sizeof(uint32_t));
    if (!generated) return false;
    float *logits = (float *)calloc(vocab_size, sizeof(float));
    if (!logits) { free(generated); return false; }

    /* Control block, boot state. generated_tokens_dev / logits_dev
     * are device addresses (buffer internal handles), never host
     * pointers copied into device memory. */
    memset(&e->autonomy_control, 0, sizeof(e->autonomy_control));
    e->autonomy_control.state = VBS_PGAE_PREFILL;
    e->autonomy_control.current_token = prompt_tokens[n_prompt - 1];
    e->autonomy_control.position = 0;
    e->autonomy_control.max_tokens = (uint32_t)max_gen;
    e->autonomy_control.vocab_size = (uint32_t)vocab_size;
    e->autonomy_control.eos_token_id = eos_token_id;
    e->autonomy_control.generated_tokens_dev = (uint64_t)(uintptr_t)e->autonomy_tokens_buffer->internal_handle;
    e->autonomy_control.logits_dev = e->logits ? (uint64_t)(uintptr_t)e->logits->internal_handle : 0;
    e->backend->copy_to_device(e->autonomy_control_buffer, &e->autonomy_control, sizeof(vbs_gpu_control_t));

    /* Prefill: prompt tokens, one forward each, as in --generate. */
    for (size_t i = 0; i < n_prompt; i++) {
        if (!vbs_engine_forward(e, &prompt_tokens[i], 1, logits)) {
            free(generated); free(logits);
            return false;
        }
    }

    /* Generate: engine owns argmax, position, stop. */
    e->autonomy_control.state = VBS_PGAE_GENERATE;
    bool eos_hit = false;
    size_t n_generated = 0;
    uint32_t cur_token = 0;
    {
        float max_logit = -1e9f;
        for (size_t i = 0; i < vocab_size; i++) {
            if (logits[i] > max_logit) { max_logit = logits[i]; cur_token = (uint32_t)i; }
        }
    }
    while (n_generated < max_gen) {
        if (eos_token_id != VBS_PGAE_EOS_NONE && cur_token == eos_token_id) { eos_hit = true; break; }
        generated[n_generated] = cur_token;
        n_generated++;
        e->autonomy_control.current_token = cur_token;
        e->autonomy_control.position = (uint32_t)n_generated;
        /* Mirror the generated token array to the device token buffer
         * as generation proceeds, so the device buffer is not stale. */
        e->backend->copy_to_device(e->autonomy_tokens_buffer, generated, n_generated * sizeof(uint32_t));
        if (n_generated >= max_gen) break;
        if (!vbs_engine_forward(e, &cur_token, 1, logits)) {
            free(generated); free(logits);
            return false;
        }
        float max_logit = -1e9f;
        cur_token = 0;
        for (size_t i = 0; i < vocab_size; i++) {
            if (logits[i] > max_logit) { max_logit = logits[i]; cur_token = (uint32_t)i; }
        }
    }

    e->autonomy_control.state = VBS_PGAE_EXIT;
    e->autonomy_control.position = (uint32_t)n_generated;
    e->backend->copy_to_device(e->autonomy_control_buffer, &e->autonomy_control, sizeof(vbs_gpu_control_t));

    *out_tokens = generated;
    *out_n = n_generated;
    if (stopped_on_eos) *stopped_on_eos = eos_hit;
    free(logits);
    return true;
}

double vbs_engine_evaluate_prompt_nll(vbs_engine_t *e, const uint32_t *prompt_tokens, size_t n_prompt, double *acc_out) {
    if (!e || !prompt_tokens || n_prompt < 2) return -1.0;
    if (!e->specialized || !e->specialized->forward_fn) return -1.0;
    size_t vocab_size = e->model_profile ? e->model_profile->vocab_size : 0;
    if (vocab_size == 0) return -1.0;
    float *logits = (float *)calloc(vocab_size, sizeof(float));
    if (!logits) return -1.0;
    if (e->specialized->reset_cache_fn) e->specialized->reset_cache_fn();
    double nll_sum = 0.0;
    size_t count = 0;
    size_t correct = 0;
    for (size_t i = 0; i + 1 < n_prompt; i++) {
        if (!vbs_engine_forward(e, &prompt_tokens[i], 1, logits)) { free(logits); return -1.0; }
        uint32_t target = prompt_tokens[i + 1];
        if (target >= vocab_size) continue;
        float max_logit = logits[0];
        for (size_t j = 1; j < vocab_size; j++) if (logits[j] > max_logit) max_logit = logits[j];
        double sum_exp = 0.0;
        for (size_t j = 0; j < vocab_size; j++) sum_exp += exp((double)logits[j] - (double)max_logit);
        if (sum_exp <= 0.0) continue;
        double log_prob = (double)logits[target] - (double)max_logit - log(sum_exp);
        nll_sum += -log_prob;
        count++;
        uint32_t best = 0;
        float best_logit = logits[0];
        for (size_t j = 1; j < vocab_size; j++) if (logits[j] > best_logit) { best_logit = logits[j]; best = (uint32_t)j; }
        if (best == target) correct++;
    }
    free(logits);
    if (count == 0) return -1.0;
    if (acc_out) *acc_out = (double)correct / (double)count;
    return nll_sum / (double)count;
}

void vbs_engine_destroy(vbs_engine_t *e) {
    if (!e) return;
    if (e->specialized) vbs_specializer_free(e->specialized);
    if (e->plan) vbs_planner_free(e->plan);
    if (e->autonomy_control_buffer) e->backend->free_buffer(e->autonomy_control_buffer);
    if (e->autonomy_tokens_buffer) e->backend->free_buffer(e->autonomy_tokens_buffer);
    if (e->act_a) e->backend->free_buffer(e->act_a);
    if (e->act_b) e->backend->free_buffer(e->act_b);
    if (e->logits) e->backend->free_buffer(e->logits);
    if (e->kv_cache_k) e->backend->free_buffer(e->kv_cache_k);
    if (e->kv_cache_v) e->backend->free_buffer(e->kv_cache_v);
    for (size_t i = 0; i < VBS_SCRATCH_COUNT; i++) {
        if (e->scratch[i]) e->backend->free_buffer(e->scratch[i]);
    }
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

/* --- Evolutionary / Hydra support ---------------------------------------
 * Mutation is FP32-only, by design. Quantized weights are packed bytes,
 * not floats, so reinterpreting them as float[] would corrupt the model.
 * The GGUF model file is also mmap'd PROT_READ, so the host tensor bytes
 * are not writable: mutation therefore works on a writable host mirror
 * (allocated on first mutation of a block) and pushes that mirror to VRAM
 * when the block is VRAM-resident. The mirror is reused on later mutations
 * of the same block; the original mmap pointer is the model map offset,
 * so a mirror is recognised by not pointing at the model map.
 *
 * Energy: reset sets every layer to 1.0 (alive). No per-layer energy
 * measurement is wired into forward yet, so Hydra pruning is dormant
 * until a caller explicitly sets g_vbs_layer_energy[layer] < 0.0001f.
 * Resetting to 0 would prune every layer on the next recompile, which
 * would delete the whole forward pass, so 0 is not a safe reset value.
 */
float g_vbs_layer_energy[VBS_MAX_LAYERS];

void vbs_engine_reset_energy(void) {
    for (size_t i = 0; i < VBS_MAX_LAYERS; i++) g_vbs_layer_energy[i] = 1.0f;
}

void vbs_engine_mutate_layer(vbs_engine_t *e, size_t layer_index, float mutation_rate) {
    if (!e || !e->plan || layer_index >= e->plan->n_plans) return;
    vbs_op_plan_t *plan = &e->plan->plans[layer_index];
    if (!plan->node || !plan->node->weight) return;
    if (plan->node->weight->type != VBS_GGML_TYPE_F32) {
        fprintf(stderr, "Mutate refused: tensor %s is type %d, not FP32; quantized bytes are not floats\n",
                plan->node->weight->name, (int)plan->node->weight->type);
        return;
    }
    if (!e->fabric || plan->node->tensor_index >= e->fabric->n_blocks) return;
    vbs_memory_block_t *block = e->fabric->blocks[plan->node->tensor_index];
    if (!block || !block->data) return;

    /* Writable mirror: first mutation copies the read-only mmap bytes. */
    const void *orig = NULL;
    if (e->model && e->model->map) orig = (const uint8_t *)e->model->map + plan->node->weight->data_offset;
    float *weights = NULL;
    if (block->data == orig) {
        weights = (float *)malloc(block->size);
        if (!weights) return;
        memcpy(weights, block->data, block->size);
        block->data = weights;
    } else {
        weights = (float *)block->data;
    }

    size_t num_elements = block->size / sizeof(float);
    for (size_t i = 0; i < num_elements; i++) {
        float noise = ((float)rand() / (float)RAND_MAX - 0.5f) * mutation_rate;
        weights[i] += noise;
    }

    if (block->current_tier == VBS_TIER_VRAM && block->gpu_buffer && e->backend && e->backend->copy_to_device) {
        e->backend->copy_to_device(block->gpu_buffer, weights, block->size);
    }
}

void vbs_engine_recompile(vbs_engine_t *e, char *err_buf, size_t err_cap) {
    if (!e || !e->plan || !e->graph) return;

    char src_path[640], bin_path[640], hash[65];
    const char *cache_dir = getenv("VBS_CACHE_DIR");
    if (!cache_dir) cache_dir = "/tmp/verum_cache";
    mkdir(cache_dir, 0755);

    vbs_specializer_compute_hash(e->graph, e->sys_profile, e->plan, hash, sizeof(hash));
    snprintf(src_path, sizeof(src_path), "%s/verum_%s.c", cache_dir, hash);
    snprintf(bin_path, sizeof(bin_path), "%s/verum_%s.so", cache_dir, hash);

    if (!vbs_specializer_emit_source(e->graph, e->sys_profile, e->plan, e->model_profile, src_path)) {
        if (err_buf) snprintf(err_buf, err_cap, "Failed to emit mutated source");
        return;
    }

    double compile_sec = 0;
    if (!vbs_specializer_compile(src_path, bin_path, e->sys_profile, &compile_sec)) {
        if (err_buf) snprintf(err_buf, err_cap, "Failed to compile mutated binary");
        return;
    }

    if (e->specialized) vbs_specializer_free(e->specialized);
    e->specialized = vbs_specializer_load(bin_path, hash);
    if (e->specialized) {
        e->specialized->compile_time_sec = compile_sec;
        e->specialized->was_cached = false;
        e->specialized->n_kernels_emitted = e->plan->n_plans;
        e->specialized->init_fn(e->fabric, (void *)e->backend);
        if (err_buf) err_buf[0] = '\0';
    } else if (err_buf) {
        snprintf(err_buf, err_cap, "Failed to load recompiled binary");
    }
}
