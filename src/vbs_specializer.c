#include "vbs_specializer.h"
#include "vbs_sha256.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dlfcn.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

void vbs_specializer_compute_hash(
    const vbs_graph_t *graph,
    const vbs_system_profile_t *sys,
    const vbs_execution_plan_t *plan,
    char *out_hex, size_t out_cap
) {
    vbs_sha256_ctx_t ctx;
    vbs_sha256_init(&ctx);
    vbs_sha256_update(&ctx, graph->fingerprint, strlen(graph->fingerprint));

    for (size_t i = 0; i < sys->n_gpus; i++) {
        vbs_sha256_update(&ctx, sys->gpus[i].name, strlen(sys->gpus[i].name));
        vbs_sha256_update(&ctx, &sys->gpus[i].total_vram_bytes, sizeof(uint64_t));
    }
    vbs_sha256_update(&ctx, &sys->cpu.total_ram_bytes, sizeof(uint64_t));

    for (size_t i = 0; i < plan->n_plans; i++) {
        vbs_sha256_update(&ctx, &plan->plans[i].strategy, sizeof(int));
        vbs_sha256_update(&ctx, &plan->plans[i].tile_rows, sizeof(uint32_t));
        vbs_sha256_update(&ctx, &plan->plans[i].tile_cols, sizeof(uint32_t));
    }

    uint8_t digest[32];
    vbs_sha256_final(&ctx, digest);
    for (int i = 0; i < 32 && (size_t)(i * 2 + 2) < out_cap; i++) {
        sprintf(out_hex + i * 2, "%02x", digest[i]);
    }
    out_hex[64] = '\0';
}

static bool file_exists(const char *p) {
    struct stat st;
    return stat(p, &st) == 0;
}

vbs_specialized_binary_t *vbs_specializer_load_cached(const char *hash, const char *cache_dir) {
    char bin_path[640];
    snprintf(bin_path, sizeof(bin_path), "%s/verum_%s.so", cache_dir, hash);
    if (!file_exists(bin_path)) return NULL;
    return vbs_specializer_load(bin_path, hash);
}

static const char *ggml_type_name(vbs_ggml_type_t t) {
    switch (t) {
        case VBS_GGML_TYPE_F32:   return "f32";
        case VBS_GGML_TYPE_F16:   return "f16";
        case VBS_GGML_TYPE_Q4_0:  return "q4_0";
        case VBS_GGML_TYPE_Q4_1:  return "q4_1";
        case VBS_GGML_TYPE_Q5_0:  return "q5_0";
        case VBS_GGML_TYPE_Q5_1:  return "q5_1";
        case VBS_GGML_TYPE_Q8_0:  return "q8_0";
        case VBS_GGML_TYPE_Q8_1:  return "q8_1";
        case VBS_GGML_TYPE_Q2_K:  return "q2_k";
        case VBS_GGML_TYPE_Q3_K:  return "q3_k";
        case VBS_GGML_TYPE_Q4_K:  return "q4_k";
        case VBS_GGML_TYPE_Q5_K:  return "q5_k";
        case VBS_GGML_TYPE_Q6_K:  return "q6_k";
        case VBS_GGML_TYPE_Q8_K:  return "q8_k";
        case VBS_GGML_TYPE_LNS8:  return "lns8";
        default:                  return "unknown";
    }
}

// Only these weight types have a real vbs_gpu_dispatch_vram_* implementation
// in the bridge. Anything else must refuse loudly instead of silently
// running through the wrong kernel.
static bool ggml_type_has_dispatch(vbs_ggml_type_t t) {
    return t == VBS_GGML_TYPE_F32 || t == VBS_GGML_TYPE_F16 || t == VBS_GGML_TYPE_Q4_0 || t == VBS_GGML_TYPE_Q4_1 || t == VBS_GGML_TYPE_Q6_K || t == VBS_GGML_TYPE_Q8_0 || t == VBS_GGML_TYPE_LNS8;
}

// Sequenced emitter: instead of chaining every weight through two ping-pong
// buffers (which cannot express attention), the forward pass is generated as
// one token step with named buffers and the transformer block structure
// written out per layer: norm -> Q/K/V -> rope -> KV store -> attention ->
// output projection -> residual add -> norm -> gate/up -> SiLU*up -> down ->
// residual add. Tensor roles are read from GGUF names (blk.<L>.*); a weight
// the sequencer cannot place stops the compile loudly, like the graph
// compiler's refusal of unresolved matrices.
typedef enum {
    SEQ_EMBED, SEQ_ATTN_NORM, SEQ_Q, SEQ_K, SEQ_V, SEQ_ATTN_OUT,
    SEQ_FFN_NORM, SEQ_GATE, SEQ_UP, SEQ_DOWN, SEQ_FINAL_NORM, SEQ_FINAL_LINEAR
} seq_role_t;

static bool seq_classify(const vbs_node_t *n, seq_role_t *role, int *layer) {
    const char *name = n->weight ? n->weight->name : "";
    *layer = -1;
    const char *blk = strstr(name, "blk.");
    if (blk) {
        *layer = atoi(blk + 4);
        if (strstr(name, "attn_norm"))   { *role = SEQ_ATTN_NORM; return true; }
        if (strstr(name, "attn_q"))      { *role = SEQ_Q; return true; }
        if (strstr(name, "attn_k"))      { *role = SEQ_K; return true; }
        if (strstr(name, "attn_v"))      { *role = SEQ_V; return true; }
        if (strstr(name, "attn_output")) { *role = SEQ_ATTN_OUT; return true; }
        if (strstr(name, "ffn_norm"))    { *role = SEQ_FFN_NORM; return true; }
        if (strstr(name, "ffn_gate"))    { *role = SEQ_GATE; return true; }
        if (strstr(name, "ffn_up"))      { *role = SEQ_UP; return true; }
        if (strstr(name, "ffn_down"))    { *role = SEQ_DOWN; return true; }
        return false;
    }
    if (strstr(name, "token_embd")) { *role = SEQ_EMBED; return true; }
    if (strstr(name, "output_norm")) { *role = SEQ_FINAL_NORM; return true; }
    if (strstr(name, "output"))     { *role = SEQ_FINAL_LINEAR; return true; }
    return false;
}

#ifndef VBS_MAX_LAYERS
#define VBS_MAX_LAYERS 512
#endif

bool vbs_specializer_emit_source(
    const vbs_graph_t *graph,
    const vbs_system_profile_t *sys,
    const vbs_execution_plan_t *plan,
    const vbs_model_profile_t *mp,
    const char *out_path
) {
    (void)sys;
    (void)graph;

    // Classify every planned node up front; refuse what cannot be placed or
    // has no real kernel, rather than emitting a wrong forward pass.
    seq_role_t *roles = (seq_role_t *)calloc(plan->n_plans, sizeof(seq_role_t));
    int *layers = (int *)calloc(plan->n_plans, sizeof(int));
    long embed_slot = -1, final_norm_slot = -1, final_linear_slot = -1;
    uint64_t gate_rows = 0;
    for (size_t i = 0; i < plan->n_plans; i++) {
        const vbs_node_t *n = plan->plans[i].node;
        if (!n || !n->weight) {
            fprintf(stderr, "Specializer refused: plan node %zu has no weight tensor\n", i);
            free(roles); free(layers);
            return false;
        }
        if ((n->kind == VBS_OP_LINEAR || n->kind == VBS_OP_PROJECTION || n->kind == VBS_OP_EMBEDDING) && !ggml_type_has_dispatch(n->weight->type)) {
            fprintf(stderr, "Specializer refused: no GPU dispatch exists for weight type %s (tensor %s)\n",
                    ggml_type_name(n->weight->type), n->weight->name);
            free(roles); free(layers);
            return false;
        }
        if (!seq_classify(n, &roles[i], &layers[i])) {
            fprintf(stderr, "Specializer refused: cannot place tensor %s in a transformer block sequence\n", n->weight->name);
            free(roles); free(layers);
            return false;
        }
        if (roles[i] == SEQ_EMBED) embed_slot = (long)i;
        if (roles[i] == SEQ_FINAL_NORM) final_norm_slot = (long)i;
        if (roles[i] == SEQ_FINAL_LINEAR) final_linear_slot = (long)i;
        if (roles[i] == SEQ_GATE) gate_rows = n->weight->rows;
    }
    if (embed_slot < 0) {
        fprintf(stderr, "Specializer refused: no token embedding tensor found\n");
        free(roles); free(layers);
        return false;
    }
    if (final_linear_slot < 0) final_linear_slot = embed_slot; // tied embeddings
    {
        const vbs_node_t *fn_ = plan->plans[final_linear_slot].node;
        if (!ggml_type_has_dispatch(fn_->weight->type)) {
            fprintf(stderr, "Specializer refused: no GPU dispatch exists for weight type %s (tensor %s)\n",
                    ggml_type_name(fn_->weight->type), fn_->weight->name);
            free(roles); free(layers);
            return false;
        }
    }

    uint32_t kv_capacity = mp->kv_ctx_capacity ? mp->kv_ctx_capacity : mp->context_length;

    FILE *f = fopen(out_path, "w");
    if (!f) { free(roles); free(layers); return false; }

    fprintf(f,
        "// AUTO-GENERATED VERUM JIT SPECIALIZER MODULE\n"
        "#include <stdint.h>\n"
        "#include <stddef.h>\n"
        "#include <string.h>\n\n"
        "#define N_LAYERS       %u\n"
        "#define N_HEADS        %u\n"
        "#define N_KV_HEADS     %u\n"
        "#define HEAD_DIM       %u\n"
        "#define EMBED_DIM      %u\n"
        "#define CONTEXT_LEN    %u\n"
        "#define VOCAB_SIZE     %u\n"
        "#define KV_CAPACITY    %u\n"
        "#define KV_DIM         %u\n"
        "#define ROPE_THETA     %.9ef\n"
        "#define RMS_EPS        %.9ef\n\n"
        "typedef struct { void *device_ptr; size_t size; } vbs_block_t;\n"
        "static vbs_block_t g_blocks[%zu];\n"
        "static size_t g_current_pos = 0;\n\n"
        "extern void *vbs_bridge_get_block_dptr(size_t index);\n"
        "extern void *vbs_gpu_get_act_buffer_a(void);\n"
        "extern void *vbs_gpu_get_act_buffer_b(void);\n"
        "extern void *vbs_gpu_get_kv_k(void);\n"
        "extern void *vbs_gpu_get_kv_v(void);\n"
        "extern void *vbs_gpu_get_logits(void);\n"
        "extern void *vbs_gpu_get_scratch(size_t index);\n"
        "extern void vbs_gpu_dispatch_embed_f32(void *w, uint32_t token, void *out, size_t dim);\n"
        "extern void vbs_gpu_dispatch_embed_f16(void *w, uint32_t token, void *out, size_t dim);\n"
        "extern void vbs_gpu_dispatch_embed_q8_0(void *w, uint32_t token, void *out, size_t dim);\n"
        "extern void vbs_gpu_dispatch_embed_q4_0(void *w, uint32_t token, void *out, size_t dim);\n"
        "extern void vbs_gpu_dispatch_embed_q4_1(void *w, uint32_t token, void *out, size_t dim);\n"
        "extern void vbs_gpu_dispatch_embed_q6_k(void *w, uint32_t token, void *out, size_t dim);\n"
        "extern void vbs_gpu_dispatch_norm(void *w, void *in, void *out, size_t dim, float eps);\n"
        "extern void vbs_gpu_dispatch_rope(void *in, uint32_t pos, size_t hdim, size_t nh, float theta);\n"
        "extern void vbs_gpu_dispatch_cache(void *k, void *v, void *kc, void *vc, uint32_t pos, uint32_t layer, uint32_t max_ctx, size_t kv_dim);\n"
        "extern void vbs_gpu_dispatch_attn(void *q, void *k_cache, void *v_cache, void *out, uint32_t pos, uint32_t layer, uint32_t max_ctx, size_t nqh, size_t nkh, size_t hd);\n"
        "extern void vbs_gpu_dispatch_silu(void *gate, void *up, void *out, size_t n);\n"
        "extern void vbs_gpu_dispatch_add(void *a, void *b, void *out, size_t n);\n"
        "extern void vbs_gpu_dispatch_vram_q8_0(void *w, void *in, void *out, size_t r, size_t c);\n"
        "extern void vbs_gpu_dispatch_vram_q4_0(void *w, void *in, void *out, size_t r, size_t c);\n"
        "extern void vbs_gpu_dispatch_vram_q4_1(void *w, void *in, void *out, size_t r, size_t c);\n"
        "extern void vbs_gpu_dispatch_vram_q6_k(void *w, void *in, void *out, size_t r, size_t c);\n"
        "extern void vbs_gpu_dispatch_vram_f16(void *w, void *in, void *out, size_t r, size_t c);\n"
        "extern void vbs_gpu_dispatch_vram_f32(void *w, void *in, void *out, size_t r, size_t c);\n"
        "extern void vbs_gpu_dispatch_vram_lns8(void *w, void *in, void *out, size_t r, size_t c);\n"
        "extern void vbs_gpu_readback_logits(void *d_src, float *h_dst, size_t n);\n\n",
        mp->n_layers, mp->n_heads, mp->n_kv_heads, mp->head_dim,
        mp->embedding_dim, mp->context_length, mp->vocab_size,
        kv_capacity, mp->n_kv_heads * mp->head_dim,
        mp->rope_theta ? mp->rope_theta : 10000.0f,
        mp->rms_eps ? mp->rms_eps : 1e-5f,
        plan->n_plans
    );

    fprintf(f,
        "static void forward_token(uint32_t tok) {\n"
        "    void *STREAM = vbs_gpu_get_act_buffer_a();\n"
        "    void *TMP    = vbs_gpu_get_act_buffer_b();\n"
        "    void *QB     = vbs_gpu_get_scratch(0);\n"
        "    void *KB     = vbs_gpu_get_scratch(1);\n"
        "    void *VB     = vbs_gpu_get_scratch(2);\n"
        "    void *AB     = vbs_gpu_get_scratch(3);\n"
        "    void *GATE   = vbs_gpu_get_scratch(4);\n"
        "    void *UP     = vbs_gpu_get_scratch(5);\n"
        "    void *HIDDEN = vbs_gpu_get_scratch(6);\n"
        "    void *NORMED = vbs_gpu_get_scratch(7);\n"
        "    void *KC     = vbs_gpu_get_kv_k();\n"
        "    void *VC     = vbs_gpu_get_kv_v();\n"
        "    uint32_t pos = (uint32_t)g_current_pos;\n"
    );

    /* HYDRA PROTOCOL: prune whole transformer layers, not single nodes.
     * layers[i] is the blk.<L> layer number (-1 for embed/final, which are
     * never pruned: deleting embed or the output projection would delete
     * the forward pass itself). A layer is pruned only when its energy
     * was explicitly set below threshold; reset sets all energies to 1.0,
     * so a plain recompile prunes nothing. Skipping a whole blk layer
     * leaves the residual stream untouched, which is the coherent
     * pruning semantics; skipping one Q/K/V/down node alone would not be.
     */
    extern float g_vbs_layer_energy[VBS_MAX_LAYERS];
    for (size_t i = 0; i < plan->n_plans; i++) {
        const vbs_node_t *n = plan->plans[i].node;
        int L = layers[i];
        if (L >= 0 && L < (int)VBS_MAX_LAYERS && g_vbs_layer_energy[L] < 0.0001f) {
            fprintf(f, "    // HYDRA PROTOCOL: transformer layer %d pruned (energy: %f)\n", L, g_vbs_layer_energy[L]);
            continue;
        }
        switch (roles[i]) {
            case SEQ_EMBED:
                fprintf(f, "    vbs_gpu_dispatch_embed_%s(g_blocks[%zu].device_ptr, tok, STREAM, EMBED_DIM);\n",
                        ggml_type_name(n->weight->type), i);
                break;
            case SEQ_ATTN_NORM:
            case SEQ_FFN_NORM:
                fprintf(f, "    vbs_gpu_dispatch_norm(g_blocks[%zu].device_ptr, STREAM, NORMED, EMBED_DIM, RMS_EPS);\n", i);
                break;
            case SEQ_Q:
                fprintf(f, "    vbs_gpu_dispatch_vram_%s(g_blocks[%zu].device_ptr, NORMED, QB, %llu, %llu);\n",
                        ggml_type_name(n->weight->type), i,
                        (unsigned long long)n->weight->rows, (unsigned long long)n->weight->columns);
                fprintf(f, "    vbs_gpu_dispatch_rope(QB, pos, HEAD_DIM, N_HEADS, ROPE_THETA);\n");
                break;
            case SEQ_K:
                fprintf(f, "    vbs_gpu_dispatch_vram_%s(g_blocks[%zu].device_ptr, NORMED, KB, %llu, %llu);\n",
                        ggml_type_name(n->weight->type), i,
                        (unsigned long long)n->weight->rows, (unsigned long long)n->weight->columns);
                fprintf(f, "    vbs_gpu_dispatch_rope(KB, pos, HEAD_DIM, N_KV_HEADS, ROPE_THETA);\n");
                break;
            case SEQ_V:
                fprintf(f, "    vbs_gpu_dispatch_vram_%s(g_blocks[%zu].device_ptr, NORMED, VB, %llu, %llu);\n",
                        ggml_type_name(n->weight->type), i,
                        (unsigned long long)n->weight->rows, (unsigned long long)n->weight->columns);
                fprintf(f, "    vbs_gpu_dispatch_cache(KB, VB, KC, VC, pos, %d, KV_CAPACITY, KV_DIM);\n", L);
                fprintf(f, "    vbs_gpu_dispatch_attn(QB, KC, VC, AB, pos, %d, KV_CAPACITY, N_HEADS, N_KV_HEADS, HEAD_DIM);\n", L);
                break;
            case SEQ_ATTN_OUT:
                fprintf(f, "    vbs_gpu_dispatch_vram_%s(g_blocks[%zu].device_ptr, AB, TMP, %llu, %llu);\n",
                        ggml_type_name(n->weight->type), i,
                        (unsigned long long)n->weight->rows, (unsigned long long)n->weight->columns);
                fprintf(f, "    vbs_gpu_dispatch_add(STREAM, TMP, STREAM, EMBED_DIM);\n");
                break;
            case SEQ_GATE:
                fprintf(f, "    vbs_gpu_dispatch_vram_%s(g_blocks[%zu].device_ptr, NORMED, GATE, %llu, %llu);\n",
                        ggml_type_name(n->weight->type), i,
                        (unsigned long long)n->weight->rows, (unsigned long long)n->weight->columns);
                break;
            case SEQ_UP:
                fprintf(f, "    vbs_gpu_dispatch_vram_%s(g_blocks[%zu].device_ptr, NORMED, UP, %llu, %llu);\n",
                        ggml_type_name(n->weight->type), i,
                        (unsigned long long)n->weight->rows, (unsigned long long)n->weight->columns);
                fprintf(f, "    vbs_gpu_dispatch_silu(GATE, UP, HIDDEN, %llu);\n", (unsigned long long)gate_rows);
                break;
            case SEQ_DOWN:
                fprintf(f, "    vbs_gpu_dispatch_vram_%s(g_blocks[%zu].device_ptr, HIDDEN, TMP, %llu, %llu);\n",
                        ggml_type_name(n->weight->type), i,
                        (unsigned long long)n->weight->rows, (unsigned long long)n->weight->columns);
                fprintf(f, "    vbs_gpu_dispatch_add(STREAM, TMP, STREAM, EMBED_DIM);\n");
                break;
            case SEQ_FINAL_NORM:
            case SEQ_FINAL_LINEAR:
                break; // emitted at the tail below
        }
    }

    if (final_norm_slot >= 0) {
        fprintf(f, "    vbs_gpu_dispatch_norm(g_blocks[%ld].device_ptr, STREAM, NORMED, EMBED_DIM, RMS_EPS);\n", final_norm_slot);
    }
    {
        const vbs_node_t *n = plan->plans[final_linear_slot].node;
        fprintf(f, "    vbs_gpu_dispatch_vram_%s(g_blocks[%ld].device_ptr, NORMED, vbs_gpu_get_logits(), VOCAB_SIZE, EMBED_DIM);\n",
                ggml_type_name(n->weight->type), final_linear_slot);
    }

    fprintf(f,
        "}\n\n"
        "void verum_forward(const uint32_t *tokens, size_t n_tokens, float *logits_out) {\n"
        "    for (size_t t = 0; t < n_tokens; t++) {\n"
        "        forward_token(tokens[t]);\n"
        "        g_current_pos++;\n"
        "    }\n"
        "    vbs_gpu_readback_logits(vbs_gpu_get_logits(), logits_out, VOCAB_SIZE);\n"
        "}\n\n"
        "void verum_init(void *fabric, void *backend) {\n"
        "    (void)fabric; (void)backend;\n"
    );
    for (size_t i = 0; i < plan->n_plans; i++) {
        fprintf(f, "    g_blocks[%zu].device_ptr = vbs_bridge_get_block_dptr(%u);\n",
                i, plan->plans[i].node->tensor_index);
    }
    fprintf(f,
        "    g_current_pos = 0;\n"
        "}\n\n"
        "void verum_reset_cache(void) { g_current_pos = 0; }\n"
        "void verum_shutdown(void) { memset(g_blocks, 0, sizeof(g_blocks)); }\n"
    );

    fclose(f);
    free(roles);
    free(layers);
    return true;
}

bool vbs_specializer_compile(const char *src_path, const char *bin_path, const vbs_system_profile_t *sys, double *out_sec) {
    (void)sys;
    char cmd[2048];
    double t0 = now_sec();
    const char *compiler = (system("command -v clang >/dev/null 2>&1") == 0) ? "clang" : "gcc";
    snprintf(cmd, sizeof(cmd), "%s -O3 -fomit-frame-pointer -flto -shared -fPIC -o %s %s 2>&1", compiler, bin_path, src_path);
    int rc = system(cmd);
    *out_sec = now_sec() - t0;
    return rc == 0;
}

vbs_specialized_binary_t *vbs_specializer_load(const char *bin_path, const char *hash) {
    double t0 = now_sec();
    void *h = dlopen(bin_path, RTLD_NOW | RTLD_LOCAL);
    if (!h) return NULL;

    vbs_specialized_binary_t *b = (vbs_specialized_binary_t *)calloc(1, sizeof(*b));
    b->loaded_handle = h;
    snprintf(b->binary_path, sizeof(b->binary_path), "%s", bin_path);
    snprintf(b->specialization_hash, sizeof(b->specialization_hash), "%s", hash);

    b->init_fn        = (void (*)(void *, void *))dlsym(h, "verum_init");
    b->forward_fn     = (void (*)(const uint32_t *, size_t, float *))dlsym(h, "verum_forward");
    b->reset_cache_fn = (void (*)(void))dlsym(h, "verum_reset_cache");
    b->shutdown_fn    = (void (*)(void))dlsym(h, "verum_shutdown");

    struct stat st;
    if (stat(bin_path, &st) == 0) b->binary_size = (size_t)st.st_size;
    b->load_time_sec = now_sec() - t0;

    if (!b->init_fn || !b->forward_fn) {
        dlclose(h);
        free(b);
        return NULL;
    }
    return b;
}

void vbs_specializer_free(vbs_specialized_binary_t *b) {
    if (!b) return;
    if (b->shutdown_fn) b->shutdown_fn();
    if (b->loaded_handle) dlclose(b->loaded_handle);
    free(b);
}

void vbs_specializer_print_stats(const vbs_specialized_binary_t *b) {
    printf("\n=== Verum Specialized Binary (GPU Native) ===\n");
    printf("Hash:         %.16s...\n", b->specialization_hash);
    printf("Binary:       %s (%.1f KiB)\n", b->binary_path, b->binary_size / 1024.0);
    printf("Cache hit:    %s\n", b->was_cached ? "YES" : "NO");
    printf("Kernels:      %zu bespoke GPU operations emitted\n", b->n_kernels_emitted);
    printf("=============================================\n\n");
}
