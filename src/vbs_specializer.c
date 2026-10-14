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
        case VBS_GGML_TYPE_Q8_0:  return "q8_0";
        default:                  return "q8_0";
    }
}

bool vbs_specializer_emit_source(
    const vbs_graph_t *graph,
    const vbs_system_profile_t *sys,
    const vbs_execution_plan_t *plan,
    const vbs_model_profile_t *mp,
    const char *out_path
) {
    (void)sys;
    FILE *f = fopen(out_path, "w");
    if (!f) return false;

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
        "#define VOCAB_SIZE     %u\n\n"
        "typedef struct { void *device_ptr; size_t size; } vbs_block_t;\n"
        "static vbs_block_t g_blocks[%zu];\n"
        "static size_t g_current_pos = 0;\n\n"
        "extern void *vbs_bridge_get_block_dptr(size_t index);\n"
        "extern void *vbs_gpu_get_act_buffer_a(void);\n"
        "extern void *vbs_gpu_get_act_buffer_b(void);\n"
        "extern void vbs_gpu_dispatch_embed(void *w, uint32_t token, void *out, size_t dim);\n"
        "extern void vbs_gpu_dispatch_norm(void *w, void *in, void *out, size_t dim, float eps);\n"
        "extern void vbs_gpu_dispatch_rope(void *in, uint32_t pos, size_t hdim, size_t nh, float theta);\n"
        "extern void vbs_gpu_dispatch_attn(void *q, void *k_cache, void *v_cache, void *out, uint32_t pos, size_t nqh, size_t nkh, size_t hd);\n"
        "extern void vbs_gpu_dispatch_silu(void *gate, void *up, void *out, size_t n);\n"
        "extern void vbs_gpu_dispatch_add(void *a, void *b, void *out, size_t n);\n"
        "extern void vbs_gpu_dispatch_vram_q8_0(void *w, void *in, void *out, size_t r, size_t c);\n"
        "extern void vbs_gpu_dispatch_vram_q4_0(void *w, void *in, void *out, size_t r, size_t c);\n"
        "extern void vbs_gpu_dispatch_vram_f32(void *w, void *in, void *out, size_t r, size_t c);\n"
        "extern void vbs_gpu_readback_logits(void *d_src, float *h_dst, size_t n);\n\n",
        mp->n_layers, mp->n_heads, mp->n_kv_heads, mp->head_dim,
        mp->embedding_dim, mp->context_length, mp->vocab_size ? mp->vocab_size : 32000,
        graph->n_nodes
    );

    for (size_t i = 0; i < plan->n_plans; i++) {
        const vbs_node_t *n = plan->plans[i].node;
        fprintf(f, "static inline void op_%zu(uint32_t tok, void *in_dptr, void *out_dptr) {\n", i);

        switch (n->kind) {
            case VBS_OP_EMBEDDING:
                fprintf(f, "    vbs_gpu_dispatch_embed(g_blocks[%zu].device_ptr, tok, in_dptr, EMBED_DIM);\n", i);
                break;
            case VBS_OP_NORMALIZE:
                fprintf(f, "    vbs_gpu_dispatch_norm(g_blocks[%zu].device_ptr, in_dptr, out_dptr, EMBED_DIM, 1e-5f);\n", i);
                break;
            case VBS_OP_ROTARY:
                fprintf(f, "    vbs_gpu_dispatch_rope(in_dptr, (uint32_t)g_current_pos, HEAD_DIM, N_HEADS, 10000.0f);\n");
                break;
            case VBS_OP_ATTENTION:
                fprintf(f, "    vbs_gpu_dispatch_attn(in_dptr, NULL, NULL, out_dptr, (uint32_t)g_current_pos, N_HEADS, N_KV_HEADS, HEAD_DIM);\n");
                break;
            case VBS_OP_ACTIVATION:
                fprintf(f, "    vbs_gpu_dispatch_silu(in_dptr, out_dptr, out_dptr, EMBED_DIM);\n");
                break;
            case VBS_OP_ELEMENTWISE:
                fprintf(f, "    vbs_gpu_dispatch_add(in_dptr, out_dptr, out_dptr, EMBED_DIM);\n");
                break;
            case VBS_OP_LINEAR:
                if (n->weight) {
                    fprintf(f, "    vbs_gpu_dispatch_vram_%s(g_blocks[%zu].device_ptr, in_dptr, out_dptr, %llu, %llu);\n",
                        ggml_type_name(n->weight->type), i,
                        (unsigned long long)n->weight->rows, (unsigned long long)n->weight->columns);
                }
                break;
            default:
                break;
        }
        fprintf(f, "}\n\n");
    }

    fprintf(f,
        "void verum_forward(const uint32_t *tokens, size_t n_tokens, float *logits_out) {\n"
        "    void *d_in = vbs_gpu_get_act_buffer_a();\n"
        "    void *d_out = vbs_gpu_get_act_buffer_b();\n"
        "    for (size_t t = 0; t < n_tokens; t++) {\n"
        "        uint32_t tok = tokens[t];\n"
    );

    for (size_t i = 0; i < plan->n_plans; i++) {
        fprintf(f, "        op_%zu(tok, d_in, d_out);\n", i);
        fprintf(f, "        { void *tmp = d_in; d_in = d_out; d_out = tmp; }\n");
    }

    fprintf(f,
        "        g_current_pos++;\n"
        "    }\n"
        "    vbs_gpu_readback_logits(d_in, logits_out, VOCAB_SIZE);\n"
        "}\n\n"
        "void verum_init(void *fabric, void *backend) {\n"
        "    (void)fabric; (void)backend;\n"
        "    for (size_t i = 0; i < %zu; i++) {\n"
        "        g_blocks[i].device_ptr = vbs_bridge_get_block_dptr(i);\n"
        "    }\n"
        "    g_current_pos = 0;\n"
        "}\n\n"
        "void verum_reset_cache(void) { g_current_pos = 0; }\n"
        "void verum_shutdown(void) { memset(g_blocks, 0, sizeof(g_blocks)); }\n",
        graph->n_nodes
    );

    fclose(f);
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
