#include "vbs_specializer_bridge.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static vbs_device_t *g_dev = NULL;
static const vbs_backend_ops_t *g_backend = NULL;
static vbs_fabric_t *g_fabric = NULL;

static vbs_buffer_t *g_act_a = NULL;
static vbs_buffer_t *g_act_b = NULL;
static vbs_buffer_t *g_kv_k = NULL;
static vbs_buffer_t *g_kv_v = NULL;
static vbs_buffer_t *g_logits = NULL;

void vbs_bridge_bind_runtime(vbs_device_t *dev, const vbs_backend_ops_t *backend, vbs_fabric_t *fabric) {
    g_dev = dev;
    g_backend = backend;
    g_fabric = fabric;
}

void vbs_bridge_set_activations(vbs_buffer_t *a, vbs_buffer_t *b, vbs_buffer_t *kv_k, vbs_buffer_t *kv_v, vbs_buffer_t *logits) {
    g_act_a = a;
    g_act_b = b;
    g_kv_k = kv_k;
    g_kv_v = kv_v;
    g_logits = logits;
}

void *vbs_gpu_get_act_buffer_a(void) { return g_act_a ? g_act_a->internal_handle : NULL; }
void *vbs_gpu_get_act_buffer_b(void) { return g_act_b ? g_act_b->internal_handle : NULL; }

void *vbs_bridge_get_block_dptr(size_t index) {
    if (!g_fabric || index >= g_fabric->n_blocks) return NULL;
    vbs_memory_block_t *b = g_fabric->blocks[index];
    if (!b) return NULL;
    if (!b->gpu_buffer) {
        vbs_fabric_fault_in_to_device(g_fabric, b, g_dev);
    }
    return b->gpu_buffer ? b->gpu_buffer->internal_handle : NULL;
}

void vbs_gpu_dispatch_vram_q8_0(void *w, void *in, void *out, size_t r, size_t c) {
    char err[256];
    void *fn = g_backend->compile_linear_pipeline(g_dev, VBS_GGML_TYPE_Q8_0, err, sizeof(err));
    if (!fn) return;
    vbs_buffer_t bw = { .internal_handle = w, .device = g_dev };
    vbs_buffer_t bin = { .internal_handle = in, .device = g_dev };
    vbs_buffer_t bout = { .internal_handle = out, .device = g_dev };
    g_backend->dispatch_linear_tile(g_dev, fn, &bw, NULL, 0, &bin, &bout, (uint32_t)c, (uint32_t)r, 0, 0, false);
}

void vbs_gpu_dispatch_vram_q4_0(void *w, void *in, void *out, size_t r, size_t c) {
    char err[256];
    void *fn = g_backend->compile_linear_pipeline(g_dev, VBS_GGML_TYPE_Q4_0, err, sizeof(err));
    if (!fn) return;
    vbs_buffer_t bw = { .internal_handle = w, .device = g_dev };
    vbs_buffer_t bin = { .internal_handle = in, .device = g_dev };
    vbs_buffer_t bout = { .internal_handle = out, .device = g_dev };
    g_backend->dispatch_linear_tile(g_dev, fn, &bw, NULL, 0, &bin, &bout, (uint32_t)c, (uint32_t)r, 0, 0, false);
}

void vbs_gpu_dispatch_vram_f32(void *w, void *in, void *out, size_t r, size_t c) {
    char err[256];
    void *fn = g_backend->compile_linear_pipeline(g_dev, VBS_GGML_TYPE_F32, err, sizeof(err));
    if (!fn) return;
    vbs_buffer_t bw = { .internal_handle = w, .device = g_dev };
    vbs_buffer_t bin = { .internal_handle = in, .device = g_dev };
    vbs_buffer_t bout = { .internal_handle = out, .device = g_dev };
    g_backend->dispatch_linear_tile(g_dev, fn, &bw, NULL, 0, &bin, &bout, (uint32_t)c, (uint32_t)r, 0, 0, false);
}

void vbs_gpu_dispatch_embed(void *w, uint32_t token, void *out, size_t dim) {
    (void)w; (void)token; (void)out; (void)dim;
}

void vbs_gpu_dispatch_norm(void *w, void *in, void *out, size_t dim, float eps) {
    (void)w; (void)in; (void)out; (void)dim; (void)eps;
}

void vbs_gpu_dispatch_rope(void *in, uint32_t pos, size_t hdim, size_t nh, float theta) {
    (void)in; (void)pos; (void)hdim; (void)nh; (void)theta;
}

void vbs_gpu_dispatch_attn(void *q, void *k_cache, void *v_cache, void *out, uint32_t pos, size_t nqh, size_t nkh, size_t hd) {
    (void)q; (void)k_cache; (void)v_cache; (void)out; (void)pos; (void)nqh; (void)nkh; (void)hd;
}

void vbs_gpu_dispatch_silu(void *gate, void *up, void *out, size_t n) {
    (void)gate; (void)up; (void)out; (void)n;
}

void vbs_gpu_dispatch_add(void *a, void *b, void *out, size_t n) {
    (void)a; (void)b; (void)out; (void)n;
}

void vbs_gpu_readback_logits(void *d_src, float *h_dst, size_t n) {
    vbs_buffer_t src_buf = { .internal_handle = d_src, .device = g_dev, .size = n * sizeof(float) };
    if (g_backend->copy_to_host) {
        g_backend->copy_to_host(h_dst, &src_buf, n * sizeof(float));
    }
}
