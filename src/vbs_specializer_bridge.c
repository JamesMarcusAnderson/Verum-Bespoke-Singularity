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
void *vbs_gpu_get_kv_k(void) { return g_kv_k ? g_kv_k->internal_handle : NULL; }
void *vbs_gpu_get_kv_v(void) { return g_kv_v ? g_kv_v->internal_handle : NULL; }

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
    char err[256];
    void *fn = g_backend->compile_primitive_pipeline(g_dev, VBS_OP_EMBEDDING, err, sizeof(err));
    if (!fn || !g_backend->dispatch_primitive) return;
    uint32_t d = (uint32_t)dim;
    void *args[] = { &w, &token, &out, &d };
    g_backend->dispatch_primitive(g_dev, fn, args, 4, (dim + 255) / 256, 256);
}

void vbs_gpu_dispatch_norm(void *w, void *in, void *out, size_t dim, float eps) {
    char err[256];
    void *fn = g_backend->compile_primitive_pipeline(g_dev, VBS_OP_NORMALIZE, err, sizeof(err));
    if (!fn || !g_backend->dispatch_primitive) return;
    uint32_t d = (uint32_t)dim;
    void *args[] = { &in, &w, &out, &d, &eps };
    g_backend->dispatch_primitive(g_dev, fn, args, 5, (dim + 255) / 256, 256);
}

void vbs_gpu_dispatch_rope(void *in, uint32_t pos, size_t hdim, size_t nh, float theta) {
    char err[256];
    void *fn = g_backend->compile_primitive_pipeline(g_dev, VBS_OP_ROTARY, err, sizeof(err));
    if (!fn || !g_backend->dispatch_primitive) return;
    uint32_t hd = (uint32_t)hdim, nheads = (uint32_t)nh;
    void *args[] = { &in, &pos, &hd, &nheads, &theta };
    size_t total_pairs = (hdim / 2) * nh;
    g_backend->dispatch_primitive(g_dev, fn, args, 5, (total_pairs + 255) / 256, 256);
}

void vbs_gpu_dispatch_attn(void *q, void *k_cache, void *v_cache, void *out, uint32_t pos, size_t nqh, size_t nkh, size_t hd) {
    char err[256];
    void *fn = g_backend->compile_primitive_pipeline(g_dev, VBS_OP_ATTENTION, err, sizeof(err));
    if (!fn || !g_backend->dispatch_primitive) return;
    uint32_t sl = pos + 1, qh = (uint32_t)nqh, kvh = (uint32_t)nkh, hdim = (uint32_t)hd;
    float scale = 1.0f / sqrtf((float)hd);
    void *args[] = { &q, &k_cache, &v_cache, &out, &sl, &qh, &kvh, &hdim, &scale };
    g_backend->dispatch_primitive(g_dev, fn, args, 9, nqh, hd > 32 ? hd : 32);
}

void vbs_gpu_dispatch_silu(void *gate, void *up, void *out, size_t n) {
    char err[256];
    void *fn = g_backend->compile_primitive_pipeline(g_dev, VBS_OP_ACTIVATION, err, sizeof(err));
    if (!fn || !g_backend->dispatch_primitive) return;
    uint32_t len = (uint32_t)n;
    void *args[] = { &gate, &up, &out, &len };
    g_backend->dispatch_primitive(g_dev, fn, args, 4, (n + 255) / 256, 256);
}

void vbs_gpu_dispatch_add(void *a, void *b, void *out, size_t n) {
    char err[256];
    void *fn = g_backend->compile_primitive_pipeline(g_dev, VBS_OP_ELEMENTWISE, err, sizeof(err));
    if (!fn || !g_backend->dispatch_primitive) return;
    uint32_t len = (uint32_t)n;
    void *args[] = { &a, &b, &out, &len };
    g_backend->dispatch_primitive(g_dev, fn, args, 4, (n + 255) / 256, 256);
}

void vbs_gpu_readback_logits(void *d_src, float *h_dst, size_t n) {
    vbs_buffer_t src_buf = { .internal_handle = d_src, .device = g_dev, .size = n * sizeof(float) };
    if (g_backend->copy_to_host) {
        g_backend->copy_to_host(h_dst, &src_buf, n * sizeof(float));
    }
}
