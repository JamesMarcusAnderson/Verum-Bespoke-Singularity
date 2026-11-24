/* vbs_specializer_bridge.c — the generated code's dispatch seam.
 *
 * The specializer emits a forward pass that calls these functions;
 * each one marshals its arguments into the canonical bridge order
 * (see the header) and dispatches through the bound backend.
 * Pipelines are per-(device, kind/type) constants: compiled once
 * and cached, instead of re-created on every dispatch. A compile
 * failure is swallowed here (the error text is discarded) and the
 * cache entry stays NULL, so the next dispatch retries the compile
 * and a failed pipeline silently no-ops the dispatch.
 */

#include "vbs_specializer_bridge.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static vbs_device_t *g_dev = NULL;
static const vbs_backend_ops_t *g_backend = NULL;
static vbs_fabric_t *g_fabric = NULL;

static vbs_buffer_t *g_act_a = NULL;
static vbs_buffer_t *g_act_b = NULL;
static vbs_buffer_t *g_kv_k = NULL;
static vbs_buffer_t *g_kv_v = NULL;
static vbs_buffer_t *g_logits = NULL;
static vbs_buffer_t *g_scratch[VBS_SCRATCH_COUNT];

void vbs_bridge_bind_runtime(vbs_device_t *dev,
                             const vbs_backend_ops_t *backend,
                             vbs_fabric_t *fabric) {
    g_dev = dev;
    g_backend = backend;
    g_fabric = fabric;
}

void vbs_bridge_set_activations(vbs_buffer_t *a, vbs_buffer_t *b,
                                vbs_buffer_t *kv_k, vbs_buffer_t *kv_v,
                                vbs_buffer_t *logits) {
    g_act_a = a;
    g_act_b = b;
    g_kv_k = kv_k;
    g_kv_v = kv_v;
    g_logits = logits;
}

void vbs_bridge_set_scratch(vbs_buffer_t **bufs, size_t n) {
    for (size_t i = 0; i < VBS_SCRATCH_COUNT; i++)
        g_scratch[i] = (bufs && i < n) ? bufs[i] : NULL;
}

static void *buffer_handle(vbs_buffer_t *b) {
    return b ? b->internal_handle : NULL;
}

void *vbs_gpu_get_act_buffer_a(void) { return buffer_handle(g_act_a); }
void *vbs_gpu_get_act_buffer_b(void) { return buffer_handle(g_act_b); }
void *vbs_gpu_get_kv_k(void) { return buffer_handle(g_kv_k); }
void *vbs_gpu_get_kv_v(void) { return buffer_handle(g_kv_v); }
void *vbs_gpu_get_logits(void) { return buffer_handle(g_logits); }

void *vbs_gpu_get_scratch(size_t index) {
    if (index >= VBS_SCRATCH_COUNT) return NULL;
    return buffer_handle(g_scratch[index]);
}

void *vbs_bridge_get_block_dptr(size_t index) {
    if (!g_fabric || index >= g_fabric->n_blocks) return NULL;
    vbs_memory_block_t *b = g_fabric->blocks[index];
    if (!b) return NULL;
    if (!b->gpu_buffer)
        vbs_fabric_fault_in_to_device(g_fabric, b, g_dev);
    return b->gpu_buffer ? b->gpu_buffer->internal_handle : NULL;
}

/* Pipeline caches, one per family. Compile failures leave a NULL
 * entry, so the next dispatch retries. */
static void *cached_linear_pipeline(vbs_model_type_t type) {
    static void *cache[64];
    if ((unsigned)type >= 64 || !g_backend ||
        !g_backend->compile_linear_pipeline)
        return NULL;
    if (!cache[type]) {
        char err[256];
        cache[type] =
            g_backend->compile_linear_pipeline(g_dev, type, err, sizeof(err));
    }
    return cache[type];
}

static void *cached_prim_pipeline(vbs_op_kind_t kind) {
    static void *cache[VBS_OP_PROJECTION + 1];
    if ((unsigned)kind > VBS_OP_PROJECTION || !g_backend ||
        !g_backend->compile_primitive_pipeline)
        return NULL;
    if (!cache[kind]) {
        char err[256];
        cache[kind] = g_backend->compile_primitive_pipeline(g_dev, kind, err,
                                                            sizeof(err));
    }
    return cache[kind];
}

static void *cached_embed_pipeline(vbs_model_type_t type) {
    static void *cache[16];
    if ((unsigned)type >= 16 || !g_backend ||
        !g_backend->compile_embed_pipeline)
        return NULL;
    if (!cache[type]) {
        char err[256];
        cache[type] =
            g_backend->compile_embed_pipeline(g_dev, type, err, sizeof(err));
    }
    return cache[type];
}

/* Wrap a raw device handle in a stack buffer struct for dispatch. */
static vbs_buffer_t wrap_handle(void *handle) {
    vbs_buffer_t b;
    memset(&b, 0, sizeof(b));
    b.internal_handle = handle;
    b.device = g_dev;
    return b;
}

static void run_linear(vbs_model_type_t type, void *w, void *in, void *out,
                       size_t r, size_t c) {
    void *fn = cached_linear_pipeline(type);
    if (!fn) return;
    vbs_buffer_t bw = wrap_handle(w);
    vbs_buffer_t bin = wrap_handle(in);
    vbs_buffer_t bout = wrap_handle(out);
    g_backend->dispatch_linear_tile(g_dev, fn, &bw, NULL, 0, &bin, &bout,
                                    (uint32_t)c, (uint32_t)r, 0, 0, false);
}

void vbs_gpu_dispatch_vram_q8_0(void *w, void *in, void *out, size_t r, size_t c) {
    run_linear(VBS_MODEL_TYPE_Q8_0, w, in, out, r, c);
}

void vbs_gpu_dispatch_vram_q4_0(void *w, void *in, void *out, size_t r, size_t c) {
    run_linear(VBS_MODEL_TYPE_Q4_0, w, in, out, r, c);
}

void vbs_gpu_dispatch_vram_q4_1(void *w, void *in, void *out, size_t r, size_t c) {
    run_linear(VBS_MODEL_TYPE_Q4_1, w, in, out, r, c);
}

void vbs_gpu_dispatch_vram_q6_k(void *w, void *in, void *out, size_t r, size_t c) {
    run_linear(VBS_MODEL_TYPE_Q6_K, w, in, out, r, c);
}

void vbs_gpu_dispatch_vram_f16(void *w, void *in, void *out, size_t r, size_t c) {
    run_linear(VBS_MODEL_TYPE_F16, w, in, out, r, c);
}

void vbs_gpu_dispatch_vram_f32(void *w, void *in, void *out, size_t r, size_t c) {
    run_linear(VBS_MODEL_TYPE_F32, w, in, out, r, c);
}

void vbs_gpu_dispatch_vram_lns8(void *w, void *in, void *out, size_t r, size_t c) {
    run_linear(VBS_MODEL_TYPE_LNS8, w, in, out, r, c);
}

static void run_embed(vbs_model_type_t type, void *w, uint32_t token,
                      void *out, size_t dim) {
    void *fn = cached_embed_pipeline(type);
    if (!fn || !g_backend->dispatch_primitive) return;
    uint32_t tok = token, d = (uint32_t)dim;
    vbs_arg_t args[] = { vbs_arg_buf(w), vbs_arg_u32(&tok),
                         vbs_arg_buf(out), vbs_arg_u32(&d) };
    g_backend->dispatch_primitive(g_dev, fn, args, 4, (dim + 255) / 256, 256);
}

void vbs_gpu_dispatch_embed(void *w, uint32_t token, void *out, size_t dim) {
    run_embed(VBS_MODEL_TYPE_F32, w, token, out, dim);
}

void vbs_gpu_dispatch_embed_f32(void *w, uint32_t token, void *out, size_t dim) {
    run_embed(VBS_MODEL_TYPE_F32, w, token, out, dim);
}

void vbs_gpu_dispatch_embed_f16(void *w, uint32_t token, void *out, size_t dim) {
    run_embed(VBS_MODEL_TYPE_F16, w, token, out, dim);
}

void vbs_gpu_dispatch_embed_q8_0(void *w, uint32_t token, void *out, size_t dim) {
    run_embed(VBS_MODEL_TYPE_Q8_0, w, token, out, dim);
}

void vbs_gpu_dispatch_embed_q4_0(void *w, uint32_t token, void *out, size_t dim) {
    run_embed(VBS_MODEL_TYPE_Q4_0, w, token, out, dim);
}

void vbs_gpu_dispatch_embed_q4_1(void *w, uint32_t token, void *out, size_t dim) {
    run_embed(VBS_MODEL_TYPE_Q4_1, w, token, out, dim);
}

void vbs_gpu_dispatch_embed_q6_k(void *w, uint32_t token, void *out, size_t dim) {
    run_embed(VBS_MODEL_TYPE_Q6_K, w, token, out, dim);
}

void vbs_gpu_dispatch_norm(void *w, void *in, void *out, size_t dim, float eps) {
    void *fn = cached_prim_pipeline(VBS_OP_NORMALIZE);
    if (!fn || !g_backend->dispatch_primitive) return;
    uint32_t d = (uint32_t)dim;
    float e = eps;
    vbs_arg_t args[] = { vbs_arg_buf(in), vbs_arg_buf(w), vbs_arg_buf(out),
                         vbs_arg_u32(&d), vbs_arg_f32(&e) };
    g_backend->dispatch_primitive(g_dev, fn, args, 5, (dim + 255) / 256, 256);
}

void vbs_gpu_dispatch_rope(void *in, uint32_t pos, size_t hdim, size_t nh,
                           float theta) {
    void *fn = cached_prim_pipeline(VBS_OP_ROTARY);
    if (!fn || !g_backend->dispatch_primitive) return;
    uint32_t p = pos, hd = (uint32_t)hdim, nheads = (uint32_t)nh;
    float th = theta;
    vbs_arg_t args[] = { vbs_arg_buf(in), vbs_arg_u32(&p), vbs_arg_u32(&hd),
                         vbs_arg_u32(&nheads), vbs_arg_f32(&th) };
    size_t total_pairs = (hdim / 2) * nh;
    g_backend->dispatch_primitive(g_dev, fn, args, 5,
                                  (total_pairs + 255) / 256, 256);
}

void vbs_gpu_dispatch_cache(void *k, void *v, void *kc, void *vc,
                            uint32_t pos, uint32_t layer, uint32_t max_ctx,
                            size_t kv_dim) {
    void *fn = cached_prim_pipeline(VBS_OP_CACHE);
    if (!fn || !g_backend->dispatch_primitive) return;
    uint32_t p = pos, l = layer, mc = max_ctx, d = (uint32_t)kv_dim;
    vbs_arg_t args[] = { vbs_arg_buf(k), vbs_arg_buf(v),
                         vbs_arg_buf(kc), vbs_arg_buf(vc),
                         vbs_arg_u32(&p), vbs_arg_u32(&l),
                         vbs_arg_u32(&mc), vbs_arg_u32(&d) };
    g_backend->dispatch_primitive(g_dev, fn, args, 8,
                                  (kv_dim + 255) / 256, 256);
}

void vbs_gpu_dispatch_attn(void *q, void *k_cache, void *v_cache, void *out,
                           uint32_t pos, uint32_t layer, uint32_t max_ctx,
                           size_t nqh, size_t nkh, size_t hd) {
    void *fn = cached_prim_pipeline(VBS_OP_ATTENTION);
    if (!fn || !g_backend->dispatch_primitive) return;
    uint32_t sl = pos + 1, l = layer, mc = max_ctx;
    uint32_t qh = (uint32_t)nqh, kvh = (uint32_t)nkh, hdim = (uint32_t)hd;
    float scale = 1.0f / sqrtf((float)hd);
    vbs_arg_t args[] = { vbs_arg_buf(q), vbs_arg_buf(k_cache),
                         vbs_arg_buf(v_cache), vbs_arg_buf(out),
                         vbs_arg_u32(&sl), vbs_arg_u32(&l), vbs_arg_u32(&mc),
                         vbs_arg_u32(&qh), vbs_arg_u32(&kvh),
                         vbs_arg_u32(&hdim), vbs_arg_f32(&scale) };
    g_backend->dispatch_primitive(g_dev, fn, args, 11, nqh,
                                  hd > 32 ? hd : 32);
}

void vbs_gpu_dispatch_silu(void *gate, void *up, void *out, size_t n) {
    void *fn = cached_prim_pipeline(VBS_OP_ACTIVATION);
    if (!fn || !g_backend->dispatch_primitive) return;
    uint32_t len = (uint32_t)n;
    vbs_arg_t args[] = { vbs_arg_buf(gate), vbs_arg_buf(up),
                         vbs_arg_buf(out), vbs_arg_u32(&len) };
    g_backend->dispatch_primitive(g_dev, fn, args, 4, (n + 255) / 256, 256);
}

void vbs_gpu_dispatch_add(void *a, void *b, void *out, size_t n) {
    void *fn = cached_prim_pipeline(VBS_OP_ELEMENTWISE);
    if (!fn || !g_backend->dispatch_primitive) return;
    uint32_t len = (uint32_t)n;
    vbs_arg_t args[] = { vbs_arg_buf(a), vbs_arg_buf(b),
                         vbs_arg_buf(out), vbs_arg_u32(&len) };
    g_backend->dispatch_primitive(g_dev, fn, args, 4, (n + 255) / 256, 256);
}

void vbs_gpu_readback_logits(void *d_src, float *h_dst, size_t n) {
    vbs_buffer_t src_buf = { .internal_handle = d_src, .device = g_dev,
                             .size = n * sizeof(float) };
    if (g_backend->copy_to_host)
        g_backend->copy_to_host(h_dst, &src_buf, n * sizeof(float));
}
