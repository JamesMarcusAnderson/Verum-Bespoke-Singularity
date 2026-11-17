/* vbs_backend_cpu.c — CPU fallback backend on the OLD backend interface.
 *
 * This is the backend the engine falls back to when no GPU is present.
 * It is NOT a stub: every op it advertises computes correct results on
 * host memory, using the shared reference math in vbs_host_math.h
 * (the same formulas the CUDA PTX and Vulkan GLSL kernels implement).
 *
 * Port note: the Dropbox held file src/vbs_backend_cpu.c was written
 * against the rejected NEW ops interface (vbs_backend_ops.h) with
 * different signatures (int init, raw-pointer copies, no buffers or
 * pipelines) and is NOT copied here. This file is written fresh against
 * the canonical OLD interface (include/vbs_backend.h); only the math
 * ideas (rms_norm / rope / silu / attention formulas) carry over, and
 * the linear path is new — exact dequantize-then-dot via
 * vbs_dequant_block, honoring the tile offsets and accumulation the
 * GPU backends ignore.
 *
 * Device model: "device" memory IS host memory. alloc_buffer returns a
 * buffer whose internal_handle == host_ptr, so the bridge's device
 * handles dereference directly as host pointers. Copies are memcpy.
 * init() always succeeds: a host CPU is always present.
 */

#include "vbs_backend.h"
#include "vbs_host_math.h"
#include "vbs_dequant.h"
#include "vbs_port.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static vbs_device_t g_cpu_device;
static bool g_cpu_ready = false;

/* Pipeline handles: the table returns void*, so linear/embed pipelines
 * carry their weight type in a small heap struct; primitive pipelines
 * carry the op kind (base) plus the embed type for EMBEDDING. */
typedef struct {
    vbs_ggml_type_t type;
} vbs_cpu_linear_pipeline_t;

typedef struct {
    vbs_prim_pipeline_t base; /* .native = this struct, .kind = op kind */
    vbs_ggml_type_t type;     /* EMBEDDING weight type (F32 default) */
} vbs_cpu_prim_pipeline_t;

static bool cpu_init(void) {
    if (g_cpu_ready) return true;
    memset(&g_cpu_device, 0, sizeof(g_cpu_device));
    long cores = (long)vbs_cpu_count();
    if (cores <= 0) cores = 1;
    snprintf(g_cpu_device.name, sizeof(g_cpu_device.name),
             "CPU (host, %ld cores)", cores);
    g_cpu_device.type = VBS_BACKEND_CPU;
    uint64_t ram = vbs_ram_bytes();
    g_cpu_device.max_working_set_bytes = ram;
    g_cpu_device.max_buffer_bytes = ram;
    g_cpu_device.internal_handle = NULL;
    g_cpu_ready = true;
    return true;
}

static size_t cpu_get_device_count(void) { return g_cpu_ready ? 1 : 0; }

static vbs_device_t *cpu_get_device(size_t i) {
    return (g_cpu_ready && i == 0) ? &g_cpu_device : NULL;
}

static vbs_buffer_t *cpu_alloc_buffer(vbs_device_t *dev, size_t bytes,
                                      bool shared) {
    (void)shared;
    if (bytes == 0) bytes = 1;
    void *mem = calloc(1, bytes);
    if (!mem) return NULL;
    vbs_buffer_t *b = (vbs_buffer_t *)calloc(1, sizeof(vbs_buffer_t));
    if (!b) { free(mem); return NULL; }
    b->internal_handle = mem;
    b->host_ptr = mem;
    b->size = bytes;
    b->device = dev;
    return b;
}

static void cpu_free_buffer(vbs_buffer_t *b) {
    if (!b) return;
    free(b->host_ptr);
    free(b);
}

static bool cpu_copy_to_device(vbs_buffer_t *dst, const void *src,
                               size_t bytes) {
    if (!dst || !src) return false;
    if (dst->host_ptr != src) memcpy(dst->host_ptr, src, bytes);
    return true;
}

static bool cpu_copy_to_host(void *dst, const vbs_buffer_t *src,
                             size_t bytes) {
    if (!dst || !src) return false;
    /* The bridge sometimes passes a stack wrapper whose host_ptr is
     * NULL but whose internal_handle is the real pointer. */
    const void *p = src->host_ptr ? src->host_ptr : src->internal_handle;
    if (!p) return false;
    if (dst != p) memcpy(dst, p, bytes);
    return true;
}

static void *cpu_compile_linear_pipeline(vbs_device_t *dev,
                                         vbs_ggml_type_t type,
                                         char *err_buf, size_t err_cap) {
    (void)dev;
    if (!vbs_host_type_supported(type)) {
        snprintf(err_buf, err_cap,
                 "CPU backend: no linear kernel for weight type %d", (int)type);
        return NULL;
    }
    vbs_cpu_linear_pipeline_t *p =
        (vbs_cpu_linear_pipeline_t *)malloc(sizeof(*p));
    if (!p) return NULL;
    p->type = type;
    return p;
}

static vbs_cpu_prim_pipeline_t *cpu_prim_pipeline(vbs_op_kind_t kind,
                                                  vbs_ggml_type_t type) {
    vbs_cpu_prim_pipeline_t *p =
        (vbs_cpu_prim_pipeline_t *)malloc(sizeof(*p));
    if (!p) return NULL;
    p->base.native = p;
    p->base.kind = kind;
    p->type = type;
    return p;
}

static void *cpu_compile_embed_pipeline(vbs_device_t *dev,
                                        vbs_ggml_type_t type,
                                        char *err_buf, size_t err_cap) {
    (void)dev;
    if (!vbs_host_type_supported(type)) {
        snprintf(err_buf, err_cap,
                 "CPU backend: no embedding kernel for weight type %d", (int)type);
        return NULL;
    }
    return cpu_prim_pipeline(VBS_OP_EMBEDDING, type);
}

static void *cpu_compile_primitive_pipeline(vbs_device_t *dev,
                                            vbs_op_kind_t kind,
                                            char *err_buf, size_t err_cap) {
    (void)dev;
    switch (kind) {
        case VBS_OP_EMBEDDING:   /* F32 table, like the CUDA path */
        case VBS_OP_NORMALIZE:
        case VBS_OP_ROTARY:
        case VBS_OP_CACHE:
        case VBS_OP_ATTENTION:
        case VBS_OP_ACTIVATION:
        case VBS_OP_ELEMENTWISE:
            return cpu_prim_pipeline(kind, VBS_GGML_TYPE_F32);
        default:
            snprintf(err_buf, err_cap,
                     "CPU backend: no primitive kernel for op kind %d", (int)kind);
            return NULL;
    }
}

static bool cpu_dispatch_linear_tile(
    vbs_device_t *dev, void *pipeline,
    vbs_buffer_t *wp, vbs_buffer_t *ws, size_t copied_bytes,
    vbs_buffer_t *in, vbs_buffer_t *out,
    uint32_t cols, uint32_t rows, uint32_t col_off, uint32_t row_off,
    bool accumulate
) {
    (void)dev; (void)copied_bytes;
    vbs_cpu_linear_pipeline_t *p = (vbs_cpu_linear_pipeline_t *)pipeline;
    if (!p || !wp || !in || !out) return false;
    /* Two calling conventions, both supported:
     *  - slice executor: ws->host_ptr holds the packed tile.
     *  - bridge (VRAM path): ws is NULL; wp->internal_handle IS the
     *    resident packed tensor (host memory on this backend). */
    const uint8_t *packed = NULL;
    if (ws && ws->host_ptr) packed = (const uint8_t *)ws->host_ptr;
    else packed = (const uint8_t *)wp->internal_handle;
    if (!packed) return false;
    const float *x = (const float *)(in->host_ptr ? in->host_ptr
                                                  : in->internal_handle);
    float *y = (float *)(out->host_ptr ? out->host_ptr : out->internal_handle);
    if (!x || !y) return false;
    return vbs_host_linear_tile(packed, p->type, x, y,
                                cols, rows, col_off, row_off, accumulate);
}

static bool cpu_dispatch_primitive(vbs_device_t *dev, void *pipeline,
                                   const vbs_arg_t *args, size_t n_args,
                                   size_t grid_x, size_t block_x) {
    (void)dev; (void)grid_x; (void)block_x;
    vbs_cpu_prim_pipeline_t *pp = (vbs_cpu_prim_pipeline_t *)pipeline;
    if (!pp || !args) return false;

    /* Canonical bridge arg order (vbs_specializer_bridge.h). Buffer
     * args carry host pointers on this backend (internal_handle). */
    switch (pp->base.kind) {
        case VBS_OP_EMBEDDING: {
            if (n_args != 4) return false;
            const uint8_t *w = (const uint8_t *)args[0].value;
            uint32_t token = *(const uint32_t *)args[1].value;
            float *o = (float *)args[2].value;
            uint32_t dim = *(const uint32_t *)args[3].value;
            if (!w || !o) return false;
            return vbs_host_embed_row(w, pp->type, token, o, dim);
        }
        case VBS_OP_NORMALIZE: {
            if (n_args != 5) return false;
            const float *in = (const float *)args[0].value;
            const float *w = (const float *)args[1].value;
            float *o = (float *)args[2].value;
            uint32_t dim = *(const uint32_t *)args[3].value;
            float eps = *(const float *)args[4].value;
            if (!in || !w || !o) return false;
            vbs_host_rms_norm(in, w, o, dim, eps);
            return true;
        }
        case VBS_OP_ROTARY: {
            if (n_args != 5) return false;
            float *x = (float *)args[0].value;
            uint32_t pos = *(const uint32_t *)args[1].value;
            uint32_t hd = *(const uint32_t *)args[2].value;
            uint32_t nh = *(const uint32_t *)args[3].value;
            float theta = *(const float *)args[4].value;
            if (!x || hd == 0 || nh == 0) return false;
            vbs_host_rope(x, pos, hd, nh, theta);
            return true;
        }
        case VBS_OP_CACHE: {
            if (n_args != 8) return false;
            const float *k = (const float *)args[0].value;
            const float *v = (const float *)args[1].value;
            uint16_t *kc = (uint16_t *)args[2].value;
            uint16_t *vc = (uint16_t *)args[3].value;
            uint32_t pos = *(const uint32_t *)args[4].value;
            uint32_t layer = *(const uint32_t *)args[5].value;
            uint32_t max_ctx = *(const uint32_t *)args[6].value;
            uint32_t kv_dim = *(const uint32_t *)args[7].value;
            if (!k || !v || !kc || !vc) return false;
            vbs_host_cache_store(k, v, kc, vc, pos, layer, max_ctx, kv_dim);
            return true;
        }
        case VBS_OP_ATTENTION: {
            if (n_args != 11) return false;
            const float *q = (const float *)args[0].value;
            const uint16_t *kc = (const uint16_t *)args[1].value;
            const uint16_t *vc = (const uint16_t *)args[2].value;
            float *o = (float *)args[3].value;
            uint32_t seq_len = *(const uint32_t *)args[4].value;
            uint32_t layer = *(const uint32_t *)args[5].value;
            uint32_t max_ctx = *(const uint32_t *)args[6].value;
            uint32_t nqh = *(const uint32_t *)args[7].value;
            uint32_t nkh = *(const uint32_t *)args[8].value;
            uint32_t hd = *(const uint32_t *)args[9].value;
            float scale = *(const float *)args[10].value;
            if (!q || !kc || !vc || !o) return false;
            if (seq_len == 0 || nqh == 0 || nkh == 0 || hd == 0) return false;
            vbs_host_attention(q, kc, vc, o, seq_len, layer, max_ctx,
                               nqh, nkh, hd, scale);
            return true;
        }
        case VBS_OP_ACTIVATION: { /* silu_mul: gate, up, out, n */
            if (n_args != 4) return false;
            const float *gate = (const float *)args[0].value;
            const float *up = (const float *)args[1].value;
            float *o = (float *)args[2].value;
            uint32_t n = *(const uint32_t *)args[3].value;
            if (!gate || !up || !o) return false;
            vbs_host_silu_mul(gate, up, o, n);
            return true;
        }
        case VBS_OP_ELEMENTWISE: { /* add: a, b, out, n */
            if (n_args != 4) return false;
            const float *a = (const float *)args[0].value;
            const float *b = (const float *)args[1].value;
            float *o = (float *)args[2].value;
            uint32_t n = *(const uint32_t *)args[3].value;
            if (!a || !b || !o) return false;
            vbs_host_add(a, b, o, n);
            return true;
        }
        default:
            return false;
    }
}

static void cpu_synchronize(vbs_device_t *dev) {
    (void)dev; /* CPU execution is synchronous by definition. */
}

const vbs_backend_ops_t g_vbs_cpu_backend = {
    .init = cpu_init,
    .get_device_count = cpu_get_device_count,
    .get_device = cpu_get_device,
    .alloc_buffer = cpu_alloc_buffer,
    .free_buffer = cpu_free_buffer,
    .copy_to_device = cpu_copy_to_device,
    .copy_to_host = cpu_copy_to_host,
    .compile_linear_pipeline = cpu_compile_linear_pipeline,
    .compile_embed_pipeline = cpu_compile_embed_pipeline,
    .compile_primitive_pipeline = cpu_compile_primitive_pipeline,
    .dispatch_linear_tile = cpu_dispatch_linear_tile,
    .dispatch_primitive = cpu_dispatch_primitive,
    .synchronize_device = cpu_synchronize,
    .launch_persistent_kernel = NULL,
};
