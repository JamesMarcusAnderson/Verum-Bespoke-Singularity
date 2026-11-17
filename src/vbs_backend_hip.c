/*
 * vbs_backend_hip.c — HIP backend on the OLD interface (ported).
 *
 * PORTED, not copied, from two sources:
 *   base ......... ~/workspace/vbs-adapter/src/vbs_backend_hip.c
 *                  (itself a port of the repo stub + the held Dropbox
 *                  HIP rewrite onto the old interface; see ADAPTER.md)
 *   held Dropbox . src/vbs_backend_hip.c (READ FOR REFERENCE ONLY —
 *                  never copied: it has TWO definitions of
 *                  vbs_backend_hip_init, includes the rejected
 *                  vbs_backend_ops.h, and exports only the new-
 *                  interface init. Its defects are fixed BY THE PORT:
 *                  one init, no ops header, the const table
 *                  g_vbs_hip_backend.)
 *
 * What this port adds over the adapter staging:
 *   - rope honors the bridge theta (the adapter used a fixed 10000
 *     base, ignoring args[4]).
 *   - dispatch_linear_tile COMPUTES (host-staged, exact dequantize +
 *     dot via vbs_host_math.h) instead of the repo stub's
 *     copy-weights-and-return-true. Both calling conventions work
 *     (staging tile or resident tensor) and col/row offsets +
 *     accumulation are honored.
 *   - EMBEDDING / CACHE / ATTENTION primitives are implemented
 *     (host-staged) instead of refused.
 *   - init is idempotent and records a failure reason for the
 *     backend-selection log (vbs_hip_last_error).
 *   - host math is shared with the CPU backend (vbs_host_math.h),
 *     so the two backends cannot drift apart numerically.
 *
 * Honest limits (unchanged from the adapter):
 *   - No ROCm hardware is present in the build VM, so the HIP calls
 *     themselves are verified by compile only; the MATH is verified
 *     through the CPU backend, which runs the same code.
 *   - vbs_hip_load_kernel() loads a HIP code object (HSACO), not raw
 *     SPIR-V: hipModuleLoadData does not accept SPIR-V. (The adapter
 *     comment saying SPIR-V could be loaded here was wrong.) A real
 *     GPU kernel path for HIP needs the GCN emitter output assembled
 *     to HSACO offline — a future step, not claimed here.
 *   - Primitives run host-staged (device -> host -> compute -> device),
 *     so they are correct but not fast. They exist so a ROCm machine
 *     produces right answers everywhere, not fast answers in some ops
 *     and refusals in others.
 */

#include "vbs_backend.h"
#include "vbs_host_math.h"
#include "vbs_port.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int hipError_t;
typedef void *hipDevice_t;
typedef void *hipCtx_t;
typedef void *hipModule_t;
typedef void *hipFunction_t;
typedef void *hipStream_t;
typedef void *hipDeviceptr_t;

#define HIP_SUCCESS 0

typedef hipError_t (*hipInit_fn)(unsigned int);
typedef hipError_t (*hipDeviceGet_fn)(hipDevice_t *, int);
typedef hipError_t (*hipGetDeviceCount_fn)(int *);
typedef hipError_t (*hipDeviceGetName_fn)(char *, int, int);
typedef hipError_t (*hipCtxCreate_fn)(hipCtx_t *, unsigned int, hipDevice_t);
typedef hipError_t (*hipModuleLoadData_fn)(hipModule_t *, const void *);
typedef hipError_t (*hipModuleGetFunction_fn)(hipFunction_t *, hipModule_t, const char *);
typedef hipError_t (*hipLaunchKernel_fn)(hipFunction_t, unsigned int, unsigned int,
                                         unsigned int, unsigned int, unsigned int,
                                         unsigned int, unsigned int, hipStream_t,
                                         void **, void **);
typedef hipError_t (*hipMalloc_fn)(hipDeviceptr_t *, size_t);
typedef hipError_t (*hipFree_fn)(hipDeviceptr_t);
typedef hipError_t (*hipMemcpyHtoD_fn)(hipDeviceptr_t, const void *, size_t);
typedef hipError_t (*hipMemcpyDtoH_fn)(void *, hipDeviceptr_t, size_t);
typedef hipError_t (*hipCtxSynchronize_fn)(void);

static struct {
    void *handle;
    hipInit_fn hipInit;
    hipDeviceGet_fn hipDeviceGet;
    hipGetDeviceCount_fn hipGetDeviceCount;
    hipDeviceGetName_fn hipDeviceGetName;
    hipCtxCreate_fn hipCtxCreate;
    hipModuleLoadData_fn hipModuleLoadData;
    hipModuleGetFunction_fn hipModuleGetFunction;
    hipLaunchKernel_fn hipLaunchKernel;
    hipMalloc_fn hipMalloc;
    hipFree_fn hipFree;
    hipMemcpyHtoD_fn hipMemcpyHtoD;
    hipMemcpyDtoH_fn hipMemcpyDtoH;
    hipCtxSynchronize_fn hipCtxSynchronize;
    hipDevice_t device;
    hipCtx_t ctx;
    hipModule_t module;
    bool probed;
    bool ready;
} g_hip;

static char g_hip_last_error[256] = "not probed yet";
static vbs_device_t g_hip_devices[8];
static int g_hip_dev_count = 0;

const char *vbs_hip_last_error(void) { return g_hip_last_error; }

/* Pipeline handles carry the weight type (linear/embed) or op kind. */
typedef struct {
    vbs_ggml_type_t type;
} vbs_hip_linear_pipeline_t;

typedef struct {
    vbs_prim_pipeline_t base;
    vbs_ggml_type_t type; /* EMBEDDING weight type (F32 default) */
} vbs_hip_prim_pipeline_t;

/* ------------------------------------------------------------------ */
/* Device staging: device ptr <-> host memory, with byte offsets       */
/* ------------------------------------------------------------------ */

static void *dev_to_host_bytes(void *dptr, size_t byte_off, size_t n_bytes) {
    uint8_t *h = (uint8_t *)malloc(n_bytes ? n_bytes : 1);
    if (!h) return NULL;
    if (n_bytes == 0) return h;
    if (g_hip.hipMemcpyDtoH(h, (hipDeviceptr_t)((uint8_t *)dptr + byte_off),
                            n_bytes) != HIP_SUCCESS) {
        free(h);
        return NULL;
    }
    return h;
}

static bool host_to_dev_bytes(void *dptr, size_t byte_off,
                              const void *h, size_t n_bytes) {
    if (n_bytes == 0) return true;
    return g_hip.hipMemcpyHtoD((hipDeviceptr_t)((uint8_t *)dptr + byte_off),
                               h, n_bytes) == HIP_SUCCESS;
}

/* ------------------------------------------------------------------ */
/* Init / devices / buffers                                            */
/* ------------------------------------------------------------------ */

static bool hip_init(void) {
    if (g_hip.probed) return g_hip.ready;
    g_hip.probed = true;

    static const char *paths[] = {
        "libamdhip64.so", "libamdhip64.so.5", "libamdhip64.so.6",
        "/opt/rocm/lib/libamdhip64.so",
        "/opt/rocm-6.0.0/lib/libamdhip64.so",
        "amdhip64.dll", NULL
    };
    for (int i = 0; paths[i]; i++) {
        g_hip.handle = vbs_dlopen(paths[i], 0);
        if (g_hip.handle) break;
    }
    if (!g_hip.handle) {
        snprintf(g_hip_last_error, sizeof(g_hip_last_error),
                 "ROCm runtime not found (libamdhip64.so missing)");
        return false;
    }

#define LOAD_REQ(name)                                                     \
    g_hip.name = (name##_fn)vbs_dlsym(g_hip.handle, #name);                \
    if (!g_hip.name) {                                                     \
        snprintf(g_hip_last_error, sizeof(g_hip_last_error),               \
                 "ROCm runtime lacks %s", #name);                          \
        return false;                                                      \
    }
#define LOAD_OPT(name)                                                     \
    g_hip.name = (name##_fn)vbs_dlsym(g_hip.handle, #name);
    LOAD_REQ(hipInit);
    LOAD_REQ(hipGetDeviceCount);
    LOAD_REQ(hipDeviceGetName);
    LOAD_REQ(hipMalloc);
    LOAD_REQ(hipFree);
    LOAD_REQ(hipMemcpyHtoD);
    LOAD_REQ(hipMemcpyDtoH);
    LOAD_REQ(hipCtxSynchronize);
    LOAD_OPT(hipDeviceGet);
    LOAD_OPT(hipCtxCreate);
    LOAD_OPT(hipModuleLoadData);
    LOAD_OPT(hipModuleGetFunction);
    LOAD_OPT(hipLaunchKernel);
#undef LOAD_REQ
#undef LOAD_OPT

    if (g_hip.hipInit(0) != HIP_SUCCESS) {
        snprintf(g_hip_last_error, sizeof(g_hip_last_error),
                 "hipInit failed");
        return false;
    }
    int count = 0;
    g_hip.hipGetDeviceCount(&count);
    if (count <= 0) {
        snprintf(g_hip_last_error, sizeof(g_hip_last_error),
                 "no AMD GPU found by ROCm");
        return false;
    }
    if (count > 8) count = 8;

    /* Context for device 0, when the loader provides it. */
    if (g_hip.hipDeviceGet && g_hip.hipCtxCreate) {
        if (g_hip.hipDeviceGet(&g_hip.device, 0) == HIP_SUCCESS)
            g_hip.hipCtxCreate(&g_hip.ctx, 0, g_hip.device);
    }

    for (int i = 0; i < count; i++) {
        char name[128] = {0};
        g_hip.hipDeviceGetName(name, sizeof(name) - 1, i);
        snprintf(g_hip_devices[i].name, 128, "AMD %s", name);
        g_hip_devices[i].type = VBS_BACKEND_HIP;
        g_hip_devices[i].internal_handle = (void *)(intptr_t)i;
    }
    g_hip_dev_count = count;
    g_hip.ready = true;
    snprintf(g_hip_last_error, sizeof(g_hip_last_error), "ok");
    return true;
}

static size_t hip_get_device_count(void) { return (size_t)g_hip_dev_count; }
static vbs_device_t *hip_get_device(size_t i) { return &g_hip_devices[i]; }

static vbs_buffer_t *hip_alloc_buffer(vbs_device_t *dev, size_t bytes,
                                      bool shared) {
    (void)shared;
    if (bytes == 0) bytes = 1;
    hipDeviceptr_t dptr = NULL;
    if (g_hip.hipMalloc(&dptr, bytes) != HIP_SUCCESS) return NULL;
    vbs_buffer_t *b = (vbs_buffer_t *)calloc(1, sizeof(vbs_buffer_t));
    if (!b) { g_hip.hipFree(dptr); return NULL; }
    b->internal_handle = dptr;
    b->size = bytes;
    b->device = dev;
    b->host_ptr = calloc(1, bytes);
    return b;
}

static void hip_free_buffer(vbs_buffer_t *b) {
    if (!b) return;
    g_hip.hipFree((hipDeviceptr_t)b->internal_handle);
    free(b->host_ptr);
    free(b);
}

static bool hip_copy_to_device(vbs_buffer_t *dst, const void *src,
                               size_t bytes) {
    if (!dst || !src) return false;
    return g_hip.hipMemcpyHtoD((hipDeviceptr_t)dst->internal_handle, src,
                               bytes) == HIP_SUCCESS;
}

static bool hip_copy_to_host(void *dst, const vbs_buffer_t *src,
                             size_t bytes) {
    if (!dst || !src) return false;
    return g_hip.hipMemcpyDtoH(dst, (hipDeviceptr_t)src->internal_handle,
                               bytes) == HIP_SUCCESS;
}

/* Load a HIP code object (HSACO) at runtime. This does NOT accept raw
 * SPIR-V — hipModuleLoadData takes code objects; SPIR-V needs offline
 * compilation to HSACO first (a future step for the GCN emitter). */
int vbs_hip_load_kernel(const uint8_t *code, size_t len) {
    (void)len;
    if (!code || !g_hip.hipModuleLoadData) return -1;
    if (g_hip.hipModuleLoadData(&g_hip.module, code) != HIP_SUCCESS) {
        fprintf(stderr, "[VBS/HIP] hipModuleLoadData failed\n");
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Pipelines                                                           */
/* ------------------------------------------------------------------ */

static void *hip_compile_linear_pipeline(vbs_device_t *dev,
                                         vbs_ggml_type_t type,
                                         char *err_buf, size_t err_cap) {
    (void)dev;
    if (!vbs_host_type_supported(type)) {
        snprintf(err_buf, err_cap,
                 "HIP backend: no linear kernel for weight type %d", (int)type);
        return NULL;
    }
    vbs_hip_linear_pipeline_t *p =
        (vbs_hip_linear_pipeline_t *)malloc(sizeof(*p));
    if (!p) return NULL;
    p->type = type;
    return p;
}

static vbs_hip_prim_pipeline_t *hip_prim_pipeline(vbs_op_kind_t kind,
                                                  vbs_ggml_type_t type) {
    vbs_hip_prim_pipeline_t *p =
        (vbs_hip_prim_pipeline_t *)malloc(sizeof(*p));
    if (!p) return NULL;
    p->base.native = p;
    p->base.kind = kind;
    p->type = type;
    return p;
}

static void *hip_compile_embed_pipeline(vbs_device_t *dev,
                                        vbs_ggml_type_t type,
                                        char *err_buf, size_t err_cap) {
    (void)dev;
    if (!vbs_host_type_supported(type)) {
        snprintf(err_buf, err_cap,
                 "HIP backend: no embedding kernel for weight type %d", (int)type);
        return NULL;
    }
    return hip_prim_pipeline(VBS_OP_EMBEDDING, type);
}

static void *hip_compile_primitive_pipeline(vbs_device_t *dev,
                                            vbs_op_kind_t kind,
                                            char *err_buf, size_t err_cap) {
    (void)dev;
    switch (kind) {
        case VBS_OP_EMBEDDING: /* F32 table, like the CUDA path */
        case VBS_OP_NORMALIZE:
        case VBS_OP_ROTARY:
        case VBS_OP_CACHE:
        case VBS_OP_ATTENTION:
        case VBS_OP_ACTIVATION:
        case VBS_OP_ELEMENTWISE:
            return hip_prim_pipeline(kind, VBS_GGML_TYPE_F32);
        default:
            snprintf(err_buf, err_cap,
                     "HIP backend: no primitive kernel for op kind %d", (int)kind);
            return NULL;
    }
}

/* ------------------------------------------------------------------ */
/* Dispatch (host-staged: device -> host math -> device)               */
/* ------------------------------------------------------------------ */

static bool hip_dispatch_linear_tile(
    vbs_device_t *dev, void *pipeline,
    vbs_buffer_t *wp, vbs_buffer_t *ws, size_t copied_bytes,
    vbs_buffer_t *in, vbs_buffer_t *out,
    uint32_t cols, uint32_t rows, uint32_t col_off, uint32_t row_off,
    bool accumulate) {
    (void)dev;
    vbs_hip_linear_pipeline_t *p = (vbs_hip_linear_pipeline_t *)pipeline;
    if (!p || !wp || !in || !out) return false;
    vbs_block_geometry_t g;
    if (!vbs_get_geometry(p->type, &g)) return false;
    if (cols % g.elements_per_block != 0) return false;
    uint64_t row_bytes = (uint64_t)(cols / g.elements_per_block) * g.bytes_per_block;

    /* Preserve the old stub's side effect: stage -> private copy. */
    if (ws && ws->host_ptr && copied_bytes > 0)
        g_hip.hipMemcpyHtoD((hipDeviceptr_t)wp->internal_handle,
                            ws->host_ptr, copied_bytes);

    /* Packed weights: staging tile (already host) or resident tensor. */
    const uint8_t *packed = NULL;
    uint8_t *packed_tmp = NULL;
    if (ws && ws->host_ptr) {
        packed = (const uint8_t *)ws->host_ptr;
    } else {
        packed_tmp = (uint8_t *)dev_to_host_bytes(wp->internal_handle, 0,
                                                  (size_t)(row_bytes * rows));
        if (!packed_tmp) return false;
        packed = packed_tmp;
    }
    float *x = (float *)dev_to_host_bytes(in->internal_handle,
                                          (size_t)col_off * sizeof(float),
                                          (size_t)cols * sizeof(float));
    float *y = (float *)dev_to_host_bytes(out->internal_handle,
                                          (size_t)row_off * sizeof(float),
                                          (size_t)rows * sizeof(float));
    if (!x || !y) { free(packed_tmp); free(x); free(y); return false; }
    /* Slices are already offset, so col/row offsets are 0 here. */
    bool ok = vbs_host_linear_tile(packed, p->type, x, y,
                                   cols, rows, 0, 0, accumulate);
    if (ok)
        ok = host_to_dev_bytes(out->internal_handle,
                               (size_t)row_off * sizeof(float),
                               y, (size_t)rows * sizeof(float));
    free(packed_tmp);
    free(x);
    free(y);
    return ok;
}

static bool hip_dispatch_primitive(vbs_device_t *dev, void *pipeline,
                                   const vbs_arg_t *args, size_t n_args,
                                   size_t grid_x, size_t block_x) {
    (void)dev; (void)grid_x; (void)block_x;
    vbs_hip_prim_pipeline_t *pp = (vbs_hip_prim_pipeline_t *)pipeline;
    if (!pp || !args) return false;

    /* Canonical bridge arg order (vbs_specializer_bridge.h). */
    switch (pp->base.kind) {
        case VBS_OP_EMBEDDING: {
            if (n_args != 4) return false;
            uint32_t token = *(const uint32_t *)args[1].value;
            uint32_t dim = *(const uint32_t *)args[3].value;
            vbs_block_geometry_t g;
            if (!vbs_get_geometry(pp->type, &g)) return false;
            uint64_t row_bytes = (uint64_t)(dim / g.elements_per_block) * g.bytes_per_block;
            uint8_t *row = (uint8_t *)dev_to_host_bytes((void *)args[0].value,
                                                        (size_t)((uint64_t)token * row_bytes),
                                                        (size_t)row_bytes);
            float *o = (float *)malloc((size_t)dim * sizeof(float));
            if (!row || !o) { free(row); free(o); return false; }
            bool ok = vbs_host_dequant_row(row, pp->type, dim, o);
            if (ok) ok = host_to_dev_bytes((void *)args[2].value, 0, o,
                                           (size_t)dim * sizeof(float));
            free(row);
            free(o);
            return ok;
        }
        case VBS_OP_NORMALIZE: {
            if (n_args != 5) return false;
            uint32_t dim = *(const uint32_t *)args[3].value;
            float eps = *(const float *)args[4].value;
            float *in = (float *)dev_to_host_bytes((void *)args[0].value, 0,
                                                   (size_t)dim * sizeof(float));
            float *w = (float *)dev_to_host_bytes((void *)args[1].value, 0,
                                                  (size_t)dim * sizeof(float));
            float *out = (float *)malloc((size_t)dim * sizeof(float));
            if (!in || !w || !out) { free(in); free(w); free(out); return false; }
            vbs_host_rms_norm(in, w, out, dim, eps);
            bool ok = host_to_dev_bytes((void *)args[2].value, 0, out,
                                        (size_t)dim * sizeof(float));
            free(in); free(w); free(out);
            return ok;
        }
        case VBS_OP_ROTARY: {
            if (n_args != 5) return false;
            uint32_t pos = *(const uint32_t *)args[1].value;
            uint32_t hd = *(const uint32_t *)args[2].value;
            uint32_t nh = *(const uint32_t *)args[3].value;
            float theta = *(const float *)args[4].value;
            float *x = (float *)dev_to_host_bytes((void *)args[0].value, 0,
                                                  (size_t)hd * nh * sizeof(float));
            if (!x) return false;
            vbs_host_rope(x, pos, hd, nh, theta);
            bool ok = host_to_dev_bytes((void *)args[0].value, 0, x,
                                        (size_t)hd * nh * sizeof(float));
            free(x);
            return ok;
        }
        case VBS_OP_CACHE: {
            if (n_args != 8) return false;
            uint32_t pos = *(const uint32_t *)args[4].value;
            uint32_t layer = *(const uint32_t *)args[5].value;
            uint32_t max_ctx = *(const uint32_t *)args[6].value;
            uint32_t kv_dim = *(const uint32_t *)args[7].value;
            float *k = (float *)dev_to_host_bytes((void *)args[0].value, 0,
                                                  (size_t)kv_dim * sizeof(float));
            float *v = (float *)dev_to_host_bytes((void *)args[1].value, 0,
                                                  (size_t)kv_dim * sizeof(float));
            uint16_t *kc = (uint16_t *)malloc((size_t)kv_dim * sizeof(uint16_t));
            uint16_t *vc = (uint16_t *)malloc((size_t)kv_dim * sizeof(uint16_t));
            if (!k || !v || !kc || !vc) {
                free(k); free(v); free(kc); free(vc);
                return false;
            }
            /* Host arrays are slice-local (base 0); the device offset
             * is applied on the copy back. */
            vbs_host_cache_store(k, v, kc, vc, 0, 0, 1, kv_dim);
            size_t byte_off = (size_t)(((uint64_t)layer * max_ctx + pos) * kv_dim * 2);
            bool ok = host_to_dev_bytes((void *)args[2].value, byte_off, kc,
                                        (size_t)kv_dim * sizeof(uint16_t));
            if (ok) ok = host_to_dev_bytes((void *)args[3].value, byte_off, vc,
                                           (size_t)kv_dim * sizeof(uint16_t));
            free(k); free(v); free(kc); free(vc);
            return ok;
        }
        case VBS_OP_ATTENTION: {
            if (n_args != 11) return false;
            uint32_t seq_len = *(const uint32_t *)args[4].value;
            uint32_t layer = *(const uint32_t *)args[5].value;
            uint32_t max_ctx = *(const uint32_t *)args[6].value;
            uint32_t nqh = *(const uint32_t *)args[7].value;
            uint32_t nkh = *(const uint32_t *)args[8].value;
            uint32_t hd = *(const uint32_t *)args[9].value;
            float scale = *(const float *)args[10].value;
            if (seq_len == 0 || nqh == 0 || nkh == 0 || hd == 0) return false;
            float *q = (float *)dev_to_host_bytes((void *)args[0].value, 0,
                                                  (size_t)nqh * hd * sizeof(float));
            size_t slice_elems = (size_t)seq_len * nkh * hd;
            size_t layer_off = (size_t)((uint64_t)layer * max_ctx * nkh * hd);
            uint16_t *kc = (uint16_t *)dev_to_host_bytes((void *)args[1].value,
                                                         layer_off * 2, slice_elems * 2);
            uint16_t *vc = (uint16_t *)dev_to_host_bytes((void *)args[2].value,
                                                         layer_off * 2, slice_elems * 2);
            float *o = (float *)malloc((size_t)nqh * hd * sizeof(float));
            if (!q || !kc || !vc || !o) {
                free(q); free(kc); free(vc); free(o);
                return false;
            }
            /* Slices are already layer-offset: pass layer 0. */
            vbs_host_attention(q, kc, vc, o, seq_len, 0, seq_len,
                               nqh, nkh, hd, scale);
            bool ok = host_to_dev_bytes((void *)args[3].value, 0, o,
                                        (size_t)nqh * hd * sizeof(float));
            free(q); free(kc); free(vc); free(o);
            return ok;
        }
        case VBS_OP_ACTIVATION: { /* silu_mul: gate, up, out, n */
            if (n_args != 4) return false;
            uint32_t n = *(const uint32_t *)args[3].value;
            float *gate = (float *)dev_to_host_bytes((void *)args[0].value, 0,
                                                     (size_t)n * sizeof(float));
            float *up = (float *)dev_to_host_bytes((void *)args[1].value, 0,
                                                   (size_t)n * sizeof(float));
            float *out = (float *)malloc((size_t)n * sizeof(float));
            if (!gate || !up || !out) { free(gate); free(up); free(out); return false; }
            vbs_host_silu_mul(gate, up, out, n);
            bool ok = host_to_dev_bytes((void *)args[2].value, 0, out,
                                        (size_t)n * sizeof(float));
            free(gate); free(up); free(out);
            return ok;
        }
        case VBS_OP_ELEMENTWISE: { /* add: a, b, out, n */
            if (n_args != 4) return false;
            uint32_t n = *(const uint32_t *)args[3].value;
            float *a = (float *)dev_to_host_bytes((void *)args[0].value, 0,
                                                  (size_t)n * sizeof(float));
            float *b = (float *)dev_to_host_bytes((void *)args[1].value, 0,
                                                  (size_t)n * sizeof(float));
            float *out = (float *)malloc((size_t)n * sizeof(float));
            if (!a || !b || !out) { free(a); free(b); free(out); return false; }
            vbs_host_add(a, b, out, n);
            bool ok = host_to_dev_bytes((void *)args[2].value, 0, out,
                                        (size_t)n * sizeof(float));
            free(a); free(b); free(out);
            return ok;
        }
        default:
            return false;
    }
}

static void hip_synchronize(vbs_device_t *dev) {
    (void)dev;
    g_hip.hipCtxSynchronize();
}

const vbs_backend_ops_t g_vbs_hip_backend = {
    .init = hip_init,
    .get_device_count = hip_get_device_count,
    .get_device = hip_get_device,
    .alloc_buffer = hip_alloc_buffer,
    .free_buffer = hip_free_buffer,
    .copy_to_device = hip_copy_to_device,
    .copy_to_host = hip_copy_to_host,
    .compile_linear_pipeline = hip_compile_linear_pipeline,
    .compile_embed_pipeline = hip_compile_embed_pipeline,
    .compile_primitive_pipeline = hip_compile_primitive_pipeline,
    .dispatch_linear_tile = hip_dispatch_linear_tile,
    .dispatch_primitive = hip_dispatch_primitive,
    .synchronize_device = hip_synchronize,
    .launch_persistent_kernel = NULL,
};
