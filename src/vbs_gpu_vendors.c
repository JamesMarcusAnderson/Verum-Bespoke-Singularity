/* =============================================================================
 * vbs_gpu_vendors.c
 * Verum-Bespoke-Singularity — vendor GPU driver adapters.
 *
 * One file, three drivers: CUDA, HIP, Metal. Each section loads its
 * driver library at runtime, probes for devices, and exposes one const
 * vbs_backend_ops_t table (g_vbs_cuda_backend, g_vbs_hip_backend,
 * g_vbs_metal_backend). The GPU path lives in vbs_backend_gpu.c and
 * the CPU path in vbs_backend_cpu.c; selection happens in
 * vbs_backend_select.c.
 *
 * The Metal section compiles only on Darwin (VBS_ENABLE_METAL). The
 * CUDA and HIP sections are always compiled and fail closed at runtime
 * when their driver library is absent.
 *
 * Kept external interface names: the driver entry-point names
 * (cuInit, hipInit, MTLCreateSystemDefaultDevice, ...), the loader
 * library filenames (libcuda.so.1, libamdhip64.so, nvcuda.dll,
 * amdhip64.dll), and the driver type names (CUresult, hipError_t,
 * MTLSize) are the drivers' own vocabularies — the code must speak
 * them to reach the drivers. The backend table names
 * (g_vbs_*_backend) and the error accessors (vbs_cuda_last_error,
 * vbs_hip_last_error) are link-visible and used by
 * vbs_backend_select.c and vbs_backend.h; they stay per James's
 * decision (sweep items 9-10).
 * ========================================================================== */

#include "vbs_backend.h"
#include "vbs_host_math.h"
#include "vbs_port.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Shared kernel-name registry.
 *
 * CUDA and Metal ship kernels under the same names: matvec_<type>
 * for linear, embedding_lookup[_<type>] for embedding, and one
 * name per primitive op. The tables below are the union both
 * backends agree on; each backend keeps its own fallback policy
 * for types or kinds missing from a table (Metal falls back to
 * the F32 kernel names and to rms_norm). */
typedef struct {
    vbs_model_type_t type;
    const char *kernel;
} vbs_type_kernel_t;

typedef struct {
    vbs_op_kind_t kind;
    const char *kernel;
} vbs_kind_kernel_t;

static const vbs_type_kernel_t k_vbs_linear_kernels[] = {
    { VBS_MODEL_TYPE_F32,  "matvec_f32"  },
    { VBS_MODEL_TYPE_F16,  "matvec_f16"  },
    { VBS_MODEL_TYPE_Q8_0, "matvec_q8_0" },
    { VBS_MODEL_TYPE_Q4_0, "matvec_q4_0" },
    { VBS_MODEL_TYPE_Q4_1, "matvec_q4_1" },
    { VBS_MODEL_TYPE_Q6_K, "matvec_q6_k" },
    { VBS_MODEL_TYPE_LNS8, "matvec_lns8" },
};

static const vbs_type_kernel_t k_vbs_embed_kernels[] = {
    { VBS_MODEL_TYPE_F32,  "embedding_lookup"      },
    { VBS_MODEL_TYPE_F16,  "embedding_lookup_f16"  },
    { VBS_MODEL_TYPE_Q8_0, "embedding_lookup_q8_0" },
    { VBS_MODEL_TYPE_Q4_0, "embedding_lookup_q4_0" },
    { VBS_MODEL_TYPE_Q4_1, "embedding_lookup_q4_1" },
    { VBS_MODEL_TYPE_Q6_K, "embedding_lookup_q6_k" },
};

static const vbs_kind_kernel_t k_vbs_prim_kernels[] = {
    { VBS_OP_EMBEDDING,   "embedding_lookup" },
    { VBS_OP_NORMALIZE,   "rms_norm"         },
    { VBS_OP_ROTARY,      "rope"             },
    { VBS_OP_CACHE,       "kv_cache_store"   },
    { VBS_OP_ATTENTION,   "attention_gqa"    },
    { VBS_OP_ACTIVATION,  "silu_mul"         },
    { VBS_OP_ELEMENTWISE, "residual_add"     },
};

static const char *vbs_kernel_for_type(const vbs_type_kernel_t *table, size_t count,
                                       vbs_model_type_t type) {
    for (size_t i = 0; i < count; i++) {
        if (table[i].type == type) return table[i].kernel;
    }
    return NULL;
}

static const char *vbs_kernel_for_kind(const vbs_kind_kernel_t *table, size_t count,
                                       vbs_op_kind_t kind) {
    for (size_t i = 0; i < count; i++) {
        if (table[i].kind == kind) return table[i].kernel;
    }
    return NULL;
}

/* Validate a primitive pipeline handle: non-NULL with a live
 * native object (CUfunction / pipeline state). Returns the
 * pipeline, or NULL. Shared by the CUDA and Metal dispatchers. */
static vbs_prim_pipeline_t *vbs_prim_pipeline_checked(void *pipeline) {
    vbs_prim_pipeline_t *pp = (vbs_prim_pipeline_t *)pipeline;
    if (!pp || !pp->native) return NULL;
    return pp;
}

/* =============================================================================
 * Section 1 — CUDA driver adapter
 * Loads libcuda at runtime; linear/embed/primitive pipelines run the
 * precompiled PTX in kernels/vendor_kernels.ptx.h.
 * ========================================================================== */


typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUdeviceptr;
typedef void *CUstream;

#define CUDA_SUCCESS 0

typedef CUresult (*cuInit_fn)(unsigned int);
typedef CUresult (*cuDeviceGet_fn)(CUdevice *, int);
typedef CUresult (*cuDeviceGetCount_fn)(int *);
typedef CUresult (*cuDeviceGetName_fn)(char *, int, CUdevice);
typedef CUresult (*cuCtxCreate_fn)(CUcontext *, unsigned int, CUdevice);
typedef CUresult (*cuModuleLoadData_fn)(CUmodule *, const void *);
typedef CUresult (*cuModuleGetFunction_fn)(CUfunction *, CUmodule, const char *);
typedef CUresult (*cuMemAlloc_fn)(CUdeviceptr *, size_t);
typedef CUresult (*cuMemFree_fn)(CUdeviceptr);
typedef CUresult (*cuMemcpyHtoD_fn)(CUdeviceptr, const void *, size_t);
typedef CUresult (*cuMemcpyDtoH_fn)(void *, CUdeviceptr, size_t);
typedef CUresult (*cuLaunchKernel_fn)(CUfunction, unsigned int, unsigned int, unsigned int,
                                       unsigned int, unsigned int, unsigned int,
                                       unsigned int, CUstream, void **, void **);
typedef CUresult (*cuCtxSynchronize_fn)(void);

static struct {
    void *handle;
    cuInit_fn cuInit;
    cuDeviceGet_fn cuDeviceGet;
    cuDeviceGetCount_fn cuDeviceGetCount;
    cuDeviceGetName_fn cuDeviceGetName;
    cuCtxCreate_fn cuCtxCreate;
    cuModuleLoadData_fn cuModuleLoadData;
    cuModuleGetFunction_fn cuModuleGetFunction;
    cuMemAlloc_fn cuMemAlloc;
    cuMemFree_fn cuMemFree;
    cuMemcpyHtoD_fn cuMemcpyHtoD;
    cuMemcpyDtoH_fn cuMemcpyDtoH;
    cuLaunchKernel_fn cuLaunchKernel;
    cuCtxSynchronize_fn cuCtxSynchronize;
} g_cu;

typedef struct {
    CUdevice dev;
    CUcontext ctx;
    CUmodule mod;
} vbs_cuda_internal_t;

static vbs_device_t g_cuda_devices[8];
static vbs_cuda_internal_t g_cuda_internals[8];
static int g_cuda_dev_count = 0;
static bool g_cuda_probed = false;
static bool g_cuda_ready = false;
static char g_cuda_last_error[256] = "not probed yet";

const char *vbs_cuda_last_error(void) { return g_cuda_last_error; }

#include "../kernels/vendor_kernels.ptx.h"

/* Resolve one driver entry point, recording which one is missing. */
static bool cuda_load_fn(const char *name, void **slot) {
    *slot = vbs_dlsym(g_cu.handle, name);
    if (!*slot) {
        snprintf(g_cuda_last_error, sizeof(g_cuda_last_error),
                 "CUDA driver lacks %s", name);
        return false;
    }
    return true;
}

static bool cuda_init(void) {
    if (g_cuda_probed) return g_cuda_ready;
    g_cuda_probed = true;
    static const char *k_loader_names[] = { "libcuda.so.1", "libcuda.so", "nvcuda.dll", NULL };
    for (int i = 0; k_loader_names[i]; i++) {
        g_cu.handle = vbs_dlopen(k_loader_names[i], 0);
        if (g_cu.handle) break;
    }
    if (!g_cu.handle) {
        snprintf(g_cuda_last_error, sizeof(g_cuda_last_error),
                 "CUDA driver not found (libcuda.so.1 missing)");
        return false;
    }

    if (!cuda_load_fn("cuInit", (void **)&g_cu.cuInit) ||
        !cuda_load_fn("cuDeviceGet", (void **)&g_cu.cuDeviceGet) ||
        !cuda_load_fn("cuDeviceGetCount", (void **)&g_cu.cuDeviceGetCount) ||
        !cuda_load_fn("cuDeviceGetName", (void **)&g_cu.cuDeviceGetName) ||
        !cuda_load_fn("cuCtxCreate", (void **)&g_cu.cuCtxCreate) ||
        !cuda_load_fn("cuModuleLoadData", (void **)&g_cu.cuModuleLoadData) ||
        !cuda_load_fn("cuModuleGetFunction", (void **)&g_cu.cuModuleGetFunction) ||
        !cuda_load_fn("cuMemAlloc", (void **)&g_cu.cuMemAlloc) ||
        !cuda_load_fn("cuMemFree", (void **)&g_cu.cuMemFree) ||
        !cuda_load_fn("cuMemcpyHtoD", (void **)&g_cu.cuMemcpyHtoD) ||
        !cuda_load_fn("cuMemcpyDtoH", (void **)&g_cu.cuMemcpyDtoH) ||
        !cuda_load_fn("cuLaunchKernel", (void **)&g_cu.cuLaunchKernel) ||
        !cuda_load_fn("cuCtxSynchronize", (void **)&g_cu.cuCtxSynchronize)) {
        return false;
    }

    if (g_cu.cuInit(0) != CUDA_SUCCESS) {
        snprintf(g_cuda_last_error, sizeof(g_cuda_last_error), "cuInit failed");
        return false;
    }

    int count = 0;
    g_cu.cuDeviceGetCount(&count);
    if (count <= 0) {
        snprintf(g_cuda_last_error, sizeof(g_cuda_last_error),
                 "no NVIDIA GPU found by CUDA driver");
        return false;
    }
    if (count > 8) count = 8;

    for (int i = 0; i < count; i++) {
        CUdevice dev;
        if (g_cu.cuDeviceGet(&dev, i) != CUDA_SUCCESS) continue;

        char name[128] = {0};
        g_cu.cuDeviceGetName(name, sizeof(name) - 1, dev);
        snprintf(g_cuda_devices[g_cuda_dev_count].name, 128, "NVIDIA %s", name);
        g_cuda_devices[g_cuda_dev_count].type = VBS_BACKEND_CUDA;
        g_cuda_devices[g_cuda_dev_count].internal_handle = &g_cuda_internals[g_cuda_dev_count];

        vbs_cuda_internal_t *ci = &g_cuda_internals[g_cuda_dev_count];
        ci->dev = dev;
        if (g_cu.cuCtxCreate(&ci->ctx, 0, dev) != CUDA_SUCCESS) continue;
        if (g_cu.cuModuleLoadData(&ci->mod, g_vendor_kernels_ptx) != CUDA_SUCCESS) continue;

        g_cuda_dev_count++;
    }
    g_cuda_ready = g_cuda_dev_count > 0;
    if (g_cuda_ready) {
        snprintf(g_cuda_last_error, sizeof(g_cuda_last_error), "ok");
    } else {
        snprintf(g_cuda_last_error, sizeof(g_cuda_last_error),
                 "no CUDA device initialized (context or PTX module failed)");
    }
    return g_cuda_ready;
}

static size_t cuda_get_device_count(void) { return (size_t)g_cuda_dev_count; }
static vbs_device_t *cuda_get_device(size_t i) { return &g_cuda_devices[i]; }

static vbs_buffer_t *cuda_alloc_buffer(vbs_device_t *dev, size_t bytes, bool shared) {
    (void)shared;
    CUdeviceptr dptr = 0;
    if (g_cu.cuMemAlloc(&dptr, bytes) != CUDA_SUCCESS) return NULL;
    vbs_buffer_t *b = (vbs_buffer_t *)calloc(1, sizeof(vbs_buffer_t));
    if (!b) {
        g_cu.cuMemFree(dptr);
        return NULL;
    }
    b->internal_handle = (void *)(uintptr_t)dptr;
    b->size = bytes;
    b->device = dev;
    b->host_ptr = calloc(1, bytes);
    return b;
}

static void cuda_free_buffer(vbs_buffer_t *b) {
    if (!b) return;
    g_cu.cuMemFree((CUdeviceptr)(uintptr_t)b->internal_handle);
    free(b->host_ptr);
    free(b);
}

static bool cuda_copy_to_device(vbs_buffer_t *dst, const void *src, size_t bytes) {
    return g_cu.cuMemcpyHtoD((CUdeviceptr)(uintptr_t)dst->internal_handle, src, bytes) == CUDA_SUCCESS;
}

static bool cuda_copy_to_host(void *dst, const vbs_buffer_t *src, size_t bytes) {
    return g_cu.cuMemcpyDtoH(dst, (CUdeviceptr)(uintptr_t)src->internal_handle, bytes) == CUDA_SUCCESS;
}

/* Kernel names come from the shared registry at the top of this
 * file (vbs_kernel_for_type / vbs_kernel_for_kind). */

static CUfunction cuda_get_function(vbs_device_t *dev, const char *kname,
                                    char *err_buf, size_t err_cap, const char *fmt) {
    vbs_cuda_internal_t *ci = (vbs_cuda_internal_t *)dev->internal_handle;
    CUfunction fn = NULL;
    if (g_cu.cuModuleGetFunction(&fn, ci->mod, kname) != CUDA_SUCCESS) {
        snprintf(err_buf, err_cap, fmt, kname);
        return NULL;
    }
    return fn;
}

static vbs_prim_pipeline_t *cuda_wrap_prim(CUfunction fn, vbs_op_kind_t kind) {
    vbs_prim_pipeline_t *pp = (vbs_prim_pipeline_t *)malloc(sizeof(vbs_prim_pipeline_t));
    if (!pp) return NULL;
    pp->native = (void *)fn;
    pp->kind = kind;
    return pp;
}

static void *cuda_compile_linear_pipeline(vbs_device_t *dev, vbs_model_type_t type, char *err_buf, size_t err_cap) {
    const char *kname = vbs_kernel_for_type(k_vbs_linear_kernels,
                                            sizeof(k_vbs_linear_kernels) / sizeof(k_vbs_linear_kernels[0]),
                                            type);
    if (!kname) {
        snprintf(err_buf, err_cap, "CUDA linear pipeline refusal: no kernel for weight type %d", (int)type);
        return NULL;
    }
    return (void *)cuda_get_function(dev, kname, err_buf, err_cap, "CUDA kernel %s not found in PTX");
}

static void *cuda_compile_embed_pipeline(vbs_device_t *dev, vbs_model_type_t type, char *err_buf, size_t err_cap) {
    const char *kname = vbs_kernel_for_type(k_vbs_embed_kernels,
                                            sizeof(k_vbs_embed_kernels) / sizeof(k_vbs_embed_kernels[0]),
                                            type);
    if (!kname) {
        snprintf(err_buf, err_cap, "CUDA embedding pipeline refusal: no kernel for type %d", (int)type);
        return NULL;
    }
    CUfunction fn = cuda_get_function(dev, kname, err_buf, err_cap, "CUDA embedding kernel %s not found");
    if (!fn) return NULL;
    return cuda_wrap_prim(fn, VBS_OP_EMBEDDING);
}

static void *cuda_compile_primitive_pipeline(vbs_device_t *dev, vbs_op_kind_t kind, char *err_buf, size_t err_cap) {
    const char *kname = vbs_kernel_for_kind(k_vbs_prim_kernels,
                                            sizeof(k_vbs_prim_kernels) / sizeof(k_vbs_prim_kernels[0]),
                                            kind);
    if (!kname) {
        snprintf(err_buf, err_cap, "CUDA has no primitive kernel for op kind %d", (int)kind);
        return NULL;
    }
    CUfunction fn = cuda_get_function(dev, kname, err_buf, err_cap, "CUDA primitive %s not found");
    if (!fn) return NULL;
    return cuda_wrap_prim(fn, kind);
}

static bool cuda_dispatch_linear_tile(
    vbs_device_t *dev, void *pipeline,
    vbs_buffer_t *wp, vbs_buffer_t *ws, size_t copied_bytes,
    vbs_buffer_t *in, vbs_buffer_t *out,
    uint32_t cols, uint32_t rows, uint32_t col_off, uint32_t row_off,
    bool accumulate
) {
    (void)dev; (void)ws; (void)copied_bytes; (void)col_off; (void)row_off; (void)accumulate;
    void *args[] = { &wp->internal_handle, &in->internal_handle, &out->internal_handle, &cols, &rows };
    unsigned int grid = (rows + 31) / 32;
    return g_cu.cuLaunchKernel((CUfunction)pipeline, grid, 1, 1, 32, 1, 1, 0, NULL, args, NULL) == CUDA_SUCCESS;
}

static bool cuda_dispatch_primitive(
    vbs_device_t *dev, void *pipeline,
    const vbs_arg_t *args, size_t n_args,
    size_t grid_x, size_t block_x
) {
    (void)dev;
    vbs_prim_pipeline_t *pp = vbs_prim_pipeline_checked(pipeline);
    if (!pp) return false;

    uint64_t bufvals[16];
    void *params[16];

    if (pp->kind == VBS_OP_ATTENTION) {
        /* Canonical order (bridge): q, kc, vc, out, seq_len, layer,
         * max_ctx, n_q_heads, n_kv_heads, head_dim, scale. PTX
         * attention_gqa takes pre-offset cache pointers and no
         * layer/max_ctx: fold the layer slice offset into the
         * pointers. */
        if (n_args != 11) return false;
        uint32_t layer   = *(const uint32_t *)args[5].value;
        uint32_t max_ctx = *(const uint32_t *)args[6].value;
        uint32_t nkh     = *(const uint32_t *)args[8].value;
        uint32_t hd      = *(const uint32_t *)args[9].value;
        uint64_t off = (uint64_t)layer * max_ctx * nkh * hd * 2; /* f16 elements -> bytes */
        bufvals[0] = (uint64_t)(uintptr_t)args[0].value;
        bufvals[1] = (uint64_t)(uintptr_t)args[1].value + off;
        bufvals[2] = (uint64_t)(uintptr_t)args[2].value + off;
        bufvals[3] = (uint64_t)(uintptr_t)args[3].value;
        params[0] = &bufvals[0]; params[1] = &bufvals[1];
        params[2] = &bufvals[2]; params[3] = &bufvals[3];
        params[4] = (void *)args[4].value;  /* seq_len */
        params[5] = (void *)args[7].value;  /* n_q_heads */
        params[6] = (void *)args[8].value;  /* n_kv_heads */
        params[7] = (void *)args[9].value;  /* head_dim */
        params[8] = (void *)args[10].value; /* scale */
    } else {
        if (n_args > 16) return false;
        for (size_t i = 0; i < n_args; i++) {
            if (args[i].tag == VBS_ARG_BUFFER) {
                bufvals[i] = (uint64_t)(uintptr_t)args[i].value;
                params[i] = &bufvals[i];
            } else {
                params[i] = (void *)args[i].value;
            }
        }
    }

    return g_cu.cuLaunchKernel((CUfunction)pp->native, (unsigned int)grid_x, 1, 1, (unsigned int)block_x, 1, 1, 0, NULL, params, NULL) == CUDA_SUCCESS;
}

static void cuda_synchronize(vbs_device_t *dev) {
    (void)dev;
    g_cu.cuCtxSynchronize();
}

const vbs_backend_ops_t g_vbs_cuda_backend = {
    .init = cuda_init,
    .get_device_count = cuda_get_device_count,
    .get_device = cuda_get_device,
    .alloc_buffer = cuda_alloc_buffer,
    .free_buffer = cuda_free_buffer,
    .copy_to_device = cuda_copy_to_device,
    .copy_to_host = cuda_copy_to_host,
    .compile_linear_pipeline = cuda_compile_linear_pipeline,
    .compile_embed_pipeline = cuda_compile_embed_pipeline,
    .compile_primitive_pipeline = cuda_compile_primitive_pipeline,
    .dispatch_linear_tile = cuda_dispatch_linear_tile,
    .dispatch_primitive = cuda_dispatch_primitive,
    .synchronize_device = cuda_synchronize
};

/* =============================================================================
 * Section 2 — HIP driver adapter (ROCm)
 *
 * Loads libamdhip64 at runtime. Primitives and linear tiles run
 * host-staged (device -> host -> compute -> device) through the shared
 * host math (vbs_host_math.h), so this backend cannot drift numerically
 * from the CPU backend. rope honors the bridge theta.
 *
 * Honest limits:
 *   - No ROCm hardware is present in the build VM, so the HIP calls
 *     themselves are verified by compile only; the MATH is verified
 *     through the CPU backend, which runs the same code.
 *   - vbs_hip_load_kernel() loads a HIP code object (HSACO), not raw
 *     SPIR-V: hipModuleLoadData does not accept SPIR-V. A real GPU
 *     kernel path for HIP needs the GCN emitter output assembled to
 *     HSACO offline — a future step, not claimed here.
 *   - Primitives run host-staged, so they are correct but not fast.
 *     They exist so a ROCm machine produces right answers everywhere,
 *     not fast answers in some ops and refusals in others.
 * ========================================================================== */


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
                                         unsigned int, hipStream_t,
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

/* Pipeline handles (vbs_host_linear_pipeline_t /
 * vbs_host_prim_pipeline_t) and the compile_* wrappers live in
 * vbs_host_math.h, shared with the CPU backend; this file only
 * delegates with its own label. */

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

/* Resolve a required driver entry point, recording which is missing. */
static bool hip_load_required(const char *name, void **slot) {
    *slot = vbs_dlsym(g_hip.handle, name);
    if (!*slot) {
        snprintf(g_hip_last_error, sizeof(g_hip_last_error),
                 "ROCm runtime lacks %s", name);
        return false;
    }
    return true;
}

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

    if (!hip_load_required("hipInit", (void **)&g_hip.hipInit) ||
        !hip_load_required("hipGetDeviceCount", (void **)&g_hip.hipGetDeviceCount) ||
        !hip_load_required("hipDeviceGetName", (void **)&g_hip.hipDeviceGetName) ||
        !hip_load_required("hipMalloc", (void **)&g_hip.hipMalloc) ||
        !hip_load_required("hipFree", (void **)&g_hip.hipFree) ||
        !hip_load_required("hipMemcpyHtoD", (void **)&g_hip.hipMemcpyHtoD) ||
        !hip_load_required("hipMemcpyDtoH", (void **)&g_hip.hipMemcpyDtoH) ||
        !hip_load_required("hipCtxSynchronize", (void **)&g_hip.hipCtxSynchronize)) {
        return false;
    }
    /* Optional: context/module/kernel entry points, used only when present. */
    g_hip.hipDeviceGet = (hipDeviceGet_fn)vbs_dlsym(g_hip.handle, "hipDeviceGet");
    g_hip.hipCtxCreate = (hipCtxCreate_fn)vbs_dlsym(g_hip.handle, "hipCtxCreate");
    g_hip.hipModuleLoadData = (hipModuleLoadData_fn)vbs_dlsym(g_hip.handle, "hipModuleLoadData");
    g_hip.hipModuleGetFunction = (hipModuleGetFunction_fn)vbs_dlsym(g_hip.handle, "hipModuleGetFunction");
    g_hip.hipLaunchKernel = (hipLaunchKernel_fn)vbs_dlsym(g_hip.handle, "hipLaunchKernel");

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
                                         vbs_model_type_t type,
                                         char *err_buf, size_t err_cap) {
    (void)dev;
    return vbs_host_compile_linear_pipeline("HIP backend", type, err_buf, err_cap);
}

static void *hip_compile_embed_pipeline(vbs_device_t *dev,
                                        vbs_model_type_t type,
                                        char *err_buf, size_t err_cap) {
    (void)dev;
    return vbs_host_compile_embed_pipeline("HIP backend", type, err_buf, err_cap);
}

static void *hip_compile_primitive_pipeline(vbs_device_t *dev,
                                            vbs_op_kind_t kind,
                                            char *err_buf, size_t err_cap) {
    (void)dev;
    return vbs_host_compile_primitive_pipeline("HIP backend", kind, err_buf, err_cap);
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
    /* The device handle is not dereferenced here (staging runs
     * through the loaded HIP entry points), but a dispatch with no
     * device is a caller bug: refuse it instead of proceeding. */
    if (!dev) return false;
    vbs_host_linear_pipeline_t *p = (vbs_host_linear_pipeline_t *)pipeline;
    if (!p || !wp || !in || !out) return false;
    vbs_block_geometry_t g;
    if (!vbs_get_geometry(p->type, &g)) return false;
    if (cols % g.elements_per_block != 0) return false;
    uint64_t row_bytes = (uint64_t)(cols / g.elements_per_block) * g.bytes_per_block;

    /* Preserve the earlier copy step: stage -> private copy. */
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

/* One host-staged primitive each; args follow the canonical bridge
 * order (vbs_specializer_bridge.h). Each stages its inputs to the
 * host, runs the shared host math, and stages the result back. */

static bool hip_prim_embed(const vbs_host_prim_pipeline_t *pp,
                           const vbs_arg_t *args, size_t n_args) {
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

static bool hip_prim_norm(const vbs_arg_t *args, size_t n_args) {
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

static bool hip_prim_rope(const vbs_arg_t *args, size_t n_args) {
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

static bool hip_prim_cache(const vbs_arg_t *args, size_t n_args) {
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
    /* Host arrays are slice-local (base 0); the device offset is
     * applied on the copy back. */
    vbs_host_cache_store(k, v, kc, vc, 0, 0, 1, kv_dim);
    size_t byte_off = (size_t)(((uint64_t)layer * max_ctx + pos) * kv_dim * 2);
    bool ok = host_to_dev_bytes((void *)args[2].value, byte_off, kc,
                                (size_t)kv_dim * sizeof(uint16_t));
    if (ok) ok = host_to_dev_bytes((void *)args[3].value, byte_off, vc,
                                   (size_t)kv_dim * sizeof(uint16_t));
    free(k); free(v); free(kc); free(vc);
    return ok;
}

static bool hip_prim_attention(const vbs_arg_t *args, size_t n_args) {
    if (n_args != 11) return false;
    vbs_attn_shape_t sh = vbs_attention_shape(args);
    if (sh.seq_len == 0 || sh.nqh == 0 || sh.nkh == 0 || sh.hd == 0) return false;
    float *q = (float *)dev_to_host_bytes((void *)args[0].value, 0,
                                          (size_t)sh.nqh * sh.hd * sizeof(float));
    size_t slice_elems = (size_t)sh.seq_len * sh.nkh * sh.hd;
    size_t layer_off = (size_t)((uint64_t)sh.layer * sh.max_ctx * sh.nkh * sh.hd);
    uint16_t *kc = (uint16_t *)dev_to_host_bytes((void *)args[1].value,
                                                 layer_off * 2, slice_elems * 2);
    uint16_t *vc = (uint16_t *)dev_to_host_bytes((void *)args[2].value,
                                                 layer_off * 2, slice_elems * 2);
    float *o = (float *)malloc((size_t)sh.nqh * sh.hd * sizeof(float));
    if (!q || !kc || !vc || !o) {
        free(q); free(kc); free(vc); free(o);
        return false;
    }
    /* Slices are already layer-offset: pass layer 0. */
    vbs_host_attention(q, kc, vc, o, sh.seq_len, 0, sh.seq_len,
                       sh.nqh, sh.nkh, sh.hd, sh.scale);
    bool ok = host_to_dev_bytes((void *)args[3].value, 0, o,
                                (size_t)sh.nqh * sh.hd * sizeof(float));
    free(q); free(kc); free(vc); free(o);
    return ok;
}

static bool hip_prim_silu(const vbs_arg_t *args, size_t n_args) {
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

static bool hip_prim_add(const vbs_arg_t *args, size_t n_args) {
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

static bool hip_dispatch_primitive(vbs_device_t *dev, void *pipeline,
                                   const vbs_arg_t *args, size_t n_args,
                                   size_t grid_x, size_t block_x) {
    (void)dev; (void)grid_x; (void)block_x;
    vbs_host_prim_pipeline_t *pp = (vbs_host_prim_pipeline_t *)pipeline;
    if (!pp || !args) return false;
    switch (pp->base.kind) {
        case VBS_OP_EMBEDDING:  return hip_prim_embed(pp, args, n_args);
        case VBS_OP_NORMALIZE:  return hip_prim_norm(args, n_args);
        case VBS_OP_ROTARY:     return hip_prim_rope(args, n_args);
        case VBS_OP_CACHE:      return hip_prim_cache(args, n_args);
        case VBS_OP_ATTENTION:  return hip_prim_attention(args, n_args);
        case VBS_OP_ACTIVATION: return hip_prim_silu(args, n_args);
        case VBS_OP_ELEMENTWISE: return hip_prim_add(args, n_args);
        default: return false;
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

/* =============================================================================
 * Section 3 — Metal driver adapter (Darwin only)
 * Loads the Metal framework at runtime via the Objective-C runtime;
 * a single command stream stays open across a whole forward pass.
 * ========================================================================== */

#if defined(VBS_ENABLE_METAL) && VBS_ENABLE_METAL

#include <dlfcn.h>
#include <objc/runtime.h>
#include <objc/message.h>

#ifndef NSUInteger
typedef unsigned long NSUInteger;
#endif

#ifndef nil
#define nil ((id)0)
#endif

typedef struct {
    NSUInteger width;
    NSUInteger height;
    NSUInteger depth;
} MTLSize;

static vbs_device_t g_metal_device;
static id g_mtl_device_handle = nil;
static id g_mtl_queue_handle = nil;
static id g_mtl_library = nil;

/* Single command-stream state (James): one command buffer + one
 * compute encoder stay open across a whole forward pass; dispatches
 * encode into it with no CPU stall, and the stream is committed +
 * waited exactly once -- at synchronize_device or before any host
 * copy. This replaces the old per-dispatch command buffer +
 * waitUntilCompleted (thousands of synchronous round-trips per
 * token). */
static id g_active_cmd_buffer = nil;
static id g_active_encoder = nil;

static id metal_active_encoder(void) {
    if (g_active_encoder == nil) {
        g_active_cmd_buffer = ((id (*)(id, SEL))objc_msgSend)(g_mtl_queue_handle, sel_registerName("commandBuffer"));
        g_active_encoder = ((id (*)(id, SEL))objc_msgSend)(g_active_cmd_buffer, sel_registerName("computeCommandEncoder"));
    }
    return g_active_encoder;
}

static void metal_flush_stream(void) {
    if (g_active_encoder != nil) {
        ((void (*)(id, SEL))objc_msgSend)(g_active_encoder, sel_registerName("endEncoding"));
        ((void (*)(id, SEL))objc_msgSend)(g_active_cmd_buffer, sel_registerName("commit"));
        ((void (*)(id, SEL))objc_msgSend)(g_active_cmd_buffer, sel_registerName("waitUntilCompleted"));
        g_active_encoder = nil;
        g_active_cmd_buffer = nil;
    }
}

/* MSL kernels. One thread per output row (matvec) or element
 * (primitives). Buffer indices and kernel names are the interface
 * the C dispatch code calls by name; the math mirrors the GLSL
 * kernels and the host math. */
static const char *g_metal_shader_source =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct q80 { half d; char q[32]; };\n"
"struct q40 { half d; uchar q[16]; };\n"
"struct q41 { half d; half m; uchar q[16]; };\n"
"struct q6k { uchar ql[128]; uchar qh[64]; char s[16]; half d; };\n"
/* One Q6_K element, fully dequantized: the 4-bit low part plus
 * the 2-bit high part make a 6-bit value centered at 32, times
 * the per-sub-block scale and the block scale d. The matvec and
 * the embedding lookup both call it. d is a parameter so the
 * multiply order matches the inline form exactly. */
"float q6k_value(device const q6k &blk, uint n, float d) {\n"
"    uint h = n / 128, q = (n % 128) / 32, l = n % 32;\n"
"    uint qi = h * 64 + (q & 1) * 32 + l;\n"
"    uint qs = (q < 2) ? 0 : 4;\n"
"    uint lo = (uint(blk.ql[qi]) >> qs) & 15;\n"
"    uint hi = (uint(blk.qh[h * 32 + l]) >> (q * 2)) & 3;\n"
"    float sc = float(blk.s[h * 8 + q * 2 + l / 16]);\n"
"    return d * sc * float(int(lo | (hi << 4)) - 32);\n"
"}\n"
"kernel void matvec_q8_0(device const q80 *w [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint r [[thread_position_in_grid]]) {\n"
"    if (r >= rows) return;\n"
"    uint nb = cols / 32;\n"
"    float acc = 0.0f;\n"
"    for (uint b = 0; b < nb; b++) {\n"
"        device const q80 &blk = w[r * nb + b];\n"
"        float d = float(blk.d);\n"
"        for (uint i = 0; i < 32; i++) acc += d * float(blk.q[i]) * x[b * 32 + i];\n"
"    }\n"
"    y[r] = acc;\n"
"}\n"
"kernel void matvec_q4_0(device const q40 *w [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint r [[thread_position_in_grid]]) {\n"
"    if (r >= rows) return;\n"
"    uint nb = cols / 32;\n"
"    float acc = 0.0f;\n"
"    for (uint b = 0; b < nb; b++) {\n"
"        device const q40 &blk = w[r * nb + b];\n"
"        float d = float(blk.d);\n"
"        for (uint i = 0; i < 16; i++) {\n"
"            uchar q = blk.q[i];\n"
"            acc += d * (float(q & 15) - 8.0f) * x[b * 32 + i] + d * (float(q >> 4) - 8.0f) * x[b * 32 + i + 16];\n"
"        }\n"
"    }\n"
"    y[r] = acc;\n"
"}\n"
"kernel void matvec_q4_1(device const q41 *w [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint r [[thread_position_in_grid]]) {\n"
"    if (r >= rows) return;\n"
"    uint nb = cols / 32;\n"
"    float acc = 0.0f;\n"
"    for (uint b = 0; b < nb; b++) {\n"
"        device const q41 &blk = w[r * nb + b];\n"
"        float d = float(blk.d), m = float(blk.m);\n"
"        for (uint i = 0; i < 16; i++) {\n"
"            uchar q = blk.q[i];\n"
"            acc += (d * float(q & 15) + m) * x[b * 32 + i] + (d * float(q >> 4) + m) * x[b * 32 + i + 16];\n"
"        }\n"
"    }\n"
"    y[r] = acc;\n"
"}\n"
"kernel void matvec_q6_k(device const q6k *w [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint r [[thread_position_in_grid]]) {\n"
"    if (r >= rows) return;\n"
"    uint nb = cols / 256;\n"
"    float acc = 0.0f;\n"
"    for (uint b = 0; b < nb; b++) {\n"
"        device const q6k &blk = w[r * nb + b];\n"
"        float d = float(blk.d);\n"
"        for (uint n = 0; n < 256; n++) {\n"
"            acc += q6k_value(blk, n, d) * x[b * 256 + n];\n"
"        }\n"
"    }\n"
"    y[r] = acc;\n"
"}\n"
"kernel void matvec_f16(device const half *w [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint r [[thread_position_in_grid]]) {\n"
"    if (r >= rows) return;\n"
"    float acc = 0.0f;\n"
"    for (uint i = 0; i < cols; i++) acc += float(w[r * cols + i]) * x[i];\n"
"    y[r] = acc;\n"
"}\n"
"kernel void matvec_lns8(device const uchar *w [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint r [[thread_position_in_grid]]) {\n"
"    if (r >= rows) return;\n"
"    uint nb = cols / 32;\n"
"    float acc = 0.0f;\n"
"    for (uint b = 0; b < nb; b++) {\n"
"        uint off = (r * nb + b) * 36;\n"
"        float d = *(device const float *)(w + off + 32);\n"
"        for (uint i = 0; i < 32; i++) acc += d * exp2(float(w[off + i])) * x[b * 32 + i];\n"
"    }\n"
"    y[r] = acc;\n"
"}\n"
"kernel void matvec_f32(device const float *w [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]], constant uint &cols [[buffer(3)]], constant uint &rows [[buffer(4)]], uint r [[thread_position_in_grid]]) {\n"
"    if (r >= rows) return;\n"
"    float acc = 0.0f;\n"
"    for (uint i = 0; i < cols; i++) acc += w[r * cols + i] * x[i];\n"
"    y[r] = acc;\n"
"}\n"
"kernel void embedding_lookup(device const float *w [[buffer(0)]], constant uint &tok [[buffer(1)]], device float *out [[buffer(2)]], constant uint &dim [[buffer(3)]], uint i [[thread_position_in_grid]]) {\n"
"    if (i < dim) out[i] = w[ulong(tok) * dim + i];\n"
"}\n"
"kernel void embedding_lookup_f16(device const half *w [[buffer(0)]], constant uint &tok [[buffer(1)]], device float *out [[buffer(2)]], constant uint &dim [[buffer(3)]], uint i [[thread_position_in_grid]]) {\n"
"    if (i < dim) out[i] = float(w[ulong(tok) * dim + i]);\n"
"}\n"
"kernel void embedding_lookup_q8_0(device const q80 *w [[buffer(0)]], constant uint &tok [[buffer(1)]], device float *out [[buffer(2)]], constant uint &dim [[buffer(3)]], uint i [[thread_position_in_grid]]) {\n"
"    if (i >= dim) return;\n"
"    uint nb = dim / 32;\n"
"    device const q80 &blk = w[ulong(tok) * nb + (i / 32)];\n"
"    out[i] = float(blk.d) * float(blk.q[i % 32]);\n"
"}\n"
"kernel void embedding_lookup_q4_0(device const q40 *w [[buffer(0)]], constant uint &tok [[buffer(1)]], device float *out [[buffer(2)]], constant uint &dim [[buffer(3)]], uint i [[thread_position_in_grid]]) {\n"
"    if (i >= dim) return;\n"
"    uint nb = dim / 32;\n"
"    device const q40 &blk = w[ulong(tok) * nb + (i / 32)];\n"
"    uint ib = i % 32;\n"
"    uchar q = blk.q[ib & 15];\n"
"    float qv = float((ib < 16) ? (q & 15) : (q >> 4));\n"
"    out[i] = float(blk.d) * (qv - 8.0f);\n"
"}\n"
"kernel void embedding_lookup_q4_1(device const q41 *w [[buffer(0)]], constant uint &tok [[buffer(1)]], device float *out [[buffer(2)]], constant uint &dim [[buffer(3)]], uint i [[thread_position_in_grid]]) {\n"
"    if (i >= dim) return;\n"
"    uint nb = dim / 32;\n"
"    device const q41 &blk = w[ulong(tok) * nb + (i / 32)];\n"
"    uint ib = i % 32;\n"
"    uchar q = blk.q[ib & 15];\n"
"    float qv = float((ib < 16) ? (q & 15) : (q >> 4));\n"
"    out[i] = float(blk.d) * qv + float(blk.m);\n"
"}\n"
"kernel void embedding_lookup_q6_k(device const q6k *w [[buffer(0)]], constant uint &tok [[buffer(1)]], device float *out [[buffer(2)]], constant uint &dim [[buffer(3)]], uint i [[thread_position_in_grid]]) {\n"
"    if (i >= dim) return;\n"
"    uint nb = dim / 256;\n"
"    device const q6k &blk = w[ulong(tok) * nb + (i / 256)];\n"
"    uint n = i % 256;\n"
"    out[i] = q6k_value(blk, n, float(blk.d));\n"
"}\n"
"kernel void rms_norm(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]], device float *y [[buffer(2)]], constant uint &n [[buffer(3)]], constant float &eps [[buffer(4)]], uint i [[thread_position_in_grid]]) {\n"
"    if (i >= n) return;\n"
"    float ss = 0.0f;\n"
"    for (uint j = 0; j < n; j++) ss += x[j] * x[j];\n"
"    y[i] = x[i] * rsqrt(ss / float(n) + eps) * w[i];\n"
"}\n"
"kernel void rope(device float *x [[buffer(0)]], constant uint &pos [[buffer(1)]], constant uint &head_dim [[buffer(2)]], constant uint &n_heads [[buffer(3)]], constant float &theta [[buffer(4)]], uint i [[thread_position_in_grid]]) {\n"
"    uint pairs_per_head = head_dim / 2;\n"
"    uint total_pairs = n_heads * pairs_per_head;\n"
"    if (i >= total_pairs) return;\n"
"    uint h = i / pairs_per_head, p = i % pairs_per_head;\n"
"    uint i0 = h * head_dim + p * 2, i1 = i0 + 1;\n"
"    float freq = pow(theta, -(float)(p * 2) / (float)head_dim);\n"
"    float ang = float(pos) * freq;\n"
"    float c = cos(ang), s = sin(ang);\n"
"    float v0 = x[i0], v1 = x[i1];\n"
"    x[i0] = v0 * c - v1 * s;\n"
"    x[i1] = v1 * c + v0 * s;\n"
"}\n"
"kernel void kv_cache_store(device const float *k [[buffer(0)]], device const float *v [[buffer(1)]], device half *kc [[buffer(2)]], device half *vc [[buffer(3)]], constant uint &pos [[buffer(4)]], constant uint &l [[buffer(5)]], constant uint &max_ctx [[buffer(6)]], constant uint &kv_dim [[buffer(7)]], uint i [[thread_position_in_grid]]) {\n"
"    if (i < kv_dim) {\n"
"        ulong off = (ulong(l) * max_ctx + pos) * kv_dim + i;\n"
"        kc[off] = half(k[i]);\n"
"        vc[off] = half(v[i]);\n"
"    }\n"
"}\n"
"kernel void attention_gqa(device const float *q [[buffer(0)]], device const half *kc [[buffer(1)]], device const half *vc [[buffer(2)]], device float *out [[buffer(3)]], constant uint &seq_len [[buffer(4)]], constant uint &l [[buffer(5)]], constant uint &max_ctx [[buffer(6)]], constant uint &nqh [[buffer(7)]], constant uint &nkh [[buffer(8)]], constant uint &hd [[buffer(9)]], constant float &sc [[buffer(10)]], uint hid [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tgs [[threads_per_threadgroup]]) {\n"
"    if (hid >= nqh) return;\n"
"    uint kh = hid / (nqh / nkh);\n"
"    uint kvd = nkh * hd;\n"
"    device const float *qh = q + hid * hd;\n"
"    threadgroup float partial[32];\n"
"    float mxv = -1e30f, den = 0.0f, accv = 0.0f;\n"
"    for (uint pp = 0; pp < seq_len; pp++) {\n"
"        ulong off = (ulong(l) * max_ctx + pp) * kvd + kh * hd;\n"
"        float dot = 0.0f;\n"
"        for (uint d = tid; d < hd; d += tgs) dot += qh[d] * float(kc[off + d]);\n"
"        dot = simd_sum(dot);\n"
"        if ((tid & 31) == 0) partial[tid / 32] = dot;\n"
"        threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"        float full = 0.0f;\n"
"        for (uint g = 0; g < (tgs + 31) / 32; g++) full += partial[g];\n"
"        float score = full * sc;\n"
"        float nmx = max(mxv, score);\n"
"        float alpha = exp(mxv - nmx);\n"
"        float beta = exp(score - nmx);\n"
"        den = den * alpha + beta;\n"
"        mxv = nmx;\n"
"        for (uint d = tid; d < hd; d += tgs) accv = accv * alpha + float(vc[off + d]) * beta;\n"
"    }\n"
"    float inv = 1.0f / (den > 0.0f ? den : 1.0f);\n"
"    for (uint d = tid; d < hd; d += tgs) out[hid * hd + d] = accv * inv;\n"
"}\n"
"kernel void silu_mul(device const float *g [[buffer(0)]], device const float *u [[buffer(1)]], device float *y [[buffer(2)]], constant uint &n [[buffer(3)]], uint i [[thread_position_in_grid]]) {\n"
"    if (i < n) y[i] = (g[i] / (1.0f + exp(-g[i]))) * u[i];\n"
"}\n"
"kernel void residual_add(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]], device float *y [[buffer(2)]], constant uint &n [[buffer(3)]], uint i [[thread_position_in_grid]]) {\n"
"    if (i < n) y[i] = a[i] + b[i];\n"
"}\n";

static bool metal_init(void) {
    typedef id (*MTLCreateSystemDefaultDeviceFn)(void);
    void *sym = dlsym(RTLD_DEFAULT, "MTLCreateSystemDefaultDevice");
    if (!sym) return false;

    g_mtl_device_handle = ((MTLCreateSystemDefaultDeviceFn)sym)();
    if (!g_mtl_device_handle) return false;

    id name_str = ((id (*)(id, SEL))objc_msgSend)(g_mtl_device_handle, sel_registerName("name"));
    const char *cstr = ((const char * (*)(id, SEL))objc_msgSend)(name_str, sel_registerName("UTF8String"));
    snprintf(g_metal_device.name, sizeof(g_metal_device.name), "%s", cstr ? cstr : "Apple Metal GPU");

    g_metal_device.type = VBS_BACKEND_METAL;
    g_metal_device.max_working_set_bytes = ((unsigned long long (*)(id, SEL))objc_msgSend)(g_mtl_device_handle, sel_registerName("recommendedMaxWorkingSetSize"));
    g_metal_device.max_buffer_bytes = ((unsigned long long (*)(id, SEL))objc_msgSend)(g_mtl_device_handle, sel_registerName("maxBufferLength"));
    g_metal_device.internal_handle = g_mtl_device_handle;

    g_mtl_queue_handle = ((id (*)(id, SEL))objc_msgSend)(g_mtl_device_handle, sel_registerName("newCommandQueue"));

    Class ns_string_cls = objc_getClass("NSString");
    id ns_src = ((id (*)(id, SEL, const char *))objc_msgSend)((id)ns_string_cls, sel_registerName("stringWithUTF8String:"), g_metal_shader_source);
    id err = nil;
    g_mtl_library = ((id (*)(id, SEL, id, id, id *))objc_msgSend)(g_mtl_device_handle, sel_registerName("newLibraryWithSource:options:error:"), ns_src, nil, &err);

    if (g_mtl_library == nil) {
        if (err) {
            id d = ((id (*)(id, SEL))objc_msgSend)(err, sel_registerName("localizedDescription"));
            const char *es = ((const char * (*)(id, SEL))objc_msgSend)(d, sel_registerName("UTF8String"));
            fprintf(stderr, "MSLERR: %s", es ? es : "?");
        } else {
            fprintf(stderr, "MSLERR: no error object");
        }
    }
    return (g_mtl_library != nil);
}

static size_t metal_get_device_count(void) { return g_mtl_device_handle ? 1 : 0; }
static vbs_device_t *metal_get_device(size_t i) { (void)i; return &g_metal_device; }

static vbs_buffer_t *metal_alloc_buffer(vbs_device_t *dev, size_t bytes, bool shared) {
    NSUInteger options = shared ? (0x0000UL) : (0x0020UL);
    id buf_handle = ((id (*)(id, SEL, NSUInteger, NSUInteger))objc_msgSend)(
        dev->internal_handle, sel_registerName("newBufferWithLength:options:"), (NSUInteger)bytes, options
    );
    if (!buf_handle) return NULL;

    vbs_buffer_t *buf = (vbs_buffer_t *)calloc(1, sizeof(vbs_buffer_t));
    if (!buf) {
        ((void (*)(id, SEL))objc_msgSend)(buf_handle, sel_registerName("release"));
        return NULL;
    }
    buf->internal_handle = buf_handle;
    buf->size = bytes;
    buf->device = dev;
    if (shared) {
        buf->host_ptr = ((void * (*)(id, SEL))objc_msgSend)(buf_handle, sel_registerName("contents"));
    }
    return buf;
}

static void metal_free_buffer(vbs_buffer_t *buf) {
    if (!buf) return;
    if (buf->internal_handle) {
        ((void (*)(id, SEL))objc_msgSend)(buf->internal_handle, sel_registerName("release"));
    }
    free(buf);
}

/* Copy `bytes` from one Metal buffer to another via a one-shot blit
 * command buffer, committed and waited. */
static void metal_blit_copy(id src_buf, id dst_buf, size_t bytes) {
    id cb = ((id (*)(id, SEL))objc_msgSend)(g_mtl_queue_handle, sel_registerName("commandBuffer"));
    id blit = ((id (*)(id, SEL))objc_msgSend)(cb, sel_registerName("blitCommandEncoder"));
    ((void (*)(id, SEL, id, NSUInteger, id, NSUInteger, NSUInteger))objc_msgSend)(
        blit, sel_registerName("copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:"),
        src_buf, (NSUInteger)0, dst_buf, (NSUInteger)0, (NSUInteger)bytes
    );
    ((void (*)(id, SEL))objc_msgSend)(blit, sel_registerName("endEncoding"));
    ((void (*)(id, SEL))objc_msgSend)(cb, sel_registerName("commit"));
    ((void (*)(id, SEL))objc_msgSend)(cb, sel_registerName("waitUntilCompleted"));
}

static bool metal_copy_to_device(vbs_buffer_t *dst, const void *src, size_t bytes) {
    metal_flush_stream();
    if (dst->host_ptr) {
        memcpy(dst->host_ptr, src, bytes);
        return true;
    }

    id stage = ((id (*)(id, SEL, const void *, NSUInteger, NSUInteger))objc_msgSend)(
        dst->device->internal_handle, sel_registerName("newBufferWithBytes:length:options:"), src, (NSUInteger)bytes, (NSUInteger)0
    );
    metal_blit_copy(stage, dst->internal_handle, bytes);
    ((void (*)(id, SEL))objc_msgSend)(stage, sel_registerName("release"));
    return true;
}

static bool metal_copy_to_host(void *dst, const vbs_buffer_t *src, size_t bytes) {
    /* Reading back commits and waits on the open stream exactly once --
     * this is the single synchronization point at the end of a token. */
    metal_flush_stream();
    if (src->host_ptr) {
        memcpy(dst, src->host_ptr, bytes);
        return true;
    }

    id stage = ((id (*)(id, SEL, NSUInteger, NSUInteger))objc_msgSend)(
        src->device->internal_handle, sel_registerName("newBufferWithLength:options:"), (NSUInteger)bytes, (NSUInteger)0
    );
    metal_blit_copy(src->internal_handle, stage, bytes);
    void *stage_contents = ((void * (*)(id, SEL))objc_msgSend)(stage, sel_registerName("contents"));
    memcpy(dst, stage_contents, bytes);
    ((void (*)(id, SEL))objc_msgSend)(stage, sel_registerName("release"));
    return true;
}

/* Look a function up in the library and build its compute pipeline
 * state. On a missing function, reports `not_found_fmt` (with the
 * name) and returns nil; a nil pipeline state is returned as-is. */
static id metal_make_pso(const char *fn_name, char *err_buf, size_t err_cap,
                         const char *not_found_fmt) {
    Class ns_string_cls = objc_getClass("NSString");
    id ns_fn = ((id (*)(id, SEL, const char *))objc_msgSend)((id)ns_string_cls, sel_registerName("stringWithUTF8String:"), fn_name);
    id fn = ((id (*)(id, SEL, id))objc_msgSend)(g_mtl_library, sel_registerName("newFunctionWithName:"), ns_fn);
    if (!fn) {
        snprintf(err_buf, err_cap, not_found_fmt, fn_name);
        return nil;
    }
    id err = nil;
    id pso = ((id (*)(id, SEL, id, id *))objc_msgSend)(
        g_mtl_device_handle, sel_registerName("newComputePipelineStateWithFunction:error:"), fn, &err
    );
    ((void (*)(id, SEL))objc_msgSend)(fn, sel_registerName("release"));
    return pso;
}

static vbs_prim_pipeline_t *metal_wrap_prim(id pso, vbs_op_kind_t kind) {
    if (!pso) return NULL;
    vbs_prim_pipeline_t *pp = (vbs_prim_pipeline_t *)malloc(sizeof(vbs_prim_pipeline_t));
    if (!pp) return NULL;
    pp->native = pso;
    pp->kind = kind;
    return pp;
}

/* Kernel names come from the shared registry at the top of this
 * file. Metal's fallbacks below only trigger for types or kinds
 * missing from the shared tables. */

static void *metal_compile_linear_pipeline(vbs_device_t *dev, vbs_model_type_t type, char *err_buf, size_t err_cap) {
    (void)dev;
    const char *fn_name = vbs_kernel_for_type(k_vbs_linear_kernels,
                                              sizeof(k_vbs_linear_kernels) / sizeof(k_vbs_linear_kernels[0]),
                                              type);
    if (!fn_name) fn_name = "matvec_f32";
    return metal_make_pso(fn_name, err_buf, err_cap, "Metal function %s not found");
}

static void *metal_compile_embed_pipeline(vbs_device_t *dev, vbs_model_type_t type, char *err_buf, size_t err_cap) {
    (void)dev;
    const char *fn_name = vbs_kernel_for_type(k_vbs_embed_kernels,
                                              sizeof(k_vbs_embed_kernels) / sizeof(k_vbs_embed_kernels[0]),
                                              type);
    if (!fn_name) {
        if (type != VBS_MODEL_TYPE_F32) {
            snprintf(err_buf, err_cap, "Metal has no embedding kernel for type %d", (int)type);
            return NULL;
        }
        fn_name = "embedding_lookup";
    }
    id pso = metal_make_pso(fn_name, err_buf, err_cap, "Metal embedding function %s not found");
    return metal_wrap_prim(pso, VBS_OP_EMBEDDING);
}

static void *metal_compile_primitive_pipeline(vbs_device_t *dev, vbs_op_kind_t kind, char *err_buf, size_t err_cap) {
    (void)dev;
    const char *fn_name = vbs_kernel_for_kind(k_vbs_prim_kernels,
                                              sizeof(k_vbs_prim_kernels) / sizeof(k_vbs_prim_kernels[0]),
                                              kind);
    if (!fn_name) fn_name = "rms_norm";
    id pso = metal_make_pso(fn_name, err_buf, err_cap, "Metal primitive %s not found");
    return metal_wrap_prim(pso, kind);
}

static void metal_set_buffer(id enc, id buf, NSUInteger idx) {
    ((void (*)(id, SEL, id, NSUInteger, NSUInteger))objc_msgSend)(
        enc, sel_registerName("setBuffer:offset:atIndex:"), buf, (NSUInteger)0, idx);
}

static void metal_set_bytes(id enc, const void *ptr, size_t len, NSUInteger idx) {
    ((void (*)(id, SEL, const void *, NSUInteger, NSUInteger))objc_msgSend)(
        enc, sel_registerName("setBytes:length:atIndex:"), ptr, (NSUInteger)len, idx);
}

static void metal_dispatch_threads(id enc, NSUInteger total, NSUInteger tg) {
    MTLSize gridSize = { total, 1, 1 };
    MTLSize tgSize = { tg, 1, 1 };
    ((void (*)(id, SEL, MTLSize, MTLSize))objc_msgSend)(
        enc, sel_registerName("dispatchThreads:threadsPerThreadgroup:"), gridSize, tgSize);
}

static bool metal_dispatch_linear_tile(
    vbs_device_t *dev, void *pipeline,
    vbs_buffer_t *wp, vbs_buffer_t *ws, size_t copied,
    vbs_buffer_t *in, vbs_buffer_t *out,
    uint32_t cols, uint32_t rows, uint32_t col_off, uint32_t row_off,
    bool acc
) {
    (void)dev; (void)ws; (void)copied; (void)col_off; (void)row_off; (void)acc;
    id enc = metal_active_encoder();

    ((void (*)(id, SEL, id))objc_msgSend)(enc, sel_registerName("setComputePipelineState:"), (id)pipeline);
    metal_set_buffer(enc, wp->internal_handle, 0);
    metal_set_buffer(enc, in->internal_handle, 1);
    metal_set_buffer(enc, out->internal_handle, 2);
    metal_set_bytes(enc, &cols, sizeof(uint32_t), 3);
    metal_set_bytes(enc, &rows, sizeof(uint32_t), 4);
    metal_dispatch_threads(enc, (NSUInteger)rows, (NSUInteger)(rows < 64 ? rows : 64));
    return true;
}

static bool metal_dispatch_primitive(
    vbs_device_t *dev, void *pipeline,
    const vbs_arg_t *args, size_t n_args,
    size_t grid_x, size_t block_x
) {
    (void)dev;
    /* Acquire the dispatch target first: the encoder is a
     * persistent cached stream, so opening it before the pipeline
     * check is harmless — an invalid pipeline just returns with
     * the stream still open for the next dispatch. */
    id enc = metal_active_encoder();
    vbs_prim_pipeline_t *pp = vbs_prim_pipeline_checked(pipeline);
    if (!pp) return false;

    ((void (*)(id, SEL, id))objc_msgSend)(enc, sel_registerName("setComputePipelineState:"), (id)pp->native);
    for (size_t i = 0; i < n_args; i++) {
        if (args[i].tag == VBS_ARG_BUFFER) {
            metal_set_buffer(enc, (id)args[i].value, (NSUInteger)i);
        } else {
            /* Scalars go as inline bytes, per the convention in
             * vbs_backend.h — binding a host pointer as a buffer
             * was the bug this replaces. */
            metal_set_bytes(enc, args[i].value, args[i].size, (NSUInteger)i);
        }
    }

    /* (grid_x, block_x) follow the CUDA blocks x threads convention
     * used by the bridge; Metal dispatches a total thread count. */
    metal_dispatch_threads(enc, (NSUInteger)(grid_x * block_x), (NSUInteger)block_x);
    return true;
}

static void metal_synchronize(vbs_device_t *dev) {
    (void)dev;
    metal_flush_stream();
}

const vbs_backend_ops_t g_vbs_metal_backend = {
    .init = metal_init,
    .get_device_count = metal_get_device_count,
    .get_device = metal_get_device,
    .alloc_buffer = metal_alloc_buffer,
    .free_buffer = metal_free_buffer,
    .copy_to_device = metal_copy_to_device,
    .copy_to_host = metal_copy_to_host,
    .compile_linear_pipeline = metal_compile_linear_pipeline,
    .compile_embed_pipeline = metal_compile_embed_pipeline,
    .compile_primitive_pipeline = metal_compile_primitive_pipeline,
    .dispatch_linear_tile = metal_dispatch_linear_tile,
    .dispatch_primitive = metal_dispatch_primitive,
    .synchronize_device = metal_synchronize
};

#endif /* VBS_ENABLE_METAL */
