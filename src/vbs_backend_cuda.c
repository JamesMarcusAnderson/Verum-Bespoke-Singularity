#include "vbs_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <stdint.h>

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

#include "../kernels/cuda_kernels.ptx.h"

static bool cuda_init(void) {
    g_cu.handle = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!g_cu.handle) g_cu.handle = dlopen("libcuda.so", RTLD_LAZY);
    if (!g_cu.handle) g_cu.handle = dlopen("nvcuda.dll", RTLD_LAZY);
    if (!g_cu.handle) return false;

#define LOAD_FN(name) g_cu.name = (name##_fn)dlsym(g_cu.handle, #name); if (!g_cu.name) return false;
    LOAD_FN(cuInit);
    LOAD_FN(cuDeviceGet);
    LOAD_FN(cuDeviceGetCount);
    LOAD_FN(cuDeviceGetName);
    LOAD_FN(cuCtxCreate);
    LOAD_FN(cuModuleLoadData);
    LOAD_FN(cuModuleGetFunction);
    LOAD_FN(cuMemAlloc);
    LOAD_FN(cuMemFree);
    LOAD_FN(cuMemcpyHtoD);
    LOAD_FN(cuMemcpyDtoH);
    LOAD_FN(cuLaunchKernel);
    LOAD_FN(cuCtxSynchronize);
#undef LOAD_FN

    if (g_cu.cuInit(0) != CUDA_SUCCESS) return false;

    int count = 0;
    g_cu.cuDeviceGetCount(&count);
    if (count <= 0) return false;
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
        if (g_cu.cuModuleLoadData(&ci->mod, g_cuda_kernels_ptx) != CUDA_SUCCESS) continue;

        g_cuda_dev_count++;
    }
    return g_cuda_dev_count > 0;
}

static size_t cuda_get_device_count(void) { return (size_t)g_cuda_dev_count; }
static vbs_device_t *cuda_get_device(size_t i) { return &g_cuda_devices[i]; }

static vbs_buffer_t *cuda_alloc_buffer(vbs_device_t *dev, size_t bytes, bool shared) {
    (void)shared;
    CUdeviceptr dptr = 0;
    if (g_cu.cuMemAlloc(&dptr, bytes) != CUDA_SUCCESS) return NULL;
    vbs_buffer_t *b = (vbs_buffer_t *)calloc(1, sizeof(vbs_buffer_t));
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

static void *cuda_compile_linear_pipeline(vbs_device_t *dev, vbs_ggml_type_t type, char *err_buf, size_t err_cap) {
    vbs_cuda_internal_t *ci = (vbs_cuda_internal_t *)dev->internal_handle;
    CUfunction fn = NULL;
    const char *kname = "matvec_f32";
    if (type == VBS_GGML_TYPE_Q8_0) kname = "matvec_q8_0";
    else if (type == VBS_GGML_TYPE_Q4_0) kname = "matvec_q4_0";
    else if (type == VBS_GGML_TYPE_Q4_1) kname = "matvec_q4_1";

    if (g_cu.cuModuleGetFunction(&fn, ci->mod, kname) != CUDA_SUCCESS) {
        snprintf(err_buf, err_cap, "CUDA kernel %s not found in PTX", kname);
        return NULL;
    }
    return (void *)fn;
}

static void *cuda_compile_primitive_pipeline(vbs_device_t *dev, vbs_op_kind_t kind, char *err_buf, size_t err_cap) {
    vbs_cuda_internal_t *ci = (vbs_cuda_internal_t *)dev->internal_handle;
    CUfunction fn = NULL;
    const char *kname = NULL;

    switch (kind) {
        case VBS_OP_EMBEDDING:   kname = "embedding_lookup"; break;
        case VBS_OP_NORMALIZE:   kname = "rms_norm";         break;
        case VBS_OP_ROTARY:      kname = "rope";             break;
        case VBS_OP_CACHE:       kname = "kv_cache_store";   break;
        case VBS_OP_ATTENTION:   kname = "attention_gqa";    break;
        case VBS_OP_ACTIVATION:  kname = "silu_mul";         break;
        case VBS_OP_ELEMENTWISE: kname = "residual_add";     break;
        default:
            snprintf(err_buf, err_cap, "CUDA has no primitive kernel for op kind %d", (int)kind);
            return NULL;
    }

    if (g_cu.cuModuleGetFunction(&fn, ci->mod, kname) != CUDA_SUCCESS) {
        snprintf(err_buf, err_cap, "CUDA primitive %s not found", kname);
        return NULL;
    }
    vbs_prim_pipeline_t *pp = (vbs_prim_pipeline_t *)malloc(sizeof(vbs_prim_pipeline_t));
    if (!pp) return NULL;
    pp->native = (void *)fn;
    pp->kind = kind;
    return pp;
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
    vbs_prim_pipeline_t *pp = (vbs_prim_pipeline_t *)pipeline;
    if (!pp || !pp->native) return false;

    uint64_t bufvals[16];
    void *params[16];

    if (pp->kind == VBS_OP_ATTENTION) {
        // Canonical order (bridge): q, kc, vc, out, seq_len, layer, max_ctx,
        // n_q_heads, n_kv_heads, head_dim, scale.
        // PTX attention_gqa takes pre-offset cache pointers and no
        // layer/max_ctx: fold the layer slice offset into the pointers.
        if (n_args != 11) return false;
        uint32_t layer   = *(const uint32_t *)args[5].value;
        uint32_t max_ctx = *(const uint32_t *)args[6].value;
        uint32_t nkh     = *(const uint32_t *)args[8].value;
        uint32_t hd      = *(const uint32_t *)args[9].value;
        uint64_t off = (uint64_t)layer * max_ctx * nkh * hd * 2; // f16 elements -> bytes
        bufvals[0] = (uint64_t)(uintptr_t)args[0].value;
        bufvals[1] = (uint64_t)(uintptr_t)args[1].value + off;
        bufvals[2] = (uint64_t)(uintptr_t)args[2].value + off;
        bufvals[3] = (uint64_t)(uintptr_t)args[3].value;
        params[0] = &bufvals[0]; params[1] = &bufvals[1];
        params[2] = &bufvals[2]; params[3] = &bufvals[3];
        params[4] = (void *)args[4].value;  // seq_len
        params[5] = (void *)args[7].value;  // n_q_heads
        params[6] = (void *)args[8].value;  // n_kv_heads
        params[7] = (void *)args[9].value;  // head_dim
        params[8] = (void *)args[10].value; // scale
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
    .compile_primitive_pipeline = cuda_compile_primitive_pipeline,
    .dispatch_linear_tile = cuda_dispatch_linear_tile,
    .dispatch_primitive = cuda_dispatch_primitive,
    .synchronize_device = cuda_synchronize
};
