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
typedef void *CUgraph;
typedef void *CUgraphExec;

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
typedef CUresult (*cuStreamCreate_fn)(CUstream *, unsigned int);
typedef CUresult (*cuStreamDestroy_fn)(CUstream);
typedef CUresult (*cuStreamBeginCapture_fn)(CUstream, int);
typedef CUresult (*cuStreamEndCapture_fn)(CUstream, CUgraph *);
typedef CUresult (*cuGraphInstantiate_fn)(CUgraphExec *, CUgraph, void *, char *, size_t);
typedef CUresult (*cuGraphLaunch_fn)(CUgraphExec, CUstream);
typedef CUresult (*cuGraphDestroy_fn)(CUgraph);
typedef CUresult (*cuGraphExecDestroy_fn)(CUgraphExec);

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
    cuStreamCreate_fn cuStreamCreate;
    cuStreamDestroy_fn cuStreamDestroy;
    cuStreamBeginCapture_fn cuStreamBeginCapture;
    cuStreamEndCapture_fn cuStreamEndCapture;
    cuGraphInstantiate_fn cuGraphInstantiate;
    cuGraphLaunch_fn cuGraphLaunch;
    cuGraphDestroy_fn cuGraphDestroy;
    cuGraphExecDestroy_fn cuGraphExecDestroy;
} g_cu;

typedef struct {
    CUdevice dev;
    CUcontext ctx;
    CUmodule mod;
    CUstream stream;
    CUgraph graph;
    CUgraphExec graph_exec;
    bool is_capturing;
} vbs_cuda_internal_t;

static vbs_device_t g_cuda_devices[8];
static vbs_cuda_internal_t g_cuda_internals[8];
static int g_cuda_dev_count = 0;

static const char *g_ptx_source =
".version 7.0\n"
".target sm_70\n"
".address_size 64\n"
".visible .entry matvec_f32(\n"
"    .param .u64 w, .param .u64 x, .param .u64 y,\n"
"    .param .u32 cols, .param .u32 rows\n"
") { ret; }\n"
".visible .entry matvec_q8_0(\n"
"    .param .u64 w, .param .u64 x, .param .u64 y,\n"
"    .param .u32 cols, .param .u32 rows\n"
") { ret; }\n";

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

    // Optional Graph API loading (driver version dependent)
    g_cu.cuStreamCreate = (cuStreamCreate_fn)dlsym(g_cu.handle, "cuStreamCreate");
    g_cu.cuStreamDestroy = (cuStreamDestroy_fn)dlsym(g_cu.handle, "cuStreamDestroy");
    g_cu.cuStreamBeginCapture = (cuStreamBeginCapture_fn)dlsym(g_cu.handle, "cuStreamBeginCapture_v2");
    if (!g_cu.cuStreamBeginCapture) g_cu.cuStreamBeginCapture = (cuStreamBeginCapture_fn)dlsym(g_cu.handle, "cuStreamBeginCapture");
    g_cu.cuStreamEndCapture = (cuStreamEndCapture_fn)dlsym(g_cu.handle, "cuStreamEndCapture");
    g_cu.cuGraphInstantiate = (cuGraphInstantiate_fn)dlsym(g_cu.handle, "cuGraphInstantiate_v2");
    if (!g_cu.cuGraphInstantiate) g_cu.cuGraphInstantiate = (cuGraphInstantiate_fn)dlsym(g_cu.handle, "cuGraphInstantiate");
    g_cu.cuGraphLaunch = (cuGraphLaunch_fn)dlsym(g_cu.handle, "cuGraphLaunch");
    g_cu.cuGraphDestroy = (cuGraphDestroy_fn)dlsym(g_cu.handle, "cuGraphDestroy");
    g_cu.cuGraphExecDestroy = (cuGraphExecDestroy_fn)dlsym(g_cu.handle, "cuGraphExecDestroy");

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
        g_cu.cuModuleLoadData(&ci->mod, g_ptx_source);
        if (g_cu.cuStreamCreate) g_cu.cuStreamCreate(&ci->stream, 0);

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

static void *cuda_compile_linear_pipeline(vbs_device_t *dev, vbs_ggml_type_t type, char *err_buf, size_t err_cap) {
    (void)type;
    vbs_cuda_internal_t *ci = (vbs_cuda_internal_t *)dev->internal_handle;
    CUfunction fn;
    if (g_cu.cuModuleGetFunction(&fn, ci->mod, "matvec_q8_0") != CUDA_SUCCESS) {
        snprintf(err_buf, err_cap, "CUDA kernel lookup failed");
        return NULL;
    }
    return (void *)fn;
}

static void *cuda_compile_primitive_pipeline(vbs_device_t *dev, vbs_op_kind_t kind, char *err_buf, size_t err_cap) {
    (void)dev; (void)kind; (void)err_buf; (void)err_cap;
    return (void *)0x1;
}

static bool cuda_dispatch_linear_tile(
    vbs_device_t *dev, void *pipeline,
    vbs_buffer_t *wp, vbs_buffer_t *ws, size_t copied_bytes,
    vbs_buffer_t *in, vbs_buffer_t *out,
    uint32_t cols, uint32_t rows, uint32_t col_off, uint32_t row_off,
    bool accumulate
) {
    (void)accumulate; (void)col_off; (void)row_off;
    vbs_cuda_internal_t *ci = (vbs_cuda_internal_t *)dev->internal_handle;
    g_cu.cuMemcpyHtoD((CUdeviceptr)(uintptr_t)wp->internal_handle, ws->host_ptr, copied_bytes);
    void *args[] = { &wp->internal_handle, &in->internal_handle, &out->internal_handle, &cols, &rows };
    unsigned int grid = (rows + 31) / 32;
    g_cu.cuLaunchKernel((CUfunction)pipeline, grid, 1, 1, 32, 1, 1, 0, ci->stream, args, NULL);
    return true;
}

static void cuda_synchronize(vbs_device_t *dev) {
    (void)dev;
    g_cu.cuCtxSynchronize();
}

// --- CUDA Graph Lifecycle APIs ---
bool vbs_cuda_begin_capture(vbs_device_t *dev) {
    if (!g_cu.cuStreamBeginCapture) return false;
    vbs_cuda_internal_t *ci = (vbs_cuda_internal_t *)dev->internal_handle;
    if (ci->is_capturing) return false;
    if (g_cu.cuStreamBeginCapture(ci->stream, 0) == CUDA_SUCCESS) {
        ci->is_capturing = true;
        return true;
    }
    return false;
}

bool vbs_cuda_end_capture(vbs_device_t *dev) {
    if (!g_cu.cuStreamEndCapture || !g_cu.cuGraphInstantiate) return false;
    vbs_cuda_internal_t *ci = (vbs_cuda_internal_t *)dev->internal_handle;
    if (!ci->is_capturing) return false;

    if (ci->graph_exec) {
        g_cu.cuGraphExecDestroy(ci->graph_exec);
        ci->graph_exec = NULL;
    }
    if (ci->graph) {
        g_cu.cuGraphDestroy(ci->graph);
        ci->graph = NULL;
    }

    if (g_cu.cuStreamEndCapture(ci->stream, &ci->graph) != CUDA_SUCCESS) {
        ci->is_capturing = false;
        return false;
    }
    ci->is_capturing = false;

    char err_log[512] = {0};
    if (g_cu.cuGraphInstantiate(&ci->graph_exec, ci->graph, NULL, err_log, sizeof(err_log)) != CUDA_SUCCESS) {
        return false;
    }
    return true;
}

bool vbs_cuda_launch_graph(vbs_device_t *dev) {
    if (!g_cu.cuGraphLaunch) return false;
    vbs_cuda_internal_t *ci = (vbs_cuda_internal_t *)dev->internal_handle;
    if (!ci->graph_exec) return false;
    return g_cu.cuGraphLaunch(ci->graph_exec, ci->stream) == CUDA_SUCCESS;
}

const vbs_backend_ops_t g_vbs_cuda_backend = {
    .init = cuda_init,
    .get_device_count = cuda_get_device_count,
    .get_device = cuda_get_device,
    .alloc_buffer = cuda_alloc_buffer,
    .free_buffer = cuda_free_buffer,
    .compile_linear_pipeline = cuda_compile_linear_pipeline,
    .compile_primitive_pipeline = cuda_compile_primitive_pipeline,
    .dispatch_linear_tile = cuda_dispatch_linear_tile,
    .synchronize_device = cuda_synchronize
};
