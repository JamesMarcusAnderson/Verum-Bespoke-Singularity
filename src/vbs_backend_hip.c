#include "vbs_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

typedef int hipError_t;
typedef void *hipDeviceptr_t;
typedef void *hipModule_t;
typedef void *hipFunction_t;
typedef void *hipStream_t;

#define HIP_SUCCESS 0

typedef hipError_t (*hipInit_fn)(unsigned int);
typedef hipError_t (*hipGetDeviceCount_fn)(int *);
typedef hipError_t (*hipDeviceGetName_fn)(char *, int, int);
typedef hipError_t (*hipMalloc_fn)(void **, size_t);
typedef hipError_t (*hipFree_fn)(void *);
typedef hipError_t (*hipMemcpyHtoD_fn)(hipDeviceptr_t, void *, size_t);
typedef hipError_t (*hipDeviceSynchronize_fn)(void);

static struct {
    void *handle;
    hipInit_fn hipInit;
    hipGetDeviceCount_fn hipGetDeviceCount;
    hipDeviceGetName_fn hipDeviceGetName;
    hipMalloc_fn hipMalloc;
    hipFree_fn hipFree;
    hipMemcpyHtoD_fn hipMemcpyHtoD;
    hipDeviceSynchronize_fn hipDeviceSynchronize;
} g_hip;

static vbs_device_t g_hip_devices[8];
static int g_hip_dev_count = 0;

static bool hip_init(void) {
    g_hip.handle = dlopen("libamdhip64.so", RTLD_LAZY);
    if (!g_hip.handle) g_hip.handle = dlopen("libamdhip64.so.5", RTLD_LAZY);
    if (!g_hip.handle) return false;

#define LOAD_FN(name) g_hip.name = (name##_fn)dlsym(g_hip.handle, #name); if (!g_hip.name) return false;
    LOAD_FN(hipInit);
    LOAD_FN(hipGetDeviceCount);
    LOAD_FN(hipDeviceGetName);
    LOAD_FN(hipMalloc);
    LOAD_FN(hipFree);
    LOAD_FN(hipMemcpyHtoD);
    LOAD_FN(hipDeviceSynchronize);
#undef LOAD_FN

    if (g_hip.hipInit(0) != HIP_SUCCESS) return false;
    int count = 0;
    g_hip.hipGetDeviceCount(&count);
    if (count <= 0) return false;
    if (count > 8) count = 8;

    for (int i = 0; i < count; i++) {
        char name[128] = {0};
        g_hip.hipDeviceGetName(name, sizeof(name) - 1, i);
        snprintf(g_hip_devices[i].name, 128, "AMD %s", name);
        g_hip_devices[i].type = VBS_BACKEND_HIP;
        g_hip_devices[i].internal_handle = (void *)(intptr_t)i;
    }
    g_hip_dev_count = count;
    return true;
}

static size_t hip_get_device_count(void) { return (size_t)g_hip_dev_count; }
static vbs_device_t *hip_get_device(size_t i) { return &g_hip_devices[i]; }
static vbs_buffer_t *hip_alloc_buffer(vbs_device_t *dev, size_t bytes, bool shared) {
    (void)shared;
    void *dptr = NULL;
    if (g_hip.hipMalloc(&dptr, bytes) != HIP_SUCCESS) return NULL;
    vbs_buffer_t *b = (vbs_buffer_t *)calloc(1, sizeof(vbs_buffer_t));
    b->internal_handle = dptr;
    b->size = bytes;
    b->device = dev;
    b->host_ptr = calloc(1, bytes);
    return b;
}
static void hip_free_buffer(vbs_buffer_t *b) {
    if (!b) return;
    g_hip.hipFree(b->internal_handle);
    free(b->host_ptr);
    free(b);
}
static void *hip_compile_linear_pipeline(vbs_device_t *dev, vbs_ggml_type_t type, char *err_buf, size_t err_cap) {
    (void)dev; (void)type; (void)err_buf; (void)err_cap;
    return (void *)0x1;
}
static void *hip_compile_embed_pipeline(vbs_device_t *dev, vbs_ggml_type_t type, char *err_buf, size_t err_cap) {
    (void)dev; (void)type;
    snprintf(err_buf, err_cap, "HIP backend is a stub: no embedding kernels");
    return NULL;
}
static void *hip_compile_primitive_pipeline(vbs_device_t *dev, vbs_op_kind_t kind, char *err_buf, size_t err_cap) {
    (void)dev; (void)kind; (void)err_buf; (void)err_cap;
    return (void *)0x1;
}
static bool hip_dispatch_linear_tile(
    vbs_device_t *dev, void *pipeline,
    vbs_buffer_t *wp, vbs_buffer_t *ws, size_t copied_bytes,
    vbs_buffer_t *in, vbs_buffer_t *out,
    uint32_t cols, uint32_t rows, uint32_t col_off, uint32_t row_off,
    bool accumulate
) {
    (void)dev; (void)pipeline; (void)wp; (void)ws; (void)copied_bytes;
    (void)in; (void)out; (void)cols; (void)rows; (void)col_off; (void)row_off; (void)accumulate;
    g_hip.hipMemcpyHtoD(wp->internal_handle, ws->host_ptr, copied_bytes);
    return true;
}
static void hip_synchronize(vbs_device_t *dev) { (void)dev; g_hip.hipDeviceSynchronize(); }

const vbs_backend_ops_t g_vbs_hip_backend = {
    .init = hip_init,
    .get_device_count = hip_get_device_count,
    .get_device = hip_get_device,
    .alloc_buffer = hip_alloc_buffer,
    .free_buffer = hip_free_buffer,
    .compile_linear_pipeline = hip_compile_linear_pipeline,
    .compile_embed_pipeline = hip_compile_embed_pipeline,
    .compile_primitive_pipeline = hip_compile_primitive_pipeline,
    .dispatch_linear_tile = hip_dispatch_linear_tile,
    .synchronize_device = hip_synchronize
};
