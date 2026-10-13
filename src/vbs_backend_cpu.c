#include "vbs_backend.h"
#include "vbs_kernels_ref.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static vbs_device_t g_cpu_device;

static bool cpu_init(void) {
    long cores = sysconf(_SC_NPROCESSORS_ONLN);
    snprintf(g_cpu_device.name, sizeof(g_cpu_device.name), "CPU (%ld cores)", cores);
    g_cpu_device.type = VBS_BACKEND_CPU;
    g_cpu_device.max_working_set_bytes = (uint64_t)-1;
    g_cpu_device.max_buffer_bytes = (uint64_t)-1;
    g_cpu_device.internal_handle = NULL;
    return true;
}

static size_t cpu_get_device_count(void) { return 1; }
static vbs_device_t *cpu_get_device(size_t i) { (void)i; return &g_cpu_device; }

static vbs_buffer_t *cpu_alloc_buffer(vbs_device_t *dev, size_t bytes, bool shared) {
    (void)shared;
    vbs_buffer_t *b = (vbs_buffer_t *)calloc(1, sizeof(vbs_buffer_t));
    void *mem = NULL;
    if (posix_memalign(&mem, 64, bytes) != 0) { free(b); return NULL; }
    b->internal_handle = mem;
    b->host_ptr = mem;
    b->size = bytes;
    b->device = dev;
    return b;
}

static void cpu_free_buffer(vbs_buffer_t *b) {
    if (!b) return;
    free(b->internal_handle);
    free(b);
}

static void *cpu_compile_linear_pipeline(vbs_device_t *dev, vbs_ggml_type_t type, char *err_buf, size_t err_cap) {
    (void)dev;
    void *fn = (void *)vbs_get_matvec_kernel(type);
    if (!fn) {
        snprintf(err_buf, err_cap, "CPU backend has no matvec kernel for type %d", type);
    }
    return fn;
}

static void *cpu_compile_primitive_pipeline(vbs_device_t *dev, vbs_op_kind_t kind, char *err_buf, size_t err_cap) {
    (void)dev; (void)kind; (void)err_buf; (void)err_cap;
    return (void *)0x1; // Non-null sentinel; CPU primitives dispatched directly
}

static bool cpu_dispatch_linear_tile(
    vbs_device_t *dev, void *pipeline,
    vbs_buffer_t *wp, vbs_buffer_t *ws, size_t copied_bytes,
    vbs_buffer_t *in, vbs_buffer_t *out,
    uint32_t cols, uint32_t rows, uint32_t col_off, uint32_t row_off,
    bool accumulate
) {
    (void)dev; (void)wp; (void)copied_bytes;
    vbs_matvec_fn_t kernel = (vbs_matvec_fn_t)pipeline;
    const float *x = (const float *)in->host_ptr + col_off;
    float *y = (float *)out->host_ptr + row_off;

    if (!accumulate) memset(y, 0, rows * sizeof(float));

    float *tmp = (float *)calloc(rows, sizeof(float));
    kernel(ws->host_ptr, x, tmp, cols, rows);
    for (uint32_t r = 0; r < rows; r++) y[r] += tmp[r];
    free(tmp);
    return true;
}

static void cpu_synchronize(vbs_device_t *dev) { (void)dev; }

const vbs_backend_ops_t g_vbs_cpu_backend = {
    .init = cpu_init,
    .get_device_count = cpu_get_device_count,
    .get_device = cpu_get_device,
    .alloc_buffer = cpu_alloc_buffer,
    .free_buffer = cpu_free_buffer,
    .compile_linear_pipeline = cpu_compile_linear_pipeline,
    .compile_primitive_pipeline = cpu_compile_primitive_pipeline,
    .dispatch_linear_tile = cpu_dispatch_linear_tile,
    .synchronize_device = cpu_synchronize
};
