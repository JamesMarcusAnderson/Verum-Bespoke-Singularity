#include "vbs_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>
#include <objc/runtime.h>
#include <objc/message.h>

#ifndef NSUInteger
typedef unsigned long NSUInteger;
#endif

typedef id (*msg_id_fn)(id, SEL);
typedef id (*msg_id_id_fn)(id, SEL, id);
typedef id (*msg_id_ptr_fn)(id, SEL, const void *);
typedef id (*msg_id_ulong_ulong_fn)(id, SEL, unsigned long, unsigned long);
typedef unsigned long long (*msg_ull_fn)(id, SEL);
typedef const char *(*msg_cstr_fn)(id, SEL);

#define OBJC_MSG(ret_type, target, sel_name, ...) \
    ((ret_type (*)(id, SEL, ##__VA_ARGS__))objc_msgSend)((id)(target), sel_registerName(sel_name), ##__VA_ARGS__)

static vbs_device_t g_metal_device;
static id g_mtl_device_handle = nil;
static id g_mtl_queue_handle = nil;

static bool metal_init(void) {
    typedef id (*MTLCreateSystemDefaultDeviceFn)(void);
    void *sym = dlsym(RTLD_DEFAULT, "MTLCreateSystemDefaultDevice");
    if (!sym) return false;

    g_mtl_device_handle = ((MTLCreateSystemDefaultDeviceFn)sym)();
    if (!g_mtl_device_handle) return false;

    id name_str = OBJC_MSG(id, g_mtl_device_handle, "name");
    const char *cstr = OBJC_MSG(const char *, name_str, "UTF8String");
    snprintf(g_metal_device.name, sizeof(g_metal_device.name), "%s", cstr ? cstr : "Apple Metal GPU");

    g_metal_device.type = VBS_BACKEND_METAL;
    g_metal_device.max_working_set_bytes = OBJC_MSG(unsigned long long, g_mtl_device_handle, "recommendedMaxWorkingSetSize");
    g_metal_device.max_buffer_bytes = OBJC_MSG(unsigned long long, g_mtl_device_handle, "maxBufferLength");
    g_metal_device.internal_handle = g_mtl_device_handle;

    g_mtl_queue_handle = OBJC_MSG(id, g_mtl_device_handle, "newCommandQueue");
    return true;
}

static size_t metal_get_device_count(void) { return g_mtl_device_handle ? 1 : 0; }
static vbs_device_t *metal_get_device(size_t i) { (void)i; return &g_metal_device; }

static vbs_buffer_t *metal_alloc_buffer(vbs_device_t *dev, size_t bytes, bool shared) {
    NSUInteger options = shared ? (0x0000UL) /* MTLResourceStorageModeShared */
                                : (0x0020UL); /* MTLResourceStorageModePrivate */
    id buf_handle = OBJC_MSG(id, dev->internal_handle, "newBufferWithLength:options:", (NSUInteger)bytes, options);
    if (!buf_handle) return NULL;

    vbs_buffer_t *buf = (vbs_buffer_t *)calloc(1, sizeof(vbs_buffer_t));
    buf->internal_handle = buf_handle;
    buf->size = bytes;
    buf->device = dev;
    if (shared) {
        buf->host_ptr = OBJC_MSG(void *, buf_handle, "contents");
    }
    return buf;
}

static void metal_free_buffer(vbs_buffer_t *buf) {
    if (!buf) return;
    if (buf->internal_handle) {
        OBJC_MSG(void, buf->internal_handle, "release");
    }
    free(buf);
}

static void *metal_compile_linear_pipeline(vbs_device_t *dev, vbs_ggml_type_t type, char *err_buf, size_t err_cap) {
    (void)type; (void)err_buf; (void)err_cap;
    return dev->internal_handle;
}

static void *metal_compile_primitive_pipeline(vbs_device_t *dev, vbs_op_kind_t kind, char *err_buf, size_t err_cap) {
    (void)kind; (void)err_buf; (void)err_cap;
    return dev->internal_handle;
}

static bool metal_dispatch_linear_tile(
    vbs_device_t *dev, void *pipeline,
    vbs_buffer_t *wp, vbs_buffer_t *ws, size_t copied,
    vbs_buffer_t *in, vbs_buffer_t *out,
    uint32_t cols, uint32_t rows, uint32_t col_off, uint32_t row_off,
    bool acc
) {
    (void)dev; (void)pipeline; (void)wp; (void)ws; (void)copied;
    (void)in; (void)out; (void)cols; (void)rows; (void)col_off; (void)row_off; (void)acc;
    return true;
}

static void metal_synchronize(vbs_device_t *dev) {
    (void)dev;
}

const vbs_backend_ops_t g_vbs_metal_backend = {
    .init = metal_init,
    .get_device_count = metal_get_device_count,
    .get_device = metal_get_device,
    .alloc_buffer = metal_alloc_buffer,
    .free_buffer = metal_free_buffer,
    .compile_linear_pipeline = metal_compile_linear_pipeline,
    .compile_primitive_pipeline = metal_compile_primitive_pipeline,
    .dispatch_linear_tile = metal_dispatch_linear_tile,
    .synchronize_device = metal_synchronize
};
