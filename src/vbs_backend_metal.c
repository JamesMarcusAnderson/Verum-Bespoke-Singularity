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

#define OBJC_MSG(ret_type, target, sel_name, ...) \
    ((ret_type (*)(id, SEL, ##__VA_ARGS__))objc_msgSend)((id)(target), sel_registerName(sel_name), ##__VA_ARGS__)

static vbs_device_t g_metal_device;
static id g_mtl_device_handle = nil;
static id g_mtl_queue_handle = nil;
static id g_mtl_library = nil;

static const char *g_metal_shader_source =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct q80{half d;char q[32];};\n"
"struct q40{half d;uchar q[16];};\n"
"kernel void matvec_q8_0(device const q80*w[[buffer(0)]],device const float*x[[buffer(1)]],device float*y[[buffer(2)]],constant uint&cols[[buffer(3)]],constant uint&rows[[buffer(4)]],uint r[[thread_position_in_grid]]){if(r>=rows)return;float a=0;uint nb=cols/32;for(uint b=0;b<nb;b++){float d=float(w[r*nb+b].d);for(uint i=0;i<32;i++)a+=d*float(w[r*nb+b].q[i])*x[b*32+i];}y[r]=a;}\n"
"kernel void matvec_q4_0(device const q40*w[[buffer(0)]],device const float*x[[buffer(1)]],device float*y[[buffer(2)]],constant uint&cols[[buffer(3)]],constant uint&rows[[buffer(4)]],uint r[[thread_position_in_grid]]){if(r>=rows)return;float a=0;uint nb=cols/32;for(uint b=0;b<nb;b++){float d=float(w[r*nb+b].d);for(uint i=0;i<16;i++){uchar q=w[r*nb+b].q[i];a+=d*(float(q&15)-8.0f)*x[b*32+i]+d*(float(q>>4)-8.0f)*x[b*32+i+16];}}y[r]=a;}\n"
"kernel void matvec_f32(device const float*w[[buffer(0)]],device const float*x[[buffer(1)]],device float*y[[buffer(2)]],constant uint&cols[[buffer(3)]],constant uint&rows[[buffer(4)]],uint r[[thread_position_in_grid]]){if(r>=rows)return;float a=0;for(uint i=0;i<cols;i++)a+=w[r*cols+i]*x[i];y[r]=a;}\n"
"kernel void rms_norm(device const float*x[[buffer(0)]],device const float*w[[buffer(1)]],device float*y[[buffer(2)]],constant uint&n[[buffer(3)]],constant float&eps[[buffer(4)]],uint i[[thread_position_in_grid]]){if(i>=n)return;float ss=0;for(uint j=0;j<n;j++)ss+=x[j]*x[j];y[i]=x[i]*rsqrt(ss/float(n)+eps)*w[i];}\n"
"kernel void silu_mul(device const float*g[[buffer(0)]],device const float*u[[buffer(1)]],device float*y[[buffer(2)]],constant uint&n[[buffer(3)]],uint i[[thread_position_in_grid]]){if(i<n)y[i]=(g[i]/(1.0f+exp(-g[i])))*u[i];}\n"
"kernel void residual_add(device const float*a[[buffer(0)]],device const float*b[[buffer(1)]],device float*y[[buffer(2)]],constant uint&n[[buffer(3)]],uint i[[thread_position_in_grid]]){if(i<n)y[i]=a[i]+b[i];}\n";

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

    // Compile embedded MSL library
    Class ns_string_cls = objc_getClass("NSString");
    id ns_src = OBJC_MSG(id, ns_string_cls, "stringWithUTF8String:", g_metal_shader_source);
    id err = nil;
    g_mtl_library = OBJC_MSG(id, g_mtl_device_handle, "newLibraryWithSource:options:error:", ns_src, nil, &err);

    return (g_mtl_library != nil);
}

static size_t metal_get_device_count(void) { return g_mtl_device_handle ? 1 : 0; }
static vbs_device_t *metal_get_device(size_t i) { (void)i; return &g_metal_device; }

static vbs_buffer_t *metal_alloc_buffer(vbs_device_t *dev, size_t bytes, bool shared) {
    NSUInteger options = shared ? (0x0000UL) : (0x0020UL);
    typedef id (*alloc_buf_fn)(id, SEL, NSUInteger, NSUInteger);
    id buf_handle = ((alloc_buf_fn)objc_msgSend)(dev->internal_handle, sel_registerName("newBufferWithLength:options:"), (NSUInteger)bytes, options);
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

static bool metal_copy_to_device(vbs_buffer_t *dst, const void *src, size_t bytes) {
    if (dst->host_ptr) {
        memcpy(dst->host_ptr, src, bytes);
        return true;
    }
    id stage = OBJC_MSG(id, dst->device->internal_handle, "newBufferWithBytes:length:options:", src, (NSUInteger)bytes, (NSUInteger)0);
    id cb = OBJC_MSG(id, g_mtl_queue_handle, "commandBuffer");
    id blit = OBJC_MSG(id, cb, "blitCommandEncoder");
    OBJC_MSG(void, blit, "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:", stage, (NSUInteger)0, dst->internal_handle, (NSUInteger)0, (NSUInteger)bytes);
    OBJC_MSG(void, blit, "endEncoding");
    OBJC_MSG(void, cb, "commit");
    OBJC_MSG(void, cb, "waitUntilCompleted");
    OBJC_MSG(void, stage, "release");
    return true;
}

static bool metal_copy_to_host(void *dst, const vbs_buffer_t *src, size_t bytes) {
    if (src->host_ptr) {
        memcpy(dst, src->host_ptr, bytes);
        return true;
    }
    id stage = OBJC_MSG(id, src->device->internal_handle, "newBufferWithLength:options:", (NSUInteger)bytes, (NSUInteger)0);
    id cb = OBJC_MSG(id, g_mtl_queue_handle, "commandBuffer");
    id blit = OBJC_MSG(id, cb, "blitCommandEncoder");
    OBJC_MSG(void, blit, "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:", src->internal_handle, (NSUInteger)0, stage, (NSUInteger)0, (NSUInteger)bytes);
    OBJC_MSG(void, blit, "endEncoding");
    OBJC_MSG(void, cb, "commit");
    OBJC_MSG(void, cb, "waitUntilCompleted");
    memcpy(dst, OBJC_MSG(void *, stage, "contents"), bytes);
    OBJC_MSG(void, stage, "release");
    return true;
}

static void *metal_compile_linear_pipeline(vbs_device_t *dev, vbs_ggml_type_t type, char *err_buf, size_t err_cap) {
    (void)dev;
    const char *fn_name = "matvec_f32";
    if (type == VBS_GGML_TYPE_Q8_0) fn_name = "matvec_q8_0";
    if (type == VBS_GGML_TYPE_Q4_0) fn_name = "matvec_q4_0";

    Class ns_string_cls = objc_getClass("NSString");
    id ns_fn = OBJC_MSG(id, ns_string_cls, "stringWithUTF8String:", fn_name);
    id fn = OBJC_MSG(id, g_mtl_library, "newFunctionWithName:", ns_fn);
    if (!fn) {
        snprintf(err_buf, err_cap, "Metal function %s not found", fn_name);
        return NULL;
    }
    id err = nil;
    id pso = OBJC_MSG(id, g_mtl_device_handle, "newComputePipelineStateWithFunction:error:", fn, &err);
    OBJC_MSG(void, fn, "release");
    return pso;
}

static void *metal_compile_primitive_pipeline(vbs_device_t *dev, vbs_op_kind_t kind, char *err_buf, size_t err_cap) {
    (void)dev;
    const char *fn_name = "rms_norm";
    if (kind == VBS_OP_ACTIVATION) fn_name = "silu_mul";
    if (kind == VBS_OP_ELEMENTWISE) fn_name = "residual_add";

    Class ns_string_cls = objc_getClass("NSString");
    id ns_fn = OBJC_MSG(id, ns_string_cls, "stringWithUTF8String:", fn_name);
    id fn = OBJC_MSG(id, g_mtl_library, "newFunctionWithName:", ns_fn);
    if (!fn) {
        snprintf(err_buf, err_cap, "Metal primitive %s not found", fn_name);
        return NULL;
    }
    id err = nil;
    id pso = OBJC_MSG(id, g_mtl_device_handle, "newComputePipelineStateWithFunction:error:", fn, &err);
    OBJC_MSG(void, fn, "release");
    return pso;
}

static bool metal_dispatch_linear_tile(
    vbs_device_t *dev, void *pipeline,
    vbs_buffer_t *wp, vbs_buffer_t *ws, size_t copied,
    vbs_buffer_t *in, vbs_buffer_t *out,
    uint32_t cols, uint32_t rows, uint32_t col_off, uint32_t row_off,
    bool acc
) {
    (void)dev; (void)ws; (void)copied; (void)col_off; (void)row_off; (void)acc;
    id cb = OBJC_MSG(id, g_mtl_queue_handle, "commandBuffer");
    id enc = OBJC_MSG(id, cb, "computeCommandEncoder");

    OBJC_MSG(void, enc, "setComputePipelineState:", pipeline);
    OBJC_MSG(void, enc, "setBuffer:offset:atIndex:", wp->internal_handle, (NSUInteger)0, (NSUInteger)0);
    OBJC_MSG(void, enc, "setBuffer:offset:atIndex:", in->internal_handle, (NSUInteger)0, (NSUInteger)1);
    OBJC_MSG(void, enc, "setBuffer:offset:atIndex:", out->internal_handle, (NSUInteger)0, (NSUInteger)2);
    OBJC_MSG(void, enc, "setBytes:length:atIndex:", &cols, sizeof(uint32_t), (NSUInteger)3);
    OBJC_MSG(void, enc, "setBytes:length:atIndex:", &rows, sizeof(uint32_t), (NSUInteger)4);

    typedef struct { NSUInteger width, height, depth; } MTLSize;
    MTLSize gridSize = { (NSUInteger)rows, 1, 1 };
    MTLSize tgSize = { (NSUInteger)MIN(rows, 64), 1, 1 };

    typedef void (*dispatch_fn)(id, SEL, MTLSize, MTLSize);
    ((dispatch_fn)objc_msgSend)(enc, sel_registerName("dispatchThreads:threadsPerThreadgroup:"), gridSize, tgSize);

    OBJC_MSG(void, enc, "endEncoding");
    OBJC_MSG(void, cb, "commit");
    OBJC_MSG(void, cb, "waitUntilCompleted");
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
    .copy_to_device = metal_copy_to_device,
    .copy_to_host = metal_copy_to_host,
    .compile_linear_pipeline = metal_compile_linear_pipeline,
    .compile_primitive_pipeline = metal_compile_primitive_pipeline,
    .dispatch_linear_tile = metal_dispatch_linear_tile,
    .synchronize_device = metal_synchronize
};
