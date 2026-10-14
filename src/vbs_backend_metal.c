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

static const char *g_metal_shader_source =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct q80{half d;char q[32];};\n"
"struct q40{half d;uchar q[16];};\n"
"struct q41{half d;half m;uchar q[16];};\n"
"kernel void matvec_q8_0(device const q80*w[[buffer(0)]],device const float*x[[buffer(1)]],device float*y[[buffer(2)]],constant uint&cols[[buffer(3)]],constant uint&rows[[buffer(4)]],uint r[[thread_position_in_grid]]){if(r>=rows)return;float a=0;uint nb=cols/32;for(uint b=0;b<nb;b++){float d=float(w[r*nb+b].d);for(uint i=0;i<32;i++)a+=d*float(w[r*nb+b].q[i])*x[b*32+i];}y[r]=a;}\n"
"kernel void matvec_q4_0(device const q40*w[[buffer(0)]],device const float*x[[buffer(1)]],device float*y[[buffer(2)]],constant uint&cols[[buffer(3)]],constant uint&rows[[buffer(4)]],uint r[[thread_position_in_grid]]){if(r>=rows)return;float a=0;uint nb=cols/32;for(uint b=0;b<nb;b++){float d=float(w[r*nb+b].d);for(uint i=0;i<16;i++){uchar q=w[r*nb+b].q[i];a+=d*(float(q&15)-8.0f)*x[b*32+i]+d*(float(q>>4)-8.0f)*x[b*32+i+16];}}y[r]=a;}\n"
"kernel void matvec_q4_1(device const q41*w[[buffer(0)]],device const float*x[[buffer(1)]],device float*y[[buffer(2)]],constant uint&cols[[buffer(3)]],constant uint&rows[[buffer(4)]],uint r[[thread_position_in_grid]]){if(r>=rows)return;float a=0;uint nb=cols/32;for(uint b=0;b<nb;b++){float d=float(w[r*nb+b].d),m=float(w[r*nb+b].m);for(uint i=0;i<16;i++){uchar q=w[r*nb+b].q[i];a+=(d*float(q&15)+m)*x[b*32+i]+(d*float(q>>4)+m)*x[b*32+i+16];}}y[r]=a;}\n"
"kernel void matvec_f32(device const float*w[[buffer(0)]],device const float*x[[buffer(1)]],device float*y[[buffer(2)]],constant uint&cols[[buffer(3)]],constant uint&rows[[buffer(4)]],uint r[[thread_position_in_grid]]){if(r>=rows)return;float a=0;for(uint i=0;i<cols;i++)a+=w[r*cols+i]*x[i];y[r]=a;}\n"
"kernel void embedding_lookup(device const float*w[[buffer(0)]],constant uint&tok[[buffer(1)]],device float*out[[buffer(2)]],constant uint&dim[[buffer(3)]],uint i[[thread_position_in_grid]]){if(i<dim)out[i]=w[ulong(tok)*dim+i];}\n"
"kernel void rms_norm(device const float*x[[buffer(0)]],device const float*w[[buffer(1)]],device float*y[[buffer(2)]],constant uint&n[[buffer(3)]],constant float&eps[[buffer(4)]],uint i[[thread_position_in_grid]]){if(i>=n)return;float ss=0;for(uint j=0;j<n;j++)ss+=x[j]*x[j];y[i]=x[i]*rsqrt(ss/float(n)+eps)*w[i];}\n"
"kernel void rope(device float*x[[buffer(0)]],constant uint&pos[[buffer(1)]],constant uint&head_dim[[buffer(2)]],constant uint&n_heads[[buffer(3)]],constant float&theta[[buffer(4)]],uint i[[thread_position_in_grid]]){uint total_pairs=n_heads*(head_dim/2);if(i>=total_pairs)return;uint h=i/(head_dim/2),p=i%(head_dim/2),i0=h*head_dim+p*2,i1=i0+1;float freq=pow(theta,-(float)(p*2)/(float)head_dim),ang=float(pos)*freq,c=cos(ang),s=sin(ang),v0=x[i0],v1=x[i1];x[i0]=v0*c-v1*s;x[i1]=v1*c+v0*s;}\n"
"kernel void kv_cache_store(device const float*k[[buffer(0)]],device const float*v[[buffer(1)]],device half*kc[[buffer(2)]],device half*vc[[buffer(3)]],constant uint&pos[[buffer(4)]],constant uint&l[[buffer(5)]],constant uint&max_ctx[[buffer(6)]],constant uint&kv_dim[[buffer(7)]],uint i[[thread_position_in_grid]]){if(i<kv_dim){ulong off=(ulong(l)*max_ctx+pos)*kv_dim+i;kc[off]=half(k[i]);vc[off]=half(v[i]);}}\n"
"kernel void attention_gqa(device const float*q[[buffer(0)]],device const half*kc[[buffer(1)]],device const half*vc[[buffer(2)]],device float*out[[buffer(3)]],constant uint&seq_len[[buffer(4)]],constant uint&l[[buffer(5)]],constant uint&max_ctx[[buffer(6)]],constant uint&nqh[[buffer(7)]],constant uint&nkh[[buffer(8)]],constant uint&hd[[buffer(9)]],constant float&sc[[buffer(10)]],uint hid[[threadgroup_position_in_grid]]){if(hid>=nqh)return;uint kh=hid/(nqh/nkh),kvd=nkh*hd;device const float*qh=q+hid*hd;threadgroup float scores[2048];for(uint p=0;p<seq_len;p++){ulong off=(ulong(l)*max_ctx+p)*kvd+kh*hd;float dot=0;for(uint d=0;d<hd;d++)dot+=qh[d]*float(kc[off+d]);scores[p]=dot*sc;}float mx=-1e9f;for(uint p=0;p<seq_len;p++)mx=max(mx,scores[p]);float sm=0;for(uint p=0;p<seq_len;p++){scores[p]=exp(scores[p]-mx);sm+=scores[p];}float inv=1.0f/sm;for(uint d=0;d<hd;d++){float acc=0;for(uint p=0;p<seq_len;p++){ulong off=(ulong(l)*max_ctx+p)*kvd+kh*hd+d;acc+=scores[p]*inv*float(vc[off]);}out[hid*hd+d]=acc;}}\n"
"kernel void silu_mul(device const float*g[[buffer(0)]],device const float*u[[buffer(1)]],device float*y[[buffer(2)]],constant uint&n[[buffer(3)]],uint i[[thread_position_in_grid]]){if(i<n)y[i]=(g[i]/(1.0f+exp(-g[i])))*u[i];}\n"
"kernel void residual_add(device const float*a[[buffer(0)]],device const float*b[[buffer(1)]],device float*y[[buffer(2)]],constant uint&n[[buffer(3)]],uint i[[thread_position_in_grid]]){if(i<n)y[i]=a[i]+b[i];}\n";

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

    if(g_mtl_library==nil){if(err){id d=((id (*)(id,SEL))objc_msgSend)(err,sel_registerName("localizedDescription"));const char *es=((const char *(*)(id,SEL))objc_msgSend)(d,sel_registerName("UTF8String"));fprintf(stderr,"MSLERR: %s",es?es:"?");}else{fprintf(stderr,"MSLERR: no error object");}}return (g_mtl_library != nil);
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

static bool metal_copy_to_device(vbs_buffer_t *dst, const void *src, size_t bytes) {
    if (dst->host_ptr) {
        memcpy(dst->host_ptr, src, bytes);
        return true;
    }

    id stage = ((id (*)(id, SEL, const void *, NSUInteger, NSUInteger))objc_msgSend)(
        dst->device->internal_handle, sel_registerName("newBufferWithBytes:length:options:"), src, (NSUInteger)bytes, (NSUInteger)0
    );
    id cb = ((id (*)(id, SEL))objc_msgSend)(g_mtl_queue_handle, sel_registerName("commandBuffer"));
    id blit = ((id (*)(id, SEL))objc_msgSend)(cb, sel_registerName("blitCommandEncoder"));

    ((void (*)(id, SEL, id, NSUInteger, id, NSUInteger, NSUInteger))objc_msgSend)(
        blit, sel_registerName("copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:"),
        stage, (NSUInteger)0, dst->internal_handle, (NSUInteger)0, (NSUInteger)bytes
    );

    ((void (*)(id, SEL))objc_msgSend)(blit, sel_registerName("endEncoding"));
    ((void (*)(id, SEL))objc_msgSend)(cb, sel_registerName("commit"));
    ((void (*)(id, SEL))objc_msgSend)(cb, sel_registerName("waitUntilCompleted"));
    ((void (*)(id, SEL))objc_msgSend)(stage, sel_registerName("release"));
    return true;
}

static bool metal_copy_to_host(void *dst, const vbs_buffer_t *src, size_t bytes) {
    if (src->host_ptr) {
        memcpy(dst, src->host_ptr, bytes);
        return true;
    }

    id stage = ((id (*)(id, SEL, NSUInteger, NSUInteger))objc_msgSend)(
        src->device->internal_handle, sel_registerName("newBufferWithLength:options:"), (NSUInteger)bytes, (NSUInteger)0
    );
    id cb = ((id (*)(id, SEL))objc_msgSend)(g_mtl_queue_handle, sel_registerName("commandBuffer"));
    id blit = ((id (*)(id, SEL))objc_msgSend)(cb, sel_registerName("blitCommandEncoder"));

    ((void (*)(id, SEL, id, NSUInteger, id, NSUInteger, NSUInteger))objc_msgSend)(
        blit, sel_registerName("copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:"),
        src->internal_handle, (NSUInteger)0, stage, (NSUInteger)0, (NSUInteger)bytes
    );

    ((void (*)(id, SEL))objc_msgSend)(blit, sel_registerName("endEncoding"));
    ((void (*)(id, SEL))objc_msgSend)(cb, sel_registerName("commit"));
    ((void (*)(id, SEL))objc_msgSend)(cb, sel_registerName("waitUntilCompleted"));

    void *stage_contents = ((void * (*)(id, SEL))objc_msgSend)(stage, sel_registerName("contents"));
    memcpy(dst, stage_contents, bytes);
    ((void (*)(id, SEL))objc_msgSend)(stage, sel_registerName("release"));
    return true;
}

static void *metal_compile_linear_pipeline(vbs_device_t *dev, vbs_ggml_type_t type, char *err_buf, size_t err_cap) {
    (void)dev;
    const char *fn_name = "matvec_f32";
    if (type == VBS_GGML_TYPE_Q8_0) fn_name = "matvec_q8_0";
    else if (type == VBS_GGML_TYPE_Q4_0) fn_name = "matvec_q4_0";
    else if (type == VBS_GGML_TYPE_Q4_1) fn_name = "matvec_q4_1";

    Class ns_string_cls = objc_getClass("NSString");
    id ns_fn = ((id (*)(id, SEL, const char *))objc_msgSend)((id)ns_string_cls, sel_registerName("stringWithUTF8String:"), fn_name);
    id fn = ((id (*)(id, SEL, id))objc_msgSend)(g_mtl_library, sel_registerName("newFunctionWithName:"), ns_fn);
    if (!fn) {
        snprintf(err_buf, err_cap, "Metal function %s not found", fn_name);
        return NULL;
    }
    id err = nil;
    id pso = ((id (*)(id, SEL, id, id *))objc_msgSend)(
        g_mtl_device_handle, sel_registerName("newComputePipelineStateWithFunction:error:"), fn, &err
    );
    ((void (*)(id, SEL))objc_msgSend)(fn, sel_registerName("release"));
    return pso;
}

static void *metal_compile_primitive_pipeline(vbs_device_t *dev, vbs_op_kind_t kind, char *err_buf, size_t err_cap) {
    (void)dev;
    const char *fn_name = "rms_norm";
    if (kind == VBS_OP_EMBEDDING)   fn_name = "embedding_lookup";
    if (kind == VBS_OP_ROTARY)      fn_name = "rope";
    if (kind == VBS_OP_CACHE)       fn_name = "kv_cache_store";
    if (kind == VBS_OP_ATTENTION)   fn_name = "attention_gqa";
    if (kind == VBS_OP_ACTIVATION)  fn_name = "silu_mul";
    if (kind == VBS_OP_ELEMENTWISE) fn_name = "residual_add";

    Class ns_string_cls = objc_getClass("NSString");
    id ns_fn = ((id (*)(id, SEL, const char *))objc_msgSend)((id)ns_string_cls, sel_registerName("stringWithUTF8String:"), fn_name);
    id fn = ((id (*)(id, SEL, id))objc_msgSend)(g_mtl_library, sel_registerName("newFunctionWithName:"), ns_fn);
    if (!fn) {
        snprintf(err_buf, err_cap, "Metal primitive %s not found", fn_name);
        return NULL;
    }
    id err = nil;
    id pso = ((id (*)(id, SEL, id, id *))objc_msgSend)(
        g_mtl_device_handle, sel_registerName("newComputePipelineStateWithFunction:error:"), fn, &err
    );
    ((void (*)(id, SEL))objc_msgSend)(fn, sel_registerName("release"));
    if (!pso) return NULL;
    vbs_prim_pipeline_t *pp = (vbs_prim_pipeline_t *)malloc(sizeof(vbs_prim_pipeline_t));
    if (!pp) return NULL;
    pp->native = pso;
    pp->kind = kind;
    return pp;
}

static bool metal_dispatch_linear_tile(
    vbs_device_t *dev, void *pipeline,
    vbs_buffer_t *wp, vbs_buffer_t *ws, size_t copied,
    vbs_buffer_t *in, vbs_buffer_t *out,
    uint32_t cols, uint32_t rows, uint32_t col_off, uint32_t row_off,
    bool acc
) {
    (void)dev; (void)ws; (void)copied; (void)col_off; (void)row_off; (void)acc;
    id cb = ((id (*)(id, SEL))objc_msgSend)(g_mtl_queue_handle, sel_registerName("commandBuffer"));
    id enc = ((id (*)(id, SEL))objc_msgSend)(cb, sel_registerName("computeCommandEncoder"));

    ((void (*)(id, SEL, id))objc_msgSend)(enc, sel_registerName("setComputePipelineState:"), (id)pipeline);
    ((void (*)(id, SEL, id, NSUInteger, NSUInteger))objc_msgSend)(enc, sel_registerName("setBuffer:offset:atIndex:"), wp->internal_handle, (NSUInteger)0, (NSUInteger)0);
    ((void (*)(id, SEL, id, NSUInteger, NSUInteger))objc_msgSend)(enc, sel_registerName("setBuffer:offset:atIndex:"), in->internal_handle, (NSUInteger)0, (NSUInteger)1);
    ((void (*)(id, SEL, id, NSUInteger, NSUInteger))objc_msgSend)(enc, sel_registerName("setBuffer:offset:atIndex:"), out->internal_handle, (NSUInteger)0, (NSUInteger)2);
    ((void (*)(id, SEL, const void *, NSUInteger, NSUInteger))objc_msgSend)(enc, sel_registerName("setBytes:length:atIndex:"), &cols, sizeof(uint32_t), (NSUInteger)3);
    ((void (*)(id, SEL, const void *, NSUInteger, NSUInteger))objc_msgSend)(enc, sel_registerName("setBytes:length:atIndex:"), &rows, sizeof(uint32_t), (NSUInteger)4);

    MTLSize gridSize = { (NSUInteger)rows, 1, 1 };
    MTLSize tgSize = { (NSUInteger)(rows < 64 ? rows : 64), 1, 1 };

    ((void (*)(id, SEL, MTLSize, MTLSize))objc_msgSend)(
        enc, sel_registerName("dispatchThreads:threadsPerThreadgroup:"), gridSize, tgSize
    );

    ((void (*)(id, SEL))objc_msgSend)(enc, sel_registerName("endEncoding"));
    ((void (*)(id, SEL))objc_msgSend)(cb, sel_registerName("commit"));
    ((void (*)(id, SEL))objc_msgSend)(cb, sel_registerName("waitUntilCompleted"));
    return true;
}

static bool metal_dispatch_primitive(
    vbs_device_t *dev, void *pipeline,
    const vbs_arg_t *args, size_t n_args,
    size_t grid_x, size_t block_x
) {
    (void)dev;
    vbs_prim_pipeline_t *pp = (vbs_prim_pipeline_t *)pipeline;
    if (!pp || !pp->native) return false;
    id cb = ((id (*)(id, SEL))objc_msgSend)(g_mtl_queue_handle, sel_registerName("commandBuffer"));
    id enc = ((id (*)(id, SEL))objc_msgSend)(cb, sel_registerName("computeCommandEncoder"));

    ((void (*)(id, SEL, id))objc_msgSend)(enc, sel_registerName("setComputePipelineState:"), (id)pp->native);
    for (size_t i = 0; i < n_args; i++) {
        if (args[i].tag == VBS_ARG_BUFFER) {
            ((void (*)(id, SEL, id, NSUInteger, NSUInteger))objc_msgSend)(
                enc, sel_registerName("setBuffer:offset:atIndex:"), (id)args[i].value, (NSUInteger)0, (NSUInteger)i
            );
        } else {
            // Scalars go as inline bytes, per the convention in vbs_backend.h —
            // binding a host pointer as a buffer was the bug this replaces.
            ((void (*)(id, SEL, const void *, NSUInteger, NSUInteger))objc_msgSend)(
                enc, sel_registerName("setBytes:length:atIndex:"), args[i].value, (NSUInteger)args[i].size, (NSUInteger)i
            );
        }
    }

    // (grid_x, block_x) follow the CUDA blocks x threads convention used by
    // the bridge; Metal dispatches a total thread count.
    MTLSize gridSize = { (NSUInteger)(grid_x * block_x), 1, 1 };
    MTLSize tgSize = { (NSUInteger)block_x, 1, 1 };

    ((void (*)(id, SEL, MTLSize, MTLSize))objc_msgSend)(
        enc, sel_registerName("dispatchThreads:threadsPerThreadgroup:"), gridSize, tgSize
    );

    ((void (*)(id, SEL))objc_msgSend)(enc, sel_registerName("endEncoding"));
    ((void (*)(id, SEL))objc_msgSend)(cb, sel_registerName("commit"));
    ((void (*)(id, SEL))objc_msgSend)(cb, sel_registerName("waitUntilCompleted"));
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
    .dispatch_primitive = metal_dispatch_primitive,
    .synchronize_device = metal_synchronize
};
