/* vbs_backend.h — the backend interface every compute device
 * speaks: devices, buffers, pipelines, and dispatch. The CPU
 * fallback, the GPU backend, and the vendor backends all publish
 * one vbs_backend_ops_t table through this header, and the engine
 * only ever talks to that table.
 *
 * The declarations here are the engine's interface contract, so
 * they are kept exactly; the commentary is new.
 */
#ifndef VBS_BACKEND_H
#define VBS_BACKEND_H

#include "vbs_model.h"
#include "vbs_graph.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VBS_BACKEND_METAL,
    VBS_BACKEND_CUDA,
    VBS_BACKEND_HIP,
    VBS_BACKEND_CPU,
    VBS_BACKEND_VULKAN,
    VBS_BACKEND_NATIVE
} vbs_backend_type_t;

typedef struct vbs_device vbs_device_t;
typedef struct vbs_buffer vbs_buffer_t;

/* A compute device: a display name, its backend family, capacity
 * limits the planner trusts, and the backend's private handle. */
struct vbs_device {
    char name[128];
    vbs_backend_type_t type;
    uint64_t max_working_set_bytes;
    uint64_t max_buffer_bytes;
    void *internal_handle;
};

/* A device buffer: the backend's private handle, a host-side
 * pointer when the memory is host-visible (NULL otherwise), its
 * size, and the device it belongs to. */
struct vbs_buffer {
    void *internal_handle;
    void *host_ptr;
    size_t size;
    vbs_device_t *device;
};

/* Primitive-dispatch argument convention — defined once here and
 * honored by every backend. Each argument is either a device
 * buffer or a scalar passed by value. CUDA passes buffer handles
 * as u64 kernel params and scalars as plain kernel params; Metal
 * binds buffers with setBuffer and scalars with setBytes. The
 * argument ORDER per op kind is fixed by the bridge (see
 * vbs_specializer_bridge.h) and matches both the PTX and MSL
 * kernel signatures; backends whose kernels differ (CUDA
 * attention takes pre-offset cache pointers and no
 * layer/max_ctx params) translate inside their own
 * dispatch_primitive. */
typedef enum { VBS_ARG_BUFFER = 0, VBS_ARG_SCALAR = 1 } vbs_arg_tag_t;

typedef struct {
    vbs_arg_tag_t tag;
    const void *value; /* BUFFER: device handle. SCALAR: pointer to the value. */
    size_t size;       /* SCALAR: byte size of the value. BUFFER: unused (0). */
} vbs_arg_t;

/* Argument constructors — the single canonical copy. */
static inline vbs_arg_t vbs_arg_buf(const void *h) {
    vbs_arg_t a = { VBS_ARG_BUFFER, h, 0 };
    return a;
}
static inline vbs_arg_t vbs_arg_u32(uint32_t *v) {
    vbs_arg_t a = { VBS_ARG_SCALAR, v, sizeof(uint32_t) };
    return a;
}
static inline vbs_arg_t vbs_arg_f32(float *v) {
    vbs_arg_t a = { VBS_ARG_SCALAR, v, sizeof(float) };
    return a;
}

/* Primitive pipeline handle returned by
 * compile_primitive_pipeline: wraps the backend-native object and
 * records the op kind so dispatch_primitive can marshal arguments
 * to the kernel's native layout. */
typedef struct {
    void *native;
    vbs_op_kind_t kind;
} vbs_prim_pipeline_t;

/* The backend operations table. */
typedef struct vbs_backend_ops {
    bool (*init)(void);
    size_t (*get_device_count)(void);
    vbs_device_t *(*get_device)(size_t index);

    vbs_buffer_t *(*alloc_buffer)(vbs_device_t *dev, size_t bytes, bool shared);
    void (*free_buffer)(vbs_buffer_t *buf);
    bool (*copy_to_device)(vbs_buffer_t *dst, const void *src, size_t bytes);
    bool (*copy_to_host)(void *dst, const vbs_buffer_t *src, size_t bytes);

    void *(*compile_linear_pipeline)(vbs_device_t *dev, vbs_model_type_t type, char *err_buf, size_t err_cap);
    void *(*compile_embed_pipeline)(vbs_device_t *dev, vbs_model_type_t type, char *err_buf, size_t err_cap);
    void *(*compile_primitive_pipeline)(vbs_device_t *dev, vbs_op_kind_t kind, char *err_buf, size_t err_cap);

    bool (*dispatch_linear_tile)(
        vbs_device_t *dev, void *pipeline,
        vbs_buffer_t *weight_private, vbs_buffer_t *weight_stage, size_t copied_bytes,
        vbs_buffer_t *input, vbs_buffer_t *output,
        uint32_t cols, uint32_t rows, uint32_t col_offset, uint32_t row_offset,
        bool accumulate
    );

    bool (*dispatch_primitive)(
        vbs_device_t *dev, void *pipeline,
        const vbs_arg_t *args, size_t n_args,
        size_t grid_x, size_t block_x
    );

    void (*synchronize_device)(vbs_device_t *dev);

    /* PGAE Phase 2 hook, currently NULL in every backend. A real
     * implementation must be a device-side persistent kernel, not
     * a host JIT function. Until one exists, the engine uses
     * Phase 1 and this pointer stays NULL; it is never called
     * blindly. */
    bool (*launch_persistent_kernel)(vbs_device_t *dev, void *pipeline, vbs_buffer_t *control_buffer);
} vbs_backend_ops_t;

/* Published backend tables. (The registry-table names stay as
 * decided by the originality sweep — James's call, not this
 * rewrite's.) */
extern const vbs_backend_ops_t g_vbs_cuda_backend;
extern const vbs_backend_ops_t g_vbs_hip_backend;
extern const vbs_backend_ops_t g_vbs_vulkan_backend;
extern const vbs_backend_ops_t g_vbs_cpu_backend;
extern const vbs_backend_ops_t g_vbs_native_backend;
#if defined(VBS_ENABLE_METAL) && VBS_ENABLE_METAL
extern const vbs_backend_ops_t g_vbs_metal_backend;
#endif

#ifdef __cplusplus
}
#endif

#endif /* VBS_BACKEND_H */
