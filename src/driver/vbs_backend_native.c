/* vbs_backend_native.c — engine backend for Verum's own GPU drivers.
 *
 * Wires the native driver framework (src/driver/vbs_drv.h) behind
 * the engine's canonical backend interface (include/vbs_backend.h).
 * The Turing, RDNA 2, and Alchemist drivers are registered at init (the
 * Alchemist ops are also auto-registered by the framework on first
 * probe; registration is idempotent, so both paths are safe).
 *
 * What works today (on a machine with a supported GPU):
 *   - probe + init + device enumeration,
 *   - buffer alloc/free and host<->device copies (real GEM buffers,
 *     real VRAM/GART, real CPU mappings),
 *   - synchronize (waits for outstanding work).
 *
 * What fails cleanly (never silently succeeds):
 *   - pipeline compilation: there are no SASS kernels yet. Shader
 *     production is a separate workstream (the shader worker derives
 *     SASS encodings from ptxas/nvdisasm per the spec digest §5.4);
 *     until a kernel exists for an op, compile_*_pipeline returns
 *     NULL with a clear error message, and dispatch_* returns false.
 *     The engine falls through to the Vulkan or CPU backend for
 *     those ops.
 *   - init on a machine with no supported GPU: returns false with a
 *     clear reason, and the engine falls back to CPU.
 *
 * Device type: VBS_BACKEND_NATIVE. The engine never branches on
 * device->type (verified: only backends assign it), so the value
 * is display-only.
 */

#include "vbs_backend.h"
#include "vbs_drv.h"
#include "vbs_drv_turing.h"
#include "vbs_drv_rdna2.h"
#include "vbs_drv_alchemist.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static vbs_drv_device_t *g_native_dev;
static vbs_device_t g_native_device;
static bool g_native_ready;
static char g_native_err[256] = "native backend not initialized";

static void set_err(const char *msg) {
    snprintf(g_native_err, sizeof(g_native_err), "%s", msg);
}

const char *vbs_native_last_error(void) {
    return g_native_err;
}

/* ------------------------------------------------------------------ */
/* Lifecycle                                                           */
/* ------------------------------------------------------------------ */

static bool native_init(void) {
    if (g_native_ready) return true;
    if (vbs_drv_turing_register() != VBS_DRV_OK) {
        set_err("native driver: failed to register the Turing driver");
        return false;
    }
    if (vbs_drv_rdna2_register() != VBS_DRV_OK) {
        set_err("native driver: failed to register the RDNA 2 driver");
        return false;
    }
    if (vbs_drv_alchemist_register() != VBS_DRV_OK) {
        set_err("native driver: failed to register the Alchemist driver");
        return false;
    }
    vbs_drv_device_info_t infos[VBS_DRV_MAX_DEVICES];
    size_t count = 0;
    if (vbs_drv_probe_all(infos, VBS_DRV_MAX_DEVICES, &count) != VBS_DRV_OK ||
        count == 0) {
        set_err("native driver: no supported GPU found "
                "(no Turing TU102 on any /dev/dri/renderD* Nouveau node, "
                "no Navi 21 on any /dev/dri/renderD* amdgpu node, "
                "no Alchemist DG2 on any /dev/dri/renderD* xe node)");
        return false;
    }
    vbs_drv_device_t *dev = NULL;
    vbs_drv_status_t open_st = vbs_drv_open(0, &dev);
    if (open_st != VBS_DRV_OK || !dev) {
        /* The failing device may belong to either vendor; report
         * whichever vendor recorded a detailed error. */
        const char *terr = vbs_drv_turing_last_error();
        const char *aerr = vbs_drv_rdna2_last_error();
        if (terr[0] != '\0' && aerr[0] != '\0') {
            snprintf(g_native_err, sizeof(g_native_err),
                     "native driver: failed to open %s (%s; turing: %s; "
                     "rdna2: %s)",
                     infos[0].name, vbs_drv_status_string(open_st), terr,
                     aerr);
        } else if (terr[0] != '\0') {
            snprintf(g_native_err, sizeof(g_native_err),
                     "native driver: failed to open %s (%s)", infos[0].name,
                     terr);
        } else if (aerr[0] != '\0') {
            snprintf(g_native_err, sizeof(g_native_err),
                     "native driver: failed to open %s (%s)", infos[0].name,
                     aerr);
        } else if (vbs_drv_alchemist_last_error()[0] != '\0') {
            snprintf(g_native_err, sizeof(g_native_err),
                     "native driver: failed to open %s (%s)", infos[0].name,
                     vbs_drv_alchemist_last_error());
        } else {
            snprintf(g_native_err, sizeof(g_native_err),
                     "native driver: failed to open %s (%s)", infos[0].name,
                     vbs_drv_status_string(open_st));
        }
        return false;
    }
    g_native_dev = dev;
    memset(&g_native_device, 0, sizeof(g_native_device));
    snprintf(g_native_device.name, sizeof(g_native_device.name), "%s",
             dev->info.name);
    g_native_device.type = VBS_BACKEND_NATIVE;
    g_native_device.max_working_set_bytes = dev->info.memory_bytes;
    g_native_device.max_buffer_bytes = dev->info.memory_bytes;
    g_native_device.internal_handle = dev;
    g_native_ready = true;
    set_err("ok");
    return true;
}

static size_t native_get_device_count(void) {
    return g_native_ready ? 1 : 0;
}

static vbs_device_t *native_get_device(size_t index) {
    if (!g_native_ready || index != 0) return NULL;
    return &g_native_device;
}

/* ------------------------------------------------------------------ */
/* Buffers                                                             */
/* ------------------------------------------------------------------ */

static vbs_buffer_t *native_alloc_buffer(vbs_device_t *dev, size_t bytes,
                                         bool shared) {
    (void)shared; /* every native buffer is host-visible by construction */
    if (!g_native_ready || !dev || dev != &g_native_device || bytes == 0)
        return NULL;
    vbs_drv_buffer_t *db = NULL;
    if (vbs_drv_alloc_buffer(g_native_dev, bytes,
                             VBS_DRV_BUF_HOST_VISIBLE |
                                 VBS_DRV_BUF_DEVICE_LOCAL,
                             &db) != VBS_DRV_OK || !db)
        return NULL;
    vbs_buffer_t *buf = (vbs_buffer_t *)calloc(1, sizeof(vbs_buffer_t));
    if (!buf) {
        vbs_drv_free_buffer(db);
        return NULL;
    }
    buf->internal_handle = db;
    buf->host_ptr = db->host_ptr;
    buf->size = bytes;
    buf->device = dev;
    return buf;
}

static void native_free_buffer(vbs_buffer_t *buf) {
    if (!buf) return;
    vbs_drv_free_buffer((vbs_drv_buffer_t *)buf->internal_handle);
    free(buf);
}

static bool native_copy_to_device(vbs_buffer_t *dst, const void *src,
                                  size_t bytes) {
    if (!dst || !dst->internal_handle || !src || bytes == 0 ||
        bytes > dst->size)
        return false;
    return vbs_drv_copy_to_device((vbs_drv_buffer_t *)dst->internal_handle,
                                  src, bytes) == VBS_DRV_OK;
}

static bool native_copy_to_host(void *dst, const vbs_buffer_t *src,
                                size_t bytes) {
    if (!dst || !src || !src->internal_handle || bytes == 0 ||
        bytes > src->size)
        return false;
    return vbs_drv_copy_to_host(dst,
                                (const vbs_drv_buffer_t *)src->internal_handle,
                                bytes) == VBS_DRV_OK;
}

/* ------------------------------------------------------------------ */
/* Pipelines — no SASS kernels exist yet, so compilation fails         */
/* cleanly with a clear error. See the file header.                    */
/* ------------------------------------------------------------------ */

static const char *op_kind_name(vbs_op_kind_t kind) {
    switch (kind) {
    case VBS_OP_EMBEDDING:   return "embedding";
    case VBS_OP_LINEAR:      return "linear";
    case VBS_OP_NORMALIZE:   return "normalize";
    case VBS_OP_ATTENTION:   return "attention";
    case VBS_OP_ROTARY:      return "rotary";
    case VBS_OP_ACTIVATION:  return "activation";
    case VBS_OP_ELEMENTWISE: return "elementwise";
    case VBS_OP_STATE_SPACE: return "state-space";
    case VBS_OP_CONVOLUTION: return "convolution";
    case VBS_OP_CACHE:       return "cache";
    case VBS_OP_PROJECTION:  return "projection";
    default:                 return "unknown op";
    }
}

static void *native_compile_linear_pipeline(vbs_device_t *dev,
                                            vbs_model_type_t type,
                                            char *err_buf, size_t err_cap) {
    (void)dev;
    if (err_buf && err_cap > 0)
        snprintf(err_buf, err_cap,
                 "native driver: no SASS kernel for linear type %d yet "
                 "(shader production is a separate workstream); "
                 "use the Vulkan or CPU backend",
                 (int)type);
    return NULL;
}

static void *native_compile_embed_pipeline(vbs_device_t *dev,
                                           vbs_model_type_t type,
                                           char *err_buf, size_t err_cap) {
    (void)dev;
    if (err_buf && err_cap > 0)
        snprintf(err_buf, err_cap,
                 "native driver: no SASS kernel for embed type %d yet "
                 "(shader production is a separate workstream); "
                 "use the Vulkan or CPU backend",
                 (int)type);
    return NULL;
}

static void *native_compile_primitive_pipeline(vbs_device_t *dev,
                                               vbs_op_kind_t kind,
                                               char *err_buf,
                                               size_t err_cap) {
    (void)dev;
    if (err_buf && err_cap > 0)
        snprintf(err_buf, err_cap,
                 "native driver: no SASS kernel for %s yet "
                 "(shader production is a separate workstream); "
                 "use the Vulkan or CPU backend",
                 op_kind_name(kind));
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Dispatch — unreachable while compilation fails, but guarded anyway  */
/* ------------------------------------------------------------------ */

static bool native_dispatch_linear_tile(
    vbs_device_t *dev, void *pipeline,
    vbs_buffer_t *weight_private, vbs_buffer_t *weight_stage,
    size_t copied_bytes,
    vbs_buffer_t *input, vbs_buffer_t *output,
    uint32_t cols, uint32_t rows, uint32_t col_offset, uint32_t row_offset,
    bool accumulate) {
    (void)dev; (void)pipeline; (void)weight_private; (void)weight_stage;
    (void)copied_bytes; (void)input; (void)output; (void)cols; (void)rows;
    (void)col_offset; (void)row_offset; (void)accumulate;
    /* No pipeline can exist (compilation fails cleanly), so there is
     * nothing to dispatch. Return false, never silently succeed. */
    return false;
}

static bool native_dispatch_primitive(vbs_device_t *dev, void *pipeline,
                                      const vbs_arg_t *args, size_t n_args,
                                      size_t grid_x, size_t block_x) {
    (void)dev; (void)pipeline; (void)args; (void)n_args;
    (void)grid_x; (void)block_x;
    return false;
}

static void native_synchronize_device(vbs_device_t *dev) {
    if (!g_native_ready || !dev || dev != &g_native_device) return;
    vbs_drv_synchronize(g_native_dev);
}

/* ------------------------------------------------------------------ */
/* Ops table                                                           */
/* ------------------------------------------------------------------ */

const vbs_backend_ops_t g_vbs_native_backend = {
    .init = native_init,
    .get_device_count = native_get_device_count,
    .get_device = native_get_device,
    .alloc_buffer = native_alloc_buffer,
    .free_buffer = native_free_buffer,
    .copy_to_device = native_copy_to_device,
    .copy_to_host = native_copy_to_host,
    .compile_linear_pipeline = native_compile_linear_pipeline,
    .compile_embed_pipeline = native_compile_embed_pipeline,
    .compile_primitive_pipeline = native_compile_primitive_pipeline,
    .dispatch_linear_tile = native_dispatch_linear_tile,
    .dispatch_primitive = native_dispatch_primitive,
    .synchronize_device = native_synchronize_device,
    .launch_persistent_kernel = NULL,
};
