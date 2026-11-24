/* vbs_drv.c — Verum native GPU driver framework: core implementation.
 *
 * Pure C11; the core (registry, VA allocator, command buffers,
 * launch validation) runs on any machine. The vendor-shared Linux
 * helpers at the bottom (ioctl wrapper, GEM/syncobj handling,
 * render-node scanning) are Linux-gated and degrade gracefully
 * elsewhere, so the file still compiles on any platform. The vendor
 * ops tables are the only part that touches a kernel interface
 * beyond those shared helpers.
 *
 * Built-in vendors (compiled into the library) are auto-registered
 * on first probe (see register_builtin_vendors below), so a caller
 * that only uses vbs_drv_probe_all sees every supported GPU without
 * naming vendors. Explicit registration (vbs_drv_<family>_register)
 * stays available and is idempotent, so backends may still register
 * vendors directly — the engine backend registers Turing that way.
 */

#include "vbs_drv.h"
#include "vbs_drv_alchemist.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Linux DRM helpers (at the bottom) need the kernel headers; they
 * are absent on other platforms, where the helpers degrade to
 * no-ops or clean failures. */
#if defined(__linux__)
#  include <errno.h>
#  include <fcntl.h>
#  include <sys/ioctl.h>
#  include <unistd.h>
#  if defined(__has_include)
#    if __has_include(<drm/drm.h>)
#      include <drm/drm.h>
#      define VBS_DRV_HAVE_DRM_H 1
#    endif
#  else
#    include <drm/drm.h>
#    define VBS_DRV_HAVE_DRM_H 1
#  endif
#endif

/* ------------------------------------------------------------------ */
/* Status strings                                                      */
/* ------------------------------------------------------------------ */

const char *vbs_drv_status_string(vbs_drv_status_t st) {
    switch (st) {
    case VBS_DRV_OK:                return "ok";
    case VBS_DRV_ERR_NO_DEVICE:     return "no usable GPU device found";
    case VBS_DRV_ERR_NO_MEMORY:     return "out of memory";
    case VBS_DRV_ERR_NOT_SUPPORTED: return "operation not supported by this driver";
    case VBS_DRV_ERR_IO:            return "kernel interface call failed";
    case VBS_DRV_ERR_INVALID_ARG:   return "invalid argument";
    case VBS_DRV_ERR_TIMEOUT:       return "timed out waiting for the GPU";
    case VBS_DRV_ERR_BUSY:          return "device busy";
    case VBS_DRV_ERR_NOT_READY:     return "device not initialized";
    case VBS_DRV_ERR_HW_FAULT:      return "GPU fault on submitted work";
    default:                        return "unknown driver error";
    }
}

/* ------------------------------------------------------------------ */
/* VA allocator — first-fit free list with coalescing                  */
/* ------------------------------------------------------------------ */

static bool is_pow2_u64(uint64_t v) {
    return v != 0 && (v & (v - 1)) == 0;
}

static uint64_t align_up_u64(uint64_t v, uint64_t align) {
    /* align is a power of two (checked by callers). */
    return (v + align - 1) & ~(align - 1);
}

vbs_drv_status_t vbs_drv_va_init(vbs_drv_va_space_t *va,
                                 uint64_t base, uint64_t size) {
    if (!va || size == 0 || base > UINT64_MAX - size)
        return VBS_DRV_ERR_INVALID_ARG;
    memset(va, 0, sizeof(*va));
    va->base = base;
    va->size = size;
    va->free_blocks[0].addr = base;
    va->free_blocks[0].size = size;
    va->free_count = 1;
    va->ready = true;
    return VBS_DRV_OK;
}

vbs_drv_status_t vbs_drv_va_alloc(vbs_drv_va_space_t *va, uint64_t size,
                                  uint64_t align, uint64_t *out_addr) {
    if (!va || !va->ready || !out_addr || size == 0 || !is_pow2_u64(align))
        return VBS_DRV_ERR_INVALID_ARG;
    for (size_t i = 0; i < va->free_count; i++) {
        uint64_t blk_addr = va->free_blocks[i].addr;
        uint64_t blk_size = va->free_blocks[i].size;
        uint64_t aligned = align_up_u64(blk_addr, align);
        if (aligned < blk_addr) continue; /* alignment overflowed */
        uint64_t head = aligned - blk_addr; /* bytes skipped for alignment */
        if (head > blk_size) continue;
        if (blk_size - head < size) continue; /* does not fit */
        uint64_t tail_addr = aligned + size;
        uint64_t tail_size = blk_size - head - size;
        *out_addr = aligned;
        if (head == 0 && tail_size == 0) {
            /* Block consumed exactly: remove it. */
            memmove(&va->free_blocks[i], &va->free_blocks[i + 1],
                    (va->free_count - i - 1) * sizeof(va->free_blocks[0]));
            va->free_count--;
        } else if (head == 0) {
            /* Allocation at the block start: shrink from the front. */
            va->free_blocks[i].addr = tail_addr;
            va->free_blocks[i].size = tail_size;
        } else if (tail_size == 0) {
            /* Allocation at the block end: shrink from the back. */
            va->free_blocks[i].size = head;
        } else {
            /* Allocation in the middle: split into head + tail. */
            if (va->free_count >= VBS_DRV_VA_MAX_BLOCKS)
                return VBS_DRV_ERR_NO_MEMORY;
            memmove(&va->free_blocks[i + 2], &va->free_blocks[i + 1],
                    (va->free_count - i - 1) * sizeof(va->free_blocks[0]));
            va->free_blocks[i].addr = blk_addr;
            va->free_blocks[i].size = head;
            va->free_blocks[i + 1].addr = tail_addr;
            va->free_blocks[i + 1].size = tail_size;
            va->free_count++;
        }
        return VBS_DRV_OK;
    }
    return VBS_DRV_ERR_NO_MEMORY;
}

vbs_drv_status_t vbs_drv_va_free(vbs_drv_va_space_t *va,
                                 uint64_t addr, uint64_t size) {
    if (!va || !va->ready || size == 0 || addr < va->base ||
        addr - va->base > va->size || size > va->size - (addr - va->base))
        return VBS_DRV_ERR_INVALID_ARG;
    /* The range must not overlap any existing free block (a double
     * free or a range that was never allocated). */
    for (size_t i = 0; i < va->free_count; i++) {
        uint64_t b0 = va->free_blocks[i].addr;
        uint64_t b1 = b0 + va->free_blocks[i].size;
        if (addr < b1 && b0 < addr + size)
            return VBS_DRV_ERR_INVALID_ARG;
    }
    if (va->free_count >= VBS_DRV_VA_MAX_BLOCKS)
        return VBS_DRV_ERR_NO_MEMORY;
    /* Insert sorted by address, then coalesce with neighbors. */
    size_t at = va->free_count;
    while (at > 0 && va->free_blocks[at - 1].addr > addr) {
        va->free_blocks[at] = va->free_blocks[at - 1];
        at--;
    }
    va->free_blocks[at].addr = addr;
    va->free_blocks[at].size = size;
    va->free_count++;
    /* Merge with the next block. */
    if (at + 1 < va->free_count &&
        va->free_blocks[at].addr + va->free_blocks[at].size ==
            va->free_blocks[at + 1].addr) {
        va->free_blocks[at].size += va->free_blocks[at + 1].size;
        memmove(&va->free_blocks[at + 1], &va->free_blocks[at + 2],
                (va->free_count - at - 2) * sizeof(va->free_blocks[0]));
        va->free_count--;
    }
    /* Merge with the previous block. */
    if (at > 0 &&
        va->free_blocks[at - 1].addr + va->free_blocks[at - 1].size ==
            va->free_blocks[at].addr) {
        va->free_blocks[at - 1].size += va->free_blocks[at].size;
        memmove(&va->free_blocks[at], &va->free_blocks[at + 1],
                (va->free_count - at - 1) * sizeof(va->free_blocks[0]));
        va->free_count--;
    }
    return VBS_DRV_OK;
}

/* ------------------------------------------------------------------ */
/* Command-buffer container                                            */
/* ------------------------------------------------------------------ */

vbs_drv_status_t vbs_drv_cmdbuf_init(vbs_drv_cmdbuf_t *cb,
                                     size_t capacity_words) {
    if (!cb || capacity_words == 0) return VBS_DRV_ERR_INVALID_ARG;
    cb->words = (uint32_t *)calloc(capacity_words, sizeof(uint32_t));
    if (!cb->words) return VBS_DRV_ERR_NO_MEMORY;
    cb->count = 0;
    cb->capacity = capacity_words;
    return VBS_DRV_OK;
}

void vbs_drv_cmdbuf_free(vbs_drv_cmdbuf_t *cb) {
    if (!cb) return;
    free(cb->words);
    cb->words = NULL;
    cb->count = 0;
    cb->capacity = 0;
}

vbs_drv_status_t vbs_drv_cmdbuf_push(vbs_drv_cmdbuf_t *cb, uint32_t word) {
    if (!cb || !cb->words) return VBS_DRV_ERR_INVALID_ARG;
    if (cb->count >= cb->capacity) return VBS_DRV_ERR_NO_MEMORY;
    cb->words[cb->count++] = word;
    return VBS_DRV_OK;
}

vbs_drv_status_t vbs_drv_cmdbuf_push_many(vbs_drv_cmdbuf_t *cb,
                                          const uint32_t *words, size_t n) {
    if (!cb || !cb->words || (n > 0 && !words))
        return VBS_DRV_ERR_INVALID_ARG;
    if (cb->capacity - cb->count < n) return VBS_DRV_ERR_NO_MEMORY;
    memcpy(&cb->words[cb->count], words, n * sizeof(uint32_t));
    cb->count += n;
    return VBS_DRV_OK;
}

void vbs_drv_cmdbuf_reset(vbs_drv_cmdbuf_t *cb) {
    if (cb) cb->count = 0;
}

/* ------------------------------------------------------------------ */
/* Vendor registry and device table                                    */
/* ------------------------------------------------------------------ */

static const vbs_drv_ops_t *g_vendors[VBS_DRV_MAX_VENDORS];
static size_t g_vendor_count;

typedef struct {
    const vbs_drv_ops_t *ops;
    size_t vendor_index; /* index within that vendor's probe results */
    vbs_drv_device_info_t info;
} vbs_drv_found_t;

static vbs_drv_found_t g_found[VBS_DRV_MAX_DEVICES];
static size_t g_found_count;

vbs_drv_status_t vbs_drv_register_ops(const vbs_drv_ops_t *ops) {
    if (!ops || !ops->name || !ops->probe || !ops->open || !ops->close ||
        !ops->alloc_buffer || !ops->free_buffer || !ops->submit ||
        !ops->fence_wait || !ops->fence_free || !ops->launch ||
        !ops->synchronize)
        return VBS_DRV_ERR_INVALID_ARG;
    for (size_t i = 0; i < g_vendor_count; i++)
        if (g_vendors[i] == ops) return VBS_DRV_OK; /* idempotent */
    if (g_vendor_count >= VBS_DRV_MAX_VENDORS)
        return VBS_DRV_ERR_NO_MEMORY;
    g_vendors[g_vendor_count++] = ops;
    return VBS_DRV_OK;
}

/* Auto-register the vendors compiled into this library, once.
 * Each vendor adds one line here when it lands (Alchemist added
 * 2026-11-17; RDNA 2 adds its line the same way). Registration is
 * idempotent, so an explicit vbs_drv_<family>_register() call
 * before or after the first probe is harmless. */
static bool g_builtins_registered;

static void register_builtin_vendors(void) {
    if (g_builtins_registered) return;
    g_builtins_registered = true;
    vbs_drv_alchemist_register();
}

vbs_drv_status_t vbs_drv_probe_all(vbs_drv_device_info_t *infos, size_t cap,
                                   size_t *count) {
    if (!infos || !count || cap == 0) return VBS_DRV_ERR_INVALID_ARG;
    register_builtin_vendors();
    g_found_count = 0;
    *count = 0;
    for (size_t v = 0; v < g_vendor_count; v++) {
        size_t room = VBS_DRV_MAX_DEVICES - g_found_count;
        if (room == 0) break;
        vbs_drv_device_info_t tmp[VBS_DRV_MAX_DEVICES];
        size_t n = 0;
        vbs_drv_status_t st = g_vendors[v]->probe(tmp, room, &n);
        if (st != VBS_DRV_OK) continue; /* one vendor failing must not
                                         * hide another vendor's devices */
        for (size_t i = 0; i < n && g_found_count < VBS_DRV_MAX_DEVICES; i++) {
            g_found[g_found_count].ops = g_vendors[v];
            g_found[g_found_count].vendor_index = i;
            g_found[g_found_count].info = tmp[i];
            g_found_count++;
        }
    }
    size_t out = g_found_count < cap ? g_found_count : cap;
    for (size_t i = 0; i < out; i++) infos[i] = g_found[i].info;
    *count = out;
    return VBS_DRV_OK;
}

vbs_drv_status_t vbs_drv_open(size_t index, vbs_drv_device_t **dev_out) {
    if (!dev_out) return VBS_DRV_ERR_INVALID_ARG;
    *dev_out = NULL;
    if (index >= g_found_count) return VBS_DRV_ERR_NO_DEVICE;
    vbs_drv_device_t *dev =
        (vbs_drv_device_t *)calloc(1, sizeof(vbs_drv_device_t));
    if (!dev) return VBS_DRV_ERR_NO_MEMORY;
    dev->ops = g_found[index].ops;
    dev->info = g_found[index].info;
    vbs_drv_status_t st = dev->ops->open(g_found[index].vendor_index, dev);
    if (st != VBS_DRV_OK) {
        dev->ops->close(dev);
        free(dev);
        return st;
    }
    *dev_out = dev;
    return VBS_DRV_OK;
}

void vbs_drv_close(vbs_drv_device_t *dev) {
    if (!dev) return;
    if (dev->ops && dev->ops->close) dev->ops->close(dev);
    free(dev);
}

/* ------------------------------------------------------------------ */
/* Buffers                                                             */
/* ------------------------------------------------------------------ */

vbs_drv_status_t vbs_drv_alloc_buffer(vbs_drv_device_t *dev, size_t size,
                                      uint32_t flags, vbs_drv_buffer_t **out) {
    if (!dev || !dev->ops || !out || size == 0)
        return VBS_DRV_ERR_INVALID_ARG;
    *out = NULL;
    vbs_drv_buffer_t *buf =
        (vbs_drv_buffer_t *)calloc(1, sizeof(vbs_drv_buffer_t));
    if (!buf) return VBS_DRV_ERR_NO_MEMORY;
    buf->device = dev;
    vbs_drv_status_t st = dev->ops->alloc_buffer(dev, size, flags, buf);
    if (st != VBS_DRV_OK) {
        free(buf);
        return st;
    }
    *out = buf;
    return VBS_DRV_OK;
}

void vbs_drv_free_buffer(vbs_drv_buffer_t *buf) {
    if (!buf) return;
    if (buf->device && buf->device->ops && buf->device->ops->free_buffer)
        buf->device->ops->free_buffer(buf);
    free(buf);
}

vbs_drv_status_t vbs_drv_map_buffer(vbs_drv_buffer_t *buf) {
    if (!buf || !buf->device || !buf->device->ops)
        return VBS_DRV_ERR_INVALID_ARG;
    if (buf->device->ops->map_buffer)
        return buf->device->ops->map_buffer(buf);
    return buf->host_ptr ? VBS_DRV_OK : VBS_DRV_ERR_NOT_SUPPORTED;
}

void vbs_drv_unmap_buffer(vbs_drv_buffer_t *buf) {
    if (!buf || !buf->device || !buf->device->ops) return;
    if (buf->device->ops->unmap_buffer) buf->device->ops->unmap_buffer(buf);
}

vbs_drv_status_t vbs_drv_cpu_begin(vbs_drv_buffer_t *buf, bool for_write) {
    if (!buf || !buf->device || !buf->device->ops)
        return VBS_DRV_ERR_INVALID_ARG;
    if (buf->device->ops->cpu_begin)
        return buf->device->ops->cpu_begin(buf, for_write);
    return VBS_DRV_OK;
}

void vbs_drv_cpu_end(vbs_drv_buffer_t *buf) {
    if (!buf || !buf->device || !buf->device->ops) return;
    if (buf->device->ops->cpu_end) buf->device->ops->cpu_end(buf);
}

vbs_drv_status_t vbs_drv_copy_to_device(vbs_drv_buffer_t *dst,
                                        const void *src, size_t bytes) {
    if (!dst || !src || bytes == 0 || bytes > dst->size)
        return VBS_DRV_ERR_INVALID_ARG;
    vbs_drv_status_t st = vbs_drv_map_buffer(dst);
    if (st != VBS_DRV_OK) return st;
    if (!dst->host_ptr) return VBS_DRV_ERR_NOT_SUPPORTED;
    st = vbs_drv_cpu_begin(dst, true);
    if (st != VBS_DRV_OK) return st;
    memcpy(dst->host_ptr, src, bytes);
    vbs_drv_cpu_end(dst);
    return VBS_DRV_OK;
}

vbs_drv_status_t vbs_drv_copy_to_host(void *dst,
                                      const vbs_drv_buffer_t *src,
                                      size_t bytes) {
    if (!dst || !src || bytes == 0 || bytes > src->size)
        return VBS_DRV_ERR_INVALID_ARG;
    /* The const cast is safe: map/cpu_begin do not modify buffer
     * contents, only the vendor's coherency state. */
    vbs_drv_buffer_t *mut = (vbs_drv_buffer_t *)src;
    vbs_drv_status_t st = vbs_drv_map_buffer(mut);
    if (st != VBS_DRV_OK) return st;
    if (!src->host_ptr) return VBS_DRV_ERR_NOT_SUPPORTED;
    st = vbs_drv_cpu_begin(mut, false);
    if (st != VBS_DRV_OK) return st;
    memcpy(dst, src->host_ptr, bytes);
    vbs_drv_cpu_end(mut);
    return VBS_DRV_OK;
}

/* ------------------------------------------------------------------ */
/* Submission, fences, launch, synchronize                             */
/* ------------------------------------------------------------------ */

vbs_drv_status_t vbs_drv_submit(vbs_drv_device_t *dev,
                                const vbs_drv_cmdbuf_t *cb,
                                vbs_drv_fence_t *fence_out) {
    if (!dev || !dev->ops || !cb || !cb->words || cb->count == 0 ||
        !fence_out)
        return VBS_DRV_ERR_INVALID_ARG;
    *fence_out = VBS_DRV_FENCE_NONE;
    return dev->ops->submit(dev, cb, fence_out);
}

vbs_drv_status_t vbs_drv_fence_wait(vbs_drv_device_t *dev,
                                    vbs_drv_fence_t fence,
                                    uint32_t timeout_ms) {
    if (!dev || !dev->ops) return VBS_DRV_ERR_INVALID_ARG;
    if (fence == VBS_DRV_FENCE_NONE) return VBS_DRV_OK; /* nothing to wait on */
    return dev->ops->fence_wait(dev, fence, timeout_ms);
}

void vbs_drv_fence_free(vbs_drv_device_t *dev, vbs_drv_fence_t fence) {
    if (!dev || !dev->ops || fence == VBS_DRV_FENCE_NONE) return;
    dev->ops->fence_free(dev, fence);
}

vbs_drv_status_t vbs_drv_submit_and_wait(vbs_drv_device_t *dev,
                                         const vbs_drv_cmdbuf_t *cb,
                                         uint32_t timeout_ms) {
    vbs_drv_fence_t fence = VBS_DRV_FENCE_NONE;
    vbs_drv_status_t st = vbs_drv_submit(dev, cb, &fence);
    if (st != VBS_DRV_OK) return st;
    st = vbs_drv_fence_wait(dev, fence, timeout_ms);
    vbs_drv_fence_free(dev, fence);
    return st;
}

/* Shared launch validation: every vendor gets the same checks, so a
 * bad description fails identically on every driver. The shape
 * checks and the semaphore check are separate functions because
 * one encoder (RDNA 2) declines semaphores outright (NOT_SUPPORTED)
 * rather than validating them, while the others validate them. */
vbs_drv_status_t vbs_drv_validate_launch_desc(
    const vbs_drv_launch_desc_t *desc) {
    if (!desc || !desc->code || desc->code->gpu_va == 0 ||
        desc->code_size == 0 || desc->register_count == 0 ||
        desc->grid_x == 0 || desc->grid_y == 0 || desc->grid_z == 0 ||
        desc->block_x == 0 || desc->block_y == 0 || desc->block_z == 0)
        return VBS_DRV_ERR_INVALID_ARG;
    if (desc->code_offset + desc->code_size > desc->code->size)
        return VBS_DRV_ERR_INVALID_ARG;
    if (desc->const_buf_count > VBS_DRV_MAX_CONST_BUFFERS)
        return VBS_DRV_ERR_INVALID_ARG;
    for (uint32_t i = 0; i < desc->const_buf_count; i++) {
        if (!desc->const_bufs[i] || desc->const_bufs[i]->gpu_va == 0)
            return VBS_DRV_ERR_INVALID_ARG;
    }
    return VBS_DRV_OK;
}

vbs_drv_status_t vbs_drv_validate_launch_semaphore(
    const vbs_drv_launch_desc_t *desc) {
    if (!desc) return VBS_DRV_ERR_INVALID_ARG;
    if (desc->use_semaphore) {
        if (!desc->sem_buf || desc->sem_buf->gpu_va == 0 ||
            desc->sem_offset + sizeof(uint32_t) > desc->sem_buf->size)
            return VBS_DRV_ERR_INVALID_ARG;
    }
    return VBS_DRV_OK;
}

vbs_drv_status_t vbs_drv_launch_buffers_on_device(
    const vbs_drv_device_t *dev, const vbs_drv_launch_desc_t *desc) {
    if (!dev || !desc || !desc->code) return VBS_DRV_ERR_INVALID_ARG;
    if (desc->code->device != dev) return VBS_DRV_ERR_INVALID_ARG;
    for (uint32_t i = 0; i < desc->const_buf_count; i++) {
        if (!desc->const_bufs[i] || desc->const_bufs[i]->device != dev)
            return VBS_DRV_ERR_INVALID_ARG;
    }
    if (desc->use_semaphore &&
        (!desc->sem_buf || desc->sem_buf->device != dev))
        return VBS_DRV_ERR_INVALID_ARG;
    return VBS_DRV_OK;
}

vbs_drv_status_t vbs_drv_launch(vbs_drv_device_t *dev,
                                const vbs_drv_launch_desc_t *desc) {
    if (!dev || !dev->ops) return VBS_DRV_ERR_INVALID_ARG;
    vbs_drv_status_t st = vbs_drv_validate_launch_desc(desc);
    if (st != VBS_DRV_OK) return st;
    st = vbs_drv_validate_launch_semaphore(desc);
    if (st != VBS_DRV_OK) return st;
    return dev->ops->launch(dev, desc);
}

vbs_drv_status_t vbs_drv_synchronize(vbs_drv_device_t *dev) {
    if (!dev || !dev->ops) return VBS_DRV_ERR_INVALID_ARG;
    return dev->ops->synchronize(dev);
}

/* ------------------------------------------------------------------ */
/* Shared helpers for the vendor drivers                               */
/* ------------------------------------------------------------------ */

void vbs_drv_note_error(char *buf, size_t cap, const char *what,
                        int errnum) {
    if (!buf || cap == 0) return;
    snprintf(buf, cap, "%s: %s", what ? what : "?",
             errnum ? strerror(errnum) : "unknown error");
}

void vbs_drv_note_error_msg(char *buf, size_t cap, const char *msg) {
    if (!buf || cap == 0) return;
    snprintf(buf, cap, "%s", msg ? msg : "?");
}

int vbs_drv_ioctl(int fd, unsigned long req, void *arg) {
#if defined(__linux__)
    int r;
    do {
        r = ioctl(fd, req, arg);
    } while (r < 0 && errno == EINTR);
    return r < 0 ? -errno : 0;
#else
    (void)fd; (void)req; (void)arg;
    return -1; /* no kernel interface on this platform */
#endif
}

void vbs_drv_gem_close(int fd, uint32_t handle) {
    if (fd < 0 || handle == 0) return;
#if defined(VBS_DRV_HAVE_DRM_H)
    struct drm_gem_close gc;
    memset(&gc, 0, sizeof(gc));
    gc.handle = handle;
    vbs_drv_ioctl(fd, DRM_IOCTL_GEM_CLOSE, &gc);
#endif
}

vbs_drv_status_t vbs_drv_drop_failed_buffer(vbs_drv_buffer_t *buf,
                                            void *vendor_shell,
                                            vbs_drv_status_t st) {
    free(vendor_shell);
    buf->vendor = NULL;
    return st;
}

vbs_drv_status_t vbs_drv_syncobj_create(int fd, uint32_t *handle_out,
                                        char *err_buf, size_t err_cap) {
    if (!handle_out) return VBS_DRV_ERR_INVALID_ARG;
#if defined(VBS_DRV_HAVE_DRM_H)
    struct drm_syncobj_create sc;
    memset(&sc, 0, sizeof(sc));
    sc.flags = 0;
    int r = vbs_drv_ioctl(fd, DRM_IOCTL_SYNCOBJ_CREATE, &sc);
    if (r != 0) {
        vbs_drv_note_error(err_buf, err_cap, "DRM_IOCTL_SYNCOBJ_CREATE",
                           -r);
        return VBS_DRV_ERR_IO;
    }
    *handle_out = sc.handle;
    return VBS_DRV_OK;
#else
    (void)fd;
    vbs_drv_note_error_msg(err_buf, err_cap,
                           "syncobj not available on this platform");
    return VBS_DRV_ERR_NOT_SUPPORTED;
#endif
}

vbs_drv_status_t vbs_drv_syncobj_wait(int fd, uint32_t handle,
                                      uint32_t timeout_ms,
                                      char *err_buf, size_t err_cap) {
#if defined(VBS_DRV_HAVE_DRM_H)
    struct drm_syncobj_wait sw;
    memset(&sw, 0, sizeof(sw));
    uint32_t h = handle;
    sw.handles = (uint64_t)(uintptr_t)&h;
    sw.count_handles = 1;
    sw.flags = 0;
    /* timeout_ms == 0 waits forever (INT64_MAX ns); otherwise
     * convert ms to ns. */
    sw.timeout_nsec = timeout_ms == 0
        ? (int64_t)0x7fffffffffffffffLL
        : (int64_t)timeout_ms * 1000000LL;
    int r = vbs_drv_ioctl(fd, DRM_IOCTL_SYNCOBJ_WAIT, &sw);
    if (r == -ETIME) return VBS_DRV_ERR_TIMEOUT;
    if (r != 0) {
        vbs_drv_note_error(err_buf, err_cap, "DRM_IOCTL_SYNCOBJ_WAIT",
                           -r);
        return VBS_DRV_ERR_IO;
    }
    return VBS_DRV_OK;
#else
    (void)fd; (void)handle; (void)timeout_ms;
    vbs_drv_note_error_msg(err_buf, err_cap,
                           "syncobj not available on this platform");
    return VBS_DRV_ERR_NOT_SUPPORTED;
#endif
}

void vbs_drv_syncobj_destroy(int fd, uint32_t handle) {
    if (fd < 0 || handle == 0) return;
#if defined(VBS_DRV_HAVE_DRM_H)
    struct drm_syncobj_destroy sd;
    memset(&sd, 0, sizeof(sd));
    sd.handle = handle;
    vbs_drv_ioctl(fd, DRM_IOCTL_SYNCOBJ_DESTROY, &sd);
#endif
}

/* Platform-fallback ops: every operation reports "no device". One
 * shared implementation serves every family; only the `name`
 * string differs per table. */
static vbs_drv_status_t unavailable_probe(vbs_drv_device_info_t *infos,
                                          size_t cap, size_t *count) {
    (void)infos; (void)cap;
    if (!count) return VBS_DRV_ERR_INVALID_ARG;
    *count = 0; /* no device can exist without the kernel interface */
    return VBS_DRV_OK;
}

static vbs_drv_status_t unavailable_open(size_t index,
                                         vbs_drv_device_t *dev) {
    (void)index; (void)dev;
    return VBS_DRV_ERR_NO_DEVICE;
}

static void unavailable_close(vbs_drv_device_t *dev) { (void)dev; }

static vbs_drv_status_t unavailable_alloc(vbs_drv_device_t *dev,
                                          size_t size, uint32_t flags,
                                          vbs_drv_buffer_t *buf) {
    (void)dev; (void)size; (void)flags; (void)buf;
    return VBS_DRV_ERR_NO_DEVICE;
}

static void unavailable_free_buf(vbs_drv_buffer_t *buf) { (void)buf; }

static vbs_drv_status_t unavailable_submit(vbs_drv_device_t *dev,
                                           const vbs_drv_cmdbuf_t *cb,
                                           vbs_drv_fence_t *fence_out) {
    (void)dev; (void)cb; (void)fence_out;
    return VBS_DRV_ERR_NO_DEVICE;
}

static vbs_drv_status_t unavailable_fence_wait(vbs_drv_device_t *dev,
                                               vbs_drv_fence_t fence,
                                               uint32_t timeout_ms) {
    (void)dev; (void)fence; (void)timeout_ms;
    return VBS_DRV_ERR_NO_DEVICE;
}

static void unavailable_fence_free(vbs_drv_device_t *dev,
                                   vbs_drv_fence_t fence) {
    (void)dev; (void)fence;
}

static vbs_drv_status_t unavailable_launch(vbs_drv_device_t *dev,
                                           const vbs_drv_launch_desc_t *desc) {
    (void)dev; (void)desc;
    return VBS_DRV_ERR_NO_DEVICE;
}

static vbs_drv_status_t unavailable_sync(vbs_drv_device_t *dev) {
    (void)dev;
    return VBS_DRV_ERR_NO_DEVICE;
}

static vbs_drv_ops_t g_unavailable_ops[VBS_DRV_MAX_VENDORS];
static size_t g_unavailable_count;

const vbs_drv_ops_t *vbs_drv_unavailable_ops(const char *family) {
    if (!family) return NULL;
    for (size_t i = 0; i < g_unavailable_count; i++)
        if (strcmp(g_unavailable_ops[i].name, family) == 0)
            return &g_unavailable_ops[i];
    if (g_unavailable_count >= VBS_DRV_MAX_VENDORS) return NULL;
    vbs_drv_ops_t *ops = &g_unavailable_ops[g_unavailable_count++];
    memset(ops, 0, sizeof(*ops));
    ops->name = family;
    ops->probe = unavailable_probe;
    ops->open = unavailable_open;
    ops->close = unavailable_close;
    ops->alloc_buffer = unavailable_alloc;
    ops->free_buffer = unavailable_free_buf;
    ops->map_buffer = NULL;
    ops->unmap_buffer = NULL;
    ops->cpu_begin = NULL;
    ops->cpu_end = NULL;
    ops->submit = unavailable_submit;
    ops->fence_wait = unavailable_fence_wait;
    ops->fence_free = unavailable_fence_free;
    ops->launch = unavailable_launch;
    ops->synchronize = unavailable_sync;
    return ops;
}

vbs_drv_status_t vbs_drv_scan_render_nodes(
    int first_node, int node_count, const char *driver_name,
    vbs_drv_node_check_t check, void *ctx,
    vbs_drv_device_info_t *infos, size_t cap, size_t *count,
    char (*node_paths)[VBS_DRV_NODE_PATH_LEN]) {
    if (!infos || !count || cap == 0 || !driver_name || !check ||
        !node_paths)
        return VBS_DRV_ERR_INVALID_ARG;
    *count = 0;
#if defined(VBS_DRV_HAVE_DRM_H)
    size_t found = 0;
    for (int n = 0; n < node_count; n++) {
        if (found >= cap || found >= VBS_DRV_MAX_DEVICES) break;
        char path[VBS_DRV_NODE_PATH_LEN];
        snprintf(path, sizeof(path), "/dev/dri/renderD%d",
                 first_node + n);
        int fd = open(path, O_RDWR);
        if (fd < 0) continue; /* node absent or not accessible */
        /* SAFETY: never issue a vendor ioctl to a node before
         * confirming it belongs to that vendor's kernel driver.
         * Ioctl numbers are per-driver, so sending one to the
         * wrong driver could do something unintended.
         * DRM_IOCTL_VERSION is generic and safe on every node. */
        char drv_name[32];
        struct drm_version ver;
        memset(&ver, 0, sizeof(ver));
        memset(drv_name, 0, sizeof(drv_name));
        ver.name_len = sizeof(drv_name);
        ver.name = drv_name;
        if (vbs_drv_ioctl(fd, DRM_IOCTL_VERSION, &ver) != 0 ||
            strcmp(drv_name, driver_name) != 0) {
            close(fd);
            continue;
        }
        vbs_drv_device_info_t *info = &infos[found];
        memset(info, 0, sizeof(*info));
        if (!check(fd, info, ctx)) {
            close(fd);
            continue;
        }
        snprintf(node_paths[found], VBS_DRV_NODE_PATH_LEN, "%s", path);
        found++;
        close(fd);
    }
    *count = found;
#endif
    return VBS_DRV_OK;
}
