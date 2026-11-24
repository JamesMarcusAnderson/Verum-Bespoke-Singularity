/* vbs_drv_turing.c — Turing driver: Nouveau DRM userspace layer.
 *
 * Talks to the GPU through the Nouveau DRM UAPI (/dev/dri/renderD*)
 * only. The kernel owns bring-up, firmware, channels, page tables,
 * and VRAM management; this layer owns the userspace side (digest
 * §8): open the render node, VM_INIT, channel alloc with the compute
 * subchannel, GEM buffers + VM_BIND, pushbuffer construction (via
 * the encoder), EXEC submission, syncobj waits.
 *
 * All ioctl numbers, struct fields, and domain bits come from the
 * spec digest §2 (source S7, include/uapi/drm/nouveau_drm.h) and are
 * come from the system UAPI header <drm/nouveau_drm.h> — included,
 * never vendored.
 *
 * Platform: Linux only. On any other platform (or a Linux system
 * without the Nouveau UAPI headers) this file compiles to the
 * framework's fallback ops table that reports "no device", so the top-level build keeps
 * working on macOS / Windows / MSVC.
 *
 * Originality: written from the digest's documented facts. The code
 * is our own — nothing here comes from Nouveau, Mesa, libdrm, or any
 * other implementation.
 */

#include "vbs_drv_turing.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Platform gate                                                       */
/* ------------------------------------------------------------------ */

#if defined(__linux__)
#  if defined(__has_include)
#    if __has_include(<drm/nouveau_drm.h>) && __has_include(<drm/drm.h>)
#      define VBS_DRV_TURING_HAVE_UAPI 1
#    endif
#  else
#    define VBS_DRV_TURING_HAVE_UAPI 1
#  endif
#endif

#ifdef VBS_DRV_TURING_HAVE_UAPI
/* ---- Real implementation: Linux + Nouveau DRM UAPI ---------------- */

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <drm/drm.h>
#include <drm/nouveau_drm.h>

/* Last detailed failure message (single-threaded use, like the rest
 * of v1). */
static char g_last_error[256];

const char *vbs_drv_turing_last_error(void) { return g_last_error; }

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */

/* First render node is renderD128 (digest §2.1; S7 drm-uapi.rst).
 * Scan 16 nodes; a machine with more GPUs than that is out of scope. */
#define VBS_DRV_TURING_FIRST_RENDER_NODE 128
#define VBS_DRV_TURING_MAX_RENDER_NODES  16

/* VA-space split (a driver choice, not a hardware fact): the kernel
 * manages the low 4 GiB; the userspace driver manages the rest of
 * the 47-bit GPU VA space (digest §1 MMU width; §2.4 VM_INIT split). */
#define VBS_DRV_TURING_KERNEL_VA_SIZE (1ULL << 32)
#define VBS_DRV_TURING_UMD_VA_BASE    (1ULL << 32)
#define VBS_DRV_TURING_UMD_VA_SIZE \
    ((1ULL << VBS_DRV_TURING_VA_BITS) - VBS_DRV_TURING_UMD_VA_BASE)

/* Pushbuffer / QMD persistent buffers (created at open). */
#define VBS_DRV_TURING_PUSHBUF_BYTES (64u * 1024u)
#define VBS_DRV_TURING_QMD_BUF_BYTES 4096u

/* Launch completion timeout: a single kernel launch that runs over
 * a minute is treated as hung rather than hanging the engine
 * forever. */
#define VBS_DRV_TURING_LAUNCH_TIMEOUT_MS 60000u

/* Subchannel the compute class is bound to (0..7; digest §3.2). */
#define VBS_DRV_TURING_COMPUTE_SUBCHAN 0u

/* ------------------------------------------------------------------ */
/* Turing PCI device ids — digest §1, S9 "Compatible GPUs"            */
/* ------------------------------------------------------------------ */

typedef struct {
    uint16_t pci_device_id;
    const char *name;
} turing_pci_entry_t;

static const turing_pci_entry_t g_turing_pci_ids[] = {
    { 0x1E02, "TITAN RTX" },
    { 0x1E04, "GeForce RTX 2080 Ti" },
    { 0x1E07, "GeForce RTX 2080 Ti" },
    { 0x1E30, "Quadro RTX 6000/8000" }, /* shared id, see S9 table */
    { 0x1E36, "Quadro RTX 6000" },
    { 0x1E78, "Quadro RTX 8000" },
};

static const char *turing_pci_name(uint16_t id) {
    for (size_t i = 0; i < sizeof(g_turing_pci_ids) / sizeof(g_turing_pci_ids[0]); i++)
        if (g_turing_pci_ids[i].pci_device_id == id)
            return g_turing_pci_ids[i].name;
    return NULL;
}

/* Node paths found by the last probe, indexed by vendor-local index. */
static char g_node_paths[VBS_DRV_MAX_DEVICES][VBS_DRV_NODE_PATH_LEN];

/* ------------------------------------------------------------------ */
/* Small DRM helpers                                                   */
/* ------------------------------------------------------------------ */

static int turing_getparam(int fd, uint64_t param, uint64_t *value) {
    struct drm_nouveau_getparam gp;
    memset(&gp, 0, sizeof(gp));
    gp.param = param;
    int r = vbs_drv_ioctl(fd, DRM_IOCTL_NOUVEAU_GETPARAM, &gp);
    if (r == 0) *value = gp.value;
    return r;
}

/* ------------------------------------------------------------------ */
/* Probe                                                               */
/* ------------------------------------------------------------------ */

/* Node check for the framework's render-node scan: fd is an open
 * Nouveau node (the framework already confirmed the driver name).
 * Query the PCI device id (digest §2.2;
 * NOUVEAU_GETPARAM_PCI_DEVICE = 4) and match the Turing TU102 list
 * (digest §1; S9). Fills *info and returns true for a device this
 * driver handles. */
static bool turing_check_node(int fd, vbs_drv_device_info_t *info,
                              void *ctx) {
    (void)ctx;
    uint64_t pci_device = 0;
    if (turing_getparam(fd, NOUVEAU_GETPARAM_PCI_DEVICE, &pci_device) != 0)
        return false;
    const char *name = turing_pci_name((uint16_t)pci_device);
    if (!name)
        return false; /* Nouveau GPU, but not a TU102 we drive */
    snprintf(info->name, sizeof(info->name), "%s (Turing TU102, sm_75)",
             name);
    uint64_t v = 0;
    if (turing_getparam(fd, NOUVEAU_GETPARAM_PCI_VENDOR, &v) == 0)
        info->pci_vendor_id = (uint16_t)v;
    info->pci_device_id = (uint16_t)pci_device;
    if (turing_getparam(fd, NOUVEAU_GETPARAM_CHIPSET_ID, &v) == 0)
        info->chipset_id = (uint32_t)v;
    if (turing_getparam(fd, NOUVEAU_GETPARAM_FB_SIZE, &v) == 0)
        info->memory_bytes = v;
    /* GRAPH_UNITS (13): the digest names the query but does not
     * define its semantics (SM count vs GPC count); the value
     * is reported as compute_units and flagged in the driver
     * report for hardware confirmation. */
    if (turing_getparam(fd, NOUVEAU_GETPARAM_GRAPH_UNITS, &v) == 0)
        info->compute_units = (uint32_t)v;
    if (turing_getparam(fd, NOUVEAU_GETPARAM_EXEC_PUSH_MAX, &v) == 0)
        info->max_push_count = (uint32_t)v;
    return true;
}

static vbs_drv_status_t turing_probe(vbs_drv_device_info_t *infos,
                                     size_t cap, size_t *count) {
    return vbs_drv_scan_render_nodes(VBS_DRV_TURING_FIRST_RENDER_NODE,
        VBS_DRV_TURING_MAX_RENDER_NODES, "nouveau", turing_check_node,
        NULL, infos, cap, count, g_node_paths);
}

/* ------------------------------------------------------------------ */
/* Per-buffer and per-device vendor state                              */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t handle;     /* GEM handle (0 = none) */
    uint64_t map_handle; /* mmap offset from GEM_INFO */
    uint64_t va;         /* bound GPU VA (0 = unbound) */
    size_t   va_size;    /* bound size (page-rounded) */
    bool     mapped;     /* host_ptr is a live mmap */
} turing_bo_t;

typedef struct {
    int fd;              /* render-node fd (-1 = closed) */
    int channel;         /* channel id (-1 = none) */
    turing_bo_t pushbuf; /* persistent pushbuffer BO */
    turing_bo_t qmdbuf;  /* persistent QMD destination BO */
    void *pushbuf_ptr;   /* CPU mapping of the pushbuffer */
    vbs_drv_fence_t last_fence; /* fence of the last submit, if any */
} turing_dev_t;

static size_t round_up_page(size_t size) {
    return (size + 4095u) & ~(size_t)4095u;
}

/* Create a GEM buffer and bind it at a framework-allocated GPU VA.
 * `domain` is the preferred GEM domain; on failure the caller may
 * retry with another domain. When cpu_ptr_out is non-NULL the buffer
 * is also mapped for the CPU and *cpu_ptr_out receives the pointer
 * (NULL on mapping failure); when NULL no CPU mapping is made. */
static vbs_drv_status_t turing_bo_create(vbs_drv_device_t *dev,
                                         turing_bo_t *bo, size_t size,
                                         uint32_t domain,
                                         void **cpu_ptr_out) {
    turing_dev_t *td = (turing_dev_t *)dev->vendor;
    memset(bo, 0, sizeof(*bo));
    size_t alloc_size = round_up_page(size);
    if (alloc_size < size) return VBS_DRV_ERR_INVALID_ARG; /* overflow */
    /* GEM_NEW (digest §2.2/§2.7; S7 nouveau_drm.h). */
    struct drm_nouveau_gem_new gn;
    memset(&gn, 0, sizeof(gn));
    gn.info.domain = domain;
    gn.info.size = alloc_size;
    gn.align = 4096;
    int r = vbs_drv_ioctl(td->fd, DRM_IOCTL_NOUVEAU_GEM_NEW, &gn);
    if (r != 0) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "DRM_IOCTL_NOUVEAU_GEM_NEW", -r);
        return VBS_DRV_ERR_IO;
    }
    bo->handle = gn.info.handle;
    /* GEM_INFO for the map handle (digest §2.2). */
    struct drm_nouveau_gem_info gi;
    memset(&gi, 0, sizeof(gi));
    gi.handle = bo->handle;
    r = vbs_drv_ioctl(td->fd, DRM_IOCTL_NOUVEAU_GEM_INFO, &gi);
    if (r != 0) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "DRM_IOCTL_NOUVEAU_GEM_INFO", -r);
        goto fail_close;
    }
    bo->map_handle = gi.map_handle;
    /* Allocate a VA in the userspace-managed region (framework
     * allocator) and bind the BO there (digest §2.4 VM_BIND). */
    r = (int)vbs_drv_va_alloc(&dev->va_space, alloc_size, 4096, &bo->va);
    if (r != VBS_DRV_OK) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "GPU VA allocation", 0);
        goto fail_close;
    }
    bo->va_size = alloc_size;
    struct drm_nouveau_vm_bind_op op;
    memset(&op, 0, sizeof(op));
    op.op = DRM_NOUVEAU_VM_BIND_OP_MAP;
    op.handle = bo->handle;
    op.addr = bo->va;
    op.bo_offset = 0;
    op.range = alloc_size;
    struct drm_nouveau_vm_bind vb;
    memset(&vb, 0, sizeof(vb));
    vb.op_count = 1;
    vb.op_ptr = (uint64_t)(uintptr_t)&op;
    r = vbs_drv_ioctl(td->fd, DRM_IOCTL_NOUVEAU_VM_BIND, &vb);
    if (r != 0) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "DRM_IOCTL_NOUVEAU_VM_BIND (MAP)", -r);
        vbs_drv_va_free(&dev->va_space, bo->va, bo->va_size);
        bo->va = 0;
        goto fail_close;
    }
    if (cpu_ptr_out) {
        *cpu_ptr_out = NULL;
        void *ptr = mmap(NULL, alloc_size, PROT_READ | PROT_WRITE,
                         MAP_SHARED, td->fd, (off_t)bo->map_handle);
        if (ptr == MAP_FAILED) {
            vbs_drv_note_error(g_last_error, sizeof(g_last_error), "mmap GEM buffer", errno);
            /* Unwind the binding, then close the handle. */
            struct drm_nouveau_vm_bind_op uop;
            memset(&uop, 0, sizeof(uop));
            uop.op = DRM_NOUVEAU_VM_BIND_OP_UNMAP;
            uop.handle = bo->handle;
            uop.addr = bo->va;
            uop.range = alloc_size;
            struct drm_nouveau_vm_bind uvb;
            memset(&uvb, 0, sizeof(uvb));
            uvb.op_count = 1;
            uvb.op_ptr = (uint64_t)(uintptr_t)&uop;
            vbs_drv_ioctl(td->fd, DRM_IOCTL_NOUVEAU_VM_BIND, &uvb);
            vbs_drv_va_free(&dev->va_space, bo->va, bo->va_size);
            bo->va = 0;
            goto fail_close;
        }
        bo->mapped = true;
        *cpu_ptr_out = ptr;
    }
    return VBS_DRV_OK;

fail_close:
    vbs_drv_gem_close(td->fd, bo->handle);
    bo->handle = 0;
    return VBS_DRV_ERR_IO;
}

/* Destroy a BO created by turing_bo_create: unbind the VA and close
 * the GEM handle. The caller unmaps any CPU mapping first (it owns
 * the pointer). */
static void turing_bo_destroy(vbs_drv_device_t *dev, turing_bo_t *bo) {
    turing_dev_t *td = (turing_dev_t *)dev->vendor;
    if (!td || td->fd < 0) return;
    bo->mapped = false;
    if (bo->va != 0) {
        struct drm_nouveau_vm_bind_op op;
        memset(&op, 0, sizeof(op));
        op.op = DRM_NOUVEAU_VM_BIND_OP_UNMAP;
        op.handle = bo->handle;
        op.addr = bo->va;
        op.range = bo->va_size;
        struct drm_nouveau_vm_bind vb;
        memset(&vb, 0, sizeof(vb));
        vb.op_count = 1;
        vb.op_ptr = (uint64_t)(uintptr_t)&op;
        vbs_drv_ioctl(td->fd, DRM_IOCTL_NOUVEAU_VM_BIND, &vb);
        vbs_drv_va_free(&dev->va_space, bo->va, bo->va_size);
        bo->va = 0;
    }
    if (bo->handle != 0) {
        vbs_drv_gem_close(td->fd, bo->handle);
        bo->handle = 0;
    }
}

/* ------------------------------------------------------------------ */
/* Open / close — digest §3.4 (channel setup order)                    */
/* ------------------------------------------------------------------ */

/* Release every resource a half-open or open device holds, in
 * reverse allocation order. Tolerates partial state (fd open but
 * no channel, no push buffer, and so on). When drop_fence is true
 * the pending submit fence is destroyed too; the open-failure
 * path passes false because no submit has happened yet. */
static void turing_teardown(vbs_drv_device_t *dev, bool drop_fence) {
    turing_dev_t *td = (turing_dev_t *)dev->vendor;
    if (drop_fence && td->last_fence != VBS_DRV_FENCE_NONE) {
        vbs_drv_syncobj_destroy(td->fd, (uint32_t)td->last_fence);
        td->last_fence = VBS_DRV_FENCE_NONE;
    }
    if (td->pushbuf_ptr) {
        munmap(td->pushbuf_ptr, td->pushbuf.va_size);
        td->pushbuf_ptr = NULL;
    }
    turing_bo_destroy(dev, &td->pushbuf);
    turing_bo_destroy(dev, &td->qmdbuf);
    if (td->channel >= 0 && td->fd >= 0) {
        struct drm_nouveau_channel_free cf;
        memset(&cf, 0, sizeof(cf));
        cf.channel = td->channel;
        vbs_drv_ioctl(td->fd, DRM_IOCTL_NOUVEAU_CHANNEL_FREE, &cf);
        td->channel = -1;
    }
    if (td->fd >= 0) {
        close(td->fd);
        td->fd = -1;
    }
    free(td);
    dev->vendor = NULL;
}

static vbs_drv_status_t turing_open(size_t index, vbs_drv_device_t *dev) {
    if (index >= VBS_DRV_MAX_DEVICES || !dev) return VBS_DRV_ERR_INVALID_ARG;
    turing_dev_t *td = (turing_dev_t *)calloc(1, sizeof(turing_dev_t));
    if (!td) return VBS_DRV_ERR_NO_MEMORY;
    td->fd = -1;
    td->channel = -1;
    td->last_fence = VBS_DRV_FENCE_NONE;
    dev->vendor = td;

    /* 1. Open the render node (digest §2.1; S7 drm-uapi.rst). */
    td->fd = open(g_node_paths[index], O_RDWR);
    if (td->fd < 0) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "open render node", errno);
        free(td);
        dev->vendor = NULL;
        return VBS_DRV_ERR_IO;
    }

    /* 2. VM_INIT — split the VA space. MUST come before any BO or
     * channel is created, else it fails with -ENOSYS (digest §2.2;
     * S7 nouveau_drm.h drm_nouveau_vm_init doc). */
    struct drm_nouveau_vm_init vi;
    memset(&vi, 0, sizeof(vi));
    vi.kernel_managed_addr = 0;
    vi.kernel_managed_size = VBS_DRV_TURING_KERNEL_VA_SIZE;
    int r = vbs_drv_ioctl(td->fd, DRM_IOCTL_NOUVEAU_VM_INIT, &vi);
    if (r != 0) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "DRM_IOCTL_NOUVEAU_VM_INIT", -r);
        goto fail;
    }
    dev->va_base = VBS_DRV_TURING_UMD_VA_BASE;
    dev->va_size = VBS_DRV_TURING_UMD_VA_SIZE;
    if (vbs_drv_va_init(&dev->va_space, dev->va_base, dev->va_size) != VBS_DRV_OK) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "GPU VA space init", 0);
        goto fail;
    }

    /* 3. CHANNEL_ALLOC with a subchannel bound to the Turing compute
     * class 0xC5C0 (digest §2.3; S7 nouveau_drm.h; class id from S1
     * clc5c0.h line 31). The kernel enforces the binding, so the
     * pushbuffer never binds the object itself.
     *
     * Driver-chosen values NOT specified in the digest (confirm on
     * hardware; see the driver report):
     *   - fb_ctxdma_handle / tt_ctxdma_handle = 0: legacy DMA-object
     *     handles, unused on Turing where the subchannel binding
     *     selects the engine.
     *   - subchan handle = 1: a client-chosen object handle; the
     *     kernel uses it as the NVIF object handle for the bound
     *     compute object.
     *   - pushbuf_domains = VRAM|GART.
     *   - notifier_handle = 0: no notifier object in v1 (completion
     *     comes from the EXEC syncobj). */
    struct drm_nouveau_channel_alloc ca;
    memset(&ca, 0, sizeof(ca));
    ca.fb_ctxdma_handle = 0;
    ca.tt_ctxdma_handle = 0;
    ca.pushbuf_domains = NOUVEAU_GEM_DOMAIN_VRAM | NOUVEAU_GEM_DOMAIN_GART;
    ca.notifier_handle = 0;
    ca.subchan[VBS_DRV_TURING_COMPUTE_SUBCHAN].handle = 1;
    ca.subchan[VBS_DRV_TURING_COMPUTE_SUBCHAN].grclass =
        VBS_DRV_TURING_COMPUTE_CLASS;
    ca.nr_subchan = 1;
    r = vbs_drv_ioctl(td->fd, DRM_IOCTL_NOUVEAU_CHANNEL_ALLOC, &ca);
    if (r != 0) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "DRM_IOCTL_NOUVEAU_CHANNEL_ALLOC", -r);
        goto fail;
    }
    td->channel = ca.channel;

    /* 4. Persistent buffers: pushbuffer (GART — the CPU rewrites it
     * every submit) and the QMD destination (VRAM — the GPU writes
     * it from the pushbuffer's inline-QMD upload, then reads it at
     * dispatch). */
    vbs_drv_status_t st = turing_bo_create(dev, &td->pushbuf,
                                           VBS_DRV_TURING_PUSHBUF_BYTES,
                                           NOUVEAU_GEM_DOMAIN_GART,
                                           &td->pushbuf_ptr);
    if (st != VBS_DRV_OK) goto fail;
    st = turing_bo_create(dev, &td->qmdbuf, VBS_DRV_TURING_QMD_BUF_BYTES,
                          NOUVEAU_GEM_DOMAIN_VRAM, NULL);
    if (st != VBS_DRV_OK) goto fail;

    /* The QMD buffer is the first VRAM allocation, so it lands at
     * the bottom of the UMD region (2^32) — well below the 2^40
     * SEND_PCAS_A limit. Verify anyway; a future allocator change
     * must not silently break launches. */
    if (td->qmdbuf.va >= VBS_DRV_TURING_QMD_MAX_VA) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "QMD buffer VA above the 2^40 SEND_PCAS_A limit", 0);
        goto fail;
    }
    return VBS_DRV_OK;

fail:
    /* Close tolerates partial state (fd open but no channel, etc.). */
    turing_teardown(dev, false);
    return VBS_DRV_ERR_IO;
}

static void turing_close(vbs_drv_device_t *dev) {
    if (!dev || !dev->vendor) return;
    turing_teardown(dev, true);
}

/* ------------------------------------------------------------------ */
/* Buffers — digest §6.1/§6.2                                          */
/* ------------------------------------------------------------------ */

static vbs_drv_status_t turing_alloc_buffer(vbs_drv_device_t *dev,
                                            size_t size, uint32_t flags,
                                            vbs_drv_buffer_t *buf) {
    if (!dev || !dev->vendor || !buf || size == 0)
        return VBS_DRV_ERR_INVALID_ARG;
    turing_bo_t *bo = (turing_bo_t *)calloc(1, sizeof(turing_bo_t));
    if (!bo) return VBS_DRV_ERR_NO_MEMORY;
    buf->vendor = bo;
    buf->size = size;
    buf->alloc_size = round_up_page(size);
    buf->flags = flags;
    bool host_visible = (flags & VBS_DRV_BUF_HOST_VISIBLE) != 0;
    /* Preferred domain: VRAM, mappable when the CPU needs access
     * (digest §2.7; MAPPABLE is the modifier bit). VRAM|MAPPABLE
     * can fail on a small BAR, so fall back to GART (system memory
     * the GPU can read — slower, but correct) rather than failing
     * the allocation outright. */
    uint32_t domain = NOUVEAU_GEM_DOMAIN_VRAM;
    if (host_visible) domain |= NOUVEAU_GEM_DOMAIN_MAPPABLE;
    void *cpu_ptr = NULL;
    vbs_drv_status_t st = turing_bo_create(
        dev, bo, size, domain, host_visible ? &cpu_ptr : NULL);
    if (st != VBS_DRV_OK)
        st = turing_bo_create(dev, bo, size, NOUVEAU_GEM_DOMAIN_GART,
                              host_visible ? &cpu_ptr : NULL);
    if (st != VBS_DRV_OK)
        return vbs_drv_drop_failed_buffer(buf, bo, st);
    buf->gpu_va = bo->va;
    buf->host_ptr = cpu_ptr; /* NULL when not host-visible */
    return VBS_DRV_OK;
}

static void turing_free_buffer(vbs_drv_buffer_t *buf) {
    if (!buf || !buf->vendor) return;
    turing_bo_t *bo = (turing_bo_t *)buf->vendor;
    vbs_drv_device_t *dev = buf->device;
    if (buf->host_ptr) {
        munmap(buf->host_ptr, bo->va_size);
        buf->host_ptr = NULL;
    }
    turing_bo_destroy(dev, bo);
    free(bo);
    buf->vendor = NULL;
}

static vbs_drv_status_t turing_map_buffer(vbs_drv_buffer_t *buf) {
    if (!buf) return VBS_DRV_ERR_INVALID_ARG;
    /* Buffers are mapped persistently at alloc time when
     * host-visible; nothing to do. */
    return buf->host_ptr ? VBS_DRV_OK : VBS_DRV_ERR_NOT_SUPPORTED;
}

static void turing_unmap_buffer(vbs_drv_buffer_t *buf) {
    (void)buf; /* persistent mapping; unmapped at free time */
}

static vbs_drv_status_t turing_cpu_begin(vbs_drv_buffer_t *buf,
                                         bool for_write) {
    if (!buf || !buf->vendor || !buf->device || !buf->device->vendor)
        return VBS_DRV_ERR_INVALID_ARG;
    turing_dev_t *td = (turing_dev_t *)buf->device->vendor;
    turing_bo_t *bo = (turing_bo_t *)buf->vendor;
    /* GEM_CPU_PREP (digest §2.2): prepare the BO for CPU access,
     * with the WRITE flag when the CPU is about to write. */
    struct drm_nouveau_gem_cpu_prep prep;
    memset(&prep, 0, sizeof(prep));
    prep.handle = bo->handle;
    prep.flags = for_write ? NOUVEAU_GEM_CPU_PREP_WRITE : 0;
    int r = vbs_drv_ioctl(td->fd, DRM_IOCTL_NOUVEAU_GEM_CPU_PREP, &prep);
    if (r != 0) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "DRM_IOCTL_NOUVEAU_GEM_CPU_PREP", -r);
        return VBS_DRV_ERR_IO;
    }
    return VBS_DRV_OK;
}

static void turing_cpu_end(vbs_drv_buffer_t *buf) {
    if (!buf || !buf->vendor || !buf->device || !buf->device->vendor) return;
    turing_dev_t *td = (turing_dev_t *)buf->device->vendor;
    turing_bo_t *bo = (turing_bo_t *)buf->vendor;
    struct drm_nouveau_gem_cpu_fini fini;
    memset(&fini, 0, sizeof(fini));
    fini.handle = bo->handle;
    vbs_drv_ioctl(td->fd, DRM_IOCTL_NOUVEAU_GEM_CPU_FINI, &fini);
}

/* ------------------------------------------------------------------ */
/* Submission + fences — digest §2.5/§2.6                              */
/* ------------------------------------------------------------------ */

static vbs_drv_status_t turing_submit(vbs_drv_device_t *dev,
                                      const vbs_drv_cmdbuf_t *cb,
                                      vbs_drv_fence_t *fence_out) {
    if (!dev || !dev->vendor || !cb || !cb->words || cb->count == 0 ||
        !fence_out)
        return VBS_DRV_ERR_INVALID_ARG;
    turing_dev_t *td = (turing_dev_t *)dev->vendor;
    if (td->fd < 0 || td->channel < 0 || !td->pushbuf_ptr)
        return VBS_DRV_ERR_NOT_READY;
    size_t bytes = cb->count * sizeof(uint32_t);
    if (bytes > VBS_DRV_TURING_PUSHBUF_BYTES)
        return VBS_DRV_ERR_INVALID_ARG;
    *fence_out = VBS_DRV_FENCE_NONE;
    /* The pushbuffer BO is single-buffered: wait for the previous
     * submit to finish before overwriting it. */
    if (td->last_fence != VBS_DRV_FENCE_NONE) {
        vbs_drv_status_t st = vbs_drv_syncobj_wait(
            td->fd, (uint32_t)td->last_fence, 0, g_last_error,
            sizeof(g_last_error));
        vbs_drv_syncobj_destroy(td->fd, (uint32_t)td->last_fence);
        td->last_fence = VBS_DRV_FENCE_NONE;
        if (st != VBS_DRV_OK) return st;
    }
    /* Copy the command words into the mapped pushbuffer BO. */
    struct drm_nouveau_gem_cpu_prep prep;
    memset(&prep, 0, sizeof(prep));
    prep.handle = td->pushbuf.handle;
    prep.flags = NOUVEAU_GEM_CPU_PREP_WRITE;
    int r = vbs_drv_ioctl(td->fd, DRM_IOCTL_NOUVEAU_GEM_CPU_PREP, &prep);
    if (r != 0) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "DRM_IOCTL_NOUVEAU_GEM_CPU_PREP (pushbuf)", -r);
        return VBS_DRV_ERR_IO;
    }
    memcpy(td->pushbuf_ptr, cb->words, bytes);
    struct drm_nouveau_gem_cpu_fini fini;
    memset(&fini, 0, sizeof(fini));
    fini.handle = td->pushbuf.handle;
    vbs_drv_ioctl(td->fd, DRM_IOCTL_NOUVEAU_GEM_CPU_FINI, &fini);

    /* Create the signal syncobj (generic DRM syncobj ioctls, digest
     * §2.6), then EXEC (digest §2.5): one push, the pushbuffer's VA
     * and length, signaled on completion. No relocations — the
     * pushbuffer refers to GPU VAs directly (digest §2.5). */
    uint32_t sync_handle = 0;
    vbs_drv_status_t st = vbs_drv_syncobj_create(td->fd, &sync_handle, g_last_error, sizeof(g_last_error));
    if (st != VBS_DRV_OK) return st;
    struct drm_nouveau_exec_push push;
    memset(&push, 0, sizeof(push));
    push.va = td->pushbuf.va;
    push.va_len = (uint32_t)bytes;
    push.flags = 0;
    struct drm_nouveau_sync sig;
    memset(&sig, 0, sizeof(sig));
    sig.flags = DRM_NOUVEAU_SYNC_SYNCOBJ;
    sig.handle = sync_handle;
    sig.timeline_value = 0;
    struct drm_nouveau_exec exec;
    memset(&exec, 0, sizeof(exec));
    exec.channel = (uint32_t)td->channel;
    exec.push_count = 1;
    exec.sig_count = 1;
    exec.sig_ptr = (uint64_t)(uintptr_t)&sig;
    exec.push_ptr = (uint64_t)(uintptr_t)&push;
    r = vbs_drv_ioctl(td->fd, DRM_IOCTL_NOUVEAU_EXEC, &exec);
    if (r != 0) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "DRM_IOCTL_NOUVEAU_EXEC", -r);
        vbs_drv_syncobj_destroy(td->fd, sync_handle);
        return VBS_DRV_ERR_IO;
    }
    *fence_out = (vbs_drv_fence_t)sync_handle;
    td->last_fence = *fence_out;
    return VBS_DRV_OK;
}

static vbs_drv_status_t turing_fence_wait(vbs_drv_device_t *dev,
                                          vbs_drv_fence_t fence,
                                          uint32_t timeout_ms) {
    if (!dev || !dev->vendor) return VBS_DRV_ERR_INVALID_ARG;
    if (fence == VBS_DRV_FENCE_NONE) return VBS_DRV_OK;
    turing_dev_t *td = (turing_dev_t *)dev->vendor;
    if (td->fd < 0) return VBS_DRV_ERR_NOT_READY;
    return vbs_drv_syncobj_wait(td->fd, (uint32_t)fence, timeout_ms, g_last_error, sizeof(g_last_error));
}

static void turing_fence_free(vbs_drv_device_t *dev, vbs_drv_fence_t fence) {
    if (!dev || !dev->vendor || fence == VBS_DRV_FENCE_NONE) return;
    turing_dev_t *td = (turing_dev_t *)dev->vendor;
    if (td->fd < 0) return;
    if (td->last_fence == fence) td->last_fence = VBS_DRV_FENCE_NONE;
    vbs_drv_syncobj_destroy(td->fd, (uint32_t)fence);
}

/* ------------------------------------------------------------------ */
/* Launch — digest §3.5 + §4                                           */
/* ------------------------------------------------------------------ */

static vbs_drv_status_t turing_launch(vbs_drv_device_t *dev,
                                      const vbs_drv_launch_desc_t *desc) {
    if (!dev || !dev->vendor || !desc) return VBS_DRV_ERR_INVALID_ARG;
    turing_dev_t *td = (turing_dev_t *)dev->vendor;
    if (td->fd < 0 || td->channel < 0) return VBS_DRV_ERR_NOT_READY;
    /* The code and constant buffers must belong to this device. */
    vbs_drv_status_t ost = vbs_drv_launch_buffers_on_device(dev, desc);
    if (ost != VBS_DRV_OK) return ost;

    vbs_drv_turing_qmd_t qmd;
    vbs_drv_status_t st = vbs_drv_turing_qmd_build(desc, &qmd);
    if (st != VBS_DRV_OK) return st;

    vbs_drv_cmdbuf_t cb;
    st = vbs_drv_cmdbuf_init(&cb, 128);
    if (st != VBS_DRV_OK) return st;
    st = vbs_drv_turing_pb_emit_launch_sequence(
        &cb, VBS_DRV_TURING_COMPUTE_SUBCHAN, td->qmdbuf.va, &qmd);
    if (st != VBS_DRV_OK) {
        vbs_drv_cmdbuf_free(&cb);
        return st;
    }
    /* Synchronous launch: submit, wait for the completion fence,
     * then free it. */
    vbs_drv_fence_t fence = VBS_DRV_FENCE_NONE;
    st = turing_submit(dev, &cb, &fence);
    vbs_drv_cmdbuf_free(&cb);
    if (st != VBS_DRV_OK) return st;
    st = turing_fence_wait(dev, fence, VBS_DRV_TURING_LAUNCH_TIMEOUT_MS);
    turing_fence_free(dev, fence);
    return st;
}

static vbs_drv_status_t turing_synchronize(vbs_drv_device_t *dev) {
    if (!dev || !dev->vendor) return VBS_DRV_ERR_INVALID_ARG;
    turing_dev_t *td = (turing_dev_t *)dev->vendor;
    if (td->last_fence == VBS_DRV_FENCE_NONE) return VBS_DRV_OK;
    vbs_drv_status_t st =
        turing_fence_wait(dev, td->last_fence, 0); /* wait forever */
    turing_fence_free(dev, td->last_fence);
    return st;
}

static const vbs_drv_ops_t g_turing_ops = {
    .name = "turing",
    .probe = turing_probe,
    .open = turing_open,
    .close = turing_close,
    .alloc_buffer = turing_alloc_buffer,
    .free_buffer = turing_free_buffer,
    .map_buffer = turing_map_buffer,
    .unmap_buffer = turing_unmap_buffer,
    .cpu_begin = turing_cpu_begin,
    .cpu_end = turing_cpu_end,
    .submit = turing_submit,
    .fence_wait = turing_fence_wait,
    .fence_free = turing_fence_free,
    .launch = turing_launch,
    .synchronize = turing_synchronize,
};

vbs_drv_status_t vbs_drv_turing_register(void) {
    return vbs_drv_register_ops(&g_turing_ops);
}

#else /* VBS_DRV_TURING_HAVE_UAPI */
/* ---- Platform fallback: no Nouveau UAPI on this platform ---------------- */

static char g_fallback_err[] = "native Turing driver needs Linux + Nouveau DRM";

const char *vbs_drv_turing_last_error(void) { return g_fallback_err; }

vbs_drv_status_t vbs_drv_turing_register(void) {
    return vbs_drv_register_ops(vbs_drv_unavailable_ops("turing"));
}

#endif /* VBS_DRV_TURING_HAVE_UAPI */
