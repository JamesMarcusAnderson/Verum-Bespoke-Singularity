/* vbs_drv_alchemist.c — Alchemist driver: Xe DRM userspace layer.
 *
 * Talks to the GPU through the Xe DRM UAPI (/dev/dri/renderD*) only.
 * The kernel owns bring-up, firmware, scheduling, page tables, and
 * memory management; this layer owns the userspace side (digest
 * §3.4): open the render node, DEVICE_QUERY (engines, memory
 * regions, config), GEM buffers + VM_BIND, VM_CREATE,
 * EXEC_QUEUE_CREATE on a COMPUTE engine, batch-buffer construction
 * (via the encoder), EXEC submission, syncobj fences.
 *
 * FIRMWARE (digest §2): compute needs exactly one firmware blob,
 * i915/dg2_guc_70.bin (GuC, API ver 70.83.0) — the KERNEL driver
 * loads it; this userspace driver NEVER loads, embeds, or ships
 * firmware. If the blob is missing from the system's linux-firmware
 * install, the xe driver will not bring the device up and open
 * fails here (DEVICE_QUERY returns an error). HuC, DMC, and GSC
 * blobs are not needed for compute on DG2 (digest §2).
 *
 * All ioctl numbers, struct fields, and enum values come from the
 * spec digest §3 (source: include/uapi/drm/xe_drm.h, Linux v7.3.5)
 * and come from the system UAPI header <drm/xe_drm.h> —
 * included, never vendored. The system header is stale
 * (digest §3.1: 63 DRM_XE defines vs 127 in v7.3.5); the struct
 * layouts for every ioctl used here were verified identical to
 * v7.3.5 by diffing, and the few v7.3.5 constants the system header
 * lacks are defined below under #ifndef guards, citing the digest.
 *
 * Platform: Linux only. On any other platform (or a Linux system
 * without the Xe UAPI headers) this file compiles to the framework's
 * fallback ops table that reports "no device", so the top-level build keeps
 * working on macOS / Windows / MSVC.
 *
 * Originality: written from the digest's documented facts. The code
 * is our own — nothing here comes from Mesa, Intel's compute runtime,
 * libdrm, or any other implementation.
 */

#include "vbs_drv_alchemist.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Platform gate                                                       */
/* ------------------------------------------------------------------ */

#if defined(__linux__)
#  if defined(__has_include)
#    if __has_include(<drm/xe_drm.h>) && __has_include(<drm/drm.h>)
#      define VBS_DRV_ALCHEMIST_HAVE_UAPI 1
#    endif
#  else
#    define VBS_DRV_ALCHEMIST_HAVE_UAPI 1
#  endif
#endif

#ifdef VBS_DRV_ALCHEMIST_HAVE_UAPI
/* ---- Real implementation: Linux + Xe DRM UAPI --------------------- */

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <drm/drm.h>
#include <drm/xe_drm.h>

/* v7.3.5 constant the stale system header lacks (digest §3.3,
 * xe_drm.h v7.3.5 lines 939-956): the PCI_BARRIER mmap flag for
 * VRAM CPU mappings. The value comes from the digest's kernel.org
 * header; where the system header differs, the digest governs.
 * Guarded so a newer system header wins. (Other v7.3.5 additions —
 * e.g. VM_BIND_FLAG_IMMEDIATE/READONLY — are not needed by v1 and
 * are deliberately not defined here.) */
#ifndef DRM_XE_MMAP_OFFSET_FLAG_PCI_BARRIER
#  define DRM_XE_MMAP_OFFSET_FLAG_PCI_BARRIER (1 << 0)
#endif

/* Last detailed failure message (single-threaded use, like the rest
 * of v1). */
static char g_last_error[256];

const char *vbs_drv_alchemist_last_error(void) { return g_last_error; }

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */

/* First render node is renderD128 (digest §3.1;
 * Documentation/gpu/drm-uapi.rst). Scan 16 nodes; a machine with
 * more GPUs than that is out of scope. */
#define VBS_DRV_ALCHEMIST_FIRST_RENDER_NODE 128
#define VBS_DRV_ALCHEMIST_MAX_RENDER_NODES  16

/* VA-space split (a driver choice, not a hardware fact): the low
 * 4 GiB is left to the kernel; the userspace driver manages
 * [4 GiB, 2^VA_BITS) of the 48-bit PPGTT (digest §7.2). Keeping
 * VA 0 out of circulation also preserves the framework's
 * gpu_va == 0 "unmapped" sentinel. */
#define VBS_DRV_ALCHEMIST_UMD_VA_BASE (1ULL << 32)

/* Persistent buffers (created at open): the batch buffer the CPU
 * rewrites every submit, and the state BO holding the binding
 * table + surface states for launches. Both live in sysmem (the
 * CPU rewrites them constantly; the GPU reads them once). The
 * batch BO's extra mapped tail also covers the hardware's
 * over-fetch past MI_BATCH_BUFFER_END (digest §4.1: "a single
 * extra page beyond the batch buffer is sufficient"). */
#define VBS_DRV_ALCHEMIST_BATCHBUF_BYTES (64u * 1024u)

/* Launch completion timeout: a single kernel launch that runs over
 * a minute is treated as hung rather than hanging the engine
 * forever. */
#define VBS_DRV_ALCHEMIST_LAUNCH_TIMEOUT_MS 60000u

/* ------------------------------------------------------------------ */
/* Alchemist PCI device ids — digest §1,                                */
/* include/drm/intel/pciids.h (Linux v7.3.5)                            */
/* ------------------------------------------------------------------ */

typedef struct {
    uint16_t pci_device_id;
    const char *die; /* G10/G11/G12 — product names are not in the
                      * digest, so the die is the honest label. */
} alchemist_pci_entry_t;

static const alchemist_pci_entry_t g_alchemist_pci_ids[] = {
    /* ACM-G10 desktop */
    { 0x56A0, "DG2-G10" }, { 0x56A1, "DG2-G10" }, { 0x56A2, "DG2-G10" },
    /* ACM-G10 engineering */
    { 0x56BE, "DG2-G10" }, { 0x56BF, "DG2-G10" },
    /* ACM-G10 mobile */
    { 0x5690, "DG2-G10" }, { 0x5691, "DG2-G10" }, { 0x5692, "DG2-G10" },
    /* Flex 170 (ATS-M150) */
    { 0x56C0, "DG2-G10" }, { 0x56C2, "DG2-G10" },
    /* ACM-G11 desktop */
    { 0x56A5, "DG2-G11" }, { 0x56A6, "DG2-G11" },
    { 0x56B0, "DG2-G11" }, { 0x56B1, "DG2-G11" },
    /* ACM-G11 engineering */
    { 0x56BA, "DG2-G11" }, { 0x56BB, "DG2-G11" },
    { 0x56BC, "DG2-G11" }, { 0x56BD, "DG2-G11" },
    /* ACM-G11 mobile */
    { 0x5693, "DG2-G11" }, { 0x5694, "DG2-G11" }, { 0x5695, "DG2-G11" },
    /* Flex 140 (ATS-M75) */
    { 0x56C1, "DG2-G11" },
    /* ACM-G12 desktop */
    { 0x56A3, "DG2-G12" }, { 0x56A4, "DG2-G12" },
    { 0x56B2, "DG2-G12" }, { 0x56B3, "DG2-G12" },
    /* ACM-G12 mobile */
    { 0x5696, "DG2-G12" }, { 0x5697, "DG2-G12" },
};

static const char *alchemist_pci_die(uint16_t id) {
    for (size_t i = 0;
         i < sizeof(g_alchemist_pci_ids) / sizeof(g_alchemist_pci_ids[0]); i++)
        if (g_alchemist_pci_ids[i].pci_device_id == id)
            return g_alchemist_pci_ids[i].die;
    return NULL;
}

/* Node paths found by the last probe, indexed by vendor-local index. */
static char g_node_paths[VBS_DRV_MAX_DEVICES][VBS_DRV_NODE_PATH_LEN];

/* ------------------------------------------------------------------ */
/* Device queries — digest §3.3 (two-call pattern)                     */
/*                                                                     */
/* First call with size = 0 learns the required size; the second
 * call with a buffer of exactly that size fills it. Returns 0 on
 * success with *out pointing at a malloc'd reply the caller frees,
 * or -errno on failure. */
/* ------------------------------------------------------------------ */

static int alchemist_query(int fd, uint32_t query, void **out,
                       uint32_t *size_out) {
    struct drm_xe_device_query q;
    memset(&q, 0, sizeof(q));
    q.query = query;
    q.size = 0;
    q.data = 0;
    int r = vbs_drv_ioctl(fd, DRM_IOCTL_XE_DEVICE_QUERY, &q);
    if (r != 0) return r;
    if (q.size == 0) return -ENODATA;
    uint32_t size = q.size;
    void *buf = malloc(size);
    if (!buf) return -ENOMEM;
    memset(&q, 0, sizeof(q));
    q.query = query;
    q.size = size;
    q.data = (uint64_t)(uintptr_t)buf;
    r = vbs_drv_ioctl(fd, DRM_IOCTL_XE_DEVICE_QUERY, &q);
    if (r != 0) {
        free(buf);
        return r;
    }
    *out = buf;
    if (size_out) *size_out = size;
    return 0;
}

/* Count set bits in a byte range (for topology masks). */
static uint32_t popcount_bytes(const uint8_t *p, size_t n) {
    uint32_t total = 0;
    for (size_t i = 0; i < n; i++) {
        uint8_t b = p[i];
        while (b) {
            total += b & 1u;
            b >>= 1;
        }
    }
    return total;
}

/* EU count from the GT_TOPOLOGY query (digest §3.3): the reply is a
 * sequence of struct drm_xe_query_topology_mask records (8-byte
 * header + num_bytes of mask). Total EUs = (compute DSS count) x
 * (EUs per DSS), from the DSS_COMPUTE and EU_PER_DSS masks. Returns
 * 0 when the query fails or the reply does not parse cleanly — the
 * count is informational only, never load-bearing. */
static uint32_t alchemist_eu_count(int fd) {
    void *reply = NULL;
    uint32_t size = 0;
    if (alchemist_query(fd, DRM_XE_DEVICE_QUERY_GT_TOPOLOGY, &reply, &size) != 0)
        return 0;
    uint32_t dss = 0, eu_per_dss = 0;
    size_t off = 0;
    while (off + sizeof(struct drm_xe_query_topology_mask) <= size) {
        const struct drm_xe_query_topology_mask *rec =
            (const struct drm_xe_query_topology_mask *)((const uint8_t *)reply +
                                                        off);
        uint32_t nbytes = rec->num_bytes;
        if (nbytes > size - off - sizeof(*rec))
            break; /* malformed: stop parsing */
        if (rec->type == DRM_XE_TOPO_DSS_COMPUTE)
            dss = popcount_bytes(rec->mask, nbytes);
        else if (rec->type == DRM_XE_TOPO_EU_PER_DSS)
            eu_per_dss = popcount_bytes(rec->mask, nbytes);
        off += sizeof(*rec) + nbytes;
    }
    free(reply);
    if (off != size) return 0; /* trailing garbage: do not trust it */
    return dss * eu_per_dss;
}

/* ------------------------------------------------------------------ */
/* Probe                                                               */
/* ------------------------------------------------------------------ */

/* Node check for the framework's render-node scan: fd is an open
 * xe node (the framework already confirmed the driver name). Read
 * the PCI device id from the CONFIG query (digest §3.3:
 * REV_AND_DEVICE_ID — device id in the lower 16 bits, revision in
 * the next 8) and match the Alchemist DG2 list (digest §1). Fills
 * *info and returns true for a device this driver handles. */
static bool alchemist_check_node(int fd, vbs_drv_device_info_t *info,
                                 void *ctx) {
    (void)ctx;
    void *cfg_reply = NULL;
    if (alchemist_query(fd, DRM_XE_DEVICE_QUERY_CONFIG, &cfg_reply,
                    NULL) != 0)
        return false;
    struct drm_xe_query_config *cfg =
        (struct drm_xe_query_config *)cfg_reply;
    if (cfg->num_params <= DRM_XE_QUERY_CONFIG_REV_AND_DEVICE_ID) {
        free(cfg_reply);
        return false;
    }
    uint64_t rev_dev =
        cfg->info[DRM_XE_QUERY_CONFIG_REV_AND_DEVICE_ID];
    uint16_t pci_device = (uint16_t)(rev_dev & 0xffffu);
    uint32_t revision = (uint32_t)((rev_dev >> 16) & 0xffu);
    const char *die = alchemist_pci_die(pci_device);
    free(cfg_reply);
    if (!die)
        return false; /* xe GPU, but not an Alchemist DG2 we drive */
    snprintf(info->name, sizeof(info->name),
             "Intel Arc Alchemist (%s, Xe-HPG)", die);
    info->pci_vendor_id = VBS_DRV_ALCHEMIST_PCI_VENDOR;
    info->pci_device_id = pci_device;
    info->chipset_id = revision;
    /* Memory: prefer VRAM total, fall back to sysmem total
     * (digest §3.3, MEM_REGIONS query). */
    void *mem_reply = NULL;
    if (alchemist_query(fd, DRM_XE_DEVICE_QUERY_MEM_REGIONS, &mem_reply,
                    NULL) == 0) {
        struct drm_xe_query_mem_regions *mr =
            (struct drm_xe_query_mem_regions *)mem_reply;
        uint64_t vram = 0, sysmem = 0;
        for (uint32_t i = 0; i < mr->num_mem_regions; i++) {
            if (mr->mem_regions[i].mem_class ==
                DRM_XE_MEM_REGION_CLASS_VRAM)
                vram += mr->mem_regions[i].total_size;
            else if (mr->mem_regions[i].mem_class ==
                     DRM_XE_MEM_REGION_CLASS_SYSMEM)
                sysmem += mr->mem_regions[i].total_size;
        }
        info->memory_bytes = vram != 0 ? vram : sysmem;
        free(mem_reply);
    }
    info->compute_units = alchemist_eu_count(fd);
    /* One batch buffer per EXEC at width 1 (digest §3.3). */
    info->max_push_count = 1;
    return true;
}

static vbs_drv_status_t alchemist_probe(vbs_drv_device_info_t *infos,
                                    size_t cap, size_t *count) {
    return vbs_drv_scan_render_nodes(VBS_DRV_ALCHEMIST_FIRST_RENDER_NODE,
        VBS_DRV_ALCHEMIST_MAX_RENDER_NODES, "xe", alchemist_check_node,
        NULL, infos, cap, count, g_node_paths);
}

/* ------------------------------------------------------------------ */
/* Per-buffer and per-device vendor state                              */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t handle;     /* GEM handle (0 = none) */
    uint64_t map_offset; /* mmap offset from GEM_MMAP_OFFSET */
    uint64_t va;         /* bound GPU VA (0 = unbound) */
    size_t   va_size;    /* bound size (page-rounded) */
    bool     mapped;     /* host_ptr is a live mmap */
    bool     in_vram;    /* placed in VRAM (needs PCI_BARRIER mmap) */
} alchemist_bo_t;

typedef struct {
    int fd;              /* render-node fd (-1 = closed) */
    uint32_t vm_id;      /* xe VM id (0 = none) */
    uint32_t exec_queue_id; /* compute exec queue id (0 = none) */
    struct drm_xe_engine_class_instance compute_engine;
    uint32_t vram_instance;   /* mem-region instance numbers */
    uint32_t sysmem_instance;
    uint32_t vram_min_page;   /* region min_page_size values */
    uint32_t sysmem_min_page;
    bool     has_vram;
    uint32_t va_bits;
    alchemist_bo_t batchbuf; /* persistent batch-buffer BO */
    alchemist_bo_t statebo;  /* persistent binding-table/surface-state BO */
    void *batchbuf_ptr;  /* CPU mapping of the batch buffer */
    void *statebo_ptr;   /* CPU mapping of the state BO */
    vbs_drv_fence_t last_fence; /* fence of the last submit, if any */
} alchemist_dev_t;

static size_t round_up_to(size_t size, size_t align) {
    return (size + align - 1) & ~(align - 1);
}

/* Bind or unbind one BO at a VA (digest §3.3, VM_BIND). The bind
 * runs on the default VM bind engine (exec_queue_id 0) and is
 * synchronous on a non-faulting VM (our VM is created with flags
 * 0, so no IMMEDIATE flag is needed — that flag only matters for
 * fault-mode VMs). pat_index is 0: the PAT index values are
 * UNSOURCED in the digest (§9 item 6), so the kernel default is
 * used. */
static int alchemist_vm_bind(alchemist_dev_t *id, uint32_t vm_id, uint32_t handle,
                         uint64_t va, uint64_t size, bool unbind) {
    struct drm_xe_vm_bind_op op;
    memset(&op, 0, sizeof(op));
    op.obj = handle;
    op.pat_index = 0;
    op.obj_offset = 0;
    op.range = size;
    op.addr = va;
    op.op = unbind ? DRM_XE_VM_BIND_OP_UNMAP : DRM_XE_VM_BIND_OP_MAP;
    op.flags = 0;
    struct drm_xe_vm_bind vb;
    memset(&vb, 0, sizeof(vb));
    vb.vm_id = vm_id;
    vb.exec_queue_id = 0; /* default VM bind engine */
    vb.num_binds = 1;
    vb.bind = op;
    vb.num_syncs = 0;
    vb.syncs = 0;
    return vbs_drv_ioctl(id->fd, DRM_IOCTL_XE_VM_BIND, &vb);
}

/* Create a GEM buffer in the given memory region, bind it at a
 * framework-allocated GPU VA, and optionally map it for the CPU.
 * `region_instance` is the mem-region instance from the
 * MEM_REGIONS query; `min_page` is that region's min_page_size;
 * `is_vram` selects the VRAM mmap path (NEEDS_VISIBLE_VRAM +
 * PCI_BARRIER). */
static vbs_drv_status_t alchemist_bo_create(vbs_drv_device_t *dev,
                                        alchemist_bo_t *bo, size_t size,
                                        uint32_t region_instance,
                                        uint32_t min_page, bool is_vram,
                                        bool host_visible,
                                        void **cpu_ptr_out) {
    alchemist_dev_t *id = (alchemist_dev_t *)dev->vendor;
    /* The placement mask is a 32-bit field of instance bits. */
    if (region_instance >= 32) return VBS_DRV_ERR_INVALID_ARG;
    memset(bo, 0, sizeof(*bo));
    size_t align = min_page > 4096 ? min_page : 4096;
    size_t alloc_size = round_up_to(size, align);
    if (alloc_size < size) return VBS_DRV_ERR_INVALID_ARG; /* overflow */
    /* GEM_CREATE (digest §3.3): size aligned to the region's
     * min_page_size; placement is a mask of memory instances. */
    struct drm_xe_gem_create gc;
    memset(&gc, 0, sizeof(gc));
    gc.size = alloc_size;
    gc.placement = 1u << region_instance;
    gc.flags = 0;
    if (is_vram && host_visible)
        gc.flags |= DRM_XE_GEM_CREATE_FLAG_NEEDS_VISIBLE_VRAM;
    int r = vbs_drv_ioctl(id->fd, DRM_IOCTL_XE_GEM_CREATE, &gc);
    if (r != 0) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "DRM_IOCTL_XE_GEM_CREATE", -r);
        return VBS_DRV_ERR_IO;
    }
    bo->handle = gc.handle;
    bo->in_vram = is_vram;
    /* Allocate a VA in the userspace-managed region and bind. */
    vbs_drv_status_t st =
        vbs_drv_va_alloc(&dev->va_space, alloc_size, align, &bo->va);
    if (st != VBS_DRV_OK) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "GPU VA allocation", 0);
        goto fail_close;
    }
    bo->va_size = alloc_size;
    r = alchemist_vm_bind(id, id->vm_id, bo->handle, bo->va, alloc_size, false);
    if (r != 0) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "DRM_IOCTL_XE_VM_BIND (MAP)", -r);
        vbs_drv_va_free(&dev->va_space, bo->va, bo->va_size);
        bo->va = 0;
        goto fail_close;
    }
    if (host_visible && cpu_ptr_out) {
        *cpu_ptr_out = NULL;
        /* GEM_MMAP_OFFSET + mmap (digest §3.3, §7.1). VRAM CPU
         * mappings use the PCI_BARRIER flag. */
        struct drm_xe_gem_mmap_offset mo;
        memset(&mo, 0, sizeof(mo));
        mo.handle = bo->handle;
        mo.flags = is_vram ? DRM_XE_MMAP_OFFSET_FLAG_PCI_BARRIER : 0;
        r = vbs_drv_ioctl(id->fd, DRM_IOCTL_XE_GEM_MMAP_OFFSET, &mo);
        if (r != 0) {
            vbs_drv_note_error(g_last_error, sizeof(g_last_error), "DRM_IOCTL_XE_GEM_MMAP_OFFSET", -r);
            goto fail_unbind;
        }
        bo->map_offset = mo.offset;
        void *ptr = mmap(NULL, alloc_size, PROT_READ | PROT_WRITE,
                         MAP_SHARED, id->fd, (off_t)bo->map_offset);
        if (ptr == MAP_FAILED) {
            vbs_drv_note_error(g_last_error, sizeof(g_last_error), "mmap GEM buffer", errno);
            goto fail_unbind;
        }
        bo->mapped = true;
        *cpu_ptr_out = ptr;
    }
    return VBS_DRV_OK;

fail_unbind:
    alchemist_vm_bind(id, id->vm_id, bo->handle, bo->va, bo->va_size, true);
    vbs_drv_va_free(&dev->va_space, bo->va, bo->va_size);
    bo->va = 0;
fail_close:
    vbs_drv_gem_close(id->fd, bo->handle);
    bo->handle = 0;
    return VBS_DRV_ERR_IO;
}

/* Destroy a BO created by alchemist_bo_create: unbind the VA and close
 * the GEM handle. The caller unmaps any CPU mapping first (it owns
 * the pointer). */
static void alchemist_bo_destroy(vbs_drv_device_t *dev, alchemist_bo_t *bo) {
    alchemist_dev_t *id = (alchemist_dev_t *)dev->vendor;
    if (!id || id->fd < 0) return;
    bo->mapped = false;
    if (bo->va != 0 && id->vm_id != 0) {
        alchemist_vm_bind(id, id->vm_id, bo->handle, bo->va, bo->va_size, true);
        vbs_drv_va_free(&dev->va_space, bo->va, bo->va_size);
        bo->va = 0;
    }
    if (bo->handle != 0) {
        vbs_drv_gem_close(id->fd, bo->handle);
        bo->handle = 0;
    }
}

/* ------------------------------------------------------------------ */
/* Open / close — digest §3.4 (submission flow)                        */
/*                                                                     */
/* FIRMWARE NOTE (digest §2): compute needs i915/dg2_guc_70.bin
 * installed on the system. The kernel loads it at device bring-up;
 * this driver never touches firmware. If the blob is missing, the
 * xe driver will not bring the device up and the queries below
 * fail — open then reports NO_DEVICE/IO with the failing ioctl
 * named in the last-error string. */
/* ------------------------------------------------------------------ */

/* Release every resource a half-open or open device holds, in
 * reverse allocation order. Tolerates partial state (fd open but
 * no VM, no batch buffer, and so on). When drop_fence is true the
 * pending submit fence is destroyed too; the open-failure path
 * passes false because no submit has happened yet. */
static void alchemist_teardown(vbs_drv_device_t *dev, bool drop_fence) {
    alchemist_dev_t *id = (alchemist_dev_t *)dev->vendor;
    if (drop_fence && id->last_fence != VBS_DRV_FENCE_NONE &&
        id->fd >= 0) {
        vbs_drv_syncobj_destroy(id->fd, (uint32_t)id->last_fence);
        id->last_fence = VBS_DRV_FENCE_NONE;
    }
    if (id->batchbuf_ptr) {
        munmap(id->batchbuf_ptr, id->batchbuf.va_size);
        id->batchbuf_ptr = NULL;
    }
    if (id->statebo_ptr) {
        munmap(id->statebo_ptr, id->statebo.va_size);
        id->statebo_ptr = NULL;
    }
    alchemist_bo_destroy(dev, &id->batchbuf);
    alchemist_bo_destroy(dev, &id->statebo);
    if (id->exec_queue_id != 0 && id->fd >= 0) {
        struct drm_xe_exec_queue_destroy eqd;
        memset(&eqd, 0, sizeof(eqd));
        eqd.exec_queue_id = id->exec_queue_id;
        vbs_drv_ioctl(id->fd, DRM_IOCTL_XE_EXEC_QUEUE_DESTROY, &eqd);
        id->exec_queue_id = 0;
    }
    if (id->vm_id != 0 && id->fd >= 0) {
        struct drm_xe_vm_destroy vd;
        memset(&vd, 0, sizeof(vd));
        vd.vm_id = id->vm_id;
        vbs_drv_ioctl(id->fd, DRM_IOCTL_XE_VM_DESTROY, &vd);
        id->vm_id = 0;
    }
    if (id->fd >= 0) {
        close(id->fd);
        id->fd = -1;
    }
    free(id);
    dev->vendor = NULL;
}

static vbs_drv_status_t alchemist_open(size_t index, vbs_drv_device_t *dev) {
    if (index >= VBS_DRV_MAX_DEVICES || !dev) return VBS_DRV_ERR_INVALID_ARG;
    alchemist_dev_t *id = (alchemist_dev_t *)calloc(1, sizeof(alchemist_dev_t));
    if (!id) return VBS_DRV_ERR_NO_MEMORY;
    id->fd = -1;
    id->last_fence = VBS_DRV_FENCE_NONE;
    dev->vendor = id;

    /* 1. Open the render node (digest §3.1). */
    id->fd = open(g_node_paths[index], O_RDWR);
    if (id->fd < 0) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "open render node", errno);
        free(id);
        dev->vendor = NULL;
        return VBS_DRV_ERR_IO;
    }

    /* 2. ENGINES query: find a COMPUTE-class engine instance
     * (digest §3.3, DRM_XE_ENGINE_CLASS_COMPUTE = 4). */
    void *eng_reply = NULL;
    int r = alchemist_query(id->fd, DRM_XE_DEVICE_QUERY_ENGINES, &eng_reply, NULL);
    if (r != 0) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "DRM_IOCTL_XE_DEVICE_QUERY (ENGINES)", -r);
        goto fail;
    }
    struct drm_xe_query_engines *engines =
        (struct drm_xe_query_engines *)eng_reply;
    bool found_compute = false;
    for (uint32_t i = 0; i < engines->num_engines; i++) {
        if (engines->engines[i].instance.engine_class ==
            DRM_XE_ENGINE_CLASS_COMPUTE) {
            id->compute_engine = engines->engines[i].instance;
            found_compute = true;
            break;
        }
    }
    free(eng_reply);
    if (!found_compute) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "no COMPUTE engine instance found", 0);
        goto fail;
    }

    /* 3. MEM_REGIONS query: VRAM + sysmem region instances and
     * their min_page_size values (digest §3.3). */
    void *mem_reply = NULL;
    r = alchemist_query(id->fd, DRM_XE_DEVICE_QUERY_MEM_REGIONS, &mem_reply, NULL);
    if (r != 0) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "DRM_IOCTL_XE_DEVICE_QUERY (MEM_REGIONS)", -r);
        goto fail;
    }
    struct drm_xe_query_mem_regions *mr =
        (struct drm_xe_query_mem_regions *)mem_reply;
    bool found_sysmem = false;
    id->has_vram = false;
    for (uint32_t i = 0; i < mr->num_mem_regions; i++) {
        if (mr->mem_regions[i].mem_class == DRM_XE_MEM_REGION_CLASS_VRAM) {
            id->vram_instance = mr->mem_regions[i].instance;
            id->vram_min_page = mr->mem_regions[i].min_page_size;
            id->has_vram = true;
        } else if (mr->mem_regions[i].mem_class ==
                   DRM_XE_MEM_REGION_CLASS_SYSMEM) {
            id->sysmem_instance = mr->mem_regions[i].instance;
            id->sysmem_min_page = mr->mem_regions[i].min_page_size;
            found_sysmem = true;
        }
    }
    free(mem_reply);
    if (!found_sysmem) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "no sysmem memory region found", 0);
        goto fail;
    }

    /* 4. CONFIG query: VA_BITS + MIN_ALIGNMENT (digest §3.3). */
    void *cfg_reply = NULL;
    r = alchemist_query(id->fd, DRM_XE_DEVICE_QUERY_CONFIG, &cfg_reply, NULL);
    if (r != 0) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "DRM_IOCTL_XE_DEVICE_QUERY (CONFIG)", -r);
        goto fail;
    }
    struct drm_xe_query_config *cfg = (struct drm_xe_query_config *)cfg_reply;
    uint32_t va_bits = 0;
    if (cfg->num_params > DRM_XE_QUERY_CONFIG_VA_BITS)
        va_bits = (uint32_t)cfg->info[DRM_XE_QUERY_CONFIG_VA_BITS];
    free(cfg_reply);
    /* va_bits must leave room for the 4 GiB kernel floor and must
     * be below 64 (1ULL << 64 is undefined). */
    if (va_bits <= 32 || va_bits >= 64) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "CONFIG query returned an unusable VA_BITS", 0);
        goto fail;
    }
    id->va_bits = va_bits;

    /* 5. VM_CREATE with flags 0 (non-faulting VM: binds are
     * synchronous — digest §3.3). */
    struct drm_xe_vm_create vc;
    memset(&vc, 0, sizeof(vc));
    vc.flags = 0;
    r = vbs_drv_ioctl(id->fd, DRM_IOCTL_XE_VM_CREATE, &vc);
    if (r != 0) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "DRM_IOCTL_XE_VM_CREATE", -r);
        goto fail;
    }
    id->vm_id = vc.vm_id;
    dev->va_base = VBS_DRV_ALCHEMIST_UMD_VA_BASE;
    dev->va_size = (1ULL << va_bits) - VBS_DRV_ALCHEMIST_UMD_VA_BASE;
    if (vbs_drv_va_init(&dev->va_space, dev->va_base, dev->va_size) !=
        VBS_DRV_OK) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "GPU VA space init", 0);
        goto fail;
    }

    /* 6. EXEC_QUEUE_CREATE: width 1, one placement, the COMPUTE
     * engine instance, this VM (digest §3.3: "for a single-CCS
     * submit use width = 1, num_placements = 1"). */
    struct drm_xe_exec_queue_create eqc;
    memset(&eqc, 0, sizeof(eqc));
    eqc.width = 1;
    eqc.num_placements = 1;
    eqc.vm_id = id->vm_id;
    eqc.flags = 0;
    eqc.instances = (uint64_t)(uintptr_t)&id->compute_engine;
    r = vbs_drv_ioctl(id->fd, DRM_IOCTL_XE_EXEC_QUEUE_CREATE, &eqc);
    if (r != 0) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "DRM_IOCTL_XE_EXEC_QUEUE_CREATE", -r);
        goto fail;
    }
    id->exec_queue_id = eqc.exec_queue_id;

    /* 7. Persistent buffers: batch buffer + state BO, both in
     * sysmem and host-visible (the CPU rewrites them every
     * submit/launch). */
    vbs_drv_status_t st = alchemist_bo_create(
        dev, &id->batchbuf, VBS_DRV_ALCHEMIST_BATCHBUF_BYTES,
        id->sysmem_instance, id->sysmem_min_page, false, true,
        &id->batchbuf_ptr);
    if (st != VBS_DRV_OK) goto fail;
    st = alchemist_bo_create(dev, &id->statebo, VBS_DRV_ALCHEMIST_STATE_BYTES,
                         id->sysmem_instance, id->sysmem_min_page, false,
                         true, &id->statebo_ptr);
    if (st != VBS_DRV_OK) goto fail;
    return VBS_DRV_OK;

fail:
    /* Close tolerates partial state (fd open but no VM, etc.). */
    alchemist_teardown(dev, false);
    return VBS_DRV_ERR_IO;
}

static void alchemist_close(vbs_drv_device_t *dev) {
    if (!dev || !dev->vendor) return;
    alchemist_teardown(dev, true);
}

/* ------------------------------------------------------------------ */
/* Buffers — digest §7.1                                               */
/* ------------------------------------------------------------------ */

static vbs_drv_status_t alchemist_alloc_buffer(vbs_drv_device_t *dev,
                                           size_t size, uint32_t flags,
                                           vbs_drv_buffer_t *buf) {
    if (!dev || !dev->vendor || !buf || size == 0)
        return VBS_DRV_ERR_INVALID_ARG;
    alchemist_dev_t *id = (alchemist_dev_t *)dev->vendor;
    alchemist_bo_t *bo = (alchemist_bo_t *)calloc(1, sizeof(alchemist_bo_t));
    if (!bo) return VBS_DRV_ERR_NO_MEMORY;
    buf->vendor = bo;
    buf->size = size;
    buf->flags = flags;
    bool host_visible = (flags & VBS_DRV_BUF_HOST_VISIBLE) != 0;
    bool device_local = (flags & VBS_DRV_BUF_DEVICE_LOCAL) != 0;
    void *cpu_ptr = NULL;
    vbs_drv_status_t st;
    /* Preferred placement: VRAM when device-local memory is asked
     * for (and VRAM exists), else sysmem. A VRAM placement can
     * fail (small BAR, NEEDS_VISIBLE_VRAM pressure), so fall back
     * to sysmem — slower, but correct — rather than failing the
     * allocation outright. */
    if (device_local && id->has_vram) {
        st = alchemist_bo_create(dev, bo, size, id->vram_instance,
                             id->vram_min_page, true, host_visible,
                             host_visible ? &cpu_ptr : NULL);
        if (st == VBS_DRV_OK) {
            buf->alloc_size = bo->va_size;
            buf->gpu_va = bo->va;
            buf->host_ptr = cpu_ptr;
            return VBS_DRV_OK;
        }
        /* Fall through to the sysmem retry. */
    }
    st = alchemist_bo_create(dev, bo, size, id->sysmem_instance,
                         id->sysmem_min_page, false, host_visible,
                         host_visible ? &cpu_ptr : NULL);
    if (st != VBS_DRV_OK)
        return vbs_drv_drop_failed_buffer(buf, bo, st);
    buf->alloc_size = bo->va_size;
    buf->gpu_va = bo->va;
    buf->host_ptr = cpu_ptr; /* NULL when not host-visible */
    return VBS_DRV_OK;
}

static void alchemist_free_buffer(vbs_drv_buffer_t *buf) {
    if (!buf || !buf->vendor) return;
    alchemist_bo_t *bo = (alchemist_bo_t *)buf->vendor;
    vbs_drv_device_t *dev = buf->device;
    if (buf->host_ptr) {
        munmap(buf->host_ptr, bo->va_size);
        buf->host_ptr = NULL;
    }
    alchemist_bo_destroy(dev, bo);
    free(bo);
    buf->vendor = NULL;
}

static vbs_drv_status_t alchemist_map_buffer(vbs_drv_buffer_t *buf) {
    if (!buf) return VBS_DRV_ERR_INVALID_ARG;
    /* Buffers are mapped persistently at alloc time when
     * host-visible; nothing to do. */
    return buf->host_ptr ? VBS_DRV_OK : VBS_DRV_ERR_NOT_SUPPORTED;
}

static void alchemist_unmap_buffer(vbs_drv_buffer_t *buf) {
    (void)buf; /* persistent mapping; unmapped at free time */
}

static vbs_drv_status_t alchemist_cpu_begin(vbs_drv_buffer_t *buf,
                                        bool for_write) {
    (void)for_write;
    if (!buf || !buf->vendor || !buf->device || !buf->device->vendor)
        return VBS_DRV_ERR_INVALID_ARG;
    /* Xe has no CPU_PREP ioctl: sysmem BOs are cache-coherent with
     * the GPU on x86 (snooped), and VRAM BAR mappings use the
     * PCI_BARRIER flag at mmap time (digest §7.1). No explicit
     * coherency step exists to perform. */
    return VBS_DRV_OK;
}

static void alchemist_cpu_end(vbs_drv_buffer_t *buf) {
    (void)buf; /* see alchemist_cpu_begin */
}

/* ------------------------------------------------------------------ */
/* Submission + fences — digest §3.3/§3.4                              */
/* ------------------------------------------------------------------ */

/* Wait for the previous submit (if any) and release its fence. The
 * batch buffer and state BO are single-buffered, so the caller
 * must call this before overwriting either. */
static vbs_drv_status_t alchemist_wait_idle(alchemist_dev_t *id) {
    if (id->last_fence == VBS_DRV_FENCE_NONE) return VBS_DRV_OK;
    vbs_drv_status_t st =
        vbs_drv_syncobj_wait(id->fd, (uint32_t)id->last_fence, 0,
                             g_last_error, sizeof(g_last_error));
    vbs_drv_syncobj_destroy(id->fd, (uint32_t)id->last_fence);
    id->last_fence = VBS_DRV_FENCE_NONE;
    return st;
}

static vbs_drv_status_t alchemist_submit(vbs_drv_device_t *dev,
                                     const vbs_drv_cmdbuf_t *cb,
                                     vbs_drv_fence_t *fence_out) {
    if (!dev || !dev->vendor || !cb || !cb->words || cb->count == 0 ||
        !fence_out)
        return VBS_DRV_ERR_INVALID_ARG;
    alchemist_dev_t *id = (alchemist_dev_t *)dev->vendor;
    if (id->fd < 0 || id->exec_queue_id == 0 || !id->batchbuf_ptr)
        return VBS_DRV_ERR_NOT_READY;
    /* Batch buffers must be QWord aligned and a multiple of QWords
     * in length (digest §4.1, PRM Vol 8 p.15). */
    if ((cb->count & 1u) != 0) return VBS_DRV_ERR_INVALID_ARG;
    size_t bytes = cb->count * sizeof(uint32_t);
    if (bytes > VBS_DRV_ALCHEMIST_BATCHBUF_BYTES)
        return VBS_DRV_ERR_INVALID_ARG;
    *fence_out = VBS_DRV_FENCE_NONE;
    /* The batch buffer is single-buffered: wait for the previous
     * submit to finish before overwriting it. */
    vbs_drv_status_t st = alchemist_wait_idle(id);
    if (st != VBS_DRV_OK) return st;
    /* Copy the command words into the mapped batch-buffer BO. */
    memcpy(id->batchbuf_ptr, cb->words, bytes);
    /* Create the signal syncobj, then EXEC (digest §3.3): address
     * = batch GPU VA, num_batch_buffer = 1 (must match the queue
     * width), one SYNCOBJ sync with SIGNAL. */
    uint32_t sync_handle = 0;
    st = vbs_drv_syncobj_create(id->fd, &sync_handle, g_last_error, sizeof(g_last_error));
    if (st != VBS_DRV_OK) return st;
    struct drm_xe_sync sync;
    memset(&sync, 0, sizeof(sync));
    sync.type = DRM_XE_SYNC_TYPE_SYNCOBJ;
    sync.flags = DRM_XE_SYNC_FLAG_SIGNAL;
    sync.handle = sync_handle;
    sync.timeline_value = 0;
    struct drm_xe_exec exec;
    memset(&exec, 0, sizeof(exec));
    exec.exec_queue_id = id->exec_queue_id;
    exec.num_syncs = 1;
    exec.syncs = (uint64_t)(uintptr_t)&sync;
    exec.address = id->batchbuf.va;
    exec.num_batch_buffer = 1;
    int r = vbs_drv_ioctl(id->fd, DRM_IOCTL_XE_EXEC, &exec);
    if (r != 0) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "DRM_IOCTL_XE_EXEC", -r);
        vbs_drv_syncobj_destroy(id->fd, sync_handle);
        return VBS_DRV_ERR_IO;
    }
    *fence_out = (vbs_drv_fence_t)sync_handle;
    id->last_fence = *fence_out;
    return VBS_DRV_OK;
}

static vbs_drv_status_t alchemist_fence_wait(vbs_drv_device_t *dev,
                                         vbs_drv_fence_t fence,
                                         uint32_t timeout_ms) {
    if (!dev || !dev->vendor) return VBS_DRV_ERR_INVALID_ARG;
    if (fence == VBS_DRV_FENCE_NONE) return VBS_DRV_OK;
    alchemist_dev_t *id = (alchemist_dev_t *)dev->vendor;
    if (id->fd < 0) return VBS_DRV_ERR_NOT_READY;
    return vbs_drv_syncobj_wait(id->fd, (uint32_t)fence, timeout_ms, g_last_error, sizeof(g_last_error));
}

static void alchemist_fence_free(vbs_drv_device_t *dev, vbs_drv_fence_t fence) {
    if (!dev || !dev->vendor || fence == VBS_DRV_FENCE_NONE) return;
    alchemist_dev_t *id = (alchemist_dev_t *)dev->vendor;
    if (id->fd < 0) return;
    if (id->last_fence == fence) id->last_fence = VBS_DRV_FENCE_NONE;
    vbs_drv_syncobj_destroy(id->fd, (uint32_t)fence);
}

/* ------------------------------------------------------------------ */
/* Launch — digest §4.9 + §5                                           */
/* ------------------------------------------------------------------ */

static vbs_drv_status_t alchemist_launch(vbs_drv_device_t *dev,
                                     const vbs_drv_launch_desc_t *desc) {
    if (!dev || !dev->vendor || !desc) return VBS_DRV_ERR_INVALID_ARG;
    alchemist_dev_t *id = (alchemist_dev_t *)dev->vendor;
    if (id->fd < 0 || id->exec_queue_id == 0 || !id->statebo_ptr)
        return VBS_DRV_ERR_NOT_READY;
    /* The code and constant buffers must belong to this device. */
    vbs_drv_status_t ost = vbs_drv_launch_buffers_on_device(dev, desc);
    if (ost != VBS_DRV_OK) return ost;
    /* Alchemist-specific launch constraints (the encoder re-checks
     * them; fail early with the specific reason here). */
    if ((desc->code_offset & 0x3fu) != 0) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "launch: code_offset is not 64-byte aligned", 0);
        return VBS_DRV_ERR_INVALID_ARG;
    }
    if (desc->shared_mem_bytes > VBS_DRV_ALCHEMIST_MAX_SLM_BYTES) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "launch: shared memory exceeds the 64 KiB SLM cap", 0);
        return VBS_DRV_ERR_INVALID_ARG;
    }
    if (desc->barrier_count > 7) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "launch: barrier count exceeds 7", 0);
        return VBS_DRV_ERR_INVALID_ARG;
    }
    /* The encoder's size fields are 32-bit; reject buffers that
     * would truncate (shader code and const buffers are KiBs in
     * practice — this is a guard, not a real limit). */
    if (desc->code->alloc_size > UINT32_MAX) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "launch: code buffer too large", 0);
        return VBS_DRV_ERR_INVALID_ARG;
    }
    for (uint32_t i = 0; i < desc->const_buf_count; i++)
        if (desc->const_bufs[i]->size > UINT32_MAX) {
            vbs_drv_note_error(g_last_error, sizeof(g_last_error), "launch: constant buffer too large", 0);
            return VBS_DRV_ERR_INVALID_ARG;
        }
    uint64_t sem_va = 0;
    if (desc->use_semaphore) {
        sem_va = desc->sem_buf->gpu_va + desc->sem_offset;
        if ((sem_va & 0x7u) != 0) {
            vbs_drv_note_error(g_last_error, sizeof(g_last_error), "launch: semaphore address is not 8-byte aligned", 0);
            return VBS_DRV_ERR_INVALID_ARG;
        }
    }
    /* The state BO is single-buffered: wait for the previous
     * launch before rewriting its binding table + surface states. */
    vbs_drv_status_t st = alchemist_wait_idle(id);
    if (st != VBS_DRV_OK) return st;

    /* Write the binding table + surface states into the state BO:
     * binding table at offset 0, surface state i at surface offset
     * + i * 64 bytes (digest §5.2/§5.3). */
    uint32_t *bt = (uint32_t *)((uint8_t *)id->statebo_ptr +
                                VBS_DRV_ALCHEMIST_STATE_BINDING_TABLE_OFF);
    uint8_t *ss_base = (uint8_t *)id->statebo_ptr +
                       VBS_DRV_ALCHEMIST_STATE_SURFACE_OFF;
    for (uint32_t i = 0; i < desc->const_buf_count; i++) {
        vbs_drv_alchemist_surface_state_t ss;
        st = vbs_drv_alchemist_surface_build(desc->const_bufs[i]->gpu_va,
                                         (uint32_t)desc->const_bufs[i]->size,
                                         &ss);
        if (st != VBS_DRV_OK) return st;
        memcpy(ss_base + i * VBS_DRV_ALCHEMIST_SURFACE_STATE_BYTES, &ss,
               sizeof(ss));
        bt[i] = vbs_drv_alchemist_binding_entry(
            i * VBS_DRV_ALCHEMIST_SURFACE_STATE_BYTES);
    }

    vbs_drv_alchemist_launch_params_t lp;
    memset(&lp, 0, sizeof(lp));
    lp.code_va = desc->code->gpu_va;
    lp.code_offset = (uint32_t)desc->code_offset;
    lp.code_size = (uint32_t)desc->code->alloc_size;
    lp.state_va = id->statebo.va;
    lp.const_count = desc->const_buf_count;
    lp.grid_x = desc->grid_x;
    lp.grid_y = desc->grid_y;
    lp.grid_z = desc->grid_z;
    lp.block_x = desc->block_x;
    lp.block_y = desc->block_y;
    lp.block_z = desc->block_z;
    lp.slm_bytes = desc->shared_mem_bytes;
    lp.barrier_count = desc->barrier_count;
    lp.use_semaphore = desc->use_semaphore;
    lp.sem_va = sem_va;
    lp.sem_payload = desc->sem_payload;

    vbs_drv_cmdbuf_t cb;
    st = vbs_drv_cmdbuf_init(&cb, 128);
    if (st != VBS_DRV_OK) return st;
    st = vbs_drv_alchemist_emit_launch_sequence(&cb, &lp);
    if (st != VBS_DRV_OK) {
        vbs_drv_cmdbuf_free(&cb);
        return st;
    }
    /* Synchronous launch: submit, wait for the completion fence,
     * then free it. */
    vbs_drv_fence_t fence = VBS_DRV_FENCE_NONE;
    st = alchemist_submit(dev, &cb, &fence);
    vbs_drv_cmdbuf_free(&cb);
    if (st != VBS_DRV_OK) return st;
    st = alchemist_fence_wait(dev, fence, VBS_DRV_ALCHEMIST_LAUNCH_TIMEOUT_MS);
    alchemist_fence_free(dev, fence);
    return st;
}

static vbs_drv_status_t alchemist_synchronize(vbs_drv_device_t *dev) {
    if (!dev || !dev->vendor) return VBS_DRV_ERR_INVALID_ARG;
    alchemist_dev_t *id = (alchemist_dev_t *)dev->vendor;
    if (id->last_fence == VBS_DRV_FENCE_NONE) return VBS_DRV_OK;
    return alchemist_wait_idle(id);
}

static const vbs_drv_ops_t g_alchemist_ops = {
    .name = "alchemist",
    .probe = alchemist_probe,
    .open = alchemist_open,
    .close = alchemist_close,
    .alloc_buffer = alchemist_alloc_buffer,
    .free_buffer = alchemist_free_buffer,
    .map_buffer = alchemist_map_buffer,
    .unmap_buffer = alchemist_unmap_buffer,
    .cpu_begin = alchemist_cpu_begin,
    .cpu_end = alchemist_cpu_end,
    .submit = alchemist_submit,
    .fence_wait = alchemist_fence_wait,
    .fence_free = alchemist_fence_free,
    .launch = alchemist_launch,
    .synchronize = alchemist_synchronize,
};

vbs_drv_status_t vbs_drv_alchemist_register(void) {
    return vbs_drv_register_ops(&g_alchemist_ops);
}

#else /* VBS_DRV_ALCHEMIST_HAVE_UAPI */
/* ---- Platform fallback: no Xe UAPI on this platform ---------------- */

static char g_fallback_err[] = "native Alchemist driver needs Linux + Xe DRM";

const char *vbs_drv_alchemist_last_error(void) { return g_fallback_err; }

vbs_drv_status_t vbs_drv_alchemist_register(void) {
    return vbs_drv_register_ops(vbs_drv_unavailable_ops("alchemist"));
}

#endif /* VBS_DRV_ALCHEMIST_HAVE_UAPI */
