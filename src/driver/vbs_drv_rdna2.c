/* vbs_drv_rdna2.c — AMD RDNA 2 driver: DRM layer (amdgpu UAPI).
 *
 * Talks to the GPU through the Linux kernel's amdgpu DRM driver
 * (render node + ioctls); it never touches MMIO directly. Flow
 * (digest §1.6): open render node → INFO queries → CTX ALLOC →
 * GEM_CREATE + GEM_VA MAP for code/data/IB buffers → BO_LIST →
 * CS submit (IB chunk + BO_HANDLES chunk) → WAIT_CS.
 *
 * The kernel wraps each submitted IB in ring packets (INDIRECT_
 * BUFFER) with the VMID itself — userspace never emits
 * INDIRECT_BUFFER packets (digest §2.6 [SI-PG §2.5.12]).
 *
 * VM note: amdgpu creates one GPU VM per DRM file descriptor in
 * the kernel; there is no userspace "create VM" ioctl (the
 * AMDGPU_VM ioctl only reserves/unreserves VMIDs, which the kernel
 * assigns per context automatically). "Binding" a buffer into the
 * VM is the GEM_VA MAP step below, with an explicit VA supplied by
 * the framework's VA allocator — the digest (§7.2) marks the
 * va_address = 0 behaviour UNSOURCED, so this driver always
 * requests an explicit VA.
 *
 * Firmware: the kernel loads the Sienna Cichlid blobs from
 * linux-firmware at bring-up (digest §8). This driver embeds and
 * ships NO firmware; open() verifies via INFO FW_VERSION that the
 * CP microcontrollers' firmware (ME, PFP, CE, MEC) is actually
 * loaded, and refuses to open when it is not — without MEC
 * firmware the compute queues cannot run.
 *
 * Fences: a framework fence is the CS sequence number + 1. The
 * +1 keeps fence 0 (VBS_DRV_FENCE_NONE, "no fence") distinct from
 * a real sequence number even if the kernel's first sequence
 * number is 0. fence_wait decodes it and issues WAIT_CS.
 */

#include "vbs_drv_rdna2.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Platform gate                                                       */
/* ------------------------------------------------------------------ */

#if defined(__linux__)
#  if defined(__has_include)
#    if __has_include(<drm/amdgpu_drm.h>) && __has_include(<drm/drm.h>)
#      define VBS_DRV_RDNA2_HAVE_UAPI 1
#    endif
#  else
#    define VBS_DRV_RDNA2_HAVE_UAPI 1
#  endif
#endif

#ifdef VBS_DRV_RDNA2_HAVE_UAPI
/* ---- Real implementation: Linux + amdgpu DRM UAPI ----------------- */

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <drm/drm.h>
#include <drm/amdgpu_drm.h>

/* Last detailed failure message (single-threaded use, like the
 * rest of v1). */
static char g_last_error[256];

const char *vbs_drv_rdna2_last_error(void) { return g_last_error; }

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */

/* First render node is renderD128 (digest §1.1 [DRM-UAPI]).
 * Scan 16 nodes; a machine with more GPUs than that is out of scope. */
#define VBS_DRV_RDNA2_FIRST_RENDER_NODE 128
#define VBS_DRV_RDNA2_MAX_RENDER_NODES  16

/* Persistent IB staging buffer (created at open): one
 * single-buffered 64 KiB GTT buffer, overwritten per submit after
 * waiting for the previous submission. */
#define VBS_DRV_RDNA2_IB_BYTES (64u * 1024u)

/* Launch completion timeout: a single kernel launch that runs over
 * a minute is treated as hung rather than hanging the engine
 * forever. */
#define VBS_DRV_RDNA2_LAUNCH_TIMEOUT_MS 60000u

/* GEM page size: buffers are page-rounded to 4 KiB. */
#define VBS_DRV_RDNA2_PAGE_SIZE 4096u

/* Max BO handles in one submission's BO_HANDLES chunk: the IB +
 * code + 8 const buffers = 10; 16 leaves headroom. */
#define VBS_DRV_RDNA2_MAX_SUBMIT_BOS 16u

/* Node paths found by the last probe, indexed by vendor-local index. */
static char g_node_paths[VBS_DRV_MAX_DEVICES][VBS_DRV_NODE_PATH_LEN];

/* ------------------------------------------------------------------ */
/* Device + buffer state                                               */
/* ------------------------------------------------------------------ */

/* One GEM buffer object: its kernel handle, its VA mapping, and
 * its CPU mapping (NULL when not host-visible). */
typedef struct {
    uint32_t handle;    /* GEM handle, 0 = none */
    uint64_t va;        /* GPU VA, 0 = unmapped */
    size_t   alloc_size;/* page-rounded size */
    void    *host_ptr;  /* CPU mapping, NULL if not mapped */
} rdna2_buf_t;

typedef struct {
    int fd;                 /* render-node fd, -1 = closed */
    uint32_t ctx_id;        /* amdgpu context, 0 = none */
    uint32_t ring;          /* compute ring index in use */
    uint32_t ib_start_align;/* from HW_IP_INFO (COMPUTE) */
    uint32_t ib_size_align; /* from HW_IP_INFO (COMPUTE) */
    uint32_t cu_count;      /* active CUs, from DEV_INFO */
    uint32_t wavefront_size;/* from DEV_INFO */
    uint32_t fw_me, fw_pfp, fw_ce, fw_mec; /* loaded FW versions */
    rdna2_buf_t ib;           /* persistent IB staging buffer */
    vbs_drv_fence_t last_fence; /* encoded seqno of last submit */
} rdna2_dev_t;

/* ------------------------------------------------------------------ */
/* INFO queries — digest §1.3/§1.4 [AMDGPU-DRM]                        */
/* ------------------------------------------------------------------ */

/* Generic INFO query: INFO is DRM_IOW — the result comes back
 * through the user-supplied return_pointer buffer (digest §1.2). */
static int rdna2_info(int fd, uint32_t query, void *out, uint32_t out_size,
                    uint32_t sel0, uint32_t sel1) {
    struct drm_amdgpu_info info;
    memset(&info, 0, sizeof(info));
    info.return_pointer = (uint64_t)(uintptr_t)out;
    info.return_size = out_size;
    info.query = query;
    /* The selector union is shared by the query kinds this driver
     * uses (query_hw_ip and query_fw both start with two u32s). */
    info.query_hw_ip.type = sel0;
    info.query_hw_ip.ip_instance = sel1;
    return vbs_drv_ioctl(fd, DRM_IOCTL_AMDGPU_INFO, &info);
}

static int rdna2_info_dev(int fd, struct drm_amdgpu_info_device *dev) {
    return rdna2_info(fd, AMDGPU_INFO_DEV_INFO, dev, (uint32_t)sizeof(*dev),
                    0, 0);
}

/* GC IP version check: Navi 21 is GC 10.3.0 (digest §9
 * [DGPU-CSV]). hw_ip_version_major/minor carry it; if the kernel
 * leaves them zero, fall back to ip_discovery_version (bits
 * 23:16 major, 15:8 minor, 7:0 revision). Returns true only for
 * 10.3.x — the register table is GC 10.3.0-specific, so a Navi 10
 * (GC 10.1.0) must not be driven with these offsets. Whether the
 * kernel fills major/minor for the GFX IP is a hardware-validation
 * item (see the driver report). */
static bool rdna2_gc_version_ok(int fd) {
    struct drm_amdgpu_info_hw_ip hwip;
    memset(&hwip, 0, sizeof(hwip));
    if (rdna2_info(fd, AMDGPU_INFO_HW_IP_INFO, &hwip,
                 (uint32_t)sizeof(hwip), AMDGPU_HW_IP_GFX, 0) != 0)
        return false;
    if (hwip.hw_ip_version_major != 0 || hwip.hw_ip_version_minor != 0)
        return hwip.hw_ip_version_major == VBS_DRV_RDNA2_GC_MAJOR &&
               hwip.hw_ip_version_minor == VBS_DRV_RDNA2_GC_MINOR;
    uint32_t disc = hwip.ip_discovery_version;
    return ((disc >> 16) & 0xFFu) == VBS_DRV_RDNA2_GC_MAJOR &&
           ((disc >> 8) & 0xFFu) == VBS_DRV_RDNA2_GC_MINOR;
}

/* Query one firmware version (digest §1.4 FW_VERSION sub-IDs).
 * Returns 0 on success and writes the version; the version is 0
 * when the kernel did not load that firmware. */
static int rdna2_fw_version(int fd, uint32_t fw_type, uint32_t *ver_out) {
    struct drm_amdgpu_info_firmware fw;
    memset(&fw, 0, sizeof(fw));
    /* query_fw is at the same union offset as query_hw_ip; the
     * generic helper's sel0/sel1 fill fw_type/ip_instance. */
    int r = rdna2_info(fd, AMDGPU_INFO_FW_VERSION, &fw,
                     (uint32_t)sizeof(fw), fw_type, 0);
    if (r == 0) *ver_out = fw.ver;
    return r;
}

/* ------------------------------------------------------------------ */
/* Probe                                                               */
/* ------------------------------------------------------------------ */

/* Node check for the framework's render-node scan: fd is an open
 * amdgpu node (the framework already confirmed the driver name).
 * The DEV_INFO family must be Navi (143) (digest §1.4) and the GC
 * IP must be 10.3.0 (Navi 21). Fills *info and returns true for a
 * device this driver handles. */
static bool rdna2_check_node(int fd, vbs_drv_device_info_t *info,
                             void *ctx) {
    (void)ctx;
    struct drm_amdgpu_info_device dinfo;
    memset(&dinfo, 0, sizeof(dinfo));
    if (rdna2_info_dev(fd, &dinfo) != 0 ||
        dinfo.family != VBS_DRV_RDNA2_FAMILY_NV)
        return false;
    if (!rdna2_gc_version_ok(fd))
        return false; /* Navi, but not a GC 10.3.0 part we drive */
    snprintf(info->name, sizeof(info->name),
             "AMD Navi 21 (RDNA 2, gfx1030, device 0x%04x)",
             dinfo.device_id);
    /* DEV_INFO carries no PCI vendor id; leave it 0 (unknown)
     * rather than asserting a value the digest does not give. */
    info->pci_device_id = (uint16_t)dinfo.device_id;
    info->chipset_id = dinfo.chip_rev;
    info->compute_units = dinfo.cu_active_number;
    /* Total VRAM, from the INFO MEMORY query [AMDGPU-DRM].
     * Failure is non-fatal: memory stays 0 (unknown). */
    struct drm_amdgpu_memory_info mem;
    memset(&mem, 0, sizeof(mem));
    if (rdna2_info(fd, AMDGPU_INFO_MEMORY, &mem,
                 (uint32_t)sizeof(mem), 0, 0) == 0)
        info->memory_bytes = mem.vram.total_heap_size;
    return true;
}

static vbs_drv_status_t rdna2_probe(vbs_drv_device_info_t *infos,
                                  size_t cap, size_t *count) {
    return vbs_drv_scan_render_nodes(VBS_DRV_RDNA2_FIRST_RENDER_NODE,
        VBS_DRV_RDNA2_MAX_RENDER_NODES, "amdgpu", rdna2_check_node,
        NULL, infos, cap, count, g_node_paths);
}

/* ------------------------------------------------------------------ */
/* GEM buffers — digest §7                                             */
/* ------------------------------------------------------------------ */

static size_t rdna2_round_page(size_t size) {
    return (size + VBS_DRV_RDNA2_PAGE_SIZE - 1u) &
           ~(size_t)(VBS_DRV_RDNA2_PAGE_SIZE - 1u);
}

/* Create a GEM buffer, bind it into the device's VA space at an
 * explicitly allocated VA, and optionally map it for the CPU.
 * domains/domain_flags follow digest §7.1; the VA mapping carries
 * READABLE|WRITEABLE|EXECUTABLE (digest §7.2: shader code mappings
 * need EXECUTABLE — the framework has no per-buffer "executable"
 * flag, so every mapping carries it; see the driver report). */
static vbs_drv_status_t rdna2_create_buffer(rdna2_dev_t *ad,
                                          vbs_drv_device_t *dev,
                                          size_t size, uint64_t domains,
                                          uint64_t domain_flags,
                                          uint64_t va_align,
                                          bool host_visible,
                                          rdna2_buf_t *out) {
    memset(out, 0, sizeof(*out));
    size_t alloc_size = rdna2_round_page(size);
    if (alloc_size == 0) return VBS_DRV_ERR_INVALID_ARG;

    union drm_amdgpu_gem_create create;
    memset(&create, 0, sizeof(create));
    create.in.bo_size = alloc_size;
    create.in.alignment = VBS_DRV_RDNA2_PAGE_SIZE;
    create.in.domains = domains;
    create.in.domain_flags = domain_flags;
    int r = vbs_drv_ioctl(ad->fd, DRM_IOCTL_AMDGPU_GEM_CREATE, &create);
    if (r != 0) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "DRM_IOCTL_AMDGPU_GEM_CREATE", -r);
        return VBS_DRV_ERR_IO;
    }
    out->handle = create.out.handle;
    out->alloc_size = alloc_size;

    /* Explicit VA from the framework allocator (digest §7.2: the
     * va_address = 0 behaviour is UNSOURCED — never rely on it). */
    uint64_t va = 0;
    vbs_drv_status_t st = vbs_drv_va_alloc(
        &dev->va_space, alloc_size, va_align, &va);
    if (st != VBS_DRV_OK) {
        vbs_drv_gem_close(ad->fd, out->handle);
        out->handle = 0;
        return st;
    }
    struct drm_amdgpu_gem_va gem_va;
    memset(&gem_va, 0, sizeof(gem_va));
    gem_va.handle = out->handle;
    gem_va.operation = AMDGPU_VA_OP_MAP;
    gem_va.flags = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE |
                   AMDGPU_VM_PAGE_EXECUTABLE;
    gem_va.va_address = va;
    gem_va.offset_in_bo = 0;
    gem_va.map_size = alloc_size;
    r = vbs_drv_ioctl(ad->fd, DRM_IOCTL_AMDGPU_GEM_VA, &gem_va);
    if (r != 0) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "DRM_IOCTL_AMDGPU_GEM_VA (MAP)", -r);
        vbs_drv_va_free(&dev->va_space, va, alloc_size);
        goto fail_gem;
    }
    out->va = va;

    if (host_visible) {
        union drm_amdgpu_gem_mmap mmap_arg;
        memset(&mmap_arg, 0, sizeof(mmap_arg));
        mmap_arg.in.handle = out->handle;
        r = vbs_drv_ioctl(ad->fd, DRM_IOCTL_AMDGPU_GEM_MMAP, &mmap_arg);
        if (r != 0) {
            vbs_drv_note_error(g_last_error, sizeof(g_last_error), "DRM_IOCTL_AMDGPU_GEM_MMAP", -r);
            goto fail_va;
        }
        void *ptr = mmap(NULL, alloc_size, PROT_READ | PROT_WRITE,
                         MAP_SHARED, ad->fd,
                         (off_t)mmap_arg.out.addr_ptr);
        if (ptr == MAP_FAILED) {
            vbs_drv_note_error(g_last_error, sizeof(g_last_error), "mmap (GEM buffer)", errno);
            goto fail_va;
        }
        out->host_ptr = ptr;
    }
    return VBS_DRV_OK;

fail_va:
    memset(&gem_va, 0, sizeof(gem_va));
    gem_va.handle = out->handle;
    gem_va.operation = AMDGPU_VA_OP_UNMAP;
    gem_va.va_address = out->va;
    gem_va.map_size = alloc_size;
    vbs_drv_ioctl(ad->fd, DRM_IOCTL_AMDGPU_GEM_VA, &gem_va);
    vbs_drv_va_free(&dev->va_space, out->va, alloc_size);
    out->va = 0;
fail_gem:
    vbs_drv_gem_close(ad->fd, out->handle);
    out->handle = 0;
    return VBS_DRV_ERR_IO;
}

static void rdna2_destroy_buffer(rdna2_dev_t *ad, vbs_drv_device_t *dev,
                               rdna2_buf_t *buf) {
    if (!buf || buf->handle == 0) return;
    if (buf->host_ptr) {
        munmap(buf->host_ptr, buf->alloc_size);
        buf->host_ptr = NULL;
    }
    if (buf->va != 0) {
        struct drm_amdgpu_gem_va gem_va;
        memset(&gem_va, 0, sizeof(gem_va));
        gem_va.handle = buf->handle;
        gem_va.operation = AMDGPU_VA_OP_UNMAP;
        gem_va.va_address = buf->va;
        gem_va.map_size = buf->alloc_size;
        vbs_drv_ioctl(ad->fd, DRM_IOCTL_AMDGPU_GEM_VA, &gem_va);
        if (dev) vbs_drv_va_free(&dev->va_space, buf->va,
                                 buf->alloc_size);
        buf->va = 0;
    }
    vbs_drv_gem_close(ad->fd, buf->handle);
    buf->handle = 0;
}

/* ------------------------------------------------------------------ */
/* Open / close                                                        */
/* ------------------------------------------------------------------ */

static vbs_drv_status_t rdna2_open(size_t index, vbs_drv_device_t *dev) {
    if (!dev || index >= VBS_DRV_MAX_DEVICES)
        return VBS_DRV_ERR_INVALID_ARG;
    rdna2_dev_t *ad = (rdna2_dev_t *)calloc(1, sizeof(rdna2_dev_t));
    if (!ad) return VBS_DRV_ERR_NO_MEMORY;
    ad->fd = -1;
    ad->last_fence = VBS_DRV_FENCE_NONE;
    dev->vendor = ad;

    /* FIRMWARE REQUIREMENT (digest §8): the kernel loads the 12
     * Sienna Cichlid blobs (amdgpu/sienna_cichlid_*.bin) from the
     * linux-firmware package at bring-up — 10 of them are
     * compute-relevant (ce, me, mec, mec2, pfp, rlc, sdma, smc,
     * sos, ta; dmcub and vcn are display/video only). This driver
     * embeds and ships NO firmware. Open verifies below that the
     * CP microcontrollers' firmware is actually loaded; if the
     * target machine lacks linux-firmware, or the kernel failed to
     * load a blob, there is no userspace workaround (digest §10
     * item 7) and open fails here with a clear reason. */
    int fd = open(g_node_paths[index], O_RDWR);
    if (fd < 0) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "open (render node)", errno);
        goto fail;
    }
    ad->fd = fd;

    /* Device query (digest §1.3): family, CUs, wavefront size,
     * and the VA limits — read at runtime, never hard-coded. */
    struct drm_amdgpu_info_device dinfo;
    memset(&dinfo, 0, sizeof(dinfo));
    if (rdna2_info_dev(fd, &dinfo) != 0) {
        vbs_drv_note_error_msg(g_last_error, sizeof(g_last_error), "AMDGPU_INFO_DEV_INFO query failed");
        goto fail;
    }
    if (dinfo.family != VBS_DRV_RDNA2_FAMILY_NV) {
        vbs_drv_note_error_msg(g_last_error, sizeof(g_last_error), "device is not an AMD Navi GPU (family != 143)");
        goto fail;
    }
    if (!rdna2_gc_version_ok(fd)) {
        vbs_drv_note_error_msg(g_last_error, sizeof(g_last_error), "device is not GC 10.3.0 (Navi 21); "
                      "this driver's register table is GC 10.3.0 only");
        goto fail;
    }
    ad->cu_count = dinfo.cu_active_number;
    ad->wavefront_size = dinfo.wave_front_size;
    dev->info.compute_units = dinfo.cu_active_number;
    dev->info.chipset_id = dinfo.chip_rev;
    dev->info.pci_device_id = (uint16_t)dinfo.device_id;

    /* Compute-ring properties (digest §1.3): read the IB
     * alignments at runtime and align indirect buffers
     * accordingly; pick the lowest available compute ring. */
    struct drm_amdgpu_info_hw_ip hwip;
    memset(&hwip, 0, sizeof(hwip));
    if (rdna2_info(fd, AMDGPU_INFO_HW_IP_INFO, &hwip,
                 (uint32_t)sizeof(hwip), AMDGPU_HW_IP_COMPUTE,
                 0) != 0) {
        vbs_drv_note_error_msg(g_last_error, sizeof(g_last_error), "AMDGPU_INFO_HW_IP_INFO (COMPUTE) query failed");
        goto fail;
    }
    if (hwip.available_rings == 0) {
        vbs_drv_note_error_msg(g_last_error, sizeof(g_last_error), "no compute rings available");
        goto fail;
    }
    ad->ring = 0;
    while (((hwip.available_rings >> ad->ring) & 1u) == 0) ad->ring++;
    ad->ib_start_align = hwip.ib_start_alignment != 0
        ? hwip.ib_start_alignment : VBS_DRV_RDNA2_PAGE_SIZE;
    ad->ib_size_align = hwip.ib_size_alignment != 0
        ? hwip.ib_size_alignment : sizeof(uint32_t);

    /* Firmware verification (digest §1.4, §8.2): the CP
     * microcontrollers' firmware must be loaded — PFP/ME/CE/MEC
     * are the CP microcontrollers and MEC controls the compute
     * queues (digest §8.1 [K-DOC driver-core]). A zero version
     * means the kernel did not load that blob. */
    const struct {
        uint32_t type;
        const char *name;
        uint32_t *slot;
    } fw_checks[] = {
        { AMDGPU_INFO_FW_GFX_ME, "me", &ad->fw_me },
        { AMDGPU_INFO_FW_GFX_PFP, "pfp", &ad->fw_pfp },
        { AMDGPU_INFO_FW_GFX_CE, "ce", &ad->fw_ce },
        { AMDGPU_INFO_FW_GFX_MEC, "mec", &ad->fw_mec },
    };
    for (size_t i = 0; i < sizeof(fw_checks) / sizeof(fw_checks[0]); i++) {
        uint32_t ver = 0;
        if (rdna2_fw_version(fd, fw_checks[i].type, &ver) != 0 ||
            ver == 0) {
            char msg[256];
            snprintf(msg, sizeof(msg),
                     "kernel did not load the %s firmware "
                     "(sienna_cichlid_%s.bin) — install linux-firmware",
                     fw_checks[i].name, fw_checks[i].name);
            vbs_drv_note_error_msg(g_last_error, sizeof(g_last_error), msg);
            goto fail;
        }
        *fw_checks[i].slot = ver;
    }

    /* Context allocation (digest §1.4: CTX op ALLOC = 1). */
    union drm_amdgpu_ctx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.in.op = AMDGPU_CTX_OP_ALLOC_CTX;
    ctx.in.priority = AMDGPU_CTX_PRIORITY_NORMAL;
    if (vbs_drv_ioctl(fd, DRM_IOCTL_AMDGPU_CTX, &ctx) != 0) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "DRM_IOCTL_AMDGPU_CTX (ALLOC)", errno);
        goto fail;
    }
    ad->ctx_id = ctx.out.alloc.ctx_id;

    /* VA space (digest §7.2): the UMD-managed range comes from
     * DEV_INFO at runtime. */
    if (dinfo.virtual_address_max <= dinfo.virtual_address_offset) {
        vbs_drv_note_error_msg(g_last_error, sizeof(g_last_error), "device reports an empty userspace VA range");
        goto fail;
    }
    dev->va_base = dinfo.virtual_address_offset;
    dev->va_size = dinfo.virtual_address_max -
                   dinfo.virtual_address_offset;
    vbs_drv_status_t st = vbs_drv_va_init(&dev->va_space, dev->va_base,
                                          dev->va_size);
    if (st != VBS_DRV_OK) goto fail;

    /* Persistent IB staging buffer: GTT (CPU-written every
     * submit), host-visible, VA-aligned to the kernel's IB start
     * alignment (digest §1.3). */
    uint64_t ib_align = ad->ib_start_align > VBS_DRV_RDNA2_PAGE_SIZE
        ? ad->ib_start_align : VBS_DRV_RDNA2_PAGE_SIZE;
    st = rdna2_create_buffer(ad, dev, VBS_DRV_RDNA2_IB_BYTES,
                           AMDGPU_GEM_DOMAIN_GTT,
                           AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED,
                           ib_align, true, &ad->ib);
    if (st != VBS_DRV_OK) goto fail;
    return VBS_DRV_OK;

fail:
    /* Close tolerates partially-initialized devices. */
    if (ad->ctx_id != 0) {
        union drm_amdgpu_ctx ctx_free;
        memset(&ctx_free, 0, sizeof(ctx_free));
        ctx_free.in.op = AMDGPU_CTX_OP_FREE_CTX;
        ctx_free.in.ctx_id = ad->ctx_id;
        vbs_drv_ioctl(ad->fd, DRM_IOCTL_AMDGPU_CTX, &ctx_free);
        ad->ctx_id = 0;
    }
    rdna2_destroy_buffer(ad, dev, &ad->ib);
    if (ad->fd >= 0) close(ad->fd);
    free(ad);
    dev->vendor = NULL;
    return VBS_DRV_ERR_NO_DEVICE;
}

static void rdna2_close(vbs_drv_device_t *dev) {
    if (!dev || !dev->vendor) return;
    rdna2_dev_t *ad = (rdna2_dev_t *)dev->vendor;
    rdna2_destroy_buffer(ad, dev, &ad->ib);
    if (ad->ctx_id != 0 && ad->fd >= 0) {
        union drm_amdgpu_ctx ctx;
        memset(&ctx, 0, sizeof(ctx));
        ctx.in.op = AMDGPU_CTX_OP_FREE_CTX;
        ctx.in.ctx_id = ad->ctx_id;
        vbs_drv_ioctl(ad->fd, DRM_IOCTL_AMDGPU_CTX, &ctx);
        ad->ctx_id = 0;
    }
    if (ad->fd >= 0) close(ad->fd);
    free(ad);
    dev->vendor = NULL;
}

/* ------------------------------------------------------------------ */
/* Buffers (framework ops)                                             */
/* ------------------------------------------------------------------ */

static vbs_drv_status_t rdna2_alloc_buffer(vbs_drv_device_t *dev,
                                         size_t size, uint32_t flags,
                                         vbs_drv_buffer_t *buf) {
    if (!dev || !dev->vendor || !buf || size == 0)
        return VBS_DRV_ERR_INVALID_ARG;
    rdna2_dev_t *ad = (rdna2_dev_t *)dev->vendor;
    if (ad->fd < 0) return VBS_DRV_ERR_NOT_READY;
    bool host_visible = (flags & VBS_DRV_BUF_HOST_VISIBLE) != 0;
    bool device_local = (flags & VBS_DRV_BUF_DEVICE_LOCAL) != 0;
    /* Domains (digest §7.1): shader code and data go in VRAM for
     * performance; GTT is the staging domain the CPU reads/writes
     * directly. `domains` is a bitmask — VRAM|GTT lets the kernel
     * fall back if VRAM is exhausted. */
    uint64_t domains;
    uint64_t domain_flags = 0;
    if (host_visible && device_local) {
        domains = AMDGPU_GEM_DOMAIN_VRAM | AMDGPU_GEM_DOMAIN_GTT;
        domain_flags = AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED;
    } else if (host_visible) {
        domains = AMDGPU_GEM_DOMAIN_GTT;
        domain_flags = AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED;
    } else if (device_local) {
        domains = AMDGPU_GEM_DOMAIN_VRAM | AMDGPU_GEM_DOMAIN_GTT;
        domain_flags = AMDGPU_GEM_CREATE_NO_CPU_ACCESS;
    } else {
        domains = AMDGPU_GEM_DOMAIN_VRAM | AMDGPU_GEM_DOMAIN_GTT;
    }
    rdna2_buf_t *ab = (rdna2_buf_t *)calloc(1, sizeof(rdna2_buf_t));
    if (!ab) return VBS_DRV_ERR_NO_MEMORY;
    vbs_drv_status_t st = rdna2_create_buffer(ad, dev, size, domains,
                                            domain_flags,
                                            VBS_DRV_RDNA2_PAGE_SIZE,
                                            host_visible, ab);
    if (st != VBS_DRV_OK) {
        free(ab);
        return st;
    }
    buf->size = size;
    buf->alloc_size = ab->alloc_size;
    buf->gpu_va = ab->va;
    buf->host_ptr = ab->host_ptr;
    buf->flags = flags;
    buf->vendor = ab;
    return VBS_DRV_OK;
}

static void rdna2_free_buffer(vbs_drv_buffer_t *buf) {
    if (!buf || !buf->vendor) return;
    rdna2_buf_t *ab = (rdna2_buf_t *)buf->vendor;
    rdna2_dev_t *ad = buf->device ? (rdna2_dev_t *)buf->device->vendor
                                : NULL;
    if (ad) rdna2_destroy_buffer(ad, buf->device, ab);
    free(ab);
    buf->vendor = NULL;
    buf->host_ptr = NULL;
    buf->gpu_va = 0;
}

static vbs_drv_status_t rdna2_map_buffer(vbs_drv_buffer_t *buf) {
    /* Mappings are persistent from alloc time; there is nothing to
     * do. A buffer that was never mapped is not host-visible. */
    if (!buf) return VBS_DRV_ERR_INVALID_ARG;
    return buf->host_ptr ? VBS_DRV_OK : VBS_DRV_ERR_NOT_SUPPORTED;
}

static void rdna2_unmap_buffer(vbs_drv_buffer_t *buf) {
    (void)buf; /* persistent mapping; nothing to do */
}

/* ------------------------------------------------------------------ */
/* Submission + fences — digest §1.6, §2.6                             */
/* ------------------------------------------------------------------ */

/* Absolute WAIT_CS timeout in nanoseconds (CLOCK_MONOTONIC):
 * timeout_ms == 0 waits forever. */
static uint64_t rdna2_wait_deadline_ns(uint32_t timeout_ms) {
    if (timeout_ms == 0) return UINT64_MAX;
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return UINT64_MAX;
    uint64_t now_ns = (uint64_t)now.tv_sec * 1000000000u +
                      (uint64_t)now.tv_nsec;
    uint64_t add_ns = (uint64_t)timeout_ms * 1000000u;
    if (now_ns > UINT64_MAX - add_ns) return UINT64_MAX;
    return now_ns + add_ns;
}

/* WAIT_CS on one sequence number (digest §1.3). Returns OK when
 * the work completed, TIMEOUT when the deadline passed first. */
static vbs_drv_status_t rdna2_wait_seqno(rdna2_dev_t *ad, uint64_t seqno,
                                       uint32_t timeout_ms) {
    union drm_amdgpu_wait_cs wait;
    memset(&wait, 0, sizeof(wait));
    wait.in.handle = seqno;
    wait.in.timeout = rdna2_wait_deadline_ns(timeout_ms);
    wait.in.ip_type = AMDGPU_HW_IP_COMPUTE;
    wait.in.ip_instance = 0;
    wait.in.ring = ad->ring;
    wait.in.ctx_id = ad->ctx_id;
    int r = vbs_drv_ioctl(ad->fd, DRM_IOCTL_AMDGPU_WAIT_CS, &wait);
    if (r == -ETIME) return VBS_DRV_ERR_TIMEOUT;
    if (r != 0) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "DRM_IOCTL_AMDGPU_WAIT_CS", -r);
        return VBS_DRV_ERR_IO;
    }
    /* out.status: 0 = completed, 1 = still busy (digest §1.3). */
    return wait.out.status == 0 ? VBS_DRV_OK : VBS_DRV_ERR_TIMEOUT;
}

/* Submit the words in `cb` on the COMPUTE ring, with `extra_bos`
 * (GEM handles the IB references, e.g. code + const buffers)
 * listed alongside the IB in the BO list. On success *fence_out
 * receives the encoded CS sequence number. */
static vbs_drv_status_t rdna2_submit_ib(rdna2_dev_t *ad,
                                      const vbs_drv_cmdbuf_t *cb,
                                      const uint32_t *extra_bos,
                                      size_t extra_count,
                                      vbs_drv_fence_t *fence_out) {
    size_t bytes = cb->count * sizeof(uint32_t);
    if (bytes > VBS_DRV_RDNA2_IB_BYTES) return VBS_DRV_ERR_INVALID_ARG;
    if (extra_count > VBS_DRV_RDNA2_MAX_SUBMIT_BOS - 1)
        return VBS_DRV_ERR_INVALID_ARG;
    /* The IB buffer is single-buffered: wait for the previous
     * submit to finish before overwriting it. */
    if (ad->last_fence != VBS_DRV_FENCE_NONE) {
        vbs_drv_status_t st =
            rdna2_wait_seqno(ad, ad->last_fence - 1, 0);
        ad->last_fence = VBS_DRV_FENCE_NONE;
        if (st != VBS_DRV_OK) return st;
    }
    /* Copy the words in, padding to the kernel's IB size
     * alignment with type-2 filler dwords (digest §1.3, §2.1). */
    memcpy(ad->ib.host_ptr, cb->words, bytes);
    size_t padded = bytes;
    if (ad->ib_size_align > 1) {
        size_t rem = padded % ad->ib_size_align;
        if (rem != 0) padded += ad->ib_size_align - rem;
    }
    if (padded > VBS_DRV_RDNA2_IB_BYTES) return VBS_DRV_ERR_INVALID_ARG;
    uint32_t *fill = (uint32_t *)((uint8_t *)ad->ib.host_ptr + bytes);
    for (size_t i = bytes / 4; i < padded / 4; i++)
        fill[i - bytes / 4] = VBS_DRV_RDNA2_PM4_TYPE2_FILLER;

    /* BO list: the IB plus every BO the IB touches (digest §1.6).
     * The kernel takes its own references at CS time, so the list
     * may be destroyed once the CS ioctl returns. */
    struct drm_amdgpu_bo_list_entry entries[VBS_DRV_RDNA2_MAX_SUBMIT_BOS];
    memset(entries, 0, sizeof(entries));
    entries[0].bo_handle = ad->ib.handle;
    entries[0].bo_priority = 0;
    for (size_t i = 0; i < extra_count; i++) {
        entries[1 + i].bo_handle = extra_bos[i];
        entries[1 + i].bo_priority = 0;
    }
    union drm_amdgpu_bo_list bo_list;
    memset(&bo_list, 0, sizeof(bo_list));
    bo_list.in.operation = AMDGPU_BO_LIST_OP_CREATE;
    bo_list.in.bo_number = (uint32_t)(1 + extra_count);
    bo_list.in.bo_info_size =
        (uint32_t)sizeof(struct drm_amdgpu_bo_list_entry);
    bo_list.in.bo_info_ptr = (uint64_t)(uintptr_t)entries;
    int r = vbs_drv_ioctl(ad->fd, DRM_IOCTL_AMDGPU_BO_LIST, &bo_list);
    if (r != 0) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "DRM_IOCTL_AMDGPU_BO_LIST (CREATE)", -r);
        return VBS_DRV_ERR_IO;
    }
    uint32_t list_handle = bo_list.out.list_handle;

    /* Chunks: one IB chunk + one BO_HANDLES chunk (digest §1.4,
     * §1.6). The IB chunk submits to ip_type COMPUTE (1), ring
     * ad->ring, with no IB flags — the SET_SH_REG packets live
     * directly in the dispatch IB (no preamble IB; the digest
     * §2.7 marks the kernel's preamble replay semantics for the
     * COMPUTE ring UNSOURCED, so v1 does not use the mechanism).
     * The BO_HANDLES chunk data is an array of u32 GEM handles,
     * length_dw = handle count — that element format is not
     * stated in the digest; it is the kernel UAPI's layout and is
     * a hardware-validation item (see the driver report). */
    struct drm_amdgpu_cs_chunk_ib ib_chunk;
    memset(&ib_chunk, 0, sizeof(ib_chunk));
    ib_chunk.flags = 0;
    ib_chunk.va_start = ad->ib.va;
    ib_chunk.ib_bytes = (uint32_t)padded;
    ib_chunk.ip_type = AMDGPU_HW_IP_COMPUTE;
    ib_chunk.ip_instance = 0;
    ib_chunk.ring = ad->ring;

    uint32_t handles[VBS_DRV_RDNA2_MAX_SUBMIT_BOS];
    for (size_t i = 0; i < 1 + extra_count; i++)
        handles[i] = entries[i].bo_handle;

    struct drm_amdgpu_cs_chunk chunks[2];
    memset(chunks, 0, sizeof(chunks));
    chunks[0].chunk_id = AMDGPU_CHUNK_ID_IB;
    chunks[0].length_dw =
        (uint32_t)(sizeof(ib_chunk) / sizeof(uint32_t));
    chunks[0].chunk_data = (uint64_t)(uintptr_t)&ib_chunk;
    chunks[1].chunk_id = AMDGPU_CHUNK_ID_BO_HANDLES;
    chunks[1].length_dw = (uint32_t)(1 + extra_count);
    chunks[1].chunk_data = (uint64_t)(uintptr_t)handles;
    uint64_t chunk_ptrs[2] = {
        (uint64_t)(uintptr_t)&chunks[0],
        (uint64_t)(uintptr_t)&chunks[1],
    };

    union drm_amdgpu_cs cs;
    memset(&cs, 0, sizeof(cs));
    cs.in.ctx_id = ad->ctx_id;
    cs.in.bo_list_handle = list_handle;
    cs.in.num_chunks = 2;
    cs.in.flags = 0;
    cs.in.chunks = (uint64_t)(uintptr_t)chunk_ptrs;
    r = vbs_drv_ioctl(ad->fd, DRM_IOCTL_AMDGPU_CS, &cs);
    uint64_t seqno = cs.out.handle;
    /* Destroy the BO list now (see the note above); the CS job
     * already holds its references. */
    union drm_amdgpu_bo_list bo_list_free;
    memset(&bo_list_free, 0, sizeof(bo_list_free));
    bo_list_free.in.operation = AMDGPU_BO_LIST_OP_DESTROY;
    bo_list_free.in.list_handle = list_handle;
    vbs_drv_ioctl(ad->fd, DRM_IOCTL_AMDGPU_BO_LIST, &bo_list_free);
    if (r != 0) {
        vbs_drv_note_error(g_last_error, sizeof(g_last_error), "DRM_IOCTL_AMDGPU_CS", -r);
        return VBS_DRV_ERR_IO;
    }
    /* Encode the fence as seqno + 1 so it can never equal
     * VBS_DRV_FENCE_NONE (0). */
    *fence_out = (vbs_drv_fence_t)(seqno + 1);
    ad->last_fence = *fence_out;
    return VBS_DRV_OK;
}

static vbs_drv_status_t rdna2_submit(vbs_drv_device_t *dev,
                                   const vbs_drv_cmdbuf_t *cb,
                                   vbs_drv_fence_t *fence_out) {
    if (!dev || !dev->vendor || !cb || !cb->words || cb->count == 0 ||
        !fence_out)
        return VBS_DRV_ERR_INVALID_ARG;
    rdna2_dev_t *ad = (rdna2_dev_t *)dev->vendor;
    if (ad->fd < 0 || ad->ib.handle == 0 || !ad->ib.host_ptr)
        return VBS_DRV_ERR_NOT_READY;
    *fence_out = VBS_DRV_FENCE_NONE;
    /* NOTE (framework gap): a generic submit carries no BO list,
     * so the words may only reference the IB itself. The launch
     * path below passes its buffers explicitly. */
    return rdna2_submit_ib(ad, cb, NULL, 0, fence_out);
}

static vbs_drv_status_t rdna2_fence_wait(vbs_drv_device_t *dev,
                                       vbs_drv_fence_t fence,
                                       uint32_t timeout_ms) {
    if (!dev || !dev->vendor) return VBS_DRV_ERR_INVALID_ARG;
    if (fence == VBS_DRV_FENCE_NONE) return VBS_DRV_OK;
    rdna2_dev_t *ad = (rdna2_dev_t *)dev->vendor;
    if (ad->fd < 0) return VBS_DRV_ERR_NOT_READY;
    return rdna2_wait_seqno(ad, fence - 1, timeout_ms);
}

static void rdna2_fence_free(vbs_drv_device_t *dev, vbs_drv_fence_t fence) {
    /* Sequence numbers need no cleanup. */
    (void)dev; (void)fence;
}

/* ------------------------------------------------------------------ */
/* Launch + synchronize                                                */
/* ------------------------------------------------------------------ */

static vbs_drv_status_t rdna2_launch(vbs_drv_device_t *dev,
                                   const vbs_drv_launch_desc_t *desc) {
    if (!dev || !dev->vendor || !desc)
        return VBS_DRV_ERR_INVALID_ARG;
    rdna2_dev_t *ad = (rdna2_dev_t *)dev->vendor;
    if (ad->fd < 0 || ad->ib.handle == 0 || !ad->ib.host_ptr)
        return VBS_DRV_ERR_NOT_READY;
    /* Build the launch IB (pure encoder; digest Appendix A.3). */
    vbs_drv_cmdbuf_t cb;
    vbs_drv_status_t st = vbs_drv_cmdbuf_init(&cb, 1024);
    if (st != VBS_DRV_OK) return st;
    st = vbs_drv_rdna2_build_launch_ib(desc, &cb);
    if (st != VBS_DRV_OK) {
        vbs_drv_cmdbuf_free(&cb);
        return st;
    }
    /* The IB references the code buffer and the const buffers;
     * they must all be in the submission's BO list. */
    uint32_t extra[VBS_DRV_RDNA2_MAX_SUBMIT_BOS - 1];
    size_t n_extra = 0;
    extra[n_extra++] = ((rdna2_buf_t *)desc->code->vendor)->handle;
    for (uint32_t i = 0; i < desc->const_buf_count; i++)
        extra[n_extra++] =
            ((rdna2_buf_t *)desc->const_bufs[i]->vendor)->handle;
    vbs_drv_fence_t fence = VBS_DRV_FENCE_NONE;
    st = rdna2_submit_ib(ad, &cb, extra, n_extra, &fence);
    vbs_drv_cmdbuf_free(&cb);
    if (st != VBS_DRV_OK) return st;
    /* Synchronous launch: wait for the grid to finish. */
    return rdna2_wait_seqno(ad, fence - 1,
                          VBS_DRV_RDNA2_LAUNCH_TIMEOUT_MS);
}

static vbs_drv_status_t rdna2_synchronize(vbs_drv_device_t *dev) {
    if (!dev || !dev->vendor) return VBS_DRV_ERR_INVALID_ARG;
    rdna2_dev_t *ad = (rdna2_dev_t *)dev->vendor;
    if (ad->fd < 0) return VBS_DRV_ERR_NOT_READY;
    /* WAIT_CS with handle ~0 waits for the latest sequence number
     * (digest §1.3). */
    return rdna2_wait_seqno(ad, ~0ULL, 0);
}

static const vbs_drv_ops_t g_rdna2_ops = {
    .name = "rdna2",
    .probe = rdna2_probe,
    .open = rdna2_open,
    .close = rdna2_close,
    .alloc_buffer = rdna2_alloc_buffer,
    .free_buffer = rdna2_free_buffer,
    .map_buffer = rdna2_map_buffer,
    .unmap_buffer = rdna2_unmap_buffer,
    .cpu_begin = NULL, /* GTT/VRAM CPU mappings need no prep step */
    .cpu_end = NULL,
    .submit = rdna2_submit,
    .fence_wait = rdna2_fence_wait,
    .fence_free = rdna2_fence_free,
    .launch = rdna2_launch,
    .synchronize = rdna2_synchronize,
};

vbs_drv_status_t vbs_drv_rdna2_register(void) {
    return vbs_drv_register_ops(&g_rdna2_ops);
}

#else /* VBS_DRV_RDNA2_HAVE_UAPI */
/* ---- Platform fallback: no amdgpu UAPI on this platform ---------------- */

static char g_fallback_err[] = "native RDNA 2 driver needs Linux + amdgpu DRM";

const char *vbs_drv_rdna2_last_error(void) { return g_fallback_err; }

vbs_drv_status_t vbs_drv_rdna2_register(void) {
    return vbs_drv_register_ops(vbs_drv_unavailable_ops("rdna2"));
}

#endif /* VBS_DRV_RDNA2_HAVE_UAPI */
