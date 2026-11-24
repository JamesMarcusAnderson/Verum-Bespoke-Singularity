/* vbs_drv.h — Verum native GPU driver framework (vendor-neutral).
 *
 * One framework, many vendors: every GPU vendor driver in the Verum
 * stack (Turing first, then RDNA 2, then Alchemist) plugs into this framework
 * through a single ops table (vbs_drv_ops_t). The framework owns what
 * is the same for every vendor:
 *
 *   - device discovery and open/close (via the registered ops tables),
 *   - the GPU virtual-address allocator for the userspace-managed VA
 *     region (pure logic, no hardware),
 *   - the command-buffer word container (encoding is the vendor's job),
 *   - the vendor-neutral kernel-launch description (vbs_drv_launch_desc_t),
 *   - buffer bookkeeping and the host<->device copy helpers,
 *   - fences (submit returns a vendor fence handle; the framework
 *     wraps wait/free and offers a synchronous submit-and-wait),
 *   - the shared vendor helpers below (platform-fallback ops, the
 *     ioctl wrapper, last-error formatting, generic DRM object
 *     handling, and render-node scanning), so no vendor re-implements
 *     them.
 *
 * The vendor layer owns what differs: the kernel interface (ioctls),
 * buffer creation/binding, command encoding, and translating a launch
 * description into hardware launch state (e.g. a QMD on Turing).
 *
 * All identifiers here are Verum-native (vbs_drv_*). No vendor names
 * appear in the framework API; the vendor is named only by the ops
 * table's `name` string, which identifies the hardware family it drives.
 *
 * Threading: v1 is single-threaded by design (the engine dispatches
 * from one thread). No internal locking is provided.
 *
 * This file is pure C11 and compiles on any platform; the vendor layers
 * degrade to "no device" fallback ops where their kernel interface is absent.
 */

/* C11-only header: the Verum driver stack has no C++ consumers (no
 * C++ translation unit in the tree includes it), so there are no
 * extern "C" guards here. */

#ifndef VBS_DRV_H
#define VBS_DRV_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* ------------------------------------------------------------------ */
/* Status codes                                                        */
/* ------------------------------------------------------------------ */

typedef enum {
    VBS_DRV_OK = 0,
    VBS_DRV_ERR_NO_DEVICE,    /* no usable GPU found / device not open */
    VBS_DRV_ERR_NO_MEMORY,    /* host or device allocation failed */
    VBS_DRV_ERR_NOT_SUPPORTED,/* operation not supported by this driver */
    VBS_DRV_ERR_IO,           /* kernel interface call failed */
    VBS_DRV_ERR_INVALID_ARG,  /* bad argument (NULL, zero size, out of range) */
    VBS_DRV_ERR_TIMEOUT,      /* fence wait timed out */
    VBS_DRV_ERR_BUSY,         /* device busy; retry later */
    VBS_DRV_ERR_NOT_READY,    /* device not initialized / VA space not set up */
    VBS_DRV_ERR_HW_FAULT      /* the GPU reported a fault on submitted work */
} vbs_drv_status_t;

/* Human-readable status string; never NULL. */
const char *vbs_drv_status_string(vbs_drv_status_t st);

/* ------------------------------------------------------------------ */
/* Device information (vendor-neutral)                                 */
/* ------------------------------------------------------------------ */

typedef struct {
    char     name[128];       /* human-readable device name */
    uint16_t pci_vendor_id;   /* PCI vendor id, 0 if unknown */
    uint16_t pci_device_id;   /* PCI device id, 0 if unknown */
    uint32_t chipset_id;      /* vendor-reported chip id, 0 if unknown */
    uint64_t memory_bytes;    /* total device memory, 0 if unknown */
    uint32_t compute_units;   /* SMs / CUs, 0 if unknown */
    uint32_t max_push_count;  /* max pushbuffers per submit, 0 if unknown */
} vbs_drv_device_info_t;

/* ------------------------------------------------------------------ */
/* GPU virtual-address allocator (pure logic, no hardware)             */
/*                                                                     */
/* The userspace driver manages its own VA region (the kernel owns    */
/* the rest; see the vendor layer's VM setup). This is a first-fit    */
/* free-list allocator over [base, base+size) with coalescing on     */
/* free. All addresses and sizes are in bytes; allocations are        */
/* aligned to `align` (a power of two, at least 1).                   */
/* ------------------------------------------------------------------ */

#define VBS_DRV_VA_MAX_BLOCKS 512

typedef struct {
    uint64_t base;
    uint64_t size;
    struct {
        uint64_t addr;
        uint64_t size;
    } free_blocks[VBS_DRV_VA_MAX_BLOCKS];
    size_t free_count;
    bool   ready;
} vbs_drv_va_space_t;

/* Initialize a VA space over [base, base+size). size must be > 0 and
 * base+size must not overflow. */
vbs_drv_status_t vbs_drv_va_init(vbs_drv_va_space_t *va,
                                 uint64_t base, uint64_t size);

/* Allocate `size` bytes aligned to `align`. Writes the VA to *out_addr. */
vbs_drv_status_t vbs_drv_va_alloc(vbs_drv_va_space_t *va, uint64_t size,
                                  uint64_t align, uint64_t *out_addr);

/* Return a previously allocated range. The (addr, size) pair must match
 * an earlier allocation exactly; freeing an unknown range is an error
 * and leaves the allocator untouched. */
vbs_drv_status_t vbs_drv_va_free(vbs_drv_va_space_t *va,
                                 uint64_t addr, uint64_t size);

/* ------------------------------------------------------------------ */
/* Buffers                                                             */
/* ------------------------------------------------------------------ */

#define VBS_DRV_BUF_HOST_VISIBLE (1u << 0) /* CPU can map it */
#define VBS_DRV_BUF_DEVICE_LOCAL (1u << 1) /* prefer device-local memory */

typedef struct vbs_drv_device vbs_drv_device_t;
typedef struct vbs_drv_buffer vbs_drv_buffer_t;

struct vbs_drv_buffer {
    vbs_drv_device_t *device;
    size_t   size;        /* requested size */
    size_t   alloc_size;  /* actual allocated size (page-rounded) */
    uint64_t gpu_va;      /* GPU virtual address, 0 if unmapped */
    void    *host_ptr;    /* CPU mapping, NULL if not CPU-mapped */
    uint32_t flags;       /* VBS_DRV_BUF_* */
    void    *vendor;      /* vendor-private state (GEM handle, etc.) */
};

/* ------------------------------------------------------------------ */
/* Command buffers (word container; encoding is the vendor's job)      */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t *words;
    size_t    count;
    size_t    capacity;
} vbs_drv_cmdbuf_t;

vbs_drv_status_t vbs_drv_cmdbuf_init(vbs_drv_cmdbuf_t *cb,
                                     size_t capacity_words);
void             vbs_drv_cmdbuf_free(vbs_drv_cmdbuf_t *cb);
vbs_drv_status_t vbs_drv_cmdbuf_push(vbs_drv_cmdbuf_t *cb, uint32_t word);
vbs_drv_status_t vbs_drv_cmdbuf_push_many(vbs_drv_cmdbuf_t *cb,
                                          const uint32_t *words, size_t n);
void             vbs_drv_cmdbuf_reset(vbs_drv_cmdbuf_t *cb);

/* ------------------------------------------------------------------ */
/* Kernel launch description (vendor-neutral)                          */
/*                                                                     */
/* The caller provides native shader code already uploaded into        */
/* `code` (the framework never produces shader code), plus the         */
/* constant buffers the kernel reads its arguments from. The vendor    */
/* layer translates this into hardware launch state (on Turing: a      */
/* QMD v2.3 + a pushbuffer launch sequence).                           */
/* ------------------------------------------------------------------ */

#define VBS_DRV_MAX_CONST_BUFFERS 8

typedef struct {
    vbs_drv_buffer_t *code;        /* buffer holding native shader code */
    uint64_t          code_offset; /* byte offset of the kernel entry */
    uint32_t          code_size;   /* bytes of shader code */
    uint32_t          register_count;   /* registers per thread */
    uint32_t          shared_mem_bytes; /* shared memory per block */
    uint32_t          barrier_count;    /* 0 unless the kernel uses barriers */
    uint32_t          grid_x, grid_y, grid_z;
    uint32_t          block_x, block_y, block_z;
    vbs_drv_buffer_t *const_bufs[VBS_DRV_MAX_CONST_BUFFERS];
    uint32_t          const_buf_count;  /* 0..VBS_DRV_MAX_CONST_BUFFERS */
    /* Optional hardware completion signal: when use_semaphore is set,
     * the hardware writes sem_payload to sem_buf+sem_offset when the
     * grid finishes. The vendor layer may decline (NOT_SUPPORTED) if
     * its hardware cannot express it. */
    vbs_drv_buffer_t *sem_buf;
    uint64_t          sem_offset;
    uint32_t          sem_payload;
    bool              use_semaphore;
} vbs_drv_launch_desc_t;

/* ------------------------------------------------------------------ */
/* Vendor ops table — one per hardware family                          */
/* ------------------------------------------------------------------ */

/* A fence is an opaque vendor handle (e.g. a DRM syncobj id).
 * VBS_DRV_FENCE_NONE means "no fence". */
typedef uint64_t vbs_drv_fence_t;
#define VBS_DRV_FENCE_NONE ((vbs_drv_fence_t)0)

typedef struct vbs_drv_ops {
    /* Hardware family this ops table drives, e.g. "turing". */
    const char *name;

    /* Enumerate devices this driver can handle. Fills infos[0..*count-1]
     * with at most `cap` entries and sets *count. Returns OK even when
     * no device is found (*count == 0). */
    vbs_drv_status_t (*probe)(vbs_drv_device_info_t *infos, size_t cap,
                              size_t *count);

    /* Open the `index`-th device found by probe (vendor-local index).
     * Fills dev->info and dev->vendor, sets dev->va_base/va_size for
     * the userspace-managed VA region (0/0 if the vendor does not
     * manage VA space), and initializes dev->va_space. On failure the
     * device is left in a state close() can clean up. */
    vbs_drv_status_t (*open)(size_t index, vbs_drv_device_t *dev);

    /* Release all vendor state (fd, channel, persistent buffers). Must
     * tolerate partially-initialized devices (failed open). */
    void (*close)(vbs_drv_device_t *dev);

    /* Allocate a device buffer, bind it into GPU VA space, and map it
     * for the CPU when VBS_DRV_BUF_HOST_VISIBLE is set. Fills *buf. */
    vbs_drv_status_t (*alloc_buffer)(vbs_drv_device_t *dev, size_t size,
                                     uint32_t flags, vbs_drv_buffer_t *buf);

    /* Release a buffer allocated by alloc_buffer (unbind, unmap, free). */
    void (*free_buffer)(vbs_drv_buffer_t *buf);

    /* Ensure buf->host_ptr is valid for CPU access (a no-op when the
     * vendor maps persistently at alloc time). */
    vbs_drv_status_t (*map_buffer)(vbs_drv_buffer_t *buf);
    void (*unmap_buffer)(vbs_drv_buffer_t *buf);

    /* Bracket CPU access to a mapped buffer (cache-coherency / prep).
     * for_write is true when the CPU is about to write. */
    vbs_drv_status_t (*cpu_begin)(vbs_drv_buffer_t *buf, bool for_write);
    void (*cpu_end)(vbs_drv_buffer_t *buf);

    /* Submit a command buffer. On success *fence_out receives a fence
     * signaled when the submitted work completes (or
     * VBS_DRV_FENCE_NONE if the vendor submits synchronously). */
    vbs_drv_status_t (*submit)(vbs_drv_device_t *dev,
                               const vbs_drv_cmdbuf_t *cb,
                               vbs_drv_fence_t *fence_out);

    /* Wait for a fence. timeout_ms == 0 waits forever. */
    vbs_drv_status_t (*fence_wait)(vbs_drv_device_t *dev,
                                   vbs_drv_fence_t fence,
                                   uint32_t timeout_ms);
    void (*fence_free)(vbs_drv_device_t *dev, vbs_drv_fence_t fence);

    /* Build the hardware launch state for `desc`, submit it, and wait
     * for completion. Synchronous: returns only after the grid has
     * finished (or failed). */
    vbs_drv_status_t (*launch)(vbs_drv_device_t *dev,
                               const vbs_drv_launch_desc_t *desc);

    /* Wait for all outstanding work on the device. */
    vbs_drv_status_t (*synchronize)(vbs_drv_device_t *dev);
} vbs_drv_ops_t;

/* ------------------------------------------------------------------ */
/* Device handle (framework-owned; vendor fills `vendor`)              */
/* ------------------------------------------------------------------ */

struct vbs_drv_device {
    const vbs_drv_ops_t *ops;
    vbs_drv_device_info_t info;
    void *vendor;              /* vendor-private state */
    uint64_t va_base;          /* userspace-managed VA region start */
    uint64_t va_size;          /* userspace-managed VA region size */
    vbs_drv_va_space_t va_space;
};

/* ------------------------------------------------------------------ */
/* Framework API                                                       */
/* ------------------------------------------------------------------ */

/* Register a vendor ops table. Idempotent for the same table pointer;
 * registering more than VBS_DRV_MAX_VENDORS tables fails. */
#define VBS_DRV_MAX_VENDORS 8
vbs_drv_status_t vbs_drv_register_ops(const vbs_drv_ops_t *ops);

/* ------------------------------------------------------------------ */
/* Shared helpers for the vendor drivers                               */
/*                                                                     */
/* Logic every vendor driver needs identically lives here, not in    */
/* three copies: the platform-fallback ops table, the EINTR-retry    */
/* ioctl wrapper, last-error formatting, generic DRM object          */
/* handling (GEM close, syncobjs), and render-node scanning. The     */
/* Linux-only helpers degrade gracefully where the kernel interface  */
/* is absent (ioctl returns -1, a scan finds nothing, syncobj calls */
/* fail NOT_SUPPORTED); vendors only call them from their Linux      */
/* branches.                                                           */
/* ------------------------------------------------------------------ */

/* Platform-fallback ops table for one hardware family: every
 * operation reports "no device". A vendor whose kernel interface
 * is absent on this platform registers this instead of real ops,
 * so the library still builds and probes cleanly. The returned
 * table stays valid for the life of the process; NULL only when
 * the table pool is exhausted (VBS_DRV_MAX_VENDORS families).
 * `family` must outlive the call (pass a string literal). */
const vbs_drv_ops_t *vbs_drv_unavailable_ops(const char *family);

/* ioctl with EINTR retry. Returns 0 on success, -errno on failure
 * (-1 where no kernel interface exists). */
int vbs_drv_ioctl(int fd, unsigned long req, void *arg);

/* Format a last-error message into buf: "what: <strerror(errnum)>"
 * (or "what: unknown error" when errnum is 0). Never writes past
 * cap; does nothing when buf is NULL or cap is 0. */
void vbs_drv_note_error(char *buf, size_t cap, const char *what,
                        int errnum);

/* Format a plain last-error message into buf (no errno suffix). */
void vbs_drv_note_error_msg(char *buf, size_t cap, const char *msg);

/* Close a GEM handle on fd (DRM_IOCTL_GEM_CLOSE). Best-effort:
 * ignores errors and does nothing for fd < 0 or handle 0. */
void vbs_drv_gem_close(int fd, uint32_t handle);

/* A buffer allocation failed after the vendor's retries: drop the
 * half-built vendor shell, detach it from the buffer, and hand
 * the failure status back to the caller. */
vbs_drv_status_t vbs_drv_drop_failed_buffer(vbs_drv_buffer_t *buf,
                                            void *vendor_shell,
                                            vbs_drv_status_t st);

/* Create a DRM syncobj on fd. On failure formats the reason into
 * err_buf (when non-NULL) and returns VBS_DRV_ERR_IO. */
vbs_drv_status_t vbs_drv_syncobj_create(int fd, uint32_t *handle_out,
                                        char *err_buf, size_t err_cap);

/* Wait for a syncobj: timeout_ms == 0 waits forever. Returns
 * VBS_DRV_ERR_TIMEOUT on expiry, VBS_DRV_ERR_IO on other failure
 * (reason in err_buf when non-NULL). */
vbs_drv_status_t vbs_drv_syncobj_wait(int fd, uint32_t handle,
                                      uint32_t timeout_ms,
                                      char *err_buf, size_t err_cap);

/* Destroy a syncobj. Best-effort: ignores errors (cleanup paths). */
void vbs_drv_syncobj_destroy(int fd, uint32_t handle);

/* Per-node probe callback for vbs_drv_scan_render_nodes: fd is an
 * open render node already confirmed to belong to the vendor's
 * kernel driver. Fill *info and return true to keep the device,
 * false to skip it. The framework closes fd either way. */
typedef bool (*vbs_drv_node_check_t)(int fd, vbs_drv_device_info_t *info,
                                     void *ctx);

/* Render-node path length ("/dev/dri/renderD255" fits with room). */
#define VBS_DRV_NODE_PATH_LEN 64

/* Scan render nodes [first_node, first_node+node_count): open each,
 * confirm the kernel driver name via DRM_IOCTL_VERSION (generic and
 * safe on every node), and run `check` on matches. Kept devices
 * land in infos[] (at most cap) with their node paths in
 * node_paths[] (same index, VBS_DRV_MAX_DEVICES entries of
 * VBS_DRV_NODE_PATH_LEN bytes). Sets *count; returns OK even when
 * no device is found. A scan never fails. */
vbs_drv_status_t vbs_drv_scan_render_nodes(
    int first_node, int node_count, const char *driver_name,
    vbs_drv_node_check_t check, void *ctx,
    vbs_drv_device_info_t *infos, size_t cap, size_t *count,
    char (*node_paths)[VBS_DRV_NODE_PATH_LEN]);

/* Probe all registered vendors. Fills infos[0..*count-1] with at most
 * `cap` entries (global device indices, in registration order) and
 * sets *count. Returns OK even when no device is found. A probe
 * refreshes the framework's device table; open a device by its global
 * index from the most recent probe. */
#define VBS_DRV_MAX_DEVICES 16
vbs_drv_status_t vbs_drv_probe_all(vbs_drv_device_info_t *infos, size_t cap,
                                   size_t *count);

/* Open the device at global index `index` (from the last probe).
 * On success *dev_out receives a framework-owned device handle. */
vbs_drv_status_t vbs_drv_open(size_t index, vbs_drv_device_t **dev_out);

/* Close a device opened by vbs_drv_open. NULL-safe. */
void vbs_drv_close(vbs_drv_device_t *dev);

/* Buffer lifecycle (thin wrappers over the device's ops). */
vbs_drv_status_t vbs_drv_alloc_buffer(vbs_drv_device_t *dev, size_t size,
                                      uint32_t flags, vbs_drv_buffer_t **out);
void             vbs_drv_free_buffer(vbs_drv_buffer_t *buf);
vbs_drv_status_t vbs_drv_map_buffer(vbs_drv_buffer_t *buf);
void             vbs_drv_unmap_buffer(vbs_drv_buffer_t *buf);
vbs_drv_status_t vbs_drv_cpu_begin(vbs_drv_buffer_t *buf, bool for_write);
void             vbs_drv_cpu_end(vbs_drv_buffer_t *buf);

/* Host<->device copies via the buffer's CPU mapping. Fails cleanly
 * (INVALID_ARG) when the buffer is not host-visible or bytes exceeds
 * the buffer size. */
vbs_drv_status_t vbs_drv_copy_to_device(vbs_drv_buffer_t *dst,
                                        const void *src, size_t bytes);
vbs_drv_status_t vbs_drv_copy_to_host(void *dst,
                                      const vbs_drv_buffer_t *src,
                                      size_t bytes);

/* Submission and fences. */
vbs_drv_status_t vbs_drv_submit(vbs_drv_device_t *dev,
                                const vbs_drv_cmdbuf_t *cb,
                                vbs_drv_fence_t *fence_out);
vbs_drv_status_t vbs_drv_fence_wait(vbs_drv_device_t *dev,
                                    vbs_drv_fence_t fence,
                                    uint32_t timeout_ms);
void             vbs_drv_fence_free(vbs_drv_device_t *dev,
                                    vbs_drv_fence_t fence);

/* Submit and wait for completion (timeout_ms == 0 waits forever). */
vbs_drv_status_t vbs_drv_submit_and_wait(vbs_drv_device_t *dev,
                                         const vbs_drv_cmdbuf_t *cb,
                                         uint32_t timeout_ms);

/* Validate a launch description's shape (code buffer, sizes,
 * grid/block dimensions, constant buffers) — the checks every
 * vendor and the framework agree on. Does NOT check the
 * semaphore; see vbs_drv_validate_launch_semaphore. The encoders
 * call this so a directly-tested encoder rejects bad input the
 * same way the framework path does. */
vbs_drv_status_t vbs_drv_validate_launch_desc(
    const vbs_drv_launch_desc_t *desc);

/* Validate the semaphore half of a launch description: when
 * use_semaphore is set, the semaphore buffer must exist, be
 * mapped, and hold the payload dword. Returns OK when the
 * description does not use a semaphore. */
vbs_drv_status_t vbs_drv_validate_launch_semaphore(
    const vbs_drv_launch_desc_t *desc);

/* Check that every buffer in a launch description belongs to dev
 * (code, constant buffers, semaphore buffer). Returns OK or
 * VBS_DRV_ERR_INVALID_ARG. Callers validate the shape first. */
vbs_drv_status_t vbs_drv_launch_buffers_on_device(
    const vbs_drv_device_t *dev, const vbs_drv_launch_desc_t *desc);

/* Kernel launch (validates the description, then calls the ops). */
vbs_drv_status_t vbs_drv_launch(vbs_drv_device_t *dev,
                                const vbs_drv_launch_desc_t *desc);

/* Wait for all outstanding work on the device. */
vbs_drv_status_t vbs_drv_synchronize(vbs_drv_device_t *dev);

#endif /* VBS_DRV_H */
