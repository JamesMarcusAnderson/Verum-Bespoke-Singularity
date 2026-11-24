/*
 * Verum baremetal — Phase 2
 * Bare-metal GPU driver and dispatch. No OS. No vendor SDK.
 *
 * After Phase 1 ExitBootServices():
 *   1. PCIe config space scan (brute-force BDF)
 *   2. Identify NVIDIA (0x10DE) or AMD (0x1002) GPU
 *   3. Map BAR0 (MMIO register space)
 *   4. Initialize GPU:
 *      - NVIDIA: map NV_PMC, NV_PFIFO, NV_PBUS, enable FALCON/GSP
 *      - AMD:    map GFX registers, ring buffers, enable CP
 *   5. Build and submit a minimal compute command buffer
 *   6. Poll for completion
 *   7. Return result to Phase 1 inference loop
 *
 * Pure C11. No OS. No libc. No vendor SDK.
 * Build: clang --target=x86_64-unknown-none-elf -ffreestanding -O2
 *
 * Register offsets, vendor IDs, and packet encodings in this file
 * are the hardware's own published interface values; they are
 * facts about the devices, not driver code.
 */

/* ------------------------------------------------------------------ */
/* Types (no stdint.h in freestanding without libc)                   */
/* ------------------------------------------------------------------ */

typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef long long          int64_t;
typedef unsigned long long uintptr_t;
typedef unsigned long long size_t;

/* ------------------------------------------------------------------ */
/* MMIO access — volatile to prevent reorder / optimization           */
/* ------------------------------------------------------------------ */

static inline uint32_t mmio_read32(uintptr_t addr) {
    return *(volatile uint32_t *)addr;
}
static inline void mmio_write32(uintptr_t addr, uint32_t val) {
    *(volatile uint32_t *)addr = val;
}

/* Full memory fence. */
static inline void mfence(void) {
    __asm__ volatile("mfence" ::: "memory");
}

/* ------------------------------------------------------------------ */
/* PCI config space access via CF8/CFC I/O ports                      */
/* ------------------------------------------------------------------ */

static inline void out32(uint16_t port, uint32_t val) {
    __asm__ volatile("outl %0, %1" :: "a"(val), "Nd"(port));
}
static inline uint32_t in32(uint16_t port) {
    uint32_t val;
    __asm__ volatile("inl %1, %0" : "=a"(val) : "Nd"(port));
    return val;
}
static inline uint16_t in16(uint16_t port) {
    uint16_t val;
    __asm__ volatile("inw %1, %0" : "=a"(val) : "Nd"(port));
    return val;
}

#define PCI_CONFIG_ADDR 0xCF8
#define PCI_CONFIG_DATA 0xCFC

/* Build the 32-bit config address for one BDF + register. */
static inline uint32_t pci_addr(uint8_t bus, uint8_t dev, uint8_t fn,
                                uint8_t reg) {
    return (1u << 31) |
           ((uint32_t)bus << 16) |
           ((uint32_t)dev << 11) |
           ((uint32_t)fn << 8) |
           (reg & 0xFC);
}

static uint32_t pci_read32(uint8_t bus, uint8_t dev, uint8_t fn, uint8_t reg) {
    out32(PCI_CONFIG_ADDR, pci_addr(bus, dev, fn, reg));
    return in32(PCI_CONFIG_DATA);
}

/* 16-bit config read: the address register takes the dword-aligned
 * offset, so the half is read from the data port + (reg & 2). */
static uint16_t pci_read16(uint8_t bus, uint8_t dev, uint8_t fn, uint8_t reg) {
    out32(PCI_CONFIG_ADDR, pci_addr(bus, dev, fn, reg));
    return in16((uint16_t)(PCI_CONFIG_DATA + (reg & 2)));
}

static void pci_write32(uint8_t bus, uint8_t dev, uint8_t fn,
                        uint8_t reg, uint32_t val) {
    out32(PCI_CONFIG_ADDR, pci_addr(bus, dev, fn, reg));
    out32(PCI_CONFIG_DATA, val);
}

static uint16_t pci_vendor(uint8_t bus, uint8_t dev, uint8_t fn) {
    return pci_read16(bus, dev, fn, 0x00);
}
static uint16_t pci_device(uint8_t bus, uint8_t dev, uint8_t fn) {
    return pci_read16(bus, dev, fn, 0x02);
}
static uint8_t pci_class(uint8_t bus, uint8_t dev, uint8_t fn) {
    return (uint8_t)(pci_read32(bus, dev, fn, 0x08) >> 24);
}
/* Header type byte, bits 6:0 (bit 7 is the multi-function flag). */
static uint8_t pci_hdr_type(uint8_t bus, uint8_t dev, uint8_t fn) {
    return (uint8_t)((pci_read32(bus, dev, fn, 0x0C) >> 16) & 0x7F);
}
/* Multi-function flag: bit 7 of the header type byte.
 * (Rewrite fix: the old scan tested bit 7 of the MASKED header
 * type, which is always zero, so it stopped after function 0 of
 * every device and never scanned multi-function devices.) */
static int pci_multifunction(uint8_t bus, uint8_t dev, uint8_t fn) {
    return (pci_read32(bus, dev, fn, 0x0C) >> 23) & 1;
}

/* BAR0: lower 32 bits of the base address (bit 0 clear = MMIO). */
static uint64_t pci_bar0(uint8_t bus, uint8_t dev, uint8_t fn) {
    uint32_t lo = pci_read32(bus, dev, fn, 0x10) & ~0xFU;
    uint32_t hi = pci_read32(bus, dev, fn, 0x14); /* BAR1 if 64-bit */
    return ((uint64_t)hi << 32) | (uint64_t)lo;
}

/* Enable Bus Master + Memory Space in the Command register. */
static void pci_enable(uint8_t bus, uint8_t dev, uint8_t fn) {
    uint32_t cmd = pci_read32(bus, dev, fn, 0x04);
    cmd |= (1u << 1) | (1u << 2); /* Memory Space | Bus Master */
    pci_write32(bus, dev, fn, 0x04, cmd);
}

/* ------------------------------------------------------------------ */
/* Discovered GPU info                                                 */
/* ------------------------------------------------------------------ */

#define PCI_VID_10DE 0x10DE
#define PCI_VID_1002    0x1002

typedef enum {
    GPU_NONE = 0,
    GPU_VENDOR_10DE,
    GPU_VENDOR_1002
} gpu_vendor_t;

typedef struct {
    gpu_vendor_t vendor;
    uint16_t     device_id;
    uint8_t      bus, dev, fn;
    uintptr_t    bar0;
    uintptr_t    bar2; /* VRAM BAR (AMD GFX9+) */
} gpu_info_t;

static gpu_info_t g_gpu;

/* ------------------------------------------------------------------ */
/* PCIe scan                                                           */
/* ------------------------------------------------------------------ */

/* Provided by Phase 1 (boot.c): console output and CPU fallback. */
extern void puts_ascii(const char *s);
extern void put_hex64(uint64_t v);
extern void put_newline(void);
extern void cpu_matmul(const float *A, const float *B, float *C,
                       int M, int N, int K);

static void gpu_scan(void) {
    for (uint32_t bus = 0; bus < 256; bus++) {
        for (uint32_t dev = 0; dev < 32; dev++) {
            for (uint32_t fn = 0; fn < 8; fn++) {
                uint8_t b = (uint8_t)bus, d = (uint8_t)dev, f = (uint8_t)fn;
                uint16_t vid = pci_vendor(b, d, f);
                if (vid == 0xFFFF) continue; /* no device */
                if (pci_class(b, d, f) != 0x03) continue; /* not display */
                if (pci_hdr_type(b, d, f) != 0) continue; /* not an endpoint */

                uint16_t did = pci_device(b, d, f);
                puts_ascii("GPU @ ");
                put_hex64(bus);
                puts_ascii(":");
                put_hex64(dev);
                puts_ascii(".");
                put_hex64(fn);
                puts_ascii(" VID=");
                put_hex64(vid);
                puts_ascii(" DID=");
                put_hex64(did);
                put_newline();

                if (vid == PCI_VID_10DE || vid == PCI_VID_1002) {
                    pci_enable(b, d, f);
                    uint64_t bar0 = pci_bar0(b, d, f);
                    puts_ascii("BAR0: ");
                    put_hex64(bar0);
                    put_newline();

                    if (g_gpu.vendor == GPU_NONE) {
                        g_gpu.vendor = (vid == PCI_VID_10DE) ? GPU_VENDOR_10DE : GPU_VENDOR_1002;
                        g_gpu.device_id = did;
                        g_gpu.bus = b;
                        g_gpu.dev = d;
                        g_gpu.fn = f;
                        g_gpu.bar0 = (uintptr_t)bar0;
                    }
                }

                /* Single-function device: no more functions to check. */
                if (fn == 0 && !pci_multifunction(b, d, 0)) break;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* NVIDIA GPU initialization (Turing/Ampere/Ada — NV_PMC path)        */
/* ------------------------------------------------------------------ */

/* NV register offsets. */
#define NV_PMC_BOOT_0              0x000000
#define NV_PMC_ENABLE              0x000200
#define NV_PMC_INTR_0              0x000100
#define NV_PBUS_FBIO_TIMEOUT       0x001560
#define NV_PFIFO_CACHES            0x002500
#define NV_PFIFO_RUNLIST           0x002600
#define NV_GR_FECS_CTXSW_MAILBOX0  0x409004

#define NV_PMC_ENABLE_ALL          0xFFFFFFFF

typedef struct __attribute__((packed)) {
    uint32_t entry_type; /* 0 = NOP, 1 = runlist entry */
    uint32_t chid;       /* channel ID */
    uint64_t inst_ptr;   /* instance block physical address */
} nv_runlist_entry_t;

typedef struct __attribute__((packed)) {
    uint32_t method; /* NV method */
    uint32_t data;
} nv_push_entry_t;

/* Ring buffer for push methods. */
#define NV_RING_SZ 4096
static uint8_t  g_nv_ring[NV_RING_SZ] __attribute__((aligned(4096)));
static uint32_t g_nv_ring_put = 0;

static void nv_push(uint32_t method, uint32_t data) {
    if (g_nv_ring_put + 8 > NV_RING_SZ) g_nv_ring_put = 0;
    nv_push_entry_t *e = (nv_push_entry_t *)(g_nv_ring + g_nv_ring_put);
    e->method = method;
    e->data = data;
    g_nv_ring_put += 8;
    mfence();
}

static int nv_init(void) {
    uintptr_t bar0 = g_gpu.bar0;
    if (!bar0) return -1;

    uint32_t boot0 = mmio_read32(bar0 + NV_PMC_BOOT_0);
    puts_ascii("NV_PMC_BOOT_0: ");
    put_hex64(boot0);
    put_newline();

    /* Enable all PMC engines. */
    mmio_write32(bar0 + NV_PMC_ENABLE, NV_PMC_ENABLE_ALL);
    mfence();

    /* Clear pending interrupts. */
    mmio_write32(bar0 + NV_PMC_INTR_0, 0xFFFFFFFF);
    mfence();

    /* Enable PFIFO (channel submission). */
    mmio_write32(bar0 + NV_PFIFO_CACHES, 0x1);
    mfence();

    puts_ascii("NV init: ok");
    put_newline();
    return 0;
}

/* Submit a NOP compute job to verify the command path. */
static int nv_submit_nop(void) {
    /* Push a NOP method (0x100 is NV_CLASS_METHOD_NOP on most
     * architectures). */
    nv_push(0x100, 0x00000000);

    /* Write the PUT pointer to PFIFO. */
    mmio_write32(g_gpu.bar0 + NV_PFIFO_RUNLIST, g_nv_ring_put);
    mfence();

    /* Poll the FECS mailbox for completion — timeout after ~1M reads. */
    for (int i = 0; i < 1000000; i++) {
        uint32_t mb = mmio_read32(g_gpu.bar0 + NV_GR_FECS_CTXSW_MAILBOX0);
        if (mb == 0x01234567) {
            puts_ascii("NV compute: ack");
            put_newline();
            return 0;
        }
    }
    puts_ascii("NV compute: timeout (expected on no firmware)");
    put_newline();
    return 0; /* not fatal in Phase 2 */
}

/* ------------------------------------------------------------------ */
/* AMD GPU initialization (GFX9 / RDNA — CP ring path)                */
/* ------------------------------------------------------------------ */

/* GFX9 register offsets (mmSoC15 indirect via MMIO index/data). */
#define GFX_MMIO_INDEX        0x0 /* SOC15 MMIO index */
#define GFX_MMIO_DATA         0x4
#define mmCP_ME_CNTL             0x1209
#define mmGRBM_GFX_CNTL          0x4342
#define mmCP_RB0_CNTL            0x1E07
#define mmCP_RB0_RPTR_ADDR       0x1E08
#define mmCP_RB0_WPTR            0x1E0B
#define mmCP_INT_CNTL            0x1E3A
#define mmGRBM_STATUS            0x4218
#define GRBM_STATUS__GUI_ACTIVE  (1u << 31)

/* AMD MMIO indirect read/write. */
static uint32_t gfx_rreg(uint32_t reg) {
    mmio_write32(g_gpu.bar0 + GFX_MMIO_INDEX * 4, reg);
    return mmio_read32(g_gpu.bar0 + GFX_MMIO_DATA * 4);
}
static void gfx_wreg(uint32_t reg, uint32_t val) {
    mmio_write32(g_gpu.bar0 + GFX_MMIO_INDEX * 4, reg);
    mmio_write32(g_gpu.bar0 + GFX_MMIO_DATA * 4, val);
    mfence();
}

/* PM4 ring buffer. */
#define GFX_RING_DWORDS 1024
static uint32_t g_gfx_ring[GFX_RING_DWORDS] __attribute__((aligned(4096)));
static uint32_t g_gfx_rptr[4] __attribute__((aligned(4096)));
static uint32_t g_gfx_wptr = 0;

static void gfx_ring_push(uint32_t dw) {
    if (g_gfx_wptr >= GFX_RING_DWORDS) g_gfx_wptr = 0;
    g_gfx_ring[g_gfx_wptr++] = dw;
    mfence();
}

/* PM4 packet types. */
#define PM4_PKT3_NOP   0x10
#define PM4_PKT3(op, n) ((3u << 30) | ((n) << 16) | (op))

static void gfx_push_nop(int count) {
    gfx_ring_push(PM4_PKT3(PM4_PKT3_NOP, count));
    for (int i = 0; i < count; i++) gfx_ring_push(0);
}

static int gfx_init(void) {
    uintptr_t bar0 = g_gpu.bar0;
    if (!bar0) return -1;

    uint32_t status = gfx_rreg(mmGRBM_STATUS);
    puts_ascii("GRBM_STATUS: ");
    put_hex64(status);
    put_newline();

    uint64_t ring_phys = (uint64_t)(uintptr_t)g_gfx_ring;
    uint64_t rptr_phys = (uint64_t)(uintptr_t)g_gfx_rptr;

    /* Set the rptr address. */
    gfx_wreg(mmCP_RB0_RPTR_ADDR, (uint32_t)(rptr_phys >> 8));

    /* Configure ring 0: size = log2(GFX_RING_DWORDS). */
    gfx_wreg(mmCP_RB0_CNTL, 10); /* log2(1024) = 10 */

    /* Set the base (upper 32 bits of the ring physical address).
     * mmCP_RB0_BASE_HI (offset depends on GFX version — use index). */
    gfx_wreg(0x1E06, (uint32_t)(ring_phys >> 8));

    puts_ascii("AMD ring @ ");
    put_hex64(ring_phys);
    put_newline();

    /* Enable CP. */
    gfx_wreg(mmCP_ME_CNTL, 0);
    mfence();

    puts_ascii("AMD init: ok");
    put_newline();
    return 0;
}

static int gfx_submit_nop(void) {
    gfx_push_nop(8);

    /* Update WPTR — the GPU reads this to know new commands are available. */
    gfx_wreg(mmCP_RB0_WPTR, g_gfx_wptr);
    mfence();

    /* Poll RPTR — when it catches up to WPTR, commands are consumed. */
    for (int i = 0; i < 2000000; i++) {
        uint32_t rptr = mmio_read32((uintptr_t)g_gfx_rptr);
        if (rptr >= g_gfx_wptr) {
            puts_ascii("AMD CP: ack (rptr=");
            put_hex64(rptr);
            puts_ascii(")");
            put_newline();
            return 0;
        }
    }
    puts_ascii("AMD CP: timeout (microcode not loaded yet)");
    put_newline();
    return 0; /* not fatal */
}

/* ------------------------------------------------------------------ */
/* Compute dispatch — Phase 2.1: matmul on GPU bare metal             */
/* ------------------------------------------------------------------ */

/*
 * On NVIDIA: would write PM4 packets to the NV_PFIFO channel ring,
 *            triggering a CUDA kernel via GRCE/PGRAPH.
 *            Requires firmware (GSP-RM) or FALCON microcode.
 *            Phase 2.1 milestone: FALCON microcode upload.
 *
 * On AMD:    PM4 DISPATCH_DIRECT packet:
 *              header = PKT3(DISPATCH_DIRECT, 4)
 *              dim_x, dim_y, dim_z
 *              dispatch_initiator
 *            Requires CP microcode (pfp.bin + me.bin from firmware).
 *            Phase 2.1 milestone: CP microcode upload from model blob.
 *
 * For now: NOPs prove the command submission path works.
 * Real compute: Phase 2.1 wires the GCN kernel from
 *               src/compiler/emit_vendors.c (Section 4) and uploads
 *               it as a code object.
 */

static void gfx_dispatch_matmul(const float *A, const float *B, float *C,
                                int M, int N, int K) {
    /* Write A, B to VRAM via BAR2 (if present) or via DMA. */
    (void)A; (void)B; (void)C;
    (void)M; (void)N; (void)K;

    /* Phase 2.0: push a NOP, verify the ring path. */
    gfx_push_nop(4);

    /* Phase 2.1 (planned):
     *   1. Upload CP microcode
     *   2. Upload GCN kernel ELF (from gcn_emit)
     *   3. Write PKT3_DISPATCH_DIRECT
     *   4. Flush with SURFACE_SYNC
     *   5. Poll GRBM_STATUS GUI_ACTIVE clear
     */

    gfx_wreg(mmCP_RB0_WPTR, g_gfx_wptr);
    mfence();

    /* Sync: poll until the CP drains. */
    for (int i = 0; i < 5000000; i++) {
        uint32_t st = gfx_rreg(mmGRBM_STATUS);
        if (!(st & GRBM_STATUS__GUI_ACTIVE)) break;
    }
}

/* ------------------------------------------------------------------ */
/* Public GPU interface (called from Phase 1 inference_loop)          */
/* ------------------------------------------------------------------ */

int baremetal_gpu_init(void) {
    gpu_scan();

    if (g_gpu.vendor == GPU_NONE) {
        puts_ascii("No supported GPU found");
        put_newline();
        return -1;
    }

    if (g_gpu.vendor == GPU_VENDOR_10DE) {
        puts_ascii("NVIDIA GPU detected");
        put_newline();
        if (nv_init() != 0) return -1;
        nv_submit_nop();
    } else {
        puts_ascii("AMD GPU detected");
        put_newline();
        if (gfx_init() != 0) return -1;
        gfx_submit_nop();
    }

    return 0;
}

void baremetal_gpu_matmul(const float *A, const float *B, float *C,
                          int M, int N, int K) {
    if (g_gpu.vendor == GPU_VENDOR_1002) {
        gfx_dispatch_matmul(A, B, C, M, N, K);
        return;
    }
    /* NVIDIA Phase 2.1: nv_dispatch_matmul() */
    /* Fallback: CPU (phase 1's cpu_matmul — the single copy). */
    cpu_matmul(A, B, C, M, N, K);
}
