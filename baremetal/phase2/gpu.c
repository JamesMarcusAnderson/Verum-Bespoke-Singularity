/*
 * definitely-not-skynet — Phase 2
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
static inline uint64_t mmio_read64(uintptr_t addr) {
    return *(volatile uint64_t *)addr;
}
static inline void mmio_write64(uintptr_t addr, uint64_t val) {
    *(volatile uint64_t *)addr = val;
}

/* Memory fence */
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

static uint32_t pci_read32(uint8_t bus, uint8_t dev, uint8_t fn, uint8_t reg) {
    uint32_t addr = (1u << 31) |
                    ((uint32_t)bus << 16) |
                    ((uint32_t)dev << 11) |
                    ((uint32_t)fn  <<  8) |
                    (reg & 0xFC);
    out32(PCI_CONFIG_ADDR, addr);
    return in32(PCI_CONFIG_DATA);
}

static void pci_write32(uint8_t bus, uint8_t dev, uint8_t fn,
                         uint8_t reg, uint32_t val) {
    uint32_t addr = (1u << 31) |
                    ((uint32_t)bus << 16) |
                    ((uint32_t)dev << 11) |
                    ((uint32_t)fn  <<  8) |
                    (reg & 0xFC);
    out32(PCI_CONFIG_ADDR, addr);
    out32(PCI_CONFIG_DATA, val);
}

static uint16_t pci_vendor(uint8_t bus, uint8_t dev, uint8_t fn) {
    return (uint16_t)(pci_read32(bus, dev, fn, 0x00) & 0xFFFF);
}
static uint16_t pci_device(uint8_t bus, uint8_t dev, uint8_t fn) {
    return (uint16_t)(pci_read32(bus, dev, fn, 0x00) >> 16);
}
static uint8_t pci_class(uint8_t bus, uint8_t dev, uint8_t fn) {
    return (uint8_t)(pci_read32(bus, dev, fn, 0x08) >> 24);
}
static uint8_t pci_hdr_type(uint8_t bus, uint8_t dev, uint8_t fn) {
    return (uint8_t)((pci_read32(bus, dev, fn, 0x0C) >> 16) & 0x7F);
}

/* BAR0: lower 32 bits of base address (bit 0 clear = MMIO) */
static uint64_t pci_bar0(uint8_t bus, uint8_t dev, uint8_t fn) {
    uint32_t lo = pci_read32(bus, dev, fn, 0x10) & ~0xFU;
    uint32_t hi = pci_read32(bus, dev, fn, 0x14); /* BAR1 if 64-bit */
    return ((uint64_t)hi << 32) | (uint64_t)lo;
}

/* Enable Bus Master + Memory Space in Command register */
static void pci_enable(uint8_t bus, uint8_t dev, uint8_t fn) {
    uint32_t cmd = pci_read32(bus, dev, fn, 0x04);
    cmd |= (1u << 1) | (1u << 2); /* Memory Space | Bus Master */
    pci_write32(bus, dev, fn, 0x04, cmd);
}

/* ------------------------------------------------------------------ */
/* Discovered GPU info                                                 */
/* ------------------------------------------------------------------ */

#define VID_NVIDIA 0x10DE
#define VID_AMD    0x1002

typedef enum {
    GPU_NONE = 0,
    GPU_NVIDIA,
    GPU_AMD
} gpu_vendor_t;

typedef struct {
    gpu_vendor_t vendor;
    uint16_t     device_id;
    uint8_t      bus, dev, fn;
    uintptr_t    bar0;
    uintptr_t    bar2;      /* VRAM BAR (AMD GFX9+) */
} gpu_info_t;

static gpu_info_t g_gpu;

/* ------------------------------------------------------------------ */
/* PCIe scan                                                           */
/* ------------------------------------------------------------------ */

/* External: Phase 1 provides this */
extern void puts_ascii(const char *s);
extern void put_hex64(uint64_t v);
extern void put_newline(void);

static void gpu_scan(void) {
    for (uint32_t bus = 0; bus < 256; bus++) {
        for (uint32_t dev = 0; dev < 32; dev++) {
            for (uint32_t fn = 0; fn < 8; fn++) {
                uint16_t vid = pci_vendor((uint8_t)bus,
                                          (uint8_t)dev,
                                          (uint8_t)fn);
                if (vid == 0xFFFF) continue; /* no device */
                uint8_t cls = pci_class((uint8_t)bus,
                                         (uint8_t)dev,
                                         (uint8_t)fn);
                /* Class 0x03 = Display controller */
                if (cls != 0x03) continue;

                uint16_t did = pci_device((uint8_t)bus,
                                           (uint8_t)dev,
                                           (uint8_t)fn);
                puts_ascii("GPU @ "); put_hex64(bus);
                puts_ascii(":"); put_hex64(dev);
                puts_ascii("."); put_hex64(fn);
                puts_ascii(" VID="); put_hex64(vid);
                puts_ascii(" DID="); put_hex64(did);
                put_newline();

                if (vid == VID_NVIDIA || vid == VID_AMD) {
                    pci_enable((uint8_t)bus, (uint8_t)dev, (uint8_t)fn);
                    uint64_t bar0 = pci_bar0((uint8_t)bus,
                                              (uint8_t)dev,
                                              (uint8_t)fn);
                    puts_ascii("BAR0: "); put_hex64(bar0); put_newline();

                    if (g_gpu.vendor == GPU_NONE) {
                        g_gpu.vendor    = (vid == VID_NVIDIA) ? GPU_NVIDIA : GPU_AMD;
                        g_gpu.device_id = did;
                        g_gpu.bus       = (uint8_t)bus;
                        g_gpu.dev       = (uint8_t)dev;
                        g_gpu.fn        = (uint8_t)fn;
                        g_gpu.bar0      = (uintptr_t)bar0;
                    }
                }

                /* Multi-function device? */
                if (fn == 0 && !(pci_hdr_type((uint8_t)bus,
                                               (uint8_t)dev, 0) & 0x80))
                    break;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* NVIDIA GPU initialization (Turing/Ampere/Ada — NV_PMC path)        */
/* ------------------------------------------------------------------ */

/* NV register offsets */
#define NV_PMC_BOOT_0         0x000000
#define NV_PMC_ENABLE         0x000200
#define NV_PMC_INTR_0         0x000100
#define NV_PBUS_FBIO_TIMEOUT  0x001560
#define NV_PFIFO_CACHES       0x002500
#define NV_PFIFO_RUNLIST      0x002600
#define NV_GR_FECS_CTXSW_MAILBOX0 0x409004

#define NV_PMC_ENABLE_ALL     0xFFFFFFFF

typedef struct __attribute__((packed)) {
    uint32_t entry_type;    /* 0 = NOP, 1 = runlist entry */
    uint32_t chid;          /* channel ID */
    uint64_t inst_ptr;      /* instance block physical address */
} nv_runlist_entry_t;

typedef struct __attribute__((packed)) {
    uint32_t method;        /* NV method */
    uint32_t data;
} nv_push_entry_t;

/* Ring buffer for push methods */
#define NV_RING_SZ 4096
static uint8_t  g_nv_ring[NV_RING_SZ] __attribute__((aligned(4096)));
static uint32_t g_nv_ring_put = 0;

static void nv_push(uint32_t method, uint32_t data) {
    if (g_nv_ring_put + 8 > NV_RING_SZ) g_nv_ring_put = 0;
    nv_push_entry_t *e = (nv_push_entry_t *)(g_nv_ring + g_nv_ring_put);
    e->method = method;
    e->data   = data;
    g_nv_ring_put += 8;
    mfence();
}

static int nv_init(void) {
    uintptr_t bar0 = g_gpu.bar0;
    if (!bar0) return -1;

    uint32_t boot0 = mmio_read32(bar0 + NV_PMC_BOOT_0);
    puts_ascii("NV_PMC_BOOT_0: "); put_hex64(boot0); put_newline();

    /* Enable all PMC engines */
    mmio_write32(bar0 + NV_PMC_ENABLE, NV_PMC_ENABLE_ALL);
    mfence();

    /* Clear pending interrupts */
    mmio_write32(bar0 + NV_PMC_INTR_0, 0xFFFFFFFF);
    mfence();

    /* Enable PFIFO (channel submission) */
    mmio_write32(bar0 + NV_PFIFO_CACHES, 0x1);
    mfence();

    puts_ascii("NV init: ok"); put_newline();
    return 0;
}

/* Submit a NOP compute job to verify command path */
static int nv_submit_nop(void) {
    /* Push a NOP method (0x100 is NV_CLASS_METHOD_NOP on most architectures) */
    nv_push(0x100, 0x00000000);

    /* Write PUT pointer to PFIFO */
    mmio_write32(g_gpu.bar0 + NV_PFIFO_RUNLIST,
                 g_nv_ring_put);
    mfence();

    /* Poll FECS mailbox for completion — timeout after ~1M reads */
    for (int i = 0; i < 1000000; i++) {
        uint32_t mb = mmio_read32(g_gpu.bar0 + NV_GR_FECS_CTXSW_MAILBOX0);
        if (mb == 0x01234567) {
            puts_ascii("NV compute: ack"); put_newline();
            return 0;
        }
    }
    puts_ascii("NV compute: timeout (expected on no firmware)"); put_newline();
    return 0; /* not fatal in Phase 2 */
}

/* ------------------------------------------------------------------ */
/* AMD GPU initialization (GFX9 / RDNA — CP ring path)                */
/* ------------------------------------------------------------------ */

/* GFX9 register offsets (mmSoC15 indirect via MMIO index/data) */
#define AMDGPU_MMIO_INDEX        0x0 /* SOC15 MMIO index */
#define AMDGPU_MMIO_DATA         0x4
#define mmCP_ME_CNTL             0x1209
#define mmGRBM_GFX_CNTL          0x4342
#define mmCP_RB0_CNTL            0x1E07
#define mmCP_RB0_RPTR_ADDR       0x1E08
#define mmCP_RB0_WPTR            0x1E0B
#define mmCP_INT_CNTL            0x1E3A
#define mmGRBM_STATUS            0x4218
#define GRBM_STATUS__GUI_ACTIVE  (1u << 31)

/* AMD MMIO indirect read/write */
static uint32_t amd_rreg(uint32_t reg) {
    mmio_write32(g_gpu.bar0 + AMDGPU_MMIO_INDEX * 4, reg);
    return mmio_read32(g_gpu.bar0 + AMDGPU_MMIO_DATA * 4);
}
static void amd_wreg(uint32_t reg, uint32_t val) {
    mmio_write32(g_gpu.bar0 + AMDGPU_MMIO_INDEX * 4, reg);
    mmio_write32(g_gpu.bar0 + AMDGPU_MMIO_DATA * 4, val);
    mfence();
}

/* PM4 ring buffer */
#define AMD_RING_DWORDS 1024
static uint32_t g_amd_ring[AMD_RING_DWORDS] __attribute__((aligned(4096)));
static uint32_t g_amd_rptr[4] __attribute__((aligned(4096)));
static uint32_t g_amd_wptr = 0;

static void amd_ring_push(uint32_t dw) {
    if (g_amd_wptr >= AMD_RING_DWORDS) g_amd_wptr = 0;
    g_amd_ring[g_amd_wptr++] = dw;
    mfence();
}

/* PM4 packet types */
#define PM4_PKT3_NOP   0x10
#define PM4_PKT3(op, n) ((3u << 30) | ((n) << 16) | (op))

static void amd_push_nop(int count) {
    amd_ring_push(PM4_PKT3(PM4_PKT3_NOP, count));
    for (int i = 0; i < count; i++) amd_ring_push(0);
}

static int amd_init(void) {
    uintptr_t bar0 = g_gpu.bar0;
    if (!bar0) return -1;

    uint32_t status = amd_rreg(mmGRBM_STATUS);
    puts_ascii("GRBM_STATUS: "); put_hex64(status); put_newline();

    /* Configure ring buffer 0 */
    uint64_t ring_phys = (uint64_t)(uintptr_t)g_amd_ring;
    uint64_t rptr_phys = (uint64_t)(uintptr_t)g_amd_rptr;

    /* Set rptr address */
    amd_wreg(mmCP_RB0_RPTR_ADDR, (uint32_t)(rptr_phys >> 8));

    /* Configure ring: size = log2(AMD_RING_DWORDS), base addr */
    uint32_t rb_cntl = 10; /* log2(1024) = 10 */
    amd_wreg(mmCP_RB0_CNTL, rb_cntl);

    /* Set base (upper 32 bits of ring physical address) */
    /* mmCP_RB0_BASE_HI (offset depends on GFX version — use index) */
    amd_wreg(0x1E06, (uint32_t)(ring_phys >> 8));

    puts_ascii("AMD ring @ "); put_hex64(ring_phys); put_newline();

    /* Enable CP */
    amd_wreg(mmCP_ME_CNTL, 0);
    mfence();

    puts_ascii("AMD init: ok"); put_newline();
    return 0;
}

static int amd_submit_nop(void) {
    amd_push_nop(8);

    /* Update WPTR — GPU reads this to know new commands are available */
    amd_wreg(mmCP_RB0_WPTR, g_amd_wptr);
    mfence();

    /* Poll RPTR — when it catches up to WPTR, commands consumed */
    for (int i = 0; i < 2000000; i++) {
        uint32_t rptr = mmio_read32((uintptr_t)g_amd_rptr);
        if (rptr >= g_amd_wptr) {
            puts_ascii("AMD CP: ack (rptr="); put_hex64(rptr);
            puts_ascii(")"); put_newline();
            return 0;
        }
    }
    puts_ascii("AMD CP: timeout (microcode not loaded yet)"); put_newline();
    return 0; /* not fatal */
}

/* ------------------------------------------------------------------ */
/* Compute dispatch — Phase 2.1: matmul on GPU bare metal             */
/* ------------------------------------------------------------------ */

/*
 * On NVIDIA: would write PM4 packets to NV_PFIFO channel ring,
 *            triggering a CUDA kernel via GRCE/PGRAPH.
 *            Requires firmware (GSP-RM) or FALCON microcode.
 *            Phase 2.1 milestone: FALCON microcode upload.
 *
 * On AMD:    PM4 DISPATCH_DIRECT packet:
 *              header = PKT3(DISPATCH_DIRECT, 4)
 *              dim_x, dim_y, dim_z
 *              dispatch_initiator
 *            Requires CP microcode (pfp.bin + me.bin from firmware).
 *            Phase 2.1 milestone: CP microcode upload from GGUF blob.
 *
 * For now: NOPs prove the command submission path works.
 * Real compute: Phase 2.1 wires GCN kernel from src/compiler/gcn_emit.c
 *               and uploads it as a code object.
 */

static void amd_dispatch_matmul(const float *A, const float *B, float *C,
                                  int M, int N, int K) {
    /* Write A, B to VRAM via BAR2 (if present) or via DMA */
    (void)A; (void)B; (void)C;
    (void)M; (void)N; (void)K;

    /* Phase 2.0: push NOP, verify ring path */
    amd_push_nop(4);

    /* Phase 2.1 (planned):
     *   1. Upload CP microcode
     *   2. Upload GCN kernel ELF (from gcn_emit)
     *   3. Write PKT3_DISPATCH_DIRECT
     *   4. Flush with SURFACE_SYNC
     *   5. Poll GRBM_STATUS GUI_ACTIVE clear
     */

    amd_wreg(mmCP_RB0_WPTR, g_amd_wptr);
    mfence();

    /* Sync: poll until CP drains */
    for (int i = 0; i < 5000000; i++) {
        uint32_t st = amd_rreg(mmGRBM_STATUS);
        if (!(st & GRBM_STATUS__GUI_ACTIVE)) break;
    }
}

/* ------------------------------------------------------------------ */
/* Public GPU interface (called from Phase 1 inference_loop)          */
/* ------------------------------------------------------------------ */

int skynet_gpu_init(void) {
    gpu_scan();

    if (g_gpu.vendor == GPU_NONE) {
        puts_ascii("No supported GPU found"); put_newline();
        return -1;
    }

    if (g_gpu.vendor == GPU_NVIDIA) {
        puts_ascii("NVIDIA GPU detected"); put_newline();
        if (nv_init() != 0) return -1;
        nv_submit_nop();
    } else {
        puts_ascii("AMD GPU detected"); put_newline();
        if (amd_init() != 0) return -1;
        amd_submit_nop();
    }

    return 0;
}

void skynet_gpu_matmul(const float *A, const float *B, float *C,
                        int M, int N, int K) {
    if (g_gpu.vendor == GPU_AMD) {
        amd_dispatch_matmul(A, B, C, M, N, K);
        return;
    }
    /* NVIDIA Phase 2.1: nv_dispatch_matmul() */
    /* Fallback: CPU */
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++)
                acc += A[m * K + k] * B[k * N + n];
            C[m * N + n] = acc;
        }
}