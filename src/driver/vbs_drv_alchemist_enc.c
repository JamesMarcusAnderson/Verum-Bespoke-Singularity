/* vbs_drv_alchemist_enc.c — Alchemist command encoder (pure C11).
 *
 * No ioctls, no platform headers: every function here builds command
 * words from the Intel spec digest's documented facts and is fully
 * unit-testable on a machine with no GPU. All hardware facts cite the
 * digest section + source document in the comment above the code that
 * relies on them.
 *
 * Originality: written from the digest's documented facts (PRM
 * command tables, structure layouts, and computed DW0 values). The
 * code is our own - nothing here comes from Mesa, Intel's compute
 * runtime, or any other implementation.
 */

#include "vbs_drv_alchemist.h"

#include <string.h>

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

static bool is_pow2_u32(uint32_t v) {
    return v != 0 && (v & (v - 1)) == 0;
}

/* SLM_SIZE enum (digest §5.6, PRM Vol 2b p.108): 0=0KB, 1=1KB, 2=2KB,
 * 3=4KB, 4=8KB, 5=16KB, 6=32KB, 7=64KB. Rounds up; returns false when
 * the request exceeds 64 KiB. */
static bool slm_enum_for_bytes(uint32_t bytes, uint32_t *out) {
    static const uint32_t sizes_kb[8] = { 0, 1, 2, 4, 8, 16, 32, 64 };
    for (uint32_t i = 0; i < 8; i++) {
        if (bytes <= sizes_kb[i] * 1024u) {
            *out = i;
            return true;
        }
    }
    return false;
}

/* ------------------------------------------------------------------ */
/* MI commands — digest §4.2, PRM Vol 2a                               */
/* ------------------------------------------------------------------ */

vbs_drv_status_t vbs_drv_alchemist_emit_noop(vbs_drv_cmdbuf_t *cb) {
    if (!cb) return VBS_DRV_ERR_INVALID_ARG;
    return vbs_drv_cmdbuf_push(cb, VBS_DRV_ALCHEMIST_MI_NOOP);
}

vbs_drv_status_t vbs_drv_alchemist_emit_batch_end(vbs_drv_cmdbuf_t *cb) {
    if (!cb) return VBS_DRV_ERR_INVALID_ARG;
    return vbs_drv_cmdbuf_push(cb, VBS_DRV_ALCHEMIST_MI_BATCH_BUFFER_END);
}

vbs_drv_status_t vbs_drv_alchemist_emit_pipeline_select_gpgpu(
    vbs_drv_cmdbuf_t *cb) {
    if (!cb) return VBS_DRV_ERR_INVALID_ARG;
    return vbs_drv_cmdbuf_push(cb, VBS_DRV_ALCHEMIST_PIPELINE_SELECT_GPGPU);
}

/* ------------------------------------------------------------------ */
/* STATE_BASE_ADDRESS — digest §4.4, PRM Vol 2a pp.1294-1295           */
/* ------------------------------------------------------------------ */

vbs_drv_status_t vbs_drv_alchemist_emit_state_base_address(
    vbs_drv_cmdbuf_t *cb,
    uint64_t general_base, uint32_t general_size,
    uint64_t surface_base,
    uint64_t instr_base, uint32_t instr_size) {
    if (!cb) return VBS_DRV_ERR_INVALID_ARG;
    /* All bases are 4 KiB aligned (the address fields are [63:12]);
     * sizes are 4 KiB multiples in [31:12]. Addresses above 2^48 are
     * outside the documented 48-bit VA range (digest §7.2). */
    if ((general_base & 0xfffu) != 0 || (surface_base & 0xfffu) != 0 ||
        (instr_base & 0xfffu) != 0 ||
        general_base >= (1ULL << VBS_DRV_ALCHEMIST_VA_BITS) ||
        surface_base >= (1ULL << VBS_DRV_ALCHEMIST_VA_BITS) ||
        instr_base >= (1ULL << VBS_DRV_ALCHEMIST_VA_BITS) ||
        general_size == 0 || (general_size & 0xfffu) != 0 ||
        instr_size == 0 || (instr_size & 0xfffu) != 0)
        return VBS_DRV_ERR_INVALID_ARG;

    uint32_t w[VBS_DRV_ALCHEMIST_STATE_BASE_ADDRESS_WORDS];
    memset(w, 0, sizeof(w));
    w[0] = VBS_DRV_ALCHEMIST_STATE_BASE_ADDRESS_DW0;
    /* DW1-2: General State Base Address [63:12] + MOCS [10:4] (0:
     * UNSOURCED, digest §9 item 6) + Modify Enable [0] = 1. */
    w[1] = (uint32_t)(general_base & 0xfffff000u) | 1u;
    w[2] = (uint32_t)(general_base >> 32);
    /* DW3: reserved + Coherency Setting Modify Enable — 0 (no
     * coherency change). */
    /* DW4-5: Surface State Base Address + Modify Enable. */
    w[4] = (uint32_t)(surface_base & 0xfffff000u) | 1u;
    w[5] = (uint32_t)(surface_base >> 32);
    /* DW6-7: Dynamic State Base — unmodified (no samplers in v1). */
    /* DW8-9: Indirect Object Base — unmodified (no indirect data). */
    /* DW10-11: Instruction Base Address + Modify Enable. */
    w[10] = (uint32_t)(instr_base & 0xfffff000u) | 1u;
    w[11] = (uint32_t)(instr_base >> 32);
    /* DW12: General State Buffer Size [31:12] + Modify Enable. */
    w[12] = (general_size & 0xfffff000u) | 1u;
    /* DW13: Dynamic State Buffer Size — unmodified. */
    /* DW14: Indirect Object Buffer Size — unmodified. */
    /* DW15: Instruction Buffer Size [31:12] + Modify Enable. */
    w[15] = (instr_size & 0xfffff000u) | 1u;
    /* DW16-21: Bindless bases/sizes — unmodified (no bindless use). */
    return vbs_drv_cmdbuf_push_many(cb, w, sizeof(w) / sizeof(w[0]));
}

/* ------------------------------------------------------------------ */
/* 3DSTATE_BINDING_TABLE_POOL_ALLOC — digest §4.6, PRM Vol 2a p.29     */
/* ------------------------------------------------------------------ */

vbs_drv_status_t vbs_drv_alchemist_emit_binding_table_pool(
    vbs_drv_cmdbuf_t *cb, uint64_t pool_base, uint32_t pool_size) {
    if (!cb) return VBS_DRV_ERR_INVALID_ARG;
    if ((pool_base & 0xfffu) != 0 ||
        pool_base >= (1ULL << VBS_DRV_ALCHEMIST_VA_BITS) ||
        pool_size == 0 || (pool_size & 0xfffu) != 0)
        return VBS_DRV_ERR_INVALID_ARG;
    uint32_t w[VBS_DRV_ALCHEMIST_BINDING_TABLE_POOL_ALLOC_WORDS];
    w[0] = VBS_DRV_ALCHEMIST_BINDING_TABLE_POOL_ALLOC_DW0;
    /* DW1-2: Binding Table Pool Base Address [63:12] + MOCS [6:0]
     * (0: UNSOURCED, digest §9 item 6). */
    w[1] = (uint32_t)(pool_base & 0xfffff000u);
    w[2] = (uint32_t)(pool_base >> 32);
    /* DW3: Binding Table Pool Buffer Size [31:12] in 4 KiB pages. */
    w[3] = pool_size & 0xfffff000u;
    return vbs_drv_cmdbuf_push_many(cb, w, sizeof(w) / sizeof(w[0]));
}

/* ------------------------------------------------------------------ */
/* PIPE_CONTROL — digest §4.7, PRM Vol 2a pp.1155-1162                 */
/* ------------------------------------------------------------------ */

static vbs_drv_status_t emit_pipe_control(vbs_drv_cmdbuf_t *cb,
                                          uint32_t dw0_extra,
                                          uint32_t dw1_extra) {
    uint32_t w[VBS_DRV_ALCHEMIST_PIPE_CONTROL_WORDS];
    memset(w, 0, sizeof(w));
    w[0] = VBS_DRV_ALCHEMIST_PIPE_CONTROL_DW0 | dw0_extra;
    w[1] = dw1_extra;
    /* DW2-3: Destination Address — 0 (Post Sync Operation is No
     * Write in both v1 forms). DW4-5: Immediate Data — 0. */
    return vbs_drv_cmdbuf_push_many(cb, w, sizeof(w) / sizeof(w[0]));
}

vbs_drv_status_t vbs_drv_alchemist_emit_pipe_control_ic_invalidate(
    vbs_drv_cmdbuf_t *cb) {
    if (!cb) return VBS_DRV_ERR_INVALID_ARG;
    /* Digest §4.9: CS Stall + Instruction Cache Invalidate before
     * the walker. */
    return emit_pipe_control(cb, 0,
                             VBS_DRV_ALCHEMIST_PC_CS_STALL |
                                 VBS_DRV_ALCHEMIST_PC_IC_INVALIDATE);
}

vbs_drv_status_t vbs_drv_alchemist_emit_pipe_control_flush(
    vbs_drv_cmdbuf_t *cb) {
    if (!cb) return VBS_DRV_ERR_INVALID_ARG;
    /* Digest §4.7 canonical compute fence: DW0 HDC Pipeline Flush
     * + Untyped L1 Flush, DW1 CS Stall. */
    return emit_pipe_control(cb,
                             VBS_DRV_ALCHEMIST_PC_HDC_FLUSH |
                                 VBS_DRV_ALCHEMIST_PC_UNTYPED_L1_FLUSH,
                             VBS_DRV_ALCHEMIST_PC_CS_STALL);
}

/* ------------------------------------------------------------------ */
/* INTERFACE_DESCRIPTOR_DATA — digest §5.1, PRM Vol 2d pp.714-719      */
/* ------------------------------------------------------------------ */

vbs_drv_status_t vbs_drv_alchemist_iface_build(
    const vbs_drv_alchemist_iface_params_t *p,
    vbs_drv_alchemist_iface_desc_t *out) {
    if (!p || !out) return VBS_DRV_ERR_INVALID_ARG;
    /* Kernel Start Pointer is 64-byte aligned (DW0 [31:6], DW1
     * reserved MBZ — Vol 2d p.714). */
    if ((p->kernel_start_offset & 0x3fu) != 0)
        return VBS_DRV_ERR_INVALID_ARG;
    /* Binding Table Pointer is 32-byte aligned in DW4 [20:5]. */
    if ((p->binding_table_offset & 0x1fu) != 0 ||
        p->binding_table_offset > 0x1fffffe0u)
        return VBS_DRV_ERR_INVALID_ARG;
    if (p->barrier_count > 7)
        return VBS_DRV_ERR_INVALID_ARG;
    if (p->thread_count < 1 ||
        p->thread_count > VBS_DRV_ALCHEMIST_MAX_THREADS)
        return VBS_DRV_ERR_INVALID_ARG;
    uint32_t slm_enum = 0;
    if (!slm_enum_for_bytes(p->slm_bytes, &slm_enum))
        return VBS_DRV_ERR_INVALID_ARG;

    memset(out, 0, sizeof(*out));
    /* DW0: Kernel Start Pointer [31:6] = the offset itself (bits
     * [5:0] MBZ); DW1: reserved MBZ. */
    out->w[0] = p->kernel_start_offset & 0xffffffc0u;
    out->w[1] = 0;
    /* DW2: all documented defaults — FP mode IEEE-754 (bit 16 = 0),
     * denorms flushed to zero (bit 19 = 0), no exception enables. */
    out->w[2] = 0;
    /* DW3: no samplers (pointer 0, count 0). */
    out->w[3] = 0;
    /* DW4: Binding Table Pointer [20:5] + Entry Count [4:0] = 0
     * (prefetch disabled, the documented typical value). */
    out->w[4] = p->binding_table_offset & 0x1fffffe0u;
    /* DW5: BTD 0, Barriers [30:28], TG Dispatch Size 00 (8,
     * default), Rounding 00 (RTNE, default), SLM [20:16],
     * Thread Count [9:0]. */
    out->w[5] = (p->barrier_count << 28) | (slm_enum << 16) |
                (p->thread_count & 0x3ffu);
    /* DW6: Preferred SLM Allocation Size — 0 (default). DW7: MBZ. */
    out->w[6] = 0;
    out->w[7] = 0;
    return VBS_DRV_OK;
}

/* ------------------------------------------------------------------ */
/* Surface state + binding table — digest §5.2/§5.3, PRM Vol 2d        */
/* ------------------------------------------------------------------ */

vbs_drv_status_t vbs_drv_alchemist_surface_build(
    uint64_t base_va, uint32_t size_bytes,
    vbs_drv_alchemist_surface_state_t *out) {
    if (!out) return VBS_DRV_ERR_INVALID_ARG;
    if (size_bytes == 0 || (size_bytes & 0x3u) != 0 ||
        (base_va & 0x3u) != 0)
        return VBS_DRV_ERR_INVALID_ARG;
    memset(out, 0, sizeof(*out));
    /* DW0: Surface Type BUFFER (4) in [31:29]; Surface Format 0
     * (ignored for untyped data-port messages); Tile Mode Linear
     * (0); everything else 0. */
    out->w[0] = (4u << 29);
    /* DW1: MOCS [30:24] = 0 (UNSOURCED, digest §9 item 6). */
    out->w[1] = 0;
    /* DW2: Width [13:0] = element count - 1 for a buffer surface.
     * v1 buffers hold 4-byte elements; the 14-bit field caps at
     * 16384 elements (64 KiB) — larger buffers clamp (see the
     * header note and the driver report). */
    uint32_t elements = size_bytes / 4u;
    uint32_t width = elements - 1;
    if (width > 0x3fffu) width = 0x3fffu;
    out->w[2] = width;
    /* DW3-7: 0 (no pitch/depth/aux use for plain buffers). */
    /* DW8-9: Surface Base Address [63:0], the buffer's GPU VA. */
    out->w[8] = (uint32_t)(base_va & 0xffffffffu);
    out->w[9] = (uint32_t)(base_va >> 32);
    /* DW10-15: aux base + clear values — 0 for compute buffers. */
    return VBS_DRV_OK;
}

uint32_t vbs_drv_alchemist_binding_entry(uint32_t surface_state_offset) {
    /* BINDING_TABLE_STATE entry: Surface State Pointer [31:6]
     * (digest §5.2, Vol 2d p.1084) — the offset itself, 64-byte
     * aligned, relative to the Surface State Base Address. */
    return surface_state_offset & 0xffffffc0u;
}

/* ------------------------------------------------------------------ */
/* COMPUTE_WALKER — digest §4.8, PRM Vol 2a pp.485-494                 */
/* ------------------------------------------------------------------ */

vbs_drv_status_t vbs_drv_alchemist_emit_walker(
    vbs_drv_cmdbuf_t *cb, const vbs_drv_alchemist_walker_params_t *p) {
    if (!cb || !p) return VBS_DRV_ERR_INVALID_ARG;
    if (p->grid_x == 0 || p->grid_y == 0 || p->grid_z == 0)
        return VBS_DRV_ERR_INVALID_ARG;
    if (p->block_x == 0 || p->block_y == 0 || p->block_z == 0 ||
        p->block_x > VBS_DRV_ALCHEMIST_MAX_LWS ||
        p->block_y > VBS_DRV_ALCHEMIST_MAX_LWS ||
        p->block_z > VBS_DRV_ALCHEMIST_MAX_LWS)
        return VBS_DRV_ERR_INVALID_ARG;
    uint32_t lws = p->block_x * p->block_y * p->block_z;
    if (lws == 0 || lws > VBS_DRV_ALCHEMIST_MAX_LWS)
        return VBS_DRV_ERR_INVALID_ARG;
    /* The interface descriptor's thread count must satisfy
     * (SIMD)*(N-1) < LWS <= (SIMD)*N (Vol 2a p.490). */
    uint32_t threads = p->iface.w[5] & 0x3ffu;
    if (threads < 1 || threads > VBS_DRV_ALCHEMIST_MAX_THREADS)
        return VBS_DRV_ERR_INVALID_ARG;
    if (!((uint32_t)VBS_DRV_ALCHEMIST_SIMD * (threads - 1) < lws &&
          lws <= (uint32_t)VBS_DRV_ALCHEMIST_SIMD * threads))
        return VBS_DRV_ERR_INVALID_ARG;
    if (p->use_semaphore && (p->sem_va & 0x7u) != 0)
        return VBS_DRV_ERR_INVALID_ARG;

    /* Local-ID generation restriction (Vol 2a p.488): at least two
     * of the three local dimension sizes must be powers of two, and
     * the non-power-of-two dimension must be lowest priority in the
     * walk order. At most one dimension may be non-power-of-two.
     * (The PRM text says "maximum values"; the sizes reading is the
     * only one consistent with the digest's own kernel design — see
     * the driver report.) */
    bool x_pow2 = is_pow2_u32(p->block_x);
    bool y_pow2 = is_pow2_u32(p->block_y);
    bool z_pow2 = is_pow2_u32(p->block_z);
    unsigned non_pow2 = (x_pow2 ? 0u : 1u) + (y_pow2 ? 0u : 1u) +
                        (z_pow2 ? 0u : 1u);
    if (non_pow2 > 1) return VBS_DRV_ERR_INVALID_ARG;
    /* Walk order (Vol 2a p.489): 0 = Walk 012 (default, normal
     * linear), 1 = 021, 2 = 102, 3 = 120, 4 = 201, 5 = 210. The
     * digits run fastest-to-slowest, so the last digit is the
     * lowest-priority dimension: put the non-power-of-two
     * dimension last. */
    uint32_t walk_order = 0; /* Walk 012: Z slowest */
    if (!x_pow2) walk_order = 5;      /* Walk 210: X slowest */
    else if (!y_pow2) walk_order = 1; /* Walk 021: Y slowest */
    /* Z non-pow2 (or all pow2): Walk 012 already has Z last. */

    /* Emit Local (DW4 [28:26]) only takes the values 0 (None), 1 (X),
     * 3 (XY), 7 (XYZ); all others are Reserved (Vol 2a p.488). A
     * dimension may therefore exceed 1 only when every lower
     * dimension also exceeds 1 — a (1,16,1) block would need an
     * Emit-Y-only value that does not exist. */
    if (p->block_y > 1 && p->block_x == 1)
        return VBS_DRV_ERR_INVALID_ARG;
    if (p->block_z > 1 && (p->block_x == 1 || p->block_y == 1))
        return VBS_DRV_ERR_INVALID_ARG;
    /* Emit Local (DW4 [28:26]): bit 26 X, 27 Y, 28 Z. A dimension
     * whose emit bit is clear must have Local Maximum 0 in DW6
     * (Vol 2a p.488) — so emit a dimension exactly when its block
     * size exceeds 1. */
    uint32_t emit_local = 0;
    if (p->block_x > 1) emit_local |= (1u << 0); /* X */
    if (p->block_y > 1) emit_local |= (1u << 1); /* Y */
    if (p->block_z > 1) emit_local |= (1u << 2); /* Z */
    /* Block sizes are >= 1, so X is emitted whenever block_x > 1;
     * a (1,1,1) block emits nothing and Generate Local ID still
     * runs (harmless: one work item, local ID 0). */

    /* Execution Mask (DW5): used with the last thread in the
     * threadgroup; SIMD16 uses the lower 16 bits (Vol 2a p.490). */
    uint32_t remainder = lws - (uint32_t)VBS_DRV_ALCHEMIST_SIMD * (threads - 1);
    uint32_t exec_mask;
    if (remainder >= (uint32_t)VBS_DRV_ALCHEMIST_SIMD) exec_mask = 0xffffu;
    else exec_mask = (1u << remainder) - 1u;

    uint32_t w[VBS_DRV_ALCHEMIST_COMPUTE_WALKER_WORDS];
    memset(w, 0, sizeof(w));
    w[0] = VBS_DRV_ALCHEMIST_COMPUTE_WALKER_DW0;
    /* DW1: reserved MBZ. */
    /* DW2: Partition Type 0, L3 prefetch disable 0, Indirect Data
     * Length 0 (no indirect data in v1). */
    /* DW3: Indirect Data Start Address 0. */
    /* DW4: SIMD Size [31:30] = 1 (SIMD16); Generate Local ID [29];
     * Emit Local [28:26]; Emit Inline Parameter [25] = 0 (v1 passes
     * kernel arguments through binding-table surfaces, not inline
     * data); Walk Order [24:22]; Tile Layout [21:19] = 0 (Linear);
     * Message SIMD [18:17] = 1 (SIMD16, equal to the thread SIMD —
     * the documented restriction is <=, Vol 2a p.489). */
    w[4] = (1u << 30) | (1u << 29) | (emit_local << 26) |
           (walk_order << 22) | (1u << 17);
    w[5] = exec_mask;
    /* DW6: Local Z/Y/X Maximum (each = count - 1, 10-bit fields). */
    w[6] = ((p->block_z - 1) << 20) | ((p->block_y - 1) << 10) |
           (p->block_x - 1);
    /* DW7-9: Thread Group ID dimensions. */
    w[7] = p->grid_x;
    w[8] = p->grid_y;
    w[9] = p->grid_z;
    /* DW10-12: Starting group IDs 0. DW13-14: Partition Size/ID 0
     * (partitioning disabled). DW15-17: Preempt 0 (must be zero on
     * initial submission, Vol 2a p.485). */
    /* DW18-25: Interface Descriptor, inlined (not a pointer). */
    memcpy(&w[18], p->iface.w, sizeof(p->iface.w));
    /* DW26-30: PostSync (digest §5.4). Without a semaphore the
     * operation is No Write (0) and the whole block stays zero. */
    if (p->use_semaphore) {
        /* DW26: Dataport Subslice Cache Flush [12] + Dataport
         * Pipeline Flush [2] + Operation [1:0] = 1 (Write
         * Immediate Data). MOCS [10:4] = 0 (UNSOURCED, digest §9
         * item 6). The Subslice Cache Flush bit requires the
         * Pipeline Flush bit (Vol 2d p.833) — both are set. */
        w[26] = (1u << 12) | (1u << 2) | 1u;
        /* DW27-28: Destination Address (PPGTT, 8-byte aligned). */
        w[27] = (uint32_t)(p->sem_va & 0xffffffffu);
        w[28] = (uint32_t)(p->sem_va >> 32);
        /* DW29-30: Immediate Data (the 32-bit payload, zero
         * extended to 64 bits). */
        w[29] = p->sem_payload;
        w[30] = 0;
    }
    /* DW31-38: Inline Data — 0 (Emit Inline Parameter is clear). */
    return vbs_drv_cmdbuf_push_many(cb, w, sizeof(w) / sizeof(w[0]));
}

/* ------------------------------------------------------------------ */
/* Full launch sequence — digest §4.9                                  */
/* ------------------------------------------------------------------ */

vbs_drv_status_t vbs_drv_alchemist_emit_launch_sequence(
    vbs_drv_cmdbuf_t *cb, const vbs_drv_alchemist_launch_params_t *p) {
    if (!cb || !p) return VBS_DRV_ERR_INVALID_ARG;
    if (p->code_va == 0 || p->state_va == 0)
        return VBS_DRV_ERR_INVALID_ARG;
    if ((p->code_offset & 0x3fu) != 0)
        return VBS_DRV_ERR_INVALID_ARG;
    if (p->code_size == 0 || (p->code_size & 0xfffu) != 0)
        return VBS_DRV_ERR_INVALID_ARG;
    if (p->const_count > VBS_DRV_MAX_CONST_BUFFERS)
        return VBS_DRV_ERR_INVALID_ARG;
    if (p->grid_x == 0 || p->grid_y == 0 || p->grid_z == 0 ||
        p->block_x == 0 || p->block_y == 0 || p->block_z == 0)
        return VBS_DRV_ERR_INVALID_ARG;
    uint32_t lws = p->block_x * p->block_y * p->block_z;
    if (lws == 0 || lws > VBS_DRV_ALCHEMIST_MAX_LWS)
        return VBS_DRV_ERR_INVALID_ARG;

    /* Thread count: ceil(LWS / SIMD), capped by the 64-thread
     * interface-descriptor limit. */
    uint32_t threads =
        (lws + (uint32_t)VBS_DRV_ALCHEMIST_SIMD - 1u) /
        (uint32_t)VBS_DRV_ALCHEMIST_SIMD;
    if (threads > VBS_DRV_ALCHEMIST_MAX_THREADS)
        return VBS_DRV_ERR_INVALID_ARG;

    vbs_drv_alchemist_iface_params_t ip;
    memset(&ip, 0, sizeof(ip));
    ip.kernel_start_offset = p->code_offset;
    ip.binding_table_offset = VBS_DRV_ALCHEMIST_STATE_BINDING_TABLE_OFF;
    ip.slm_bytes = p->slm_bytes;
    ip.barrier_count = p->barrier_count;
    ip.thread_count = threads;
    vbs_drv_alchemist_iface_desc_t iface;
    vbs_drv_status_t st = vbs_drv_alchemist_iface_build(&ip, &iface);
    if (st != VBS_DRV_OK) return st;

    /* STATE_BASE_ADDRESS: General State Base == Instruction Base ==
     * code_va (digest §6.10 dual-reading safety choice); Surface
     * State Base = state VA + surface offset. */
    st = vbs_drv_alchemist_emit_state_base_address(
        cb, p->code_va, p->code_size,
        p->state_va + VBS_DRV_ALCHEMIST_STATE_SURFACE_OFF,
        p->code_va, p->code_size);
    if (st != VBS_DRV_OK) return st;

    /* Binding-table pool, only when the launch uses binding tables
     * (digest §4.9). */
    if (p->const_count > 0) {
        st = vbs_drv_alchemist_emit_binding_table_pool(
            cb, p->state_va + VBS_DRV_ALCHEMIST_STATE_BINDING_TABLE_OFF,
            VBS_DRV_ALCHEMIST_STATE_SURFACE_OFF);
        if (st != VBS_DRV_OK) return st;
    }

    st = vbs_drv_alchemist_emit_pipe_control_ic_invalidate(cb);
    if (st != VBS_DRV_OK) return st;

    vbs_drv_alchemist_walker_params_t wp;
    memset(&wp, 0, sizeof(wp));
    wp.grid_x = p->grid_x;
    wp.grid_y = p->grid_y;
    wp.grid_z = p->grid_z;
    wp.block_x = p->block_x;
    wp.block_y = p->block_y;
    wp.block_z = p->block_z;
    wp.iface = iface;
    wp.use_semaphore = p->use_semaphore;
    wp.sem_va = p->sem_va;
    wp.sem_payload = p->sem_payload;
    st = vbs_drv_alchemist_emit_walker(cb, &wp);
    if (st != VBS_DRV_OK) return st;

    st = vbs_drv_alchemist_emit_pipe_control_flush(cb);
    if (st != VBS_DRV_OK) return st;

    return vbs_drv_alchemist_emit_batch_end(cb);
}
