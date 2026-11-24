/* vbs_drv_alchemist.h — Verum driver for Alchemist GPUs (DG2, Xe-HPG).
 *
 * This layer implements the framework ops (vbs_drv_ops_t) for Alchemist
 * GPUs through the Xe DRM userspace API (/dev/dri/renderD*). It has two
 * parts:
 *
 *   1. The ENCODER (vbs_drv_alchemist_enc.c): pure C11, no ioctls — the
 *      batch-buffer command emitters (MI_*, STATE_BASE_ADDRESS,
 *      PIPE_CONTROL, COMPUTE_WALKER), the interface-descriptor builder,
 *      the surface-state / binding-table builders, and the full
 *      compute-launch sequence builder. Fully unit-testable on a
 *      machine with no GPU.
 *   2. The DRM LAYER (vbs_drv_alchemist.c): device open/probe, VM setup,
 *      exec-queue creation, GEM buffers, VM_BIND, EXEC submission,
 *      syncobj fences. Linux-only; on other platforms (or without
 *      the Xe UAPI headers) the ops degrade to "no device" fallback ops.
 *
 * EVERY hardware fact in this layer comes from the Intel spec digest
 * (~/workspace/verum-driver-2026-11-17/specs/intel/SPEC-DIGEST.md)
 * and is cited in comments as digest section + source document.
 * Facts the digest marks UNSOURCED are never guessed at: the code
 * either avoids them, takes the documented-safe default (flagged in
 * the comment and in the driver report), or fails cleanly.
 *
 * The digest warns the system <drm/xe_drm.h> is stale (digest §3.1).
 * The DRM layer includes the system header (its struct layouts for
 * the ioctls used here are identical to the v7.3.5 header — verified
 * by diffing them) and defines the few v7.3.5 constants the system
 * header lacks, under #ifndef guards, citing the digest.
 *
 * Naming: "alchemist" here is the hardware family name from the digest
 * (the GPU family this driver targets) — the framework API itself
 * stays vendor-neutral (vbs_drv_*).
 */

#ifndef VBS_DRV_ALCHEMIST_H
#define VBS_DRV_ALCHEMIST_H

#include "vbs_drv.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Register the Alchemist ops table with the framework. Idempotent. */
vbs_drv_status_t vbs_drv_alchemist_register(void);

/* Last detailed failure message from the Alchemist layer (e.g. which
 * ioctl failed and why). Never NULL; "" when nothing has failed. */
const char *vbs_drv_alchemist_last_error(void);

/* ------------------------------------------------------------------ */
/* Hardware constants — digest §1                                      */
/* ------------------------------------------------------------------ */

/* Intel PCI vendor id (digest §1; include/drm/intel/pciids.h). */
#define VBS_DRV_ALCHEMIST_PCI_VENDOR 0x8086u

/* Alchemist graphics virtual-address width: 48-bit PPGTT
 * (digest §1, PRM Vol 4 pp.3-4; digest §7.2, PRM Vol 6 p.17). The
 * driver prefers the VA_BITS value from the CONFIG query at open
 * time; this is the documented width used when reasoning about VA
 * layout. */
#define VBS_DRV_ALCHEMIST_VA_BITS 48

/* Full-die EU count for ACM-G10 (digest §1, PRM Vol 4 pp.3-4:
 * 8 slices, 32 DSS, 16 EU/DSS, 512 EU). Cut-down dies enable fewer;
 * the probe path prefers the kernel-reported GT_TOPOLOGY value. */
#define VBS_DRV_ALCHEMIST_G10_EUS 512

/* v1 compute threads are SIMD16 (digest §6.1: for F32 the widest
 * single instruction is SIMD16; the digest's vector-add kernel is
 * SIMD16, §6.9). The framework launch description carries no SIMD
 * field, so the driver fixes it — see the driver report
 * (framework gap F1). */
#define VBS_DRV_ALCHEMIST_SIMD 16

/* Maximum threads per threadgroup for a SIMD16 launch: the
 * interface descriptor's workaround note caps threadgroups at 64
 * threads (digest §5.1, PRM Vol 2d p.718), and with Emit Local ID
 * the SIMD16 cap is also 64 (1024 work items / 16). */
#define VBS_DRV_ALCHEMIST_MAX_THREADS 64

/* Maximum work items per threadgroup (digest §4.8, PRM Vol 2a
 * p.490: LWS must be <= 1024). */
#define VBS_DRV_ALCHEMIST_MAX_LWS 1024

/* Maximum SLM per threadgroup the SLM_SIZE enum can express:
 * 64 KB (digest §5.6, PRM Vol 2b p.108). */
#define VBS_DRV_ALCHEMIST_MAX_SLM_BYTES (64u * 1024u)

/* ------------------------------------------------------------------ */
/* Command DW0 values — digest §4, PRM Vol 2a                         */
/* ------------------------------------------------------------------ */

/* MI_NOOP, opcode 0x00 (digest §4.2, Vol 2a p.1052). */
#define VBS_DRV_ALCHEMIST_MI_NOOP 0x00000000u

/* MI_BATCH_BUFFER_END, opcode 0x0A (digest §4.2, Vol 2a p.996). */
#define VBS_DRV_ALCHEMIST_MI_BATCH_BUFFER_END 0x05000000u

/* MI_BATCH_BUFFER_START, opcode 0x31, PPGTT first-level batch
 * (digest §4.2, Vol 2a pp.997-1001): bit 8 ASI=1 (PPGTT), length 1. */
#define VBS_DRV_ALCHEMIST_MI_BATCH_BUFFER_START_PPGTT 0x18800101u

/* MI_STORE_DATA_IMM, opcode 0x20, base DW0 before length bits
 * (digest §4.2, Vol 2a pp.1082-1084). */
#define VBS_DRV_ALCHEMIST_MI_STORE_DATA_IMM 0x10000000u

/* MI_FLUSH_DW, opcode 0x26 (digest §4.2, Vol 2a pp.1019-1027).
 * BlitterCS only; ComputeCS uses PIPE_CONTROL instead. */
#define VBS_DRV_ALCHEMIST_MI_FLUSH_DW 0x13000000u

/* MI_SEMAPHORE_WAIT, opcode 0x1C (digest §4.2, Vol 2a pp.1071-1075). */
#define VBS_DRV_ALCHEMIST_MI_SEMAPHORE_WAIT 0x0E000000u

/* MI_ATOMIC, opcode 0x2F (digest §4.2, Vol 2a pp.990-995). */
#define VBS_DRV_ALCHEMIST_MI_ATOMIC 0x17800000u

/* PIPELINE_SELECT for the GPGPU pipeline (digest §4.3, Vol 2a
 * pp.1166-1168): 0x69040000 | mask 0x0300 | value 0x02. Allowed on
 * ComputeCS but not required (the ComputeCS pipeline is already
 * GPGPU); the v1 launch sequence does not emit it. */
#define VBS_DRV_ALCHEMIST_PIPELINE_SELECT_GPGPU 0x69040302u

/* STATE_BASE_ADDRESS, 22 dwords (digest §4.4, Vol 2a p.1294). */
#define VBS_DRV_ALCHEMIST_STATE_BASE_ADDRESS_DW0 0x61010014u
#define VBS_DRV_ALCHEMIST_STATE_BASE_ADDRESS_WORDS 22

/* STATE_SIP, 3 dwords (digest §4.5, Vol 2a p.1307). v1 leaves the
 * SIP unset (mid-thread preemption is not supported by the
 * hardware anyway, digest §4.5). */
#define VBS_DRV_ALCHEMIST_STATE_SIP_DW0 0x61020001u

/* 3DSTATE_BINDING_TABLE_POOL_ALLOC, 4 dwords (digest §4.6,
 * Vol 2a p.29). */
#define VBS_DRV_ALCHEMIST_BINDING_TABLE_POOL_ALLOC_DW0 0x79190002u
#define VBS_DRV_ALCHEMIST_BINDING_TABLE_POOL_ALLOC_WORDS 4

/* PIPE_CONTROL, 6 dwords (digest §4.7, Vol 2a pp.1155-1162). */
#define VBS_DRV_ALCHEMIST_PIPE_CONTROL_DW0 0x7A000004u
#define VBS_DRV_ALCHEMIST_PIPE_CONTROL_WORDS 6

/* PIPE_CONTROL DW0 bits (digest §4.7, Vol 2a pp.1155-1156). */
#define VBS_DRV_ALCHEMIST_PC_HDC_FLUSH      (1u << 9)  /* HDC Pipeline Flush */
#define VBS_DRV_ALCHEMIST_PC_UNTYPED_L1_FLUSH (1u << 11) /* Untyped L1 Flush */

/* PIPE_CONTROL DW1 bits (digest §4.7, Vol 2a pp.1158-1162). */
#define VBS_DRV_ALCHEMIST_PC_CS_STALL       (1u << 20) /* Cmd Streamer Stall */
#define VBS_DRV_ALCHEMIST_PC_IC_INVALIDATE  (1u << 11) /* Instr Cache Invalidate */

/* COMPUTE_WALKER, 39 dwords (digest §4.8, Vol 2a p.485). */
#define VBS_DRV_ALCHEMIST_COMPUTE_WALKER_DW0 0x72080025u
#define VBS_DRV_ALCHEMIST_COMPUTE_WALKER_WORDS 39

/* ------------------------------------------------------------------ */
/* State structures — digest §5, PRM Vol 2d                           */
/* ------------------------------------------------------------------ */

/* INTERFACE_DESCRIPTOR_DATA: 256 bits = 8 dwords (digest §5.1,
 * Vol 2d pp.714-719). */
#define VBS_DRV_ALCHEMIST_IFACE_DESC_WORDS 8

typedef struct {
    uint32_t w[VBS_DRV_ALCHEMIST_IFACE_DESC_WORDS];
} vbs_drv_alchemist_iface_desc_t;

/* RENDER_SURFACE_STATE: 512 bits = 16 dwords (digest §5.3,
 * Vol 2d pp.860-900). */
#define VBS_DRV_ALCHEMIST_SURFACE_STATE_WORDS 16
#define VBS_DRV_ALCHEMIST_SURFACE_STATE_BYTES \
    (VBS_DRV_ALCHEMIST_SURFACE_STATE_WORDS * 4)

typedef struct {
    uint32_t w[VBS_DRV_ALCHEMIST_SURFACE_STATE_WORDS];
} vbs_drv_alchemist_surface_state_t;

/* POSTSYNC_DATA: 160 bits = 5 dwords (digest §5.4, Vol 2d
 * pp.833-834). */
#define VBS_DRV_ALCHEMIST_POSTSYNC_WORDS 5

/* State-BO layout (a driver choice, shared by the encoder and the
 * DRM layer): the binding-table pool occupies the first 4 KiB page,
 * surface states start at the second 4 KiB page. The pool base and
 * the surface-state base must both be 4 KiB aligned (digest §4.4,
 * §4.6), and the binding-table pointer is a 32-byte-aligned offset
 * from the pool base (digest §5.1). */
#define VBS_DRV_ALCHEMIST_STATE_BINDING_TABLE_OFF 0x0000u
#define VBS_DRV_ALCHEMIST_STATE_SURFACE_OFF       0x1000u
#define VBS_DRV_ALCHEMIST_STATE_BYTES             0x2000u

/* ------------------------------------------------------------------ */
/* MI command emitters — digest §4.2                                   */
/* ------------------------------------------------------------------ */

/* Emit one MI_NOOP word. */
vbs_drv_status_t vbs_drv_alchemist_emit_noop(vbs_drv_cmdbuf_t *cb);

/* Emit MI_BATCH_BUFFER_END. Every batch buffer must end with this
 * (digest §4.1, Vol 2a p.997). */
vbs_drv_status_t vbs_drv_alchemist_emit_batch_end(vbs_drv_cmdbuf_t *cb);

/* Emit PIPELINE_SELECT(GPGPU). Not used by the v1 launch sequence
 * (ComputeCS is already GPGPU, digest §4.3); provided for tests
 * and for a future RenderCS path. */
vbs_drv_status_t vbs_drv_alchemist_emit_pipeline_select_gpgpu(
    vbs_drv_cmdbuf_t *cb);

/* ------------------------------------------------------------------ */
/* STATE_BASE_ADDRESS — digest §4.4                                    */
/*                                                                     */
/* Programs the bases the launch uses and leaves the rest
 * unmodified (Modify Enable 0):                                       */
/*   - General State Base = general_base (also the A32 stateless      */
/*     base; v1 kernels use A64, so this base is effectively unused   */
/*     — the launch sequence sets it equal to the instruction base,   */
/*     see the launch-sequence builder).                               */
/*   - Surface State Base = surface_base.                              */
/*   - Instruction Base = instr_base.                                  */
/*   - Dynamic / Indirect-Object / Bindless bases: untouched (no      */
/*     samplers, no indirect data, no bindless surfaces in v1).        */
/* Bases must be 4 KiB aligned and below 2^48; sizes must be 4 KiB
 * multiples and non-zero. MOCS is 0 everywhere (the MOCS value
 * table is UNSOURCED in the digest, §9 item 6). */
/* ------------------------------------------------------------------ */

vbs_drv_status_t vbs_drv_alchemist_emit_state_base_address(
    vbs_drv_cmdbuf_t *cb,
    uint64_t general_base, uint32_t general_size,
    uint64_t surface_base,
    uint64_t instr_base, uint32_t instr_size);

/* ------------------------------------------------------------------ */
/* 3DSTATE_BINDING_TABLE_POOL_ALLOC — digest §4.6                      */
/*                                                                     */
/* pool_base must be 4 KiB aligned; pool_size must be a non-zero
 * 4 KiB multiple ("Programming size of zero is illegal in the case
 * that the pool is enabled", Vol 2a p.29). MOCS is 0 (UNSOURCED,
 * digest §9 item 6). */
/* ------------------------------------------------------------------ */

vbs_drv_status_t vbs_drv_alchemist_emit_binding_table_pool(
    vbs_drv_cmdbuf_t *cb, uint64_t pool_base, uint32_t pool_size);

/* ------------------------------------------------------------------ */
/* PIPE_CONTROL — digest §4.7                                          */
/*                                                                     */
/* Two canonical forms, both with Command Streamer Stall set (the
 * digest notes CS Stall "must be always set" on ComputeCS):          */
/*   - ic_invalidate: leading fence, CS Stall + Instruction Cache     */
/*     Invalidate (digest §4.9: before the walker, after uploading    */
/*     new shader code).                                               */
/*   - flush: trailing fence, CS Stall + HDC Pipeline Flush +
 *     Untyped L1 Flush (digest §4.7: the canonical compute fence).   */
/* Neither writes a post-sync value (completion is signaled by the
 * EXEC syncobj; the optional launch semaphore uses the walker
 * PostSync instead). */
/* ------------------------------------------------------------------ */

vbs_drv_status_t vbs_drv_alchemist_emit_pipe_control_ic_invalidate(
    vbs_drv_cmdbuf_t *cb);
vbs_drv_status_t vbs_drv_alchemist_emit_pipe_control_flush(
    vbs_drv_cmdbuf_t *cb);

/* ------------------------------------------------------------------ */
/* INTERFACE_DESCRIPTOR_DATA builder — digest §5.1                     */
/* ------------------------------------------------------------------ */

typedef struct {
    /* Byte offset of the kernel entry from the Instruction Base
     * Address; must be 64-byte aligned and fit in 32 bits (the
     * pointer field is DW0 [31:6] with DW1 reserved MBZ — PRM
     * Vol 2d p.714, confirmed against the extracted text). */
    uint32_t kernel_start_offset;
    /* Byte offset of the binding table from the Binding Table Pool
     * Base Address; must be 32-byte aligned and fit in bits
     * [20:5] (max 2 MiB pool offset). */
    uint32_t binding_table_offset;
    /* Shared local memory per threadgroup in bytes; rounded up to
     * the SLM_SIZE enum (digest §5.6). Must be <= 64 KiB. */
    uint32_t slm_bytes;
    /* Barriers per threadgroup, 0..7 (3-bit field). */
    uint32_t barrier_count;
    /* Threads in the threadgroup, 1..64 (SIMD16 cap, see
     * VBS_DRV_ALCHEMIST_MAX_THREADS). */
    uint32_t thread_count;
} vbs_drv_alchemist_iface_params_t;

/* Build an interface descriptor. Fixed v1 choices (all documented
 * defaults): FP mode IEEE-754, denorms flushed to zero, rounding
 * RTNE, no exception enables, no samplers, no BTD mode, thread
 * group dispatch size 8 (default), binding-table prefetch count 0
 * ("Typically set to 0 to avoid prefetching on every thread
 * dispatch", Vol 2d p.716). */
vbs_drv_status_t vbs_drv_alchemist_iface_build(
    const vbs_drv_alchemist_iface_params_t *p,
    vbs_drv_alchemist_iface_desc_t *out);

/* ------------------------------------------------------------------ */
/* Surface state + binding table — digest §5.2/§5.3                    */
/* ------------------------------------------------------------------ */

/* Build a SURFTYPE_BUFFER surface state for a raw compute buffer.
 * base_va must be 4-byte aligned (the surface base "must be
 * naturally-aligned to the element size", Vol 2d p.860 — v1
 * buffers hold 4-byte elements); size_bytes must be a non-zero
 * multiple of 4. Surface Format is 0 (ignored for untyped
 * data-port messages, digest §5.3); MOCS is 0 (UNSOURCED, digest
 * §9 item 6). Width is the element count - 1, clamped to the
 * 14-bit field (0x3FFF): buffers larger than 64 KiB of F32
 * elements cannot express their full extent in this field — see
 * the driver report (needs hardware confirmation whether untyped
 * BTS messages bounds-check against Width at all). */
vbs_drv_status_t vbs_drv_alchemist_surface_build(
    uint64_t base_va, uint32_t size_bytes,
    vbs_drv_alchemist_surface_state_t *out);

/* One binding-table entry: the surface-state offset (relative to
 * the Surface State Base Address), stored in bits [31:6] (digest
 * §5.2, Vol 2d p.1084). The offset must be 64-byte aligned; the
 * low 6 bits are masked off. */
uint32_t vbs_drv_alchemist_binding_entry(uint32_t surface_state_offset);

/* ------------------------------------------------------------------ */
/* COMPUTE_WALKER — digest §4.8                                        */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t grid_x, grid_y, grid_z;
    uint32_t block_x, block_y, block_z;
    /* Pre-built interface descriptor (inlined at walker DW18-25).
     * Its thread count must satisfy (SIMD)*(N-1) < LWS <=
     * (SIMD)*N for the block size (digest §4.8, Vol 2a p.490). */
    vbs_drv_alchemist_iface_desc_t iface;
    /* Optional walker PostSync (digest §5.4): when set, the walker
     * writes sem_payload (zero-extended to 64 bits) to sem_va
     * after the grid completes, with Dataport Pipeline Flush +
     * Dataport Subslice Cache Flush. sem_va must be 8-byte
     * aligned (Write Immediate Data requires 8-byte alignment,
     * Vol 2d p.833). */
    bool     use_semaphore;
    uint64_t sem_va;
    uint32_t sem_payload;
} vbs_drv_alchemist_walker_params_t;

/* Emit one 39-dword COMPUTE_WALKER. v1 choices:
 *   - SIMD16 (VBS_DRV_ALCHEMIST_SIMD), Message SIMD = SIMD16 (the
 *     Message SIMD enum matches the SIMD Size enum, Vol 2a p.489;
 *     the "must be <= Thread SIMD size" restriction is met with
 *     equality).
 *   - Generate Local ID set, Emit Local X (+Y/+Z when the block
 *     has more than one work item in that dimension). The
 *     local-ID generation restriction (Vol 2a p.488 — at least two
 *     of the three local dimension sizes must be powers of two,
 *     the non-power-of-two dimension lowest priority in the walk
 *     order) is enforced: at most one block dimension may have a
 *     non-power-of-two size, and the walk order is chosen to put
 *     that dimension last (Walk 210 for X, 021 for Y, 012 — the
 *     default — for Z). NOTE: the PRM text says "maximum values"
 *     where the sizes reading is the only one consistent with the
 *     digest's own kernel design; see the driver report.
 *   - Block shapes are restricted by the Emit Local value table
 *     (Vol 2a p.488: only 0/1/3/7 are valid): a dimension may
 *     exceed 1 only when every lower dimension also exceeds 1, so
 *     (1,16,1) and (16,1,16) are rejected.
 *   - No indirect data, no partitioning, no predication, no
 *     systolic mode. */
vbs_drv_status_t vbs_drv_alchemist_emit_walker(
    vbs_drv_cmdbuf_t *cb, const vbs_drv_alchemist_walker_params_t *p);

/* ------------------------------------------------------------------ */
/* Full launch sequence — digest §4.9                                  */
/*                                                                     */
/*   STATE_BASE_ADDRESS                                                */
/*   3DSTATE_BINDING_TABLE_POOL_ALLOC  (only when const_count > 0)     */
/*   PIPE_CONTROL (CS Stall + IC Invalidate)                           */
/*   COMPUTE_WALKER                                                    */
/*   PIPE_CONTROL (CS Stall + HDC Flush + Untyped L1 Flush)            */
/*   MI_BATCH_BUFFER_END                                               */
/*                                                                     */
/* The caller writes the binding table + surface states into the
 * state BO (at VBS_DRV_ALCHEMIST_STATE_BINDING_TABLE_OFF /
 * VBS_DRV_ALCHEMIST_STATE_SURFACE_OFF) before emitting; this builder
 * only emits the commands. The General State Base is set equal to
 * the Instruction Base (both = code_va): the interface descriptor
 * normatively says the kernel pointer is relative to the
 * Instruction Base, while a Vol 9 sentence says instruction
 * addresses are offsets from the General State Base — setting
 * them equal satisfies both readings (digest §6.10). */
/* ------------------------------------------------------------------ */

typedef struct {
    uint64_t code_va;     /* GPU VA of the code buffer (= Instr Base) */
    uint32_t code_offset; /* kernel entry offset, 64-byte aligned */
    uint32_t code_size;   /* code buffer alloc size, 4 KiB multiple */
    uint64_t state_va;    /* GPU VA of the state BO */
    uint32_t const_count; /* binding-table entries in use, 0..8 */
    uint32_t grid_x, grid_y, grid_z;
    uint32_t block_x, block_y, block_z;
    uint32_t slm_bytes;
    uint32_t barrier_count;
    bool     use_semaphore;
    uint64_t sem_va;
    uint32_t sem_payload;
} vbs_drv_alchemist_launch_params_t;

vbs_drv_status_t vbs_drv_alchemist_emit_launch_sequence(
    vbs_drv_cmdbuf_t *cb, const vbs_drv_alchemist_launch_params_t *p);

#ifdef __cplusplus
}
#endif

#endif /* VBS_DRV_ALCHEMIST_H */
