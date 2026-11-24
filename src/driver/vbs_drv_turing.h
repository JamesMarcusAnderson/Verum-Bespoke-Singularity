/* vbs_drv_turing.h — Verum driver for Turing-family GPUs (TU102, sm_75).
 *
 * This layer implements the framework ops (vbs_drv_ops_t) for Turing
 * GPUs through the Nouveau DRM userspace API (/dev/dri/renderD*).
 * It has two parts:
 *
 *   1. The ENCODER (vbs_drv_turing_enc.c): pure C11, no ioctls — the
 *      QMD v2.3 launch-descriptor builder and the pushbuffer method
 *      encoder. Fully unit-testable on a machine with no GPU.
 *   2. The DRM LAYER (vbs_drv_turing.c): device open/probe, VM setup,
 *      channel allocation, GEM buffers, VM_BIND, EXEC submission,
 *      syncobj fences. Linux-only; on other platforms (or without
 *      the Nouveau UAPI headers) the ops degrade to "no device".
 *
 * EVERY hardware fact in this layer comes from the project spec
 * digest (~/workspace/verum-driver-2026-11-17/specs/SPEC-DIGEST.md)
 * and is cited in comments as digest section + source tag. Facts the
 * digest marks UNSOURCED are never guessed at: the code either
 * avoids them, takes them as caller-supplied values, or fails
 * cleanly. See the driver report for the full list.
 *
 * Naming: "turing" here is the hardware family name from the digest
 * (the GPU family this driver targets, like sm_75) — the framework
 * API itself stays vendor-neutral (vbs_drv_*).
 */

#ifndef VBS_DRV_TURING_H
#define VBS_DRV_TURING_H

#include "vbs_drv.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Register the Turing ops table with the framework. Idempotent. */
vbs_drv_status_t vbs_drv_turing_register(void);

/* Last detailed failure message from the Turing layer (e.g. which
 * ioctl failed and why). Never NULL; "" when nothing has failed. */
const char *vbs_drv_turing_last_error(void);

/* ------------------------------------------------------------------ */
/* Hardware constants — digest §1/§3.5, sources S1/S9/S10              */
/* ------------------------------------------------------------------ */

/* Turing compute class id (digest §3.5; S1 clc5c0.h line 31). */
#define VBS_DRV_TURING_COMPUTE_CLASS 0xC5C0u

/* Turing DMA copy class id (digest §3.5; S3 clc5b5.h). Reserved for a
 * future async-copy subchannel; v1 does not use it. */
#define VBS_DRV_TURING_DMA_COPY_CLASS 0xC5B5u

/* Compute capability of the target family (digest §1; S4 §11.1.2). */
#define VBS_DRV_TURING_SM_MAJOR 7
#define VBS_DRV_TURING_SM_MINOR 5

/* GPU MMU width: 47-bit addresses (digest §1; S13). */
#define VBS_DRV_TURING_VA_BITS 47

/* Full-die SM count for TU102 (digest §1; S10 p.7). Note: cut-down
 * TU102 boards (e.g. RTX 2080 Ti) enable fewer SMs; the probe path
 * prefers the kernel-reported GRAPH_UNITS value over this constant. */
#define VBS_DRV_TURING_TU102_SMS 72

/* QMD v2.3 size: 256 bytes = 64 x 32-bit words (digest §4; S1
 * clc5c0qmd.h header comment, "Queue Meta Data, Version 02_03"). */
#define VBS_DRV_TURING_QMD_WORDS 64
#define VBS_DRV_TURING_QMD_BYTES (VBS_DRV_TURING_QMD_WORDS * 4)

/* QMD version fields (digest §4; S1 clc5c0qmd.h; SASS_VERSION from
 * S12, cross-verified against S1). */
#define VBS_DRV_TURING_QMD_VERSION       3u /* word 18 bits 3:0 */
#define VBS_DRV_TURING_QMD_MAJOR_VERSION 2u /* word 18 bits 7:4 */
#define VBS_DRV_TURING_SASS_VERSION      0x75u /* word 31 bits 31:24 */

/* SEND_PCAS_A carries only the low 32 bits of the >>8-shifted QMD
 * address (digest §3.5; S1), so the QMD buffer must live below
 * 2^40 in GPU VA space. */
#define VBS_DRV_TURING_QMD_MAX_VA (1ULL << 40)

/* ------------------------------------------------------------------ */
/* Compute-class methods — digest §3.5, S1 clc5c0.h.                  */
/* Offsets are the x4 method-address form used by the class header.    */
/* ------------------------------------------------------------------ */

#define VBS_DRV_TURING_MTHD_SET_OBJECT                 0x0000u
#define VBS_DRV_TURING_MTHD_NO_OPERATION               0x0100u
#define VBS_DRV_TURING_MTHD_WAIT_FOR_IDLE              0x0110u
#define VBS_DRV_TURING_MTHD_LAUNCH_DMA                 0x01b0u
#define VBS_DRV_TURING_MTHD_INVALIDATE_SHADER_CACHES   0x021cu
#define VBS_DRV_TURING_MTHD_SET_QMD_VERSION            0x0288u
#define VBS_DRV_TURING_MTHD_CHECK_QMD_VERSION          0x0290u
#define VBS_DRV_TURING_MTHD_SET_SHADER_SHARED_MEM_WIN_A 0x02a0u
#define VBS_DRV_TURING_MTHD_SET_SHADER_SHARED_MEM_WIN_B 0x02a4u
#define VBS_DRV_TURING_MTHD_SEND_PCAS_A                0x02b4u
#define VBS_DRV_TURING_MTHD_SEND_PCAS_B                0x02b8u
#define VBS_DRV_TURING_MTHD_SEND_SIGNALING_PCAS_B      0x02bcu
#define VBS_DRV_TURING_MTHD_SET_SPA_VERSION            0x0310u
#define VBS_DRV_TURING_MTHD_SET_INLINE_QMD_ADDR_A      0x0318u
#define VBS_DRV_TURING_MTHD_SET_INLINE_QMD_ADDR_B      0x031cu
#define VBS_DRV_TURING_MTHD_LOAD_INLINE_QMD_DATA_BASE  0x0320u /* + i*4 */
#define VBS_DRV_TURING_MTHD_SET_SHADER_LOCAL_MEM_A     0x0790u
#define VBS_DRV_TURING_MTHD_SET_SHADER_LOCAL_MEM_B     0x0794u
#define VBS_DRV_TURING_MTHD_SET_SHADER_LOCAL_MEM_WIN_A 0x07b0u
#define VBS_DRV_TURING_MTHD_SET_SHADER_LOCAL_MEM_WIN_B 0x07b4u
#define VBS_DRV_TURING_MTHD_SET_SHADER_CACHE_CONTROL   0x0d94u
#define VBS_DRV_TURING_MTHD_SET_SM_TIMEOUT_INTERVAL    0x0de4u
#define VBS_DRV_TURING_MTHD_SET_REPORT_SEMAPHORE_A     0x1b00u
#define VBS_DRV_TURING_MTHD_SET_REPORT_SEMAPHORE_B     0x1b04u
#define VBS_DRV_TURING_MTHD_SET_REPORT_SEMAPHORE_C     0x1b08u
#define VBS_DRV_TURING_MTHD_SET_REPORT_SEMAPHORE_D     0x1b0cu
#define VBS_DRV_TURING_MTHD_SET_TRAP_HANDLER_A         0x25f8u
#define VBS_DRV_TURING_MTHD_SET_TRAP_HANDLER_B         0x25fcu

/* SEND_SIGNALING_PCAS_B value: INVALIDATE (bit 0) + SCHEDULE (bit 1)
 * = 3, the actual launch trigger (digest §3.5; S1). */
#define VBS_DRV_TURING_PCAS_LAUNCH 0x3u

/* SET_OBJECT parameter packing (digest §3.2; S1 clc5c0.h lines 33-35):
 * CLASS_ID bits 15:0, ENGINE_ID bits 20:16. PGRAPH is engine 0
 * (digest §3.1; S6 hw/fifo/intro.rst). */
#define VBS_DRV_TURING_SET_OBJECT_PARAM(class_id, engine_id) \
    (((uint32_t)(class_id) & 0xffffu) | (((uint32_t)(engine_id) & 0x1fu) << 16))

/* ------------------------------------------------------------------ */
/* QMD v2.3 builder — digest §4, S1 clc5c0qmd.h                       */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t w[VBS_DRV_TURING_QMD_WORDS];
} vbs_drv_turing_qmd_t;

/* Zero a QMD and stamp the version fields (QMD_VERSION 3,
 * QMD_MAJOR_VERSION 2, SASS_VERSION 0x75). Every builder starts here. */
void vbs_drv_turing_qmd_zero(vbs_drv_turing_qmd_t *qmd);

/* Program address + prefetch address. code_va is the GPU VA of the
 * shader code; code_size is currently recorded only for validation
 * (the prefetch-size field's unit is UNSOURCED in the digest, so the
 * builder leaves PROGRAM_PREFETCH_SIZE at 0 = no prefetch; see the
 * report). */
vbs_drv_status_t vbs_drv_turing_qmd_set_program(vbs_drv_turing_qmd_t *qmd,
                                                uint64_t code_va,
                                                uint32_t code_size);

/* Grid (CTA raster) dimensions. */
vbs_drv_status_t vbs_drv_turing_qmd_set_grid(vbs_drv_turing_qmd_t *qmd,
                                             uint32_t gx, uint32_t gy,
                                             uint32_t gz);

/* Block (CTA thread) dimensions. */
vbs_drv_status_t vbs_drv_turing_qmd_set_block(vbs_drv_turing_qmd_t *qmd,
                                              uint32_t bx, uint32_t by,
                                              uint32_t bz);

/* Per-thread register count (1..255: the REGISTER_COUNT field is 8
 * bits, digest §4.1; PTX .maxnreg caps at 255, digest §5.2),
 * shared memory per CTA in bytes (18-bit field), barrier count
 * (0 unless the kernel uses bar.sync). */
vbs_drv_status_t vbs_drv_turing_qmd_set_resources(vbs_drv_turing_qmd_t *qmd,
                                                  uint32_t regs,
                                                  uint32_t shared_bytes,
                                                  uint32_t barriers);

/* Bind constant buffer `idx` (0..7) to a GPU VA range. size must be
 * a multiple of 16 and at most 128 KiB (the SIZE_SHIFTED4 field is
 * 13 bits of 16-byte units, digest §4.1). */
vbs_drv_status_t vbs_drv_turing_qmd_set_const_buf(vbs_drv_turing_qmd_t *qmd,
                                                  unsigned idx, uint64_t va,
                                                  uint32_t size_bytes);

/* Configure completion semaphore `which` (0 or 1): when the grid
 * finishes, the hardware writes `payload` to `va`. Reduction is
 * left disabled (plain payload write). */
vbs_drv_status_t vbs_drv_turing_qmd_set_release(vbs_drv_turing_qmd_t *qmd,
                                                unsigned which, uint64_t va,
                                                uint32_t payload);

/* Set the instruction-cache and shader-constant-cache invalidate
 * bits (word 5). Call on the first launch after uploading new
 * shader code. Bit positions follow the digest §4.1 listing order
 * (see the encoder source for the assumption note). */
void vbs_drv_turing_qmd_invalidate_caches(vbs_drv_turing_qmd_t *qmd);

/* Build a complete QMD from a framework launch description. The
 * code buffer's GPU VA plus code_offset is the program address;
 * constant buffers map to CB0..CBn in order; the optional
 * semaphore maps to RELEASE0. */
vbs_drv_status_t vbs_drv_turing_qmd_build(const vbs_drv_launch_desc_t *desc,
                                          vbs_drv_turing_qmd_t *qmd);

/* ------------------------------------------------------------------ */
/* Pushbuffer encoder — digest §3.3, S6 hw/fifo/dma-pusher.rst        */
/*                                                                     */
/* v1 uses the OLD (NV4) command format, which the digest documents   */
/* completely; the GF100+ "new" format's header bitfields are         */
/* UNSOURCED in prose (digest §10 item 5) and the old format is       */
/* still supported on Turing (digest §3.3).                           */
/*                                                                     */
/* NOTE on the digest itself: its §3.3 prose says the count field     */
/* sits at "bits 29-19", but its own bit-string                        */
/* (000CCCCCCCCCCC00SSSMMMMMMMMMMM00) puts the 11-bit count at bits   */
/* 28-18, the subchannel at bits 15-13, and the 11-bit method at      */
/* bits 12-2. The encoder follows the bit-string (which also matches  */
/* the classic NV4 encoding: count<<18 | subchan<<13 | method>>2).    */
/* The discrepancy is flagged in the driver report.                   */
/* ------------------------------------------------------------------ */

/* Header word for an increasing-methods command: `count` data words
 * follow, going to methods M, M+1, ... (method numbers in x4 form).
 * count must be 1..2047, subchan 0..7. */
uint32_t vbs_drv_turing_pb_hdr_incr(uint32_t subchan, uint32_t method,
                                    uint32_t count);

/* Header word for a non-increasing-methods command: all `count` data
 * words go to the same method. */
uint32_t vbs_drv_turing_pb_hdr_nonincr(uint32_t subchan, uint32_t method,
                                       uint32_t count);

/* Emit one method with one parameter word. */
vbs_drv_status_t vbs_drv_turing_pb_emit_method1(vbs_drv_cmdbuf_t *cb,
                                                uint32_t subchan,
                                                uint32_t method,
                                                uint32_t value);

/* Emit an increasing run of methods starting at `first_method`
 * (values[i] goes to method first_method + i*4). */
vbs_drv_status_t vbs_drv_turing_pb_emit_methods(vbs_drv_cmdbuf_t *cb,
                                                uint32_t subchan,
                                                uint32_t first_method,
                                                const uint32_t *values,
                                                uint32_t count);

/* Emit a NO_OPERATION (padding / alignment). */
vbs_drv_status_t vbs_drv_turing_pb_emit_nop(vbs_drv_cmdbuf_t *cb,
                                            uint32_t subchan);

/* Emit WAIT_FOR_IDLE (stall the channel until the engine is idle). */
vbs_drv_status_t vbs_drv_turing_pb_emit_wait_idle(vbs_drv_cmdbuf_t *cb,
                                                  uint32_t subchan);

/* Emit INVALIDATE_SHADER_CACHES (needed after writing new shader
 * code; digest §3.5). */
vbs_drv_status_t vbs_drv_turing_pb_emit_invalidate_caches(vbs_drv_cmdbuf_t *cb,
                                                          uint32_t subchan);

/* Upload a QMD into GPU memory from the pushbuffer itself:
 * SET_INLINE_QMD_ADDRESS_A/B = (qmd_va >> 8), then 64x
 * LOAD_INLINE_QMD_DATA words carrying the 256-byte QMD
 * (digest §3.5 steps 2; S1). */
vbs_drv_status_t vbs_drv_turing_pb_emit_inline_qmd(vbs_drv_cmdbuf_t *cb,
                                                   uint32_t subchan,
                                                   uint64_t qmd_va,
                                                   const vbs_drv_turing_qmd_t *qmd);

/* Dispatch a launch: SEND_PCAS_A = (qmd_va >> 8), SEND_PCAS_B = 0
 * (single launch, FROM/DELTA 0), SEND_SIGNALING_PCAS_B = 3
 * (INVALIDATE + SCHEDULE). qmd_va must be below 2^40 (see
 * VBS_DRV_TURING_QMD_MAX_VA). */
vbs_drv_status_t vbs_drv_turing_pb_emit_launch(vbs_drv_cmdbuf_t *cb,
                                               uint32_t subchan,
                                               uint64_t qmd_va);

/* Emit SET_QMD_VERSION with a caller-supplied value. The value
 * encoding for QMD v2.3 is NOT in the digest, so the driver never
 * guesses it: callers that know the value (from hardware
 * validation) pass it in. */
vbs_drv_status_t vbs_drv_turing_pb_emit_set_qmd_version(vbs_drv_cmdbuf_t *cb,
                                                        uint32_t subchan,
                                                        uint32_t value);

/* Emit a completion semaphore (SET_REPORT_SEMAPHORE_A/B/C/D). The
 * control word's field layout is NOT in the digest beyond the
 * field names, so it is caller-supplied. */
vbs_drv_status_t vbs_drv_turing_pb_emit_report_semaphore(vbs_drv_cmdbuf_t *cb,
                                                         uint32_t subchan,
                                                         uint64_t addr,
                                                         uint32_t payload,
                                                         uint32_t control);

/* The full v1 launch sequence on one subchannel (digest §3.5):
 * inline-QMD upload, INVALIDATE_SHADER_CACHES, SEND_PCAS dispatch.
 * SET_QMD_VERSION is NOT emitted (value encoding unsourced — see
 * above); completion is signaled by the EXEC syncobj, not by a
 * semaphore method. */
vbs_drv_status_t vbs_drv_turing_pb_emit_launch_sequence(vbs_drv_cmdbuf_t *cb,
                                                        uint32_t subchan,
                                                        uint64_t qmd_va,
                                                        const vbs_drv_turing_qmd_t *qmd);

#ifdef __cplusplus
}
#endif

#endif /* VBS_DRV_TURING_H */
