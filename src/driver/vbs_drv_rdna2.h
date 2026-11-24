/* vbs_drv_rdna2.h — Verum driver for RDNA 2 GPUs (Navi 21, gfx1030).
 *
 * This layer implements the framework ops (vbs_drv_ops_t) for Navi 21
 * GPUs (Radeon RX 6800 / 6800 XT / 6900 XT, PRO W6800, PRO V620)
 * through the amdgpu DRM userspace API (/dev/dri/renderD*). It has
 * two parts:
 *
 *   1. The ENCODER (vbs_drv_rdna2_enc.c): pure C11, no ioctls — the
 *      PM4 packet builder (SET_SH_REG, DISPATCH_DIRECT), the
 *      register-address computation (CP address = mm + segment
 *      base), the PGM_RSRC builders, and the full launch-IB builder.
 *      Fully unit-testable on a machine with no GPU.
 *   2. The DRM LAYER (vbs_drv_rdna2.c): device open/probe, INFO
 *      queries, context allocation, GEM buffers, GEM_VA mapping,
 *      BO lists, CS submission, WAIT_CS fences. Linux-only; on
 *      other platforms (or without the amdgpu UAPI headers) the ops
 *      degrade to "no device".
 *
 * EVERY hardware fact in this layer comes from the AMD spec digest
 * (~/workspace/verum-driver-2026-11-17/specs/amd/SPEC-DIGEST.md)
 * and is cited in comments as digest section + source tag. Facts
 * the digest marks UNSOURCED are never guessed at: the code either
 * avoids them, takes the documented-safe default, or fails cleanly.
 * See the driver report for the full list — including one place
 * where the digest contradicts itself and the primary source
 * (SI-PG §2.4.7) was followed instead: the SET_SH_REG REG_OFFSET
 * is a DWORD offset from a DWORD base, so no division by 4 is
 * applied (see VBS_DRV_RDNA2_SH_REG_OFFSET below).
 *
 * Firmware: this driver never embeds or ships firmware. The kernel
 * amdgpu driver loads the 12 Sienna Cichlid blobs from the
 * linux-firmware package at bring-up (digest §8); the device-open
 * path verifies the compute-relevant ones (ME, PFP, CE, MEC) are
 * actually loaded, via INFO FW_VERSION queries, and refuses to open
 * when the CP microcontrollers have no firmware.
 *
 * Naming: "rdna2" here is the hardware family name from the digest
 * (the GPU family this driver targets, like gfx1030) — the
 * framework API itself stays vendor-neutral (vbs_drv_*).
 */

#ifndef VBS_DRV_RDNA2_H
#define VBS_DRV_RDNA2_H

#include "vbs_drv.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Register the RDNA 2 ops table with the framework. Idempotent. */
vbs_drv_status_t vbs_drv_rdna2_register(void);

/* Last detailed failure message from the RDNA 2 layer (e.g. which
 * ioctl failed and why). Never NULL; "" when nothing has failed. */
const char *vbs_drv_rdna2_last_error(void);

/* ------------------------------------------------------------------ */
/* Chip identity — digest §1.4, §9 [AMDGPU-DRM], [DGPU-CSV]           */
/* ------------------------------------------------------------------ */

/* AMDGPU_FAMILY_NV = 143 (Navi). The INFO DEV_INFO `family` field
 * returns this for Navi 21 (digest §1.4). */
#define VBS_DRV_RDNA2_FAMILY_NV 143u

/* GC IP version for Navi 21: GC 10.3.0 (digest §9 [DGPU-CSV]:
 * "SIENNA_CICHLID, DCN 3.0.0, GC 10.3.0, VCN 3.0.0, SDMA 5.2.0").
 * Checked at open from INFO HW_IP_INFO on the GFX IP. */
#define VBS_DRV_RDNA2_GC_MAJOR 10u
#define VBS_DRV_RDNA2_GC_MINOR 3u

/* Wavefront size the gfx1030 target is compiled for (digest §5.1,
 * [AMDGPUUsage] target property wavefrontsize64). v1 launches
 * wave64 kernels only; wave32 (CS_W32_EN) is a later step. */
#define VBS_DRV_RDNA2_WAVEFRONT_SIZE 64u

/* LDS limit per workgroup (digest §5.1 [RDNA2-ISA §2.3.1]): a
 * single workgroup may use at most 64 KiB of LDS. */
#define VBS_DRV_RDNA2_MAX_LDS_BYTES 65536u

/* ------------------------------------------------------------------ */
/* Register addressing model — digest §3, [UMR-SC], [GC1030-OFF]     */
/*                                                                     */
/* The kernel's register headers give each register an `mm` value     */
/* (a DWORD offset). The CP-visible address is mm + segment_base,     */
/* where the segment bases for Sienna Cichlid come from the UMR       */
/* database [UMR-SC, GC section line 233]. All COMPUTE_* registers    */
/* are BASE_IDX 0 (seg0). The digest's §3 cross-validation: the       */
/* absolute address of COMPUTE_PGM_LO is identical on GFX9            */
/* (0x0E0C + 0x2000 = 0x2E0C) and GFX10.3 (0x1BAC + 0x1260 =          */
/* 0x2E0C) — the same register, two generations, one CP address.      */
/* ------------------------------------------------------------------ */

#define VBS_DRV_RDNA2_SEG0_BASE 0x1260u /* segment 0 base, dwords */
#define VBS_DRV_RDNA2_SEG1_BASE 0xA000u /* segment 1 base, dwords */
#define VBS_DRV_RDNA2_SEG2_BASE 0x1C000u /* segment 2 base, dwords */

/* PM4 register-window bases (DWORD offsets — [SI-PG §2.4.7] notes
 * the SH base as "Byte Offset 0xB000; DWord Offset 0x2C00").
 * Digest §2.2/§2.3 [SI-PG §§2.2.2, 2.4.4, 2.4.7; NVD]. */
#define VBS_DRV_RDNA2_CONFIG_REG_BASE 0x2000u
#define VBS_DRV_RDNA2_SH_REG_BASE     0x2C00u
#define VBS_DRV_RDNA2_CONTEXT_REG_BASE 0xA000u
#define VBS_DRV_RDNA2_UCONFIG_REG_BASE 0xC000u

/* CP-visible address of a BASE_IDX-0 register: mm + seg0 base
 * (digest §3). */
uint32_t vbs_drv_rdna2_cp_addr(uint32_t mm);

/* SET_SH_REG REG_OFFSET for a CP address in the SH window.
 * [SI-PG §2.4.7]: "The REG_OFFSET field is a DWord-offset from the
 * starting address" and "Reg_Start_Address[17:2] = 0x2C00 +
 * REG_OFFSET" — the base 0x2C00 is itself a DWORD offset, and the
 * CP address is a DWORD offset (mm values and segment bases are
 * dword offsets), so the offset is the plain difference, with NO
 * division by 4. This matches the digest's §4.1 table and its
 * Appendix A.1 worked packet (COMPUTE_PGM_LO → 0x0000020C), and
 * contradicts the digest's own prose formula "(CP − 0x2C00)/4" —
 * the primary source governs; see the driver report. */
uint32_t vbs_drv_rdna2_sh_reg_offset(uint32_t cp_addr);

/* ------------------------------------------------------------------ */
/* Compute registers (GC 10.3.0) — digest §4.1, [GC1030-OFF]         */
/*                                                                     */
/* Values are the `mm` (DWORD) offsets from the kernel header. The    */
/* CP address is mm + 0x1260; the SET_SH_REG offset is CP − 0x2C00.   */
/* The digest's §4.1 table lists the CP address and offset for        */
/* every row; the unit test checks all 29 rows against that table.    */
/*                                                                     */
/* Registers whose semantics the digest marks UNSOURCED               */
/* (COMPUTE_DIM_X/Y/Z, COMPUTE_DISPATCH_TUNNEL,                       */
/* COMPUTE_DISPATCH_END) are defined here for address completeness    */
/* but are NEVER programmed by v1 (digest §4.1).                      */
/* ------------------------------------------------------------------ */

#define VBS_DRV_RDNA2_MM_DISPATCH_INITIATOR 0x1BA0u
#define VBS_DRV_RDNA2_MM_DIM_X              0x1BA1u /* UNSOURCED semantics */
#define VBS_DRV_RDNA2_MM_DIM_Y              0x1BA2u /* UNSOURCED semantics */
#define VBS_DRV_RDNA2_MM_DIM_Z              0x1BA3u /* UNSOURCED semantics */
#define VBS_DRV_RDNA2_MM_START_X            0x1BA4u
#define VBS_DRV_RDNA2_MM_START_Y            0x1BA5u
#define VBS_DRV_RDNA2_MM_START_Z            0x1BA6u
#define VBS_DRV_RDNA2_MM_NUM_THREAD_X       0x1BA7u
#define VBS_DRV_RDNA2_MM_NUM_THREAD_Y       0x1BA8u
#define VBS_DRV_RDNA2_MM_NUM_THREAD_Z       0x1BA9u
#define VBS_DRV_RDNA2_MM_PGM_LO             0x1BACu
#define VBS_DRV_RDNA2_MM_PGM_HI             0x1BADu
#define VBS_DRV_RDNA2_MM_SCRATCH_BASE_LO    0x1BB0u
#define VBS_DRV_RDNA2_MM_SCRATCH_BASE_HI    0x1BB1u
#define VBS_DRV_RDNA2_MM_PGM_RSRC1          0x1BB2u
#define VBS_DRV_RDNA2_MM_PGM_RSRC2          0x1BB3u
#define VBS_DRV_RDNA2_MM_RESOURCE_LIMITS    0x1BB5u
#define VBS_DRV_RDNA2_MM_STATIC_TM_SE0      0x1BB6u
#define VBS_DRV_RDNA2_MM_STATIC_TM_SE1      0x1BB7u
#define VBS_DRV_RDNA2_MM_TMPRING_SIZE       0x1BB8u
#define VBS_DRV_RDNA2_MM_RESTART_X          0x1BBBu
#define VBS_DRV_RDNA2_MM_RESTART_Y          0x1BBCu
#define VBS_DRV_RDNA2_MM_RESTART_Z          0x1BBDu
#define VBS_DRV_RDNA2_MM_THREAD_TRACE_EN    0x1BBEu
#define VBS_DRV_RDNA2_MM_DISPATCH_ID        0x1BC0u
#define VBS_DRV_RDNA2_MM_PGM_RSRC3          0x1BC8u
#define VBS_DRV_RDNA2_MM_USER_DATA_0        0x1BE0u /* _0.._15 consecutive */
#define VBS_DRV_RDNA2_MM_DISPATCH_TUNNEL    0x1C1Du /* UNSOURCED semantics */
#define VBS_DRV_RDNA2_MM_DISPATCH_END       0x1C1Eu /* UNSOURCED semantics */

/* Number of COMPUTE_USER_DATA registers (digest §4.1, §7.3). */
#define VBS_DRV_RDNA2_USER_DATA_COUNT 16u

/* ------------------------------------------------------------------ */
/* COMPUTE_DISPATCH_INITIATOR bits — digest §4.2, [GC1030-MASK]      */
/* ------------------------------------------------------------------ */

/* Bit 0: COMPUTE_SHADER_EN. Minimal initiator value is 0x1. */
#define VBS_DRV_RDNA2_INITIATOR_COMPUTE_EN (1u << 0)
/* Bit 15: CS_W32_EN (wave32 mode; 0 = wave64). v1 never sets it. */
#define VBS_DRV_RDNA2_INITIATOR_W32_EN (1u << 15)

/* Bits 2 (FORCE_START_AT_000) and 5 (USE_THREAD_DIMENSIONS) are
 * UNSOURCED beyond their names (digest §4.2) and are NEVER set in
 * v1; no constants are defined for them, so no caller can. */

/* ------------------------------------------------------------------ */
/* PGM_RSRC fields — digest §4.4, [GC1030-MASK] + [AMDGPUUsage]      */
/* ------------------------------------------------------------------ */

/* PGM_RSRC1: VGPRS [5:0]; SGPRS [9:6] is reserved on GFX10–12 and
 * must be 0 (128 SGPRs always allocated). */
#define VBS_DRV_RDNA2_RSRC1_VGPRS_SHIFT 0u
#define VBS_DRV_RDNA2_RSRC1_VGPRS_MASK  0x3Fu

/* PGM_RSRC2: SCRATCH_EN bit 0; USER_SGPR [5:1]; LDS_SIZE [23:15]
 * = roundup(lds_bytes / 512) (GFX7–12). */
#define VBS_DRV_RDNA2_RSRC2_SCRATCH_EN (1u << 0)
#define VBS_DRV_RDNA2_RSRC2_USER_SGPR_SHIFT 1u
#define VBS_DRV_RDNA2_RSRC2_USER_SGPR_MASK  0x1Fu
#define VBS_DRV_RDNA2_RSRC2_LDS_SIZE_SHIFT 15u
#define VBS_DRV_RDNA2_RSRC2_LDS_SIZE_MASK  0x1FFu

/* PGM_RSRC3: SHARED_VGPR_CNT [3:0]. v1 programs 0 (no shared VGPRs). */
#define VBS_DRV_RDNA2_RSRC3_SHARED_VGPR_MASK 0xFu

/* COMPUTE_NUM_THREAD_X/Y/Z: FULL [15:0], PARTIAL [31:16] (digest
 * §4.3). Ordinary kernels set PARTIAL = 0 and FULL = thread count. */
#define VBS_DRV_RDNA2_NUM_THREAD_FULL_MASK 0xFFFFu

/* Shader code entry alignment: 256 bytes (digest §4.4,
 * [AMDGPUUsage] KERNEL_CODE_ENTRY_BYTE_OFFSET). */
#define VBS_DRV_RDNA2_CODE_ENTRY_ALIGN 256u

/* VGPRS field encoding (GFX10–12, digest §4.4 [AMDGPUUsage]):
 * wave64 → max(0, ceil(vgprs_used/4) − 1);
 * wave32 → max(0, ceil(vgprs_used/8) − 1). */
uint32_t vbs_drv_rdna2_vgprs_field(uint32_t vgprs_used, bool wave32);

/* Build a COMPUTE_PGM_RSRC1 value from the VGPRs used per thread.
 * SGPRS is left 0 (reserved on GFX10–12); all other fields are 0
 * (documented-safe defaults: priority 0, float mode 0, no PRIV,
 * no DX10 clamp, no IEEE mode, no BULKY, no WGP mode, no MEM_ORDERED,
 * no FWD_PROGRESS). */
uint32_t vbs_drv_rdna2_pgm_rsrc1(uint32_t vgprs_used, bool wave32);

/* Build a COMPUTE_PGM_RSRC2 value: user_sgpr_count SGPRs loaded
 * from COMPUTE_USER_DATA_0..15 (0..16), lds_bytes of LDS
 * (0..64 KiB), scratch_en enables the scratch (tmp-ring) path.
 * TGID/TG_SIZE system SGPRs are not enabled in v1 (the
 * driver-controlled SGPR layout of digest §6.3 does not use them),
 * and no exceptions are enabled. Returns 0 and sets *ok = false
 * when an argument is out of range. */
uint32_t vbs_drv_rdna2_pgm_rsrc2(uint32_t user_sgpr_count,
                               uint32_t lds_bytes, bool scratch_en,
                               bool *ok);

/* ------------------------------------------------------------------ */
/* PM4 packet encoder — digest §2, [SI-PG] + [NVD]                    */
/* ------------------------------------------------------------------ */

/* Type-3 packet opcodes a compute driver needs (digest §2.2 [NVD]). */
#define VBS_DRV_RDNA2_PM4_NOP                0x10u
#define VBS_DRV_RDNA2_PM4_SET_BASE           0x11u
#define VBS_DRV_RDNA2_PM4_CLEAR_STATE        0x12u
#define VBS_DRV_RDNA2_PM4_DISPATCH_DIRECT    0x15u
#define VBS_DRV_RDNA2_PM4_DISPATCH_INDIRECT  0x16u
#define VBS_DRV_RDNA2_PM4_CONTEXT_CONTROL    0x28u
#define VBS_DRV_RDNA2_PM4_PREABLE_CNTL       0x4Au
#define VBS_DRV_RDNA2_PM4_WRITE_DATA         0x37u
#define VBS_DRV_RDNA2_PM4_WAIT_REG_MEM       0x3Cu
/* INDIRECT_BUFFER (0x3F) is KERNEL-ONLY: the kernel wraps the IB
 * in ring packets with the VMID itself (digest §2.6 [SI-PG
 * §2.5.12]); userspace never emits it. No constant is defined for
 * it, so no encoder can. */
#define VBS_DRV_RDNA2_PM4_COPY_DATA          0x40u
#define VBS_DRV_RDNA2_PM4_CP_DMA             0x41u
#define VBS_DRV_RDNA2_PM4_EVENT_WRITE        0x46u
#define VBS_DRV_RDNA2_PM4_EVENT_WRITE_EOP    0x47u
#define VBS_DRV_RDNA2_PM4_RELEASE_MEM        0x49u
#define VBS_DRV_RDNA2_PM4_DMA_DATA           0x50u
#define VBS_DRV_RDNA2_PM4_ACQUIRE_MEM        0x58u
#define VBS_DRV_RDNA2_PM4_LOAD_SH_REG        0x5Fu
#define VBS_DRV_RDNA2_PM4_LOAD_CONFIG_REG    0x60u
#define VBS_DRV_RDNA2_PM4_LOAD_CONTEXT_REG   0x61u
#define VBS_DRV_RDNA2_PM4_SET_CONFIG_REG     0x68u
#define VBS_DRV_RDNA2_PM4_SET_CONTEXT_REG    0x69u
#define VBS_DRV_RDNA2_PM4_SET_SH_REG         0x76u
#define VBS_DRV_RDNA2_PM4_SET_SH_REG_OFFSET  0x77u
#define VBS_DRV_RDNA2_PM4_SET_UCONFIG_REG    0x79u
#define VBS_DRV_RDNA2_PM4_SET_SH_REG_INDEX   0x9Bu

/* Type-2 filler: a single dword 0x80000000 (digest §2.1 [SI-PG
 * §2.1]). Used to pad an IB to the kernel's ib_size_alignment. */
#define VBS_DRV_RDNA2_PM4_TYPE2_FILLER 0x80000000u

/* Type-3 header word (digest §2.1 [SI-PG §2.1]):
 * bits [31:30] = 3 (type 3), bits [29:16] = COUNT (body dwords
 * minus 1), bits [15:8] = opcode, bit [1] = SHADER_TYPE
 * (1 = compute), bit [0] = PREDICATE (0 = unconditional).
 * body_dwords is the number of body dwords (1..16384); the COUNT
 * field holds body_dwords - 1. */
uint32_t vbs_drv_rdna2_pm4_hdr(uint32_t opcode, uint32_t body_dwords,
                             bool shader_type);

/* Emit a SET_SH_REG packet writing `count` consecutive SH
 * registers starting at `sh_offset` (a DWORD offset from the SH
 * window base 0x2C00 — see vbs_drv_rdna2_sh_reg_offset). The
 * header's SHADER_TYPE bit is 1: these are the compute shader's
 * SH registers (digest §2.3 [SI-PG §2.4.7]). Body layout:
 * REG_OFFSET, then one data dword per register (digest §2.3). */
vbs_drv_status_t vbs_drv_rdna2_pm4_set_sh_reg(vbs_drv_cmdbuf_t *cb,
                                            uint32_t sh_offset,
                                            const uint32_t *values,
                                            uint32_t count);

/* Emit a DISPATCH_DIRECT packet (digest §2.4 [SI-PG §2.6.1]):
 * DIM_X/Y/Z are THREAD-GROUP counts (threads per group come from
 * COMPUTE_NUM_THREAD_X/Y/Z), and the initiator is a
 * COMPUTE_DISPATCH_INITIATOR value (digest §4.2; 0x1 minimal).
 * SHADER_TYPE = 1 ("Dispatches are only used for Compute
 * Shaders"). */
vbs_drv_status_t vbs_drv_rdna2_pm4_dispatch_direct(vbs_drv_cmdbuf_t *cb,
                                                 uint32_t dim_x,
                                                 uint32_t dim_y,
                                                 uint32_t dim_z,
                                                 uint32_t initiator);

/* Emit one type-2 filler dword (padding / IB size alignment). */
vbs_drv_status_t vbs_drv_rdna2_pm4_nop(vbs_drv_cmdbuf_t *cb);

/* ------------------------------------------------------------------ */
/* Launch-IB builder — digest Appendix A.3                            */
/*                                                                     */
/* Builds the minimal IB for one compute dispatch, in the digest's    */
/* order: (1) SET_SH_REG for PGM_LO/HI, PGM_RSRC1/2/3,                */
/* NUM_THREAD_X/Y/Z; (2) SET_SH_REG for COMPUTE_USER_DATA_0..N        */
/* (each const buffer's 64-bit GPU VA in two consecutive USER_DATA    */
/* registers, low dword then high — digest §7.3); (3)                 */
/* DISPATCH_DIRECT. The kernel wraps this IB in ring packets with     */
/* the VMID (digest §2.6); no INDIRECT_BUFFER packet is emitted.      */
/*                                                                     */
/* Mapping from the framework launch description:                     */
/*   code VA + code_offset → PGM_LO/HI (must be 256-byte aligned);    */
/*   register_count → VGPRs per thread (wave64 encoding);             */
/*   shared_mem_bytes → LDS_SIZE; grid_* → DISPATCH_DIRECT dims;      */
/*   block_* → NUM_THREAD_X/Y/Z; const_bufs[i] VA → USER_DATA         */
/*   pair i (up to 8 buffers = all 16 USER_DATA registers).           */
/*   barrier_count is ignored: barriers are an instruction            */
/*   (S_BARRIER), not launch state, on this hardware.                 */
/*   use_semaphore is NOT supported in v1: the completion signal      */
/*   would need a RELEASE_MEM packet, whose body layout the digest    */
/*   does not document — completion is via WAIT_CS on the CS          */
/*   sequence number instead. Returns NOT_SUPPORTED when set.         */
/*   Scratch (TMPRING_SIZE / SCRATCH_BASE) is not programmed: the     */
/*   framework launch description carries no scratch buffer, so v1    */
/*   launches run with SCRATCH_EN = 0 (see the driver report).        */
/* ------------------------------------------------------------------ */

vbs_drv_status_t vbs_drv_rdna2_build_launch_ib(
    const vbs_drv_launch_desc_t *desc, vbs_drv_cmdbuf_t *cb);

#ifdef __cplusplus
}
#endif

#endif /* VBS_DRV_RDNA2_H */
