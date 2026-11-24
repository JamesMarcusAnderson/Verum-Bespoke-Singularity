/* vbs_drv_rdna2_enc.c — RDNA 2 PM4 encoder (pure C11, no ioctls).
 *
 * Every encoding here is built on the AMD spec digest
 * (~/workspace/verum-driver-2026-11-17/specs/amd/SPEC-DIGEST.md),
 * cited as digest section + source tag. No Mesa, ROCm, or other
 * implementation was used or transcribed.
 *
 * Addressing-model note (digest §3 vs [SI-PG §2.4.7]): the digest's
 * prose formula for the SET_SH_REG offset, "(CP − 0x2C00)/4", is
 * wrong — the SI guide says the REG_OFFSET field is a DWORD offset
 * from a starting address whose base 0x2C00 is itself a DWORD
 * offset ("Byte Offset 0xB000; DWord Offset 0x2C00"), and the mm
 * values and segment bases are DWORD offsets too. So the offset is
 * the plain difference CP − 0x2C00, which is also what the digest's
 * own §4.1 table and Appendix A.1 worked packet use (COMPUTE_PGM_LO
 * → 0x0000020C). See the driver report.
 */

#include "vbs_drv_rdna2.h"

/* ------------------------------------------------------------------ */
/* Register addressing — digest §3                                     */
/* ------------------------------------------------------------------ */

uint32_t vbs_drv_rdna2_cp_addr(uint32_t mm) {
    /* All COMPUTE_* registers are BASE_IDX 0 (seg0, base 0x1260)
     * [UMR-SC]. */
    return mm + VBS_DRV_RDNA2_SEG0_BASE;
}

uint32_t vbs_drv_rdna2_sh_reg_offset(uint32_t cp_addr) {
    /* Both operands are DWORD offsets; no division (see the file
     * header note). Callers must pass a CP address inside the SH
     * window [0x2C00, 0x3000); the encoder does not check this —
     * the constants in vbs_drv_rdna2.h are all in-window. */
    return cp_addr - VBS_DRV_RDNA2_SH_REG_BASE;
}

/* ------------------------------------------------------------------ */
/* PGM_RSRC builders — digest §4.4                                     */
/* ------------------------------------------------------------------ */

uint32_t vbs_drv_rdna2_vgprs_field(uint32_t vgprs_used, bool wave32) {
    /* [AMDGPUUsage]: wave64 → max(0, ceil(vgprs/4) − 1); wave32 →
     * max(0, ceil(vgprs/8) − 1). vgprs_used == 0 yields 0 either
     * way (ceil(0) = 0, minus 1 clamps to 0). */
    uint32_t group = wave32 ? 8u : 4u;
    uint32_t encoded = (vgprs_used + group - 1u) / group;
    if (encoded == 0) return 0;
    return encoded - 1u;
}

uint32_t vbs_drv_rdna2_pgm_rsrc1(uint32_t vgprs_used, bool wave32) {
    uint32_t v = vbs_drv_rdna2_vgprs_field(vgprs_used, wave32);
    return (v & VBS_DRV_RDNA2_RSRC1_VGPRS_MASK)
           << VBS_DRV_RDNA2_RSRC1_VGPRS_SHIFT;
    /* SGPRS [9:6] stays 0: reserved on GFX10–12 (digest §4.4).
     * All other fields stay at their documented-safe default 0. */
}

uint32_t vbs_drv_rdna2_pgm_rsrc2(uint32_t user_sgpr_count,
                               uint32_t lds_bytes, bool scratch_en,
                               bool *ok) {
    if (ok) *ok = false;
    /* USER_SGPR is a 5-bit field counting SGPRs 0..15 loaded from
     * COMPUTE_USER_DATA_0–15 (digest §4.4, §7.3): 0..16. */
    if (user_sgpr_count > VBS_DRV_RDNA2_USER_DATA_COUNT) return 0;
    /* LDS limit per workgroup is 64 KiB (digest §5.1). */
    if (lds_bytes > VBS_DRV_RDNA2_MAX_LDS_BYTES) return 0;
    /* LDS_SIZE = roundup(lds_bytes / 512), 9-bit field [23:15]
     * (digest §4.4 [AMDGPUUsage]). 64 KiB → 128, fits. */
    uint32_t lds_size = (lds_bytes + 511u) / 512u;
    if (lds_size > VBS_DRV_RDNA2_RSRC2_LDS_SIZE_MASK) return 0;
    uint32_t v = 0;
    if (scratch_en) v |= VBS_DRV_RDNA2_RSRC2_SCRATCH_EN;
    v |= (user_sgpr_count & VBS_DRV_RDNA2_RSRC2_USER_SGPR_MASK)
         << VBS_DRV_RDNA2_RSRC2_USER_SGPR_SHIFT;
    v |= lds_size << VBS_DRV_RDNA2_RSRC2_LDS_SIZE_SHIFT;
    if (ok) *ok = true;
    return v;
}

/* ------------------------------------------------------------------ */
/* PM4 packets — digest §2                                             */
/* ------------------------------------------------------------------ */

uint32_t vbs_drv_rdna2_pm4_hdr(uint32_t opcode, uint32_t body_dwords,
                             bool shader_type) {
    /* [SI-PG §2.1]: bits [31:30] = 3 (type 3); bits [29:16] =
     * COUNT = body dwords minus 1; bits [15:8] = opcode; bit [1] =
     * SHADER_TYPE; bit [0] = PREDICATE = 0 (unconditional). */
    uint32_t count = body_dwords == 0 ? 0 : body_dwords - 1u;
    return (3u << 30) | ((count & 0x3FFFu) << 16) |
           ((opcode & 0xFFu) << 8) | (shader_type ? (1u << 1) : 0u);
}

vbs_drv_status_t vbs_drv_rdna2_pm4_set_sh_reg(vbs_drv_cmdbuf_t *cb,
                                            uint32_t sh_offset,
                                            const uint32_t *values,
                                            uint32_t count) {
    if (!cb || (count > 0 && !values) || count == 0 ||
        count > 0x3FFEu)
        return VBS_DRV_ERR_INVALID_ARG;
    /* REG_OFFSET is a 16-bit field (digest §2.3 [SI-PG §2.4.7]
     * packet description: REG_OFFSET [15:0]). */
    if (sh_offset > 0xFFFFu) return VBS_DRV_ERR_INVALID_ARG;
    /* Body: REG_OFFSET + one data dword per register → COUNT =
     * count (body dwords = count + 1, minus 1). */
    vbs_drv_status_t st = vbs_drv_cmdbuf_push(
        cb, vbs_drv_rdna2_pm4_hdr(VBS_DRV_RDNA2_PM4_SET_SH_REG, count + 1u,
                                true));
    if (st != VBS_DRV_OK) return st;
    st = vbs_drv_cmdbuf_push(cb, sh_offset);
    if (st != VBS_DRV_OK) return st;
    return vbs_drv_cmdbuf_push_many(cb, values, count);
}

vbs_drv_status_t vbs_drv_rdna2_pm4_dispatch_direct(vbs_drv_cmdbuf_t *cb,
                                                 uint32_t dim_x,
                                                 uint32_t dim_y,
                                                 uint32_t dim_z,
                                                 uint32_t initiator) {
    if (!cb) return VBS_DRV_ERR_INVALID_ARG;
    /* [SI-PG §2.6.1]: header (COUNT=3, SHADER_TYPE=1), DIM_X,
     * DIM_Y, DIM_Z (thread-group counts), DISPATCH_INITIATOR. */
    vbs_drv_status_t st = vbs_drv_cmdbuf_push(
        cb, vbs_drv_rdna2_pm4_hdr(VBS_DRV_RDNA2_PM4_DISPATCH_DIRECT, 4u,
                                true));
    if (st != VBS_DRV_OK) return st;
    uint32_t body[4] = { dim_x, dim_y, dim_z, initiator };
    return vbs_drv_cmdbuf_push_many(cb, body, 4);
}

vbs_drv_status_t vbs_drv_rdna2_pm4_nop(vbs_drv_cmdbuf_t *cb) {
    if (!cb) return VBS_DRV_ERR_INVALID_ARG;
    /* Type-2 filler: a single dword 0x80000000 (digest §2.1
     * [SI-PG §2.1]). */
    return vbs_drv_cmdbuf_push(cb, VBS_DRV_RDNA2_PM4_TYPE2_FILLER);
}

/* ------------------------------------------------------------------ */
/* Launch-IB builder — digest Appendix A.3                             */
/* ------------------------------------------------------------------ */

static uint32_t sh_off(uint32_t mm) {
    return vbs_drv_rdna2_sh_reg_offset(vbs_drv_rdna2_cp_addr(mm));
}

vbs_drv_status_t vbs_drv_rdna2_build_launch_ib(
    const vbs_drv_launch_desc_t *desc, vbs_drv_cmdbuf_t *cb) {
    if (!cb) return VBS_DRV_ERR_INVALID_ARG;
    vbs_drv_status_t vst = vbs_drv_validate_launch_desc(desc);
    if (vst != VBS_DRV_OK) return vst;
    if (desc->use_semaphore)
        return VBS_DRV_ERR_NOT_SUPPORTED; /* no RELEASE_MEM in v1 */
    /* Block dimensions live in the 16-bit FULL field of
     * COMPUTE_NUM_THREAD_X/Y/Z (digest §4.3); PARTIAL stays 0. */
    if (desc->block_x > VBS_DRV_RDNA2_NUM_THREAD_FULL_MASK ||
        desc->block_y > VBS_DRV_RDNA2_NUM_THREAD_FULL_MASK ||
        desc->block_z > VBS_DRV_RDNA2_NUM_THREAD_FULL_MASK)
        return VBS_DRV_ERR_INVALID_ARG;
    /* The shader code entry must be 256-byte aligned (digest §4.4
     * [AMDGPUUsage]). */
    uint64_t code_va = desc->code->gpu_va + desc->code_offset;
    if ((code_va & (VBS_DRV_RDNA2_CODE_ENTRY_ALIGN - 1u)) != 0)
        return VBS_DRV_ERR_INVALID_ARG;
    /* Const buffers: at most 8 (16 USER_DATA registers, two per
     * 64-bit VA — digest §7.3). The framework already caps at 8. */
    if (desc->const_buf_count * 2u > VBS_DRV_RDNA2_USER_DATA_COUNT)
        return VBS_DRV_ERR_INVALID_ARG;

    /* PGM_RSRC1: VGPRs per thread, wave64 (digest §5.1: compile
     * for wave64 first). */
    uint32_t rsrc1 =
        vbs_drv_rdna2_pgm_rsrc1(desc->register_count, false);
    /* PGM_RSRC2: USER_SGPR = 2 per const buffer, LDS from
     * shared_mem_bytes, no scratch (see the header comment). */
    bool rsrc2_ok = false;
    uint32_t rsrc2 = vbs_drv_rdna2_pgm_rsrc2(
        desc->const_buf_count * 2u, desc->shared_mem_bytes, false,
        &rsrc2_ok);
    if (!rsrc2_ok) return VBS_DRV_ERR_INVALID_ARG;
    /* PGM_RSRC3: SHARED_VGPR_CNT = 0 (no shared VGPRs in v1). */
    uint32_t rsrc3 = 0;

    vbs_drv_status_t st;
    /* Step 1 (digest A.3): program registers. Consecutive
     * registers go in one packet (digest §2.3). */
    uint32_t pgm[2] = { (uint32_t)(code_va & 0xFFFFFFFFu),
                        (uint32_t)(code_va >> 32) };
    st = vbs_drv_rdna2_pm4_set_sh_reg(
        cb, sh_off(VBS_DRV_RDNA2_MM_PGM_LO), pgm, 2);
    if (st != VBS_DRV_OK) return st;
    uint32_t rsrc12[2] = { rsrc1, rsrc2 };
    st = vbs_drv_rdna2_pm4_set_sh_reg(
        cb, sh_off(VBS_DRV_RDNA2_MM_PGM_RSRC1), rsrc12, 2);
    if (st != VBS_DRV_OK) return st;
    st = vbs_drv_rdna2_pm4_set_sh_reg(
        cb, sh_off(VBS_DRV_RDNA2_MM_PGM_RSRC3), &rsrc3, 1);
    if (st != VBS_DRV_OK) return st;
    uint32_t threads[3] = { desc->block_x, desc->block_y,
                            desc->block_z };
    st = vbs_drv_rdna2_pm4_set_sh_reg(
        cb, sh_off(VBS_DRV_RDNA2_MM_NUM_THREAD_X), threads, 3);
    if (st != VBS_DRV_OK) return st;
    /* Step 2: USER_DATA — each const buffer's VA as a low/high
     * pair landing in an even-aligned SGPR pair (digest §7.3
     * [SI-PG §1.3.2.1]). */
    if (desc->const_buf_count > 0) {
        uint32_t ud[VBS_DRV_RDNA2_USER_DATA_COUNT];
        for (uint32_t i = 0; i < desc->const_buf_count; i++) {
            uint64_t va = desc->const_bufs[i]->gpu_va;
            ud[2 * i] = (uint32_t)(va & 0xFFFFFFFFu);
            ud[2 * i + 1] = (uint32_t)(va >> 32);
        }
        st = vbs_drv_rdna2_pm4_set_sh_reg(
            cb, sh_off(VBS_DRV_RDNA2_MM_USER_DATA_0), ud,
            desc->const_buf_count * 2u);
        if (st != VBS_DRV_OK) return st;
    }
    /* Step 3: dispatch. Minimal initiator: COMPUTE_SHADER_EN only
     * (digest §4.2). The USE_THREAD_DIMENSIONS and
     * FORCE_START_AT_000 bits are UNSOURCED and never set. */
    return vbs_drv_rdna2_pm4_dispatch_direct(
        cb, desc->grid_x, desc->grid_y, desc->grid_z,
        VBS_DRV_RDNA2_INITIATOR_COMPUTE_EN);
}
