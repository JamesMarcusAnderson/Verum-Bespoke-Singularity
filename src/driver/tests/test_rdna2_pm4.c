/* test_rdna2_pm4.c — RDNA 2 PM4 encoder + register-address tests.
 *
 * Pure-logic tests: every check runs on a machine with no GPU.
 * Expected values come from the AMD spec digest
 * (~/workspace/verum-driver-2026-11-17/specs/amd/SPEC-DIGEST.md):
 *
 *   - the worked packets of Appendix A.1 (SET_SH_REG) and A.2
 *     (DISPATCH_DIRECT), checked word for word;
 *   - the 29-row register table of §4.1 (mm, CP address,
 *     SET_SH_REG offset), checked for every row — USER_DATA_0..15
 *     is one table row covering 16 registers, all checked;
 *   - the PGM_RSRC formulas of §4.4 and the vector-add setup of
 *     §5.4 (3 VGPRs, USER_SGPR = 6, 64 threads/group, 16 groups).
 *
 * NOTE on the digest's SET_SH_REG offset formula: the digest's
 * prose says (CP − 0x2C00)/4, but its own §4.1 table and Appendix
 * A.1 use the plain difference (COMPUTE_PGM_LO → 0x20C), and the
 * primary source [SI-PG §2.4.7] says REG_OFFSET is a DWORD offset
 * from a DWORD base (0x2C00 = dword offset; byte 0xB000). The
 * encoder follows the table/appendix/primary source — no division
 * — and this test checks the table values, so a regression to the
 * /4 formula fails here. See the driver report.
 */

#include <stdint.h>
#include <string.h>

#include "test_util.h"
#include "vbs_drv.h"
#include "vbs_drv_rdna2.h"

/* ------------------------------------------------------------------ */
/* PM4 headers — digest §2.1 [SI-PG §2.1]                              */
/* ------------------------------------------------------------------ */

static void test_pm4_headers(void) {
    /* Appendix A.1: SET_SH_REG, COUNT=1, opcode 0x76, shader=1. */
    CHECK_EQ_U32(vbs_drv_rdna2_pm4_hdr(VBS_DRV_RDNA2_PM4_SET_SH_REG, 2,
                                     true),
                 0xC0017602u, "SET_SH_REG header (A.1)");
    /* Appendix A.2: DISPATCH_DIRECT, COUNT=3, opcode 0x15,
     * shader=1. */
    CHECK_EQ_U32(vbs_drv_rdna2_pm4_hdr(VBS_DRV_RDNA2_PM4_DISPATCH_DIRECT,
                                     4, true),
                 0xC0031502u, "DISPATCH_DIRECT header (A.2)");
    /* Shader-type bit clear for a graphics packet. */
    CHECK_EQ_U32(vbs_drv_rdna2_pm4_hdr(VBS_DRV_RDNA2_PM4_SET_SH_REG, 2,
                                     false),
                 0xC0017600u, "SET_SH_REG header, shader_type=0");
    /* COUNT field: body dwords minus 1. */
    CHECK_EQ_U32(vbs_drv_rdna2_pm4_hdr(0x10, 1, false), 0xC0001000u,
                 "NOP header, 1 body dword (COUNT=0)");
    CHECK_EQ_U32(vbs_drv_rdna2_pm4_hdr(0x10, 5, false), 0xC0041000u,
                 "NOP header, 5 body dwords (COUNT=4)");
    /* Opcode values (digest §2.2 [NVD]). */
    CHECK_EQ_U32(VBS_DRV_RDNA2_PM4_SET_SH_REG, 0x76u, "SET_SH_REG opcode");
    CHECK_EQ_U32(VBS_DRV_RDNA2_PM4_DISPATCH_DIRECT, 0x15u,
                 "DISPATCH_DIRECT opcode");
    CHECK_EQ_U32(VBS_DRV_RDNA2_PM4_SET_CONFIG_REG, 0x68u,
                 "SET_CONFIG_REG opcode");
    CHECK_EQ_U32(VBS_DRV_RDNA2_PM4_SET_CONTEXT_REG, 0x69u,
                 "SET_CONTEXT_REG opcode");
    CHECK_EQ_U32(VBS_DRV_RDNA2_PM4_SET_UCONFIG_REG, 0x79u,
                 "SET_UCONFIG_REG opcode");
    CHECK_EQ_U32(VBS_DRV_RDNA2_PM4_TYPE2_FILLER, 0x80000000u,
                 "type-2 filler dword");
}

/* ------------------------------------------------------------------ */
/* Worked packets — digest Appendix A.1 / A.2                          */
/* ------------------------------------------------------------------ */

static void test_worked_packets(void) {
    vbs_drv_cmdbuf_t cb;
    CHECK(vbs_drv_cmdbuf_init(&cb, 64) == VBS_DRV_OK,
          "cmdbuf init for worked packets");

    /* A.1: write one dword to COMPUTE_PGM_LO (offset 0x20C). */
    uint32_t pgm_lo = 0x12345678u;
    CHECK(vbs_drv_rdna2_pm4_set_sh_reg(&cb, 0x20C, &pgm_lo, 1) ==
              VBS_DRV_OK,
          "SET_SH_REG PGM_LO emits");
    CHECK(cb.count == 3, "A.1 packet is 3 dwords");
    CHECK_EQ_U32(cb.words[0], 0xC0017602u, "A.1 header");
    CHECK_EQ_U32(cb.words[1], 0x0000020Cu, "A.1 REG_OFFSET");
    CHECK_EQ_U32(cb.words[2], 0x12345678u, "A.1 PGM_LO value");

    /* A.1, consecutive run: PGM_LO + PGM_HI in one packet. */
    vbs_drv_cmdbuf_reset(&cb);
    uint32_t pgm[2] = { 0x11111111u, 0x22222222u };
    CHECK(vbs_drv_rdna2_pm4_set_sh_reg(&cb, 0x20C, pgm, 2) ==
              VBS_DRV_OK,
          "SET_SH_REG PGM_LO/HI run emits");
    CHECK(cb.count == 4, "PGM_LO/HI packet is 4 dwords");
    CHECK_EQ_U32(cb.words[0], 0xC0027602u,
                 "PGM_LO/HI header (COUNT=2)");
    CHECK_EQ_U32(cb.words[1], 0x0000020Cu, "PGM_LO/HI REG_OFFSET");
    CHECK_EQ_U32(cb.words[2], 0x11111111u, "PGM_LO value");
    CHECK_EQ_U32(cb.words[3], 0x22222222u, "PGM_HI value");

    /* A.2: dispatch a 1024-element vector-add (16 groups of 64). */
    vbs_drv_cmdbuf_reset(&cb);
    CHECK(vbs_drv_rdna2_pm4_dispatch_direct(&cb, 16, 1, 1, 0x1) ==
              VBS_DRV_OK,
          "DISPATCH_DIRECT emits");
    CHECK(cb.count == 5, "A.2 packet is 5 dwords");
    CHECK_EQ_U32(cb.words[0], 0xC0031502u, "A.2 header");
    CHECK_EQ_U32(cb.words[1], 16u, "A.2 DIM_X");
    CHECK_EQ_U32(cb.words[2], 1u, "A.2 DIM_Y");
    CHECK_EQ_U32(cb.words[3], 1u, "A.2 DIM_Z");
    CHECK_EQ_U32(cb.words[4], 0x00000001u, "A.2 DISPATCH_INITIATOR");

    /* Type-2 filler NOP. */
    vbs_drv_cmdbuf_reset(&cb);
    CHECK(vbs_drv_rdna2_pm4_nop(&cb) == VBS_DRV_OK, "NOP emits");
    CHECK_EQ_U32(cb.words[0], 0x80000000u, "NOP word");

    /* NULL / invalid arguments fail cleanly. */
    CHECK(vbs_drv_rdna2_pm4_set_sh_reg(NULL, 0x20C, pgm, 1) ==
              VBS_DRV_ERR_INVALID_ARG,
          "SET_SH_REG rejects NULL cmdbuf");
    CHECK(vbs_drv_rdna2_pm4_set_sh_reg(&cb, 0x20C, NULL, 1) ==
              VBS_DRV_ERR_INVALID_ARG,
          "SET_SH_REG rejects NULL values");
    CHECK(vbs_drv_rdna2_pm4_set_sh_reg(&cb, 0x20C, pgm, 0) ==
              VBS_DRV_ERR_INVALID_ARG,
          "SET_SH_REG rejects zero count");
    CHECK(vbs_drv_rdna2_pm4_set_sh_reg(&cb, 0x10000, pgm, 1) ==
              VBS_DRV_ERR_INVALID_ARG,
          "SET_SH_REG rejects a >16-bit offset");
    CHECK(vbs_drv_rdna2_pm4_dispatch_direct(NULL, 1, 1, 1, 1) ==
              VBS_DRV_ERR_INVALID_ARG,
          "DISPATCH_DIRECT rejects NULL cmdbuf");
    CHECK(vbs_drv_rdna2_pm4_nop(NULL) == VBS_DRV_ERR_INVALID_ARG,
          "NOP rejects NULL cmdbuf");

    vbs_drv_cmdbuf_free(&cb);
}

/* ------------------------------------------------------------------ */
/* Register table — digest §4.1 (29 rows)                              */
/*                                                                     */
/* mm and cp are the digest's table values, transcribed literally.     */
/* `hdr` is the matching constant from vbs_drv_rdna2.h, so a wrong       */
/* constant fails here too. USER_DATA_0..15 is one table row; the     */
/* loop below checks all 16 registers of that row.                     */
/* ------------------------------------------------------------------ */

typedef struct {
    uint32_t hdr;  /* constant from vbs_drv_rdna2.h */
    uint32_t mm;   /* digest §4.1 mm column */
    uint32_t cp;   /* digest §4.1 CP address column */
    uint32_t off;  /* digest §4.1 SET_SH_REG offset column */
    const char *name;
} reg_row_t;

static const reg_row_t g_reg_table[] = {
    { VBS_DRV_RDNA2_MM_DISPATCH_INITIATOR, 0x1BA0, 0x2E00, 0x200,
      "COMPUTE_DISPATCH_INITIATOR" },
    { VBS_DRV_RDNA2_MM_DIM_X, 0x1BA1, 0x2E01, 0x201, "COMPUTE_DIM_X" },
    { VBS_DRV_RDNA2_MM_DIM_Y, 0x1BA2, 0x2E02, 0x202, "COMPUTE_DIM_Y" },
    { VBS_DRV_RDNA2_MM_DIM_Z, 0x1BA3, 0x2E03, 0x203, "COMPUTE_DIM_Z" },
    { VBS_DRV_RDNA2_MM_START_X, 0x1BA4, 0x2E04, 0x204, "COMPUTE_START_X" },
    { VBS_DRV_RDNA2_MM_START_Y, 0x1BA5, 0x2E05, 0x205, "COMPUTE_START_Y" },
    { VBS_DRV_RDNA2_MM_START_Z, 0x1BA6, 0x2E06, 0x206, "COMPUTE_START_Z" },
    { VBS_DRV_RDNA2_MM_NUM_THREAD_X, 0x1BA7, 0x2E07, 0x207,
      "COMPUTE_NUM_THREAD_X" },
    { VBS_DRV_RDNA2_MM_NUM_THREAD_Y, 0x1BA8, 0x2E08, 0x208,
      "COMPUTE_NUM_THREAD_Y" },
    { VBS_DRV_RDNA2_MM_NUM_THREAD_Z, 0x1BA9, 0x2E09, 0x209,
      "COMPUTE_NUM_THREAD_Z" },
    { VBS_DRV_RDNA2_MM_PGM_LO, 0x1BAC, 0x2E0C, 0x20C, "COMPUTE_PGM_LO" },
    { VBS_DRV_RDNA2_MM_PGM_HI, 0x1BAD, 0x2E0D, 0x20D, "COMPUTE_PGM_HI" },
    { VBS_DRV_RDNA2_MM_SCRATCH_BASE_LO, 0x1BB0, 0x2E10, 0x210,
      "COMPUTE_DISPATCH_SCRATCH_BASE_LO" },
    { VBS_DRV_RDNA2_MM_SCRATCH_BASE_HI, 0x1BB1, 0x2E11, 0x211,
      "COMPUTE_DISPATCH_SCRATCH_BASE_HI" },
    { VBS_DRV_RDNA2_MM_PGM_RSRC1, 0x1BB2, 0x2E12, 0x212,
      "COMPUTE_PGM_RSRC1" },
    { VBS_DRV_RDNA2_MM_PGM_RSRC2, 0x1BB3, 0x2E13, 0x213,
      "COMPUTE_PGM_RSRC2" },
    { VBS_DRV_RDNA2_MM_RESOURCE_LIMITS, 0x1BB5, 0x2E15, 0x215,
      "COMPUTE_RESOURCE_LIMITS" },
    { VBS_DRV_RDNA2_MM_STATIC_TM_SE0, 0x1BB6, 0x2E16, 0x216,
      "COMPUTE_STATIC_THREAD_MGMT_SE0" },
    { VBS_DRV_RDNA2_MM_STATIC_TM_SE1, 0x1BB7, 0x2E17, 0x217,
      "COMPUTE_STATIC_THREAD_MGMT_SE1" },
    { VBS_DRV_RDNA2_MM_TMPRING_SIZE, 0x1BB8, 0x2E18, 0x218,
      "COMPUTE_TMPRING_SIZE" },
    { VBS_DRV_RDNA2_MM_RESTART_X, 0x1BBB, 0x2E1B, 0x21B,
      "COMPUTE_RESTART_X" },
    { VBS_DRV_RDNA2_MM_RESTART_Y, 0x1BBC, 0x2E1C, 0x21C,
      "COMPUTE_RESTART_Y" },
    { VBS_DRV_RDNA2_MM_RESTART_Z, 0x1BBD, 0x2E1D, 0x21D,
      "COMPUTE_RESTART_Z" },
    { VBS_DRV_RDNA2_MM_THREAD_TRACE_EN, 0x1BBE, 0x2E1E, 0x21E,
      "COMPUTE_THREAD_TRACE_ENABLE" },
    { VBS_DRV_RDNA2_MM_DISPATCH_ID, 0x1BC0, 0x2E20, 0x220,
      "COMPUTE_DISPATCH_ID" },
    { VBS_DRV_RDNA2_MM_PGM_RSRC3, 0x1BC8, 0x2E28, 0x228,
      "COMPUTE_PGM_RSRC3" },
    { VBS_DRV_RDNA2_MM_DISPATCH_TUNNEL, 0x1C1D, 0x2E7D, 0x27D,
      "COMPUTE_DISPATCH_TUNNEL" },
    { VBS_DRV_RDNA2_MM_DISPATCH_END, 0x1C1E, 0x2E7E, 0x27E,
      "COMPUTE_DISPATCH_END" },
};

static void test_register_table(void) {
    /* Segment bases (digest §3 [UMR-SC]). */
    CHECK_EQ_U32(VBS_DRV_RDNA2_SEG0_BASE, 0x1260u, "seg0 base");
    CHECK_EQ_U32(VBS_DRV_RDNA2_SEG1_BASE, 0xA000u, "seg1 base");
    CHECK_EQ_U32(VBS_DRV_RDNA2_SEG2_BASE, 0x1C000u, "seg2 base");
    CHECK_EQ_U32(VBS_DRV_RDNA2_SH_REG_BASE, 0x2C00u, "SH window base");

    /* The table above has 28 rows; the 29th row is USER_DATA_0..15
     * (checked in the loop below), so the digest's 29 rows are all
     * covered. */
    CHECK(sizeof(g_reg_table) / sizeof(g_reg_table[0]) == 28,
          "register table carries 28 named rows (+ USER_DATA row)");
    for (size_t i = 0; i < sizeof(g_reg_table) / sizeof(g_reg_table[0]);
         i++) {
        const reg_row_t *r = &g_reg_table[i];
        CHECK_EQ_U32(r->hdr, r->mm, r->name);
        CHECK_EQ_U32(vbs_drv_rdna2_cp_addr(r->mm), r->cp, r->name);
        CHECK_EQ_U32(vbs_drv_rdna2_sh_reg_offset(r->cp), r->off,
                     r->name);
        /* The offset is also derivable straight from mm. */
        CHECK_EQ_U32(vbs_drv_rdna2_sh_reg_offset(
                         vbs_drv_rdna2_cp_addr(r->mm)),
                     r->off, r->name);
    }
    /* USER_DATA_0..15: mm 0x1BE0–0x1BEF, CP 0x2E40–0x2E4F,
     * offsets 0x240–0x24F. */
    CHECK_EQ_U32(VBS_DRV_RDNA2_MM_USER_DATA_0, 0x1BE0u,
                 "USER_DATA_0 mm");
    CHECK_EQ_U32(VBS_DRV_RDNA2_USER_DATA_COUNT, 16u,
                 "USER_DATA register count");
    for (uint32_t i = 0; i < 16; i++) {
        uint32_t mm = 0x1BE0u + i;
        CHECK_EQ_U32(vbs_drv_rdna2_cp_addr(mm), 0x2E40u + i,
                     "USER_DATA CP address");
        CHECK_EQ_U32(vbs_drv_rdna2_sh_reg_offset(0x2E40u + i),
                     0x240u + i, "USER_DATA SH offset");
    }
    /* Digest §3 cross-validation: GFX9 (0x0E0C + 0x2000) and
     * GFX10.3 (0x1BAC + 0x1260) agree on COMPUTE_PGM_LO = 0x2E0C. */
    CHECK_EQ_U32(0x0E0Cu + 0x2000u, 0x2E0Cu, "GFX9 PGM_LO CP address");
    CHECK_EQ_U32(vbs_drv_rdna2_cp_addr(VBS_DRV_RDNA2_MM_PGM_LO), 0x2E0Cu,
                 "GFX10.3 PGM_LO CP address");
    /* Digest §4.1 non-SH references, corrected per the primary
     * source (see the file header): GRBM_STATUS mm 0x0DA4 → CP
     * 0x2004 → SET_CONFIG_REG offset (0x2004 − 0x2000) = 0x4, not
     * the digest's 0x001 (which divided by 4); SH_MEM_BASES mm
     * 0x10AA → CP 0x230A → offset 0x30A (the digest's own value
     * here, un-divided). */
    CHECK_EQ_U32(vbs_drv_rdna2_cp_addr(0x0DA4u), 0x2004u,
                 "GRBM_STATUS CP address");
    CHECK_EQ_U32(0x2004u - VBS_DRV_RDNA2_CONFIG_REG_BASE, 0x4u,
                 "GRBM_STATUS SET_CONFIG_REG offset");
    CHECK_EQ_U32(vbs_drv_rdna2_cp_addr(0x10AAu), 0x230Au,
                 "SH_MEM_BASES CP address");
    CHECK_EQ_U32(0x230Au - VBS_DRV_RDNA2_CONFIG_REG_BASE, 0x30Au,
                 "SH_MEM_BASES SET_CONFIG_REG offset");
}

/* ------------------------------------------------------------------ */
/* PGM_RSRC builders — digest §4.4                                     */
/* ------------------------------------------------------------------ */

static void test_pgm_rsrc(void) {
    /* VGPRS field, wave64: max(0, ceil(n/4) − 1). */
    CHECK_EQ_U32(vbs_drv_rdna2_vgprs_field(0, false), 0u, "VGPRS 0/64");
    CHECK_EQ_U32(vbs_drv_rdna2_vgprs_field(1, false), 0u, "VGPRS 1/64");
    CHECK_EQ_U32(vbs_drv_rdna2_vgprs_field(3, false), 0u,
                 "VGPRS 3/64 (vec-add, digest §5.4)");
    CHECK_EQ_U32(vbs_drv_rdna2_vgprs_field(4, false), 0u, "VGPRS 4/64");
    CHECK_EQ_U32(vbs_drv_rdna2_vgprs_field(5, false), 1u, "VGPRS 5/64");
    CHECK_EQ_U32(vbs_drv_rdna2_vgprs_field(8, false), 1u, "VGPRS 8/64");
    CHECK_EQ_U32(vbs_drv_rdna2_vgprs_field(9, false), 2u, "VGPRS 9/64");
    CHECK_EQ_U32(vbs_drv_rdna2_vgprs_field(256, false), 63u,
                 "VGPRS 256/64 (max VGPRs)");
    /* VGPRS field, wave32: max(0, ceil(n/8) − 1). */
    CHECK_EQ_U32(vbs_drv_rdna2_vgprs_field(8, true), 0u, "VGPRS 8/32");
    CHECK_EQ_U32(vbs_drv_rdna2_vgprs_field(9, true), 1u, "VGPRS 9/32");
    CHECK_EQ_U32(vbs_drv_rdna2_vgprs_field(256, true), 31u,
                 "VGPRS 256/32");
    /* PGM_RSRC1: VGPRS in bits [5:0], SGPRS [9:6] stays 0
     * (reserved on GFX10–12). */
    CHECK_EQ_U32(vbs_drv_rdna2_pgm_rsrc1(3, false), 0u,
                 "RSRC1 for the vec-add kernel");
    CHECK_EQ_U32(vbs_drv_rdna2_pgm_rsrc1(9, false), 2u, "RSRC1, 9 VGPRs");
    CHECK_EQ_U32(vbs_drv_rdna2_pgm_rsrc1(256, false) & 0x3C0u, 0u,
                 "RSRC1 SGPRS field is 0");
    /* PGM_RSRC2: USER_SGPR [5:1], LDS_SIZE [23:15] =
     * roundup(bytes/512), SCRATCH_EN bit 0. */
    bool ok = false;
    CHECK_EQ_U32(vbs_drv_rdna2_pgm_rsrc2(6, 0, false, &ok), 0xCu,
                 "RSRC2, USER_SGPR=6 (vec-add, digest §5.4)");
    CHECK(ok, "RSRC2 vec-add args are valid");
    CHECK_EQ_U32(vbs_drv_rdna2_pgm_rsrc2(0, 512, false, &ok), 0x8000u,
                 "RSRC2, LDS 512 B");
    CHECK(ok, "RSRC2 LDS 512 B is valid");
    CHECK_EQ_U32(vbs_drv_rdna2_pgm_rsrc2(0, 1024, false, &ok), 0x10000u,
                 "RSRC2, LDS 1024 B");
    CHECK_EQ_U32(vbs_drv_rdna2_pgm_rsrc2(0, 513, false, &ok), 0x10000u,
                 "RSRC2, LDS 513 B rounds up");
    CHECK_EQ_U32(vbs_drv_rdna2_pgm_rsrc2(16, 65536, true, &ok),
                 0x400021u, "RSRC2, max USER_SGPR + max LDS + scratch");
    CHECK(ok, "RSRC2 max args are valid");
    vbs_drv_rdna2_pgm_rsrc2(17, 0, false, &ok);
    CHECK(!ok, "RSRC2 rejects USER_SGPR > 16");
    vbs_drv_rdna2_pgm_rsrc2(0, 65537, false, &ok);
    CHECK(!ok, "RSRC2 rejects LDS > 64 KiB");
    /* Initiator bits (digest §4.2). */
    CHECK_EQ_U32(VBS_DRV_RDNA2_INITIATOR_COMPUTE_EN, 0x1u,
                 "COMPUTE_SHADER_EN bit");
    CHECK_EQ_U32(VBS_DRV_RDNA2_INITIATOR_W32_EN, 0x8000u,
                 "CS_W32_EN bit");
    CHECK_EQ_U32(VBS_DRV_RDNA2_INITIATOR_COMPUTE_EN |
                     VBS_DRV_RDNA2_INITIATOR_W32_EN,
                 0x8001u, "wave32 initiator value (digest §4.2)");
}

/* ------------------------------------------------------------------ */
/* Launch-IB builder — digest Appendix A.3, §5.4                       */
/* ------------------------------------------------------------------ */

static void test_launch_ib(void) {
    /* Fake buffers: the encoder only reads gpu_va/size, so no
     * device is needed. */
    vbs_drv_buffer_t code, a, b, c;
    memset(&code, 0, sizeof(code));
    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    memset(&c, 0, sizeof(c));
    code.gpu_va = 0x1000000u; /* 256-byte aligned */
    code.size = 4096;
    a.gpu_va = 0x2000000u; a.size = 4096;
    b.gpu_va = 0x3000000u; b.size = 4096;
    c.gpu_va = 0x4000000u; c.size = 4096;

    vbs_drv_launch_desc_t desc;
    memset(&desc, 0, sizeof(desc));
    desc.code = &code;
    desc.code_offset = 0;
    desc.code_size = 64;
    desc.register_count = 3; /* vec-add uses 3 VGPRs (§5.4) */
    desc.shared_mem_bytes = 0;
    desc.grid_x = 16; desc.grid_y = 1; desc.grid_z = 1;
    desc.block_x = 64; desc.block_y = 1; desc.block_z = 1;
    desc.const_bufs[0] = &a;
    desc.const_bufs[1] = &b;
    desc.const_bufs[2] = &c;
    desc.const_buf_count = 3;

    vbs_drv_cmdbuf_t cb;
    CHECK(vbs_drv_cmdbuf_init(&cb, 64) == VBS_DRV_OK,
          "cmdbuf init for launch IB");
    CHECK(vbs_drv_rdna2_build_launch_ib(&desc, &cb) == VBS_DRV_OK,
          "launch IB builds for the vec-add setup");
    /* Expected words, packet by packet (digest A.3 order):
     * PGM_LO/HI, RSRC1/2, RSRC3, NUM_THREAD_X/Y/Z, USER_DATA_0..5,
     * DISPATCH_DIRECT. */
    static const uint32_t want[] = {
        0xC0027602u, 0x0000020Cu, 0x01000000u, 0x00000000u, /* PGM */
        0xC0027602u, 0x00000212u, 0x00000000u, 0x0000000Cu, /* RSRC1/2 */
        0xC0017602u, 0x00000228u, 0x00000000u,             /* RSRC3 */
        0xC0037602u, 0x00000207u, 64u, 1u, 1u,             /* NUM_THREAD */
        0xC0067602u, 0x00000240u, 0x02000000u, 0x00000000u, /* USER_DATA */
        0x03000000u, 0x00000000u, 0x04000000u, 0x00000000u,
        0xC0031502u, 16u, 1u, 1u, 0x00000001u,             /* DISPATCH */
    };
    CHECK(cb.count == sizeof(want) / sizeof(want[0]),
          "launch IB word count");
    size_t n = cb.count < sizeof(want) / sizeof(want[0])
        ? cb.count : sizeof(want) / sizeof(want[0]);
    for (size_t i = 0; i < n; i++) {
        char what[32];
        snprintf(what, sizeof(what), "launch IB word %zu", i);
        CHECK_EQ_U32(cb.words[i], want[i], what);
    }
    /* No INDIRECT_BUFFER packet anywhere in the IB: the kernel
     * emits those (digest §2.6). Every type-3 header in the IB
     * must be SET_SH_REG (0x76) or DISPATCH_DIRECT (0x15). */
    for (size_t i = 0; i < cb.count; i++) {
        uint32_t w = cb.words[i];
        if ((w >> 30) == 3) {
            uint32_t op = (w >> 8) & 0xFFu;
            CHECK(op == 0x76u || op == 0x15u,
                  "IB type-3 opcode is SET_SH_REG or DISPATCH_DIRECT");
        }
    }

    /* Zero const buffers: no USER_DATA packet, USER_SGPR = 0. */
    vbs_drv_cmdbuf_reset(&cb);
    desc.const_buf_count = 0;
    CHECK(vbs_drv_rdna2_build_launch_ib(&desc, &cb) == VBS_DRV_OK,
          "launch IB builds with no const buffers");
    CHECK(cb.count == 21, "no-const-buffer IB is 21 words");
    CHECK_EQ_U32(cb.words[7], 0x00000000u,
                 "RSRC2 is 0 with no const buffers");
    desc.const_buf_count = 3;

    /* Invalid descriptions fail cleanly. */
    vbs_drv_cmdbuf_reset(&cb);
    CHECK(vbs_drv_rdna2_build_launch_ib(NULL, &cb) ==
              VBS_DRV_ERR_INVALID_ARG,
          "launch IB rejects NULL desc");
    CHECK(vbs_drv_rdna2_build_launch_ib(&desc, NULL) ==
              VBS_DRV_ERR_INVALID_ARG,
          "launch IB rejects NULL cmdbuf");
    vbs_drv_launch_desc_t bad = desc;
    bad.code->gpu_va = 0x1000104u; /* not 256-byte aligned */
    CHECK(vbs_drv_rdna2_build_launch_ib(&bad, &cb) ==
              VBS_DRV_ERR_INVALID_ARG,
          "launch IB rejects an unaligned code VA");
    code.gpu_va = 0x1000000u;
    bad = desc;
    bad.block_x = 65536; /* overflows the 16-bit FULL field */
    CHECK(vbs_drv_rdna2_build_launch_ib(&bad, &cb) ==
              VBS_DRV_ERR_INVALID_ARG,
          "launch IB rejects block_x > 65535");
    bad = desc;
    bad.shared_mem_bytes = 65537; /* over the 64 KiB LDS limit */
    CHECK(vbs_drv_rdna2_build_launch_ib(&bad, &cb) ==
              VBS_DRV_ERR_INVALID_ARG,
          "launch IB rejects shared memory > 64 KiB");
    bad = desc;
    bad.code_size = 0;
    CHECK(vbs_drv_rdna2_build_launch_ib(&bad, &cb) ==
              VBS_DRV_ERR_INVALID_ARG,
          "launch IB rejects zero code size");
    bad = desc;
    bad.grid_x = 0;
    CHECK(vbs_drv_rdna2_build_launch_ib(&bad, &cb) ==
              VBS_DRV_ERR_INVALID_ARG,
          "launch IB rejects zero grid_x");
    bad = desc;
    bad.use_semaphore = true; /* RELEASE_MEM not in the digest */
    CHECK(vbs_drv_rdna2_build_launch_ib(&bad, &cb) ==
              VBS_DRV_ERR_NOT_SUPPORTED,
          "launch IB declines the semaphore (NOT_SUPPORTED)");
    bad = desc;
    bad.code_offset = 4096; /* offset + size past the buffer end */
    CHECK(vbs_drv_rdna2_build_launch_ib(&bad, &cb) ==
              VBS_DRV_ERR_INVALID_ARG,
          "launch IB rejects code past the buffer end");

    vbs_drv_cmdbuf_free(&cb);
}

int main(void) {
    test_pm4_headers();
    test_worked_packets();
    test_register_table();
    test_pgm_rsrc();
    test_launch_ib();
    TEST_SUMMARY("test_rdna2_pm4");
}
