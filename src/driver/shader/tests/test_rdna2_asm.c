/* test_rdna2_asm.c — RDNA 2 assembler encoding tests.
 *
 * Checks every emitter against the ISA docs: the digest §5.3
 * hand-verified encodings as oracles, plus per-field checks
 * (each emitted instruction's bit fields extracted and compared
 * to the format tables in [RDNA2-ISA] §13 via digest §5.2).
 * Also checks the opcode table itself (single source of opcode
 * numbers) and the emitters' invalid-input behavior.
 *
 * No GPU needed: pure encoding arithmetic.
 */

#include <stdint.h>
#include <string.h>

#include "test_util.h"
#include "vbs_rdna2_asm.h"

/* Field extractors for the format checks. */
#define SOPP_ENC(dw0) (((dw0) >> 23) & 0x1FFu)
#define SOPP_OP(dw0) (((dw0) >> 16) & 0x7Fu)
#define SOPP_SIMM(dw0) ((dw0) & 0xFFFFu)
#define SMEM_ENC(dw0) (((dw0) >> 26) & 0x3Fu)
#define SMEM_OP(dw0) (((dw0) >> 18) & 0xFFu)
#define SMEM_SDATA(dw0) (((dw0) >> 6) & 0x7Fu)
#define SMEM_SBASE(dw0) ((dw0) & 0x3Fu)
#define VOP1_ENC(dw0) (((dw0) >> 25) & 0x7Fu)
#define VOP1_OP(dw0) (((dw0) >> 9) & 0xFFu)
#define VOP1_VDST(dw0) (((dw0) >> 17) & 0xFFu)
#define VOP1_SRC0(dw0) ((dw0) & 0x1FFu)
#define VOP2_OP(dw0) (((dw0) >> 25) & 0x3Fu)
#define VOP2_VDST(dw0) (((dw0) >> 17) & 0xFFu)
#define VOP2_VSRC1(dw0) (((dw0) >> 9) & 0xFFu)
#define VOP2_SRC0(dw0) ((dw0) & 0x1FFu)
#define VOP3_ENC(dw0) (((dw0) >> 26) & 0x3Fu)
#define VOP3_OP(dw0) (((dw0) >> 16) & 0x3FFu)
#define VOP3_VDST(dw0) ((dw0) & 0xFFu)
#define VOP3_SDST(dw0) (((dw0) >> 8) & 0x7Fu)
#define VOP3_SRC0(dw1) ((dw1) & 0x1FFu)
#define VOP3_SRC1(dw1) (((dw1) >> 9) & 0x1FFu)
#define VOP3_SRC2(dw1) (((dw1) >> 18) & 0x1FFu)
#define FLAT_ENC(dw0) (((dw0) >> 26) & 0x3Fu)
#define FLAT_OP(dw0) (((dw0) >> 18) & 0x7Fu)
#define FLAT_SEG(dw0) (((dw0) >> 14) & 0x3u)
#define FLAT_OFF(dw0) ((dw0) & 0xFFFu)
#define FLAT_ADDR(dw1) ((dw1) & 0xFFu)
#define FLAT_DATA(dw1) (((dw1) >> 8) & 0xFFu)
#define FLAT_SADDR(dw1) (((dw1) >> 16) & 0x7Fu)
#define FLAT_VDST(dw1) (((dw1) >> 24) & 0xFFu)

static void test_opcode_table(void) {
    const vbs_rdna2_op_entry_t *e;
    size_t i;
    int ok = 0;

    /* The table has exactly one row per kind, in kind order. */
    CHECK(VBS_RDNA2_OP_TABLE_COUNT == (size_t)VBS_RDNA2_OP_COUNT,
          "table count %u == op count %u",
          (unsigned)VBS_RDNA2_OP_TABLE_COUNT,
          (unsigned)VBS_RDNA2_OP_COUNT);
    for (i = 0; i < VBS_RDNA2_OP_TABLE_COUNT; i++) {
        CHECK(VBS_RDNA2_OP_TABLE[i].kind == (vbs_rdna2_op_t)i,
              "table row %u in kind order", (unsigned)i);
        /* Every opcode in the table is sourced (no guesses). */
        CHECK(VBS_RDNA2_OP_TABLE[i].sourced != 0,
              "table row %s is sourced",
              VBS_RDNA2_OP_TABLE[i].name);
        CHECK(VBS_RDNA2_OP_TABLE[i].citation != NULL &&
              VBS_RDNA2_OP_TABLE[i].citation[0] != '\0',
              "table row %s carries a citation",
              VBS_RDNA2_OP_TABLE[i].name);
    }

    /* Spot-check opcodes against the digest / ISA-XML. */
    CHECK_EQ_U32(vbs_rdna2_opcode(VBS_RDNA2_OP_S_ENDPGM, &ok), 1,
                 "S_ENDPGM opcode");
    CHECK(ok, "S_ENDPGM lookup ok");
    CHECK_EQ_U32(vbs_rdna2_opcode(VBS_RDNA2_OP_S_WAITCNT, &ok), 12,
                 "S_WAITCNT opcode");
    CHECK_EQ_U32(vbs_rdna2_opcode(VBS_RDNA2_OP_S_LOAD_DWORDX2, &ok), 1,
                 "S_LOAD_DWORDX2 opcode");
    CHECK_EQ_U32(vbs_rdna2_opcode(VBS_RDNA2_OP_V_MOV_B32, &ok), 1,
                 "V_MOV_B32 opcode");
    CHECK_EQ_U32(vbs_rdna2_opcode(VBS_RDNA2_OP_V_ADD_NC_U32, &ok), 37,
                 "V_ADD_NC_U32 opcode");
    CHECK_EQ_U32(vbs_rdna2_opcode(VBS_RDNA2_OP_V_ADD_CO_U32, &ok), 783,
                 "V_ADD_CO_U32 opcode");
    CHECK_EQ_U32(vbs_rdna2_opcode(
            VBS_RDNA2_OP_GLOBAL_LOAD_DWORD_ADDTID, &ok), 22,
                 "GLOBAL_LOAD_DWORD_ADDTID opcode");
    CHECK_EQ_U32(vbs_rdna2_opcode(
            VBS_RDNA2_OP_GLOBAL_STORE_DWORD_ADDTID, &ok), 23,
                 "GLOBAL_STORE_DWORD_ADDTID opcode");
    /* VOP3 re-encode rule: VOP2 op + 256 (ISA-XML). */
    CHECK_EQ_U32(vbs_rdna2_opcode(VBS_RDNA2_OP_V_ADD_NC_U32_VOP3,
                                  &ok), 293, "V_ADD_NC_U32 VOP3 opcode");

    /* Out-of-range kinds fail cleanly. */
    e = vbs_rdna2_op_lookup((vbs_rdna2_op_t)VBS_RDNA2_OP_COUNT);
    CHECK(e == NULL, "lookup past the end returns NULL");
    e = vbs_rdna2_op_lookup((vbs_rdna2_op_t)-1);
    CHECK(e == NULL, "lookup of -1 returns NULL");
    CHECK_EQ_U32(vbs_rdna2_opcode((vbs_rdna2_op_t)9999, &ok), 0,
                 "bad kind opcode 0");
    CHECK(!ok, "bad kind sets ok = 0");
}

/* The 7 hand-verified encodings from digest §5.3, as oracles. */
static void test_digest_oracles(void) {
    vbs_rdna2_inst_t inst;

    inst = vbs_rdna2_emit_s_endpgm();
    CHECK(!inst.is64, "s_endpgm is 32-bit");
    CHECK_EQ_U32(inst.dw0, 0xBF810000u, "s_endpgm oracle");

    inst = vbs_rdna2_emit_s_waitcnt(0, 0, 0);
    CHECK(!inst.is64, "s_waitcnt is 32-bit");
    CHECK_EQ_U32(inst.dw0, 0xBF8C0000u, "s_waitcnt 0 oracle");

    inst = vbs_rdna2_emit_v_mov_b32(0, VBS_RDNA2_SRC_SGPR(0));
    CHECK(!inst.is64, "v_mov_b32 is 32-bit");
    CHECK_EQ_U32(inst.dw0, 0x7E000200u, "v_mov_b32 v0, s0 oracle");

    inst = vbs_rdna2_emit_v_add_nc_u32(2, VBS_RDNA2_SRC_VGPR(0), 1);
    CHECK(!inst.is64, "v_add_nc_u32 is 32-bit");
    CHECK_EQ_U32(inst.dw0, 0x4A040300u, "v_add_nc_u32 v2, v0, v1 oracle");

    inst = vbs_rdna2_emit_s_load_dwordx2(4, 1, 0);
    CHECK(inst.is64, "s_load_dwordx2 is 64-bit");
    CHECK_EQ_U32(inst.dw0, 0xF4040101u, "s_load_dwordx2 DW0 oracle");
    CHECK_EQ_U32(inst.dw1, 0xFA000000u, "s_load_dwordx2 DW1 oracle");

    inst = vbs_rdna2_emit_global_load_dword_addtid(2, 0, 0);
    CHECK(inst.is64, "global_load_dword_addtid is 64-bit");
    CHECK_EQ_U32(inst.dw0, 0xDC588000u,
                 "global_load_dword_addtid DW0 oracle");
    CHECK_EQ_U32(inst.dw1, 0x02000000u,
                 "global_load_dword_addtid DW1 oracle");

    inst = vbs_rdna2_emit_global_store_dword_addtid(2, 0, 0);
    CHECK(inst.is64, "global_store_dword_addtid is 64-bit");
    CHECK_EQ_U32(inst.dw0, 0xDC5C8000u,
                 "global_store_dword_addtid DW0 oracle");
    CHECK_EQ_U32(inst.dw1, 0x00000200u,
                 "global_store_dword_addtid DW1 oracle");
}

static void test_sopp_fields(void) {
    vbs_rdna2_inst_t inst;

    inst = vbs_rdna2_emit_s_endpgm();
    CHECK_EQ_U32(SOPP_ENC(inst.dw0), 0x17Fu, "SOPP encoding tag");
    CHECK_EQ_U32(SOPP_OP(inst.dw0), 1u, "S_ENDPGM op field");
    CHECK_EQ_U32(SOPP_SIMM(inst.dw0), 0u, "S_ENDPGM SIMM16 = 0");

    inst = vbs_rdna2_emit_s_barrier();
    CHECK_EQ_U32(SOPP_ENC(inst.dw0), 0x17Fu, "S_BARRIER enc tag");
    CHECK_EQ_U32(SOPP_OP(inst.dw0), 10u, "S_BARRIER op field");
    CHECK_EQ_U32(inst.dw0, 0xBF8A0000u, "s_barrier full encoding");

    /* S_WAITCNT SIMM16 packing: vmcnt {15:14, 3:0}, expcnt [6:4],
     * lgkmcnt [13:8]. */
    inst = vbs_rdna2_emit_s_waitcnt(1, 0, 0);
    CHECK_EQ_U32(SOPP_SIMM(inst.dw0), 0x0001u, "waitcnt vmcnt 1");
    inst = vbs_rdna2_emit_s_waitcnt(16, 0, 0);
    CHECK_EQ_U32(SOPP_SIMM(inst.dw0), 0x4000u, "waitcnt vmcnt 16");
    inst = vbs_rdna2_emit_s_waitcnt(32, 0, 0);
    CHECK_EQ_U32(SOPP_SIMM(inst.dw0), 0x8000u, "waitcnt vmcnt 32");
    inst = vbs_rdna2_emit_s_waitcnt(48, 0, 0);
    CHECK_EQ_U32(SOPP_SIMM(inst.dw0), 0xC000u, "waitcnt vmcnt 48");
    inst = vbs_rdna2_emit_s_waitcnt(0, 5, 0);
    CHECK_EQ_U32(SOPP_SIMM(inst.dw0), 0x0050u, "waitcnt expcnt 5");
    inst = vbs_rdna2_emit_s_waitcnt(0, 0, 0x3F);
    CHECK_EQ_U32(SOPP_SIMM(inst.dw0), 0x3F00u, "waitcnt lgkmcnt 63");
    inst = vbs_rdna2_emit_s_waitcnt(63, 7, 63);
    CHECK_EQ_U32(SOPP_SIMM(inst.dw0), 0xFF7Fu, "waitcnt all max");
    CHECK_EQ_U32(inst.dw0, 0xBF8CFF7Fu, "waitcnt all max full");
}

static void test_smem_fields(void) {
    vbs_rdna2_inst_t inst;

    /* s_load_dwordx2 s[6:7], s[8:9], 0x100, SOFFSET NULL. */
    inst = vbs_rdna2_emit_s_load_dwordx(6, 4, 0x100,
                                        VBS_RDNA2_SRC_NULL, 2);
    CHECK_EQ_U32(SMEM_ENC(inst.dw0), 0x3Du, "SMEM enc tag");
    CHECK_EQ_U32(SMEM_OP(inst.dw0), 1u, "SMEM op = x2");
    CHECK_EQ_U32(SMEM_SDATA(inst.dw0), 6u, "SMEM SDATA");
    CHECK_EQ_U32(SMEM_SBASE(inst.dw0), 4u, "SMEM SBASE pair");
    CHECK_EQ_U32(inst.dw1 & 0x1FFFFFu, 0x100u, "SMEM OFFSET");
    CHECK_EQ_U32((inst.dw1 >> 25) & 0x7Fu, 125u, "SMEM SOFFSET NULL");
    CHECK_EQ_U32(inst.dw0, 0xF4040184u, "smem full DW0");
    CHECK_EQ_U32(inst.dw1, 0xFA000100u, "smem full DW1");

    /* SOFFSET from an SGPR (s10) and a negative offset. */
    inst = vbs_rdna2_emit_s_load_dwordx(0, 0, -4, 10, 1);
    CHECK_EQ_U32(SMEM_OP(inst.dw0), 0u, "SMEM op = x1");
    CHECK_EQ_U32(inst.dw1 & 0x1FFFFFu, 0x1FFFFCu,
                 "SMEM negative OFFSET (21-bit two's complement)");
    CHECK_EQ_U32((inst.dw1 >> 25) & 0x7Fu, 10u, "SMEM SOFFSET s10");

    /* x4/x8/x16 opcodes and alignment. */
    inst = vbs_rdna2_emit_s_load_dwordx(8, 0, 0,
                                        VBS_RDNA2_SRC_NULL, 4);
    CHECK_EQ_U32(SMEM_OP(inst.dw0), 2u, "SMEM op = x4");
    inst = vbs_rdna2_emit_s_load_dwordx(8, 0, 0,
                                        VBS_RDNA2_SRC_NULL, 8);
    CHECK_EQ_U32(SMEM_OP(inst.dw0), 3u, "SMEM op = x8");
    inst = vbs_rdna2_emit_s_load_dwordx(16, 0, 0,
                                        VBS_RDNA2_SRC_NULL, 16);
    CHECK_EQ_U32(SMEM_OP(inst.dw0), 4u, "SMEM op = x16");

    /* Alignment rule violations fail (all-zero return). */
    inst = vbs_rdna2_emit_s_load_dwordx(5, 0, 0,
                                        VBS_RDNA2_SRC_NULL, 2);
    CHECK(inst.dw0 == 0 && inst.dw1 == 0 && !inst.is64,
          "x2 with odd SDATA rejected");
    inst = vbs_rdna2_emit_s_load_dwordx(2, 0, 0,
                                        VBS_RDNA2_SRC_NULL, 4);
    CHECK(inst.dw0 == 0 && inst.dw1 == 0,
          "x4 with SDATA not multiple of 4 rejected");
    inst = vbs_rdna2_emit_s_load_dwordx(0, 0, 0,
                                        VBS_RDNA2_SRC_NULL, 3);
    CHECK(inst.dw0 == 0 && inst.dw1 == 0,
          "dwords = 3 rejected");
    inst = vbs_rdna2_emit_s_load_dwordx(0, 0, 1048576,
                                        VBS_RDNA2_SRC_NULL, 1);
    CHECK(inst.dw0 == 0 && inst.dw1 == 0,
          "OFFSET out of 21-bit range rejected");
}

static void test_vop_fields(void) {
    vbs_rdna2_inst_t inst;

    /* VOP1: v_mov_b32 v5, s10. */
    inst = vbs_rdna2_emit_v_mov_b32(5, VBS_RDNA2_SRC_SGPR(10));
    CHECK_EQ_U32(VOP1_ENC(inst.dw0), 0x3Fu, "VOP1 enc tag");
    CHECK_EQ_U32(VOP1_OP(inst.dw0), 1u, "V_MOV_B32 op");
    CHECK_EQ_U32(VOP1_VDST(inst.dw0), 5u, "VOP1 VDST");
    CHECK_EQ_U32(VOP1_SRC0(inst.dw0), 10u, "VOP1 SRC0");
    CHECK_EQ_U32(inst.dw0, 0x7E0A020Au, "v_mov_b32 v5, s10 full");

    /* VOP2: v_add_nc_u32 v10, v3, v7. */
    inst = vbs_rdna2_emit_v_add_nc_u32(10, VBS_RDNA2_SRC_VGPR(3), 7);
    CHECK_EQ_U32((inst.dw0 >> 31) & 1u, 0u, "VOP2 enc bit 31 = 0");
    CHECK_EQ_U32(VOP2_OP(inst.dw0), 37u, "VOP2 op");
    CHECK_EQ_U32(VOP2_VDST(inst.dw0), 10u, "VOP2 VDST");
    CHECK_EQ_U32(VOP2_VSRC1(inst.dw0), 7u, "VOP2 VSRC1");
    CHECK_EQ_U32(VOP2_SRC0(inst.dw0), 259u, "VOP2 SRC0 = v3 code");
    CHECK_EQ_U32(inst.dw0, 0x4A140F03u, "v_add_nc_u32 v10, v3, v7 full");

    /* VOP2 with an SGPR source and an inline constant. */
    inst = vbs_rdna2_emit_v_add_nc_u32(0, VBS_RDNA2_SRC_SGPR(4), 0);
    CHECK_EQ_U32(VOP2_SRC0(inst.dw0), 4u, "VOP2 SRC0 = s4");
    inst = vbs_rdna2_emit_v_add_nc_u32(0, VBS_RDNA2_SRC_CONST0, 0);
    CHECK_EQ_U32(VOP2_SRC0(inst.dw0), 128u, "VOP2 SRC0 = const 0");
}

static void test_vop3_fields(void) {
    vbs_rdna2_inst_t inst;

    /* VOP3A re-encode: v_add_nc_u32 v2, v0, v1 (src2 = const 0,
     * unused by this op but the field must still encode). */
    inst = vbs_rdna2_emit_vop3a(VBS_RDNA2_OP_V_ADD_NC_U32_VOP3, 2,
                                VBS_RDNA2_SRC_VGPR(0),
                                VBS_RDNA2_SRC_VGPR(1),
                                VBS_RDNA2_SRC_CONST0);
    CHECK(inst.is64, "VOP3A is 64-bit");
    CHECK_EQ_U32(VOP3_ENC(inst.dw0), 0x35u, "VOP3 enc tag");
    CHECK_EQ_U32(VOP3_OP(inst.dw0), 293u, "VOP3A op = 293 (+256)");
    CHECK_EQ_U32(VOP3_VDST(inst.dw0), 2u, "VOP3A VDST");
    CHECK_EQ_U32(VOP3_SRC0(inst.dw1), 256u, "VOP3A SRC0");
    CHECK_EQ_U32(VOP3_SRC1(inst.dw1), 257u, "VOP3A SRC1");
    CHECK_EQ_U32(VOP3_SRC2(inst.dw1), 128u, "VOP3A SRC2");
    CHECK_EQ_U32(inst.dw0, 0xD5250002u, "vop3a add full DW0");
    CHECK_EQ_U32(inst.dw1, 0x02020300u, "vop3a add full DW1");

    /* VOP3A native op: v_bfi_b32 v4, v0, v1, v2 (op 330). */
    inst = vbs_rdna2_emit_vop3a(VBS_RDNA2_OP_V_BFI_B32, 4,
                                VBS_RDNA2_SRC_VGPR(0),
                                VBS_RDNA2_SRC_VGPR(1),
                                VBS_RDNA2_SRC_VGPR(2));
    CHECK_EQ_U32(VOP3_OP(inst.dw0), 330u, "V_BFI_B32 op");
    CHECK_EQ_U32(VOP3_VDST(inst.dw0), 4u, "V_BFI_B32 VDST");
    CHECK_EQ_U32(inst.dw0, 0xD54A0004u, "v_bfi_b32 full DW0");

    /* A non-VOP3A kind is rejected. */
    inst = vbs_rdna2_emit_vop3a(VBS_RDNA2_OP_V_ADD_NC_U32, 0, 0, 0, 0);
    CHECK(inst.dw0 == 0 && inst.dw1 == 0 && !inst.is64,
          "vop3a rejects a VOP2 kind");
    inst = vbs_rdna2_emit_vop3a(VBS_RDNA2_OP_V_ADD_CO_U32, 0, 0, 0, 0);
    CHECK(inst.dw0 == 0 && inst.dw1 == 0,
          "vop3a rejects a VOP3B kind");

    /* VOP3B: v_add_co_u32 v2, vcc, v0, v1, vcc. */
    inst = vbs_rdna2_emit_v_add_co_u32(2, VBS_RDNA2_SRC_VCC_LO,
                                       VBS_RDNA2_SRC_VGPR(0),
                                       VBS_RDNA2_SRC_VGPR(1),
                                       VBS_RDNA2_SRC_VCC_LO);
    CHECK(inst.is64, "VOP3B is 64-bit");
    CHECK_EQ_U32(VOP3_ENC(inst.dw0), 0x35u, "VOP3B enc tag");
    CHECK_EQ_U32(VOP3_OP(inst.dw0), 783u, "V_ADD_CO_U32 op");
    CHECK_EQ_U32(VOP3_VDST(inst.dw0), 2u, "VOP3B VDST");
    CHECK_EQ_U32(VOP3_SDST(inst.dw0), 106u, "VOP3B SDST = VCC");
    CHECK_EQ_U32(VOP3_SRC0(inst.dw1), 256u, "VOP3B SRC0");
    CHECK_EQ_U32(VOP3_SRC1(inst.dw1), 257u, "VOP3B SRC1");
    CHECK_EQ_U32(VOP3_SRC2(inst.dw1), 106u, "VOP3B SRC2 = VCC");
    CHECK_EQ_U32(inst.dw0, 0xD70F6A02u, "v_add_co_u32 full DW0");
    CHECK_EQ_U32(inst.dw1, 0x01AA0300u, "v_add_co_u32 full DW1");
}

static void test_flat_fields(void) {
    vbs_rdna2_inst_t inst;

    /* GLOBAL_LOAD_DWORD v3, [s[6:7] + v10 + 16] (mode b). */
    inst = vbs_rdna2_emit_global_load_dword(3, 10, 6, 16);
    CHECK(inst.is64, "flat load is 64-bit");
    CHECK_EQ_U32(FLAT_ENC(inst.dw0), 0x37u, "FLAT enc tag");
    CHECK_EQ_U32(FLAT_OP(inst.dw0), 12u, "GLOBAL_LOAD_DWORD op");
    CHECK_EQ_U32(FLAT_SEG(inst.dw0), 2u, "FLAT SEG = global");
    CHECK_EQ_U32(FLAT_OFF(inst.dw0), 16u, "FLAT OFFSET");
    CHECK_EQ_U32(FLAT_ADDR(inst.dw1), 10u, "FLAT ADDR vgpr");
    CHECK_EQ_U32(FLAT_DATA(inst.dw1), 0u, "FLAT DATA unused on load");
    CHECK_EQ_U32(FLAT_SADDR(inst.dw1), 6u, "FLAT SADDR");
    CHECK_EQ_U32(FLAT_VDST(inst.dw1), 3u, "FLAT VDST");
    CHECK_EQ_U32(inst.dw0, 0xDC308010u, "global_load_dword full DW0");
    CHECK_EQ_U32(inst.dw1, 0x0306000Au, "global_load_dword full DW1");

    /* Mode (a): SADDR NULL, VGPR holds the whole address. */
    inst = vbs_rdna2_emit_global_load_dword(3, 10,
                                            VBS_RDNA2_SADDR_NULL, 0);
    CHECK_EQ_U32(FLAT_SADDR(inst.dw1), 0x7Fu, "FLAT SADDR NULL");

    /* GLOBAL_STORE_DWORD [s[6:7] + v10 - 4], v5. */
    inst = vbs_rdna2_emit_global_store_dword(5, 10, 6, -4);
    CHECK_EQ_U32(FLAT_OP(inst.dw0), 28u, "GLOBAL_STORE_DWORD op");
    CHECK_EQ_U32(FLAT_OFF(inst.dw0), 0xFFCu,
                 "FLAT negative OFFSET (12-bit two's complement)");
    CHECK_EQ_U32(FLAT_DATA(inst.dw1), 5u, "FLAT DATA = v5");
    CHECK_EQ_U32(FLAT_VDST(inst.dw1), 0u, "FLAT VDST unused on store");
    CHECK_EQ_U32(inst.dw0, 0xDC708FFCu, "global_store_dword full DW0");
    CHECK_EQ_U32(inst.dw1, 0x0006050Au, "global_store_dword full DW1");

    /* ADDTID: the ADDR field is 0 by our (UNSOURCED) choice —
     * the test asserts what we emit, not that the hardware
     * requires it. */
    inst = vbs_rdna2_emit_global_load_dword_addtid(7, 4, 0);
    CHECK_EQ_U32(FLAT_ADDR(inst.dw1), 0u, "ADDTID ADDR field = 0");
    CHECK_EQ_U32(FLAT_OP(inst.dw0), 22u, "ADDTID load op");
    CHECK_EQ_U32(FLAT_SADDR(inst.dw1), 4u, "ADDTID load SADDR");
    CHECK_EQ_U32(FLAT_VDST(inst.dw1), 7u, "ADDTID load VDST");
    inst = vbs_rdna2_emit_global_store_dword_addtid(7, 4, 0);
    CHECK_EQ_U32(FLAT_ADDR(inst.dw1), 0u, "ADDTID store ADDR = 0");
    CHECK_EQ_U32(FLAT_OP(inst.dw0), 23u, "ADDTID store op");
    CHECK_EQ_U32(FLAT_DATA(inst.dw1), 7u, "ADDTID store DATA");

    /* OFFSET out of 12-bit range is rejected. */
    inst = vbs_rdna2_emit_global_load_dword(0, 0, 0, 2048);
    CHECK(inst.dw0 == 0 && inst.dw1 == 0,
          "FLAT OFFSET +2048 rejected");
    inst = vbs_rdna2_emit_global_load_dword(0, 0, 0, -2049);
    CHECK(inst.dw0 == 0 && inst.dw1 == 0,
          "FLAT OFFSET -2049 rejected");
}

static void test_operand_codes_and_serialize(void) {
    vbs_rdna2_inst_t inst;
    uint8_t buf[8];

    CHECK_EQ_U32(VBS_RDNA2_SRC_VGPR(0), 256u, "VGPR 0 code");
    CHECK_EQ_U32(VBS_RDNA2_SRC_VGPR(255), 511u, "VGPR 255 code");
    CHECK_EQ_U32(VBS_RDNA2_SRC_SGPR(105), 105u, "SGPR 105 code");
    CHECK_EQ_U32(VBS_RDNA2_SRC_CONST0, 128u, "const 0 code");
    CHECK_EQ_U32(VBS_RDNA2_SRC_INT(64), 192u, "int 64 code");
    CHECK_EQ_U32(VBS_RDNA2_SRC_NEGINT(1), 193u, "int -1 code");
    CHECK_EQ_U32(VBS_RDNA2_SRC_NEGINT(16), 208u, "int -16 code");
    CHECK_EQ_U32(VBS_RDNA2_SRC_NULL, 125u, "NULL code");
    CHECK_EQ_U32(VBS_RDNA2_SRC_M0, 124u, "M0 code");

    /* Serialize: 32-bit, little-endian, dw0 only. */
    inst = vbs_rdna2_emit_s_endpgm();
    CHECK(vbs_rdna2_inst_size(&inst) == 4, "s_endpgm size 4");
    memset(buf, 0xAA, sizeof(buf));
    CHECK(vbs_rdna2_inst_serialize(&inst, buf) == 4,
          "s_endpgm serialize returns 4");
    CHECK(buf[0] == 0x00 && buf[1] == 0x00 && buf[2] == 0x81 &&
          buf[3] == 0xBF, "s_endpgm bytes LE");
    CHECK(buf[4] == 0xAA, "serialize does not overrun (32-bit)");

    /* Serialize: 64-bit, dw0 then dw1. */
    inst = vbs_rdna2_emit_global_load_dword_addtid(2, 0, 0);
    CHECK(vbs_rdna2_inst_size(&inst) == 8, "addtid size 8");
    CHECK(vbs_rdna2_inst_serialize(&inst, buf) == 8,
          "addtid serialize returns 8");
    CHECK(buf[0] == 0x00 && buf[1] == 0x80 && buf[2] == 0x58 &&
          buf[3] == 0xDC && buf[4] == 0x00 && buf[5] == 0x00 &&
          buf[6] == 0x00 && buf[7] == 0x02, "addtid bytes LE");
}

int main(void) {
    test_opcode_table();
    test_digest_oracles();
    test_sopp_fields();
    test_smem_fields();
    test_vop_fields();
    test_vop3_fields();
    test_flat_fields();
    test_operand_codes_and_serialize();
    TEST_SUMMARY("test_rdna2_asm");
}
