/* vbs_rdna2_asm.c — RDNA 2 (gfx1030) assembler implementation.
 *
 * One function per instruction kind; every encoding fact cites
 * SPEC-DIGEST.md §5.2 or the RDNA 2 ISA Reference Guide section
 * it came from. Opcode numbers come only from the opcode table
 * in vbs_rdna2_isa.h (vbs_rdna2_opcode) — there are no opcode
 * literals in this file. All code is original Verum work.
 */

#include "vbs_rdna2_asm.h"

/* Serialize one instruction: dw0 little-endian first, then dw1
 * for 64-bit instructions. */
size_t vbs_rdna2_inst_serialize(const vbs_rdna2_inst_t *inst,
                                uint8_t *out) {
    out[0] = (uint8_t)(inst->dw0 & 0xFFu);
    out[1] = (uint8_t)((inst->dw0 >> 8) & 0xFFu);
    out[2] = (uint8_t)((inst->dw0 >> 16) & 0xFFu);
    out[3] = (uint8_t)((inst->dw0 >> 24) & 0xFFu);
    if (!inst->is64)
        return 4;
    out[4] = (uint8_t)(inst->dw1 & 0xFFu);
    out[5] = (uint8_t)((inst->dw1 >> 8) & 0xFFu);
    out[6] = (uint8_t)((inst->dw1 >> 16) & 0xFFu);
    out[7] = (uint8_t)((inst->dw1 >> 24) & 0xFFu);
    return 8;
}

/* --- SOPP -----------------------------------------------------------
 * SIMM16 [15:0], OP [22:16], ENCODING [31:23] = 0x17F.
 * Digest §5.2 SOPP; [RDNA2-ISA] §13.1.5, p.245. */
static vbs_rdna2_inst_t emit_sopp(vbs_rdna2_op_t kind,
                                  uint16_t simm16) {
    int ok = 0;
    uint16_t op = vbs_rdna2_opcode(kind, &ok);
    vbs_rdna2_inst_t inst;
    if (!ok) {
        inst.dw0 = 0;
        inst.dw1 = 0;
        inst.is64 = 0;
        return inst;
    }
    inst.dw0 = (VBS_RDNA2_ENC_SOPP << 23) |
               ((uint32_t)op << 16) | (uint32_t)simm16;
    inst.dw1 = 0;
    inst.is64 = 0;
    return inst;
}

vbs_rdna2_inst_t vbs_rdna2_emit_s_endpgm(void) {
    return emit_sopp(VBS_RDNA2_OP_S_ENDPGM, 0);
}

vbs_rdna2_inst_t vbs_rdna2_emit_s_barrier(void) {
    return emit_sopp(VBS_RDNA2_OP_S_BARRIER, 0);
}

vbs_rdna2_inst_t vbs_rdna2_emit_s_waitcnt(unsigned vmcnt,
                                          unsigned expcnt,
                                          unsigned lgkmcnt) {
    /* SIMM16: vmcnt {bits 15:14, 3:0}, expcnt [6:4], lgkmcnt
     * [13:8]. Digest §5.2 SOPP; [RDNA2-ISA] §12.2 (S_WAITCNT).
     * vmcnt is 6 bits: low 4 go to [3:0], high 2 to [15:14]. */
    uint16_t simm = (uint16_t)(
            ((vmcnt & 0xFu)) |
            ((expcnt & 0x7u) << 4) |
            ((lgkmcnt & 0x3Fu) << 8) |
            (((vmcnt >> 4) & 0x3u) << 14));
    return emit_sopp(VBS_RDNA2_OP_S_WAITCNT, simm);
}

/* --- SMEM -----------------------------------------------------------
 * DW0: SBASE [5:0], SDATA [12:6], DLC [14], GLC [16], OP [25:18],
 *      ENCODING [31:26] = 0x3D.
 * DW1: OFFSET [52:32] (21-bit signed), SOFFSET [63:57].
 * Digest §5.2 SMEM; [RDNA2-ISA] §13.2.1, p.246; §7.1, p.51. */
vbs_rdna2_inst_t vbs_rdna2_emit_s_load_dwordx(unsigned sdata,
                                              unsigned sbase_pair,
                                              int32_t offset,
                                              unsigned soffset,
                                              unsigned dwords) {
    vbs_rdna2_op_t kind;
    int ok = 0;
    uint16_t op;
    vbs_rdna2_inst_t inst;

    /* Map the dword count to the table kind. */
    switch (dwords) {
    case 1: kind = VBS_RDNA2_OP_S_LOAD_DWORD; break;
    case 2: kind = VBS_RDNA2_OP_S_LOAD_DWORDX2; break;
    case 4: kind = VBS_RDNA2_OP_S_LOAD_DWORDX4; break;
    case 8: kind = VBS_RDNA2_OP_S_LOAD_DWORDX8; break;
    case 16: kind = VBS_RDNA2_OP_S_LOAD_DWORDX16; break;
    default:
        return VBS_RDNA2_INST_INVALID;
    }

    /* Alignment rule [RDNA2-ISA] §7.1, p.51: 2-dword reads need
     * an even SDATA, 4+ dword reads need a multiple of 4. */
    if (dwords == 2 && (sdata & 1u) != 0)
        return VBS_RDNA2_INST_INVALID;
    if (dwords >= 4 && (sdata & 3u) != 0)
        return VBS_RDNA2_INST_INVALID;

    /* OFFSET is a signed 21-bit field: -2^20 .. 2^20-1. */
    if (offset < -1048576 || offset > 1048575)
        return VBS_RDNA2_INST_INVALID;

    op = vbs_rdna2_opcode(kind, &ok);
    if (!ok)
        return VBS_RDNA2_INST_INVALID;

    inst.dw0 = (VBS_RDNA2_ENC_SMEM << 26) |
               ((uint32_t)op << 18) |
               ((uint32_t)(sdata & 0x7Fu) << 6) |
               (uint32_t)(sbase_pair & 0x3Fu);
    /* DW1 bits [20:0] = OFFSET, bits [31:25] = SOFFSET. */
    inst.dw1 = ((uint32_t)(offset & 0x1FFFFF)) |
               ((uint32_t)(soffset & 0x7Fu) << 25);
    inst.is64 = 1;
    return inst;
}

vbs_rdna2_inst_t vbs_rdna2_emit_s_load_dwordx2(unsigned sdata,
                                               unsigned sbase_pair,
                                               int32_t offset) {
    return vbs_rdna2_emit_s_load_dwordx(sdata, sbase_pair, offset,
                                        VBS_RDNA2_SRC_NULL, 2);
}

/* --- VOP1 -----------------------------------------------------------
 * SRC0 [8:0], OP [16:9], VDST [24:17], ENCODING [31:25] = 0x3F.
 * Digest §5.2 VOP1; [RDNA2-ISA] §13.3.2, pp.249-250. */
vbs_rdna2_inst_t vbs_rdna2_emit_v_mov_b32(unsigned vdst,
                                          unsigned src0) {
    int ok = 0;
    uint16_t op = vbs_rdna2_opcode(VBS_RDNA2_OP_V_MOV_B32, &ok);
    vbs_rdna2_inst_t inst;
    if (!ok)
        return VBS_RDNA2_INST_INVALID;
    inst.dw0 = (VBS_RDNA2_ENC_VOP1 << 25) |
               ((uint32_t)(vdst & 0xFFu) << 17) |
               ((uint32_t)op << 9) |
               (uint32_t)(src0 & 0x1FFu);
    inst.dw1 = 0;
    inst.is64 = 0;
    return inst;
}

/* --- VOP2 -----------------------------------------------------------
 * SRC0 [8:0], VSRC1 [16:9], VDST [24:17], OP [30:25],
 * ENCODING [31] = 0. Digest §5.2 VOP2; [RDNA2-ISA] §13.3.1,
 * pp.247-248. */
vbs_rdna2_inst_t vbs_rdna2_emit_v_add_nc_u32(unsigned vdst,
                                             unsigned src0,
                                             unsigned vsrc1) {
    int ok = 0;
    uint16_t op = vbs_rdna2_opcode(VBS_RDNA2_OP_V_ADD_NC_U32, &ok);
    vbs_rdna2_inst_t inst;
    if (!ok)
        return VBS_RDNA2_INST_INVALID;
    inst.dw0 = ((uint32_t)op << 25) |
               ((uint32_t)(vdst & 0xFFu) << 17) |
               ((uint32_t)(vsrc1 & 0xFFu) << 9) |
               (uint32_t)(src0 & 0x1FFu);
    inst.dw1 = 0;
    inst.is64 = 0;
    return inst;
}

/* Pack the three 9-bit source operands of a VOP3 instruction into
 * its second dword: SRC0 [8:0], SRC1 [17:9], SRC2 [26:18]. Shared
 * by the VOP3A and VOP3B encodings. */
static uint32_t vop3_src_word(unsigned src0, unsigned src1,
                              unsigned src2) {
    return (uint32_t)(src0 & 0x1FFu) |
           ((uint32_t)(src1 & 0x1FFu) << 9) |
           ((uint32_t)(src2 & 0x1FFu) << 18);
}

/* --- VOP3A ----------------------------------------------------------
 * DW0: VDST [7:0], ABS [10:8] = 0, OPSEL [14:11] = 0,
 *      CLMP [15] = 0, OP [25:16], ENCODING [31:26] = 0x35.
 * DW1: SRC0 [40:32], SRC1 [49:41], SRC2 [58:50], OMOD [60:59]
 *      = 0, NEG [63:61] = 0.
 * Digest §5.2 VOP3A; [RDNA2-ISA] §13.3.4, p.257. */
vbs_rdna2_inst_t vbs_rdna2_emit_vop3a(vbs_rdna2_op_t kind,
                                      unsigned vdst,
                                      unsigned src0,
                                      unsigned src1,
                                      unsigned src2) {
    const vbs_rdna2_op_entry_t *e = vbs_rdna2_op_lookup(kind);
    vbs_rdna2_inst_t inst;
    if (!e || e->fmt != VBS_RDNA2_FMT_VOP3A || !e->sourced)
        return VBS_RDNA2_INST_INVALID;
    inst.dw0 = (VBS_RDNA2_ENC_VOP3 << 26) |
               ((uint32_t)e->opcode << 16) |
               (uint32_t)(vdst & 0xFFu);
    inst.dw1 = vop3_src_word(src0, src1, src2);
    inst.is64 = 1;
    return inst;
}

/* --- VOP3B ----------------------------------------------------------
 * DW0: VDST [7:0], SDST [14:8], CLMP [15] = 0, OP [25:16],
 *      ENCODING [31:26] = 0x35.
 * DW1: SRC0 [40:32], SRC1 [49:41], SRC2 [58:50].
 * Digest §5.2 VOP3B; [RDNA2-ISA] §13.3.5, pp.260-261. */
vbs_rdna2_inst_t vbs_rdna2_emit_v_add_co_u32(unsigned vdst,
                                             unsigned sdst,
                                             unsigned src0,
                                             unsigned src1,
                                             unsigned src2) {
    int ok = 0;
    uint16_t op = vbs_rdna2_opcode(VBS_RDNA2_OP_V_ADD_CO_U32, &ok);
    vbs_rdna2_inst_t inst;
    if (!ok)
        return VBS_RDNA2_INST_INVALID;
    inst.dw0 = (VBS_RDNA2_ENC_VOP3 << 26) |
               ((uint32_t)op << 16) |
               ((uint32_t)(sdst & 0x7Fu) << 8) |
               (uint32_t)(vdst & 0xFFu);
    inst.dw1 = vop3_src_word(src0, src1, src2);
    inst.is64 = 1;
    return inst;
}

/* --- FLAT / GLOBAL --------------------------------------------------
 * DW0: OFFSET [11:0], DLC [12] = 0, LDS [13] = 0, SEG [15:14],
 *      GLC [16] = 0, SLC [17] = 0, OP [24:18], ENCODING [31:26]
 *      = 0x37.
 * DW1: ADDR [39:32], DATA [47:40], SADDR [54:48], VDST [63:56].
 * Digest §5.2 FLAT; [RDNA2-ISA] §13.8.1-2, pp.279-281. */
static vbs_rdna2_inst_t emit_flat(vbs_rdna2_op_t kind,
                                  unsigned addr_vgpr,
                                  unsigned data_vgpr,
                                  unsigned saddr,
                                  unsigned vdst,
                                  int16_t offset) {
    int ok = 0;
    uint16_t op = vbs_rdna2_opcode(kind, &ok);
    vbs_rdna2_inst_t inst;
    if (!ok)
        return VBS_RDNA2_INST_INVALID;
    /* OFFSET is a 12-bit signed field: -2048 .. 2047. */
    if (offset < -2048 || offset > 2047)
        return VBS_RDNA2_INST_INVALID;
    inst.dw0 = (VBS_RDNA2_ENC_FLAT << 26) |
               ((uint32_t)op << 18) |
               ((uint32_t)VBS_RDNA2_SEG_GLOBAL << 14) |
               (uint32_t)(offset & 0xFFF);
    inst.dw1 = (uint32_t)(addr_vgpr & 0xFFu) |
               ((uint32_t)(data_vgpr & 0xFFu) << 8) |
               ((uint32_t)(saddr & 0x7Fu) << 16) |
               ((uint32_t)(vdst & 0xFFu) << 24);
    inst.is64 = 1;
    return inst;
}

vbs_rdna2_inst_t vbs_rdna2_emit_global_load_dword(
        unsigned vdst, unsigned addr_vgpr, unsigned saddr,
        int16_t offset) {
    /* Loads: DATA field unused, set 0. */
    return emit_flat(VBS_RDNA2_OP_GLOBAL_LOAD_DWORD, addr_vgpr, 0,
                     saddr, vdst, offset);
}

vbs_rdna2_inst_t vbs_rdna2_emit_global_load_dword_addtid(
        unsigned vdst, unsigned saddr, int16_t offset) {
    /* ADDTID: no VGPR address is supplied ([RDNA2-ISA] §12.8);
     * the ADDR field value is UNSOURCED — 0 here (see the header
     * for the derivation procedure). DATA unused, 0. */
    return emit_flat(VBS_RDNA2_OP_GLOBAL_LOAD_DWORD_ADDTID, 0, 0,
                     saddr, vdst, offset);
}

vbs_rdna2_inst_t vbs_rdna2_emit_global_store_dword(
        unsigned vdata, unsigned addr_vgpr, unsigned saddr,
        int16_t offset) {
    /* Stores: VDST field unused, set 0. */
    return emit_flat(VBS_RDNA2_OP_GLOBAL_STORE_DWORD, addr_vgpr,
                     vdata, saddr, 0, offset);
}

vbs_rdna2_inst_t vbs_rdna2_emit_global_store_dword_addtid(
        unsigned vdata, unsigned saddr, int16_t offset) {
    /* ADDTID: ADDR field = 0 (UNSOURCED, see the header).
     * VDST unused, 0. */
    return emit_flat(VBS_RDNA2_OP_GLOBAL_STORE_DWORD_ADDTID, 0,
                     vdata, saddr, 0, offset);
}
