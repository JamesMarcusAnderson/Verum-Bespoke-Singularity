/* vbs_alchemist_eu.c — minimal EU assembler for Intel Arc Alchemist.
 *
 * One emitter per instruction kind; every bit placed from the PRM
 * field tables cited in vbs_alchemist_eu.h. All code here is
 * original Verum work — no transcription from Mesa, Intel's compute
 * runtime, or any other implementation.
 *
 * The emitters enforce the sourced legality rules, so an illegal
 * instruction can never be emitted silently:
 *   - ExecSize * (largest operand element size) <= 64 (Vol 9,
 *     p.387) — this is what forces all :uq ALU work into SIMD8
 *     halves (16 * 8 = 128 would be illegal).
 *   - An operand's Width must be <= ExecSize (Vol 2b, p.78).
 *   - A source operand's region must fit within two adjacent GRFs
 *     (64 bytes total; Vol 2b, p.126 VertStride note).
 *   - send EOT rules (Vol 2a, pp.1254-1255): dst null, sources in
 *     r112-r127; null dst requires Response Length 0; null src1
 *     requires Extended Message Length 0.
 * A violation returns -1 and leaves *inst untouched.
 */

#include "vbs_alchemist_eu.h"

#include <string.h>

/* Element size in bytes for a RegDataType value. */
static unsigned type_bytes(uint8_t type)
{
    switch (type) {
    case VBS_ALCHEMIST_TYPE_UB:
    case VBS_ALCHEMIST_TYPE_B:
        return 1;
    case VBS_ALCHEMIST_TYPE_UW:
    case VBS_ALCHEMIST_TYPE_W:
    case VBS_ALCHEMIST_TYPE_HF:
    case VBS_ALCHEMIST_TYPE_BF:
        return 2;
    case VBS_ALCHEMIST_TYPE_UD:
    case VBS_ALCHEMIST_TYPE_D:
    case VBS_ALCHEMIST_TYPE_F:
        return 4;
    case VBS_ALCHEMIST_TYPE_UQ:
    case VBS_ALCHEMIST_TYPE_Q:
    case VBS_ALCHEMIST_TYPE_DF:
        return 8;
    default:
        return 0; /* reserved type */
    }
}

/* Channel count for an ExecSize encoding (Vol 2b, p.78). */
static unsigned exec_channels(uint8_t exec_enc)
{
    static const unsigned n[6] = { 1, 2, 4, 8, 16, 32 };
    return exec_enc < 6 ? n[exec_enc] : 0;
}

/* Element count for a Width encoding (Vol 2b, p.127). */
static unsigned width_elems(uint8_t width_enc)
{
    static const unsigned n[5] = { 1, 2, 4, 8, 16 };
    return width_enc < 5 ? n[width_enc] : 0;
}

/* Element count for a VertStride encoding (Vol 2b, p.126). */
static unsigned vstride_elems(uint8_t v_enc)
{
    static const unsigned n[7] = { 0, 1, 2, 4, 8, 16, 32 };
    return v_enc < 7 ? n[v_enc] : 0;
}

/* Element count for a HorzStride encoding (Vol 2b, p.82). */
static unsigned hstride_elems(uint8_t h_enc)
{
    static const unsigned n[4] = { 0, 1, 2, 4 };
    return h_enc < 4 ? n[h_enc] : 0;
}

uint16_t vbs_alchemist_direct_operand(uint8_t reg, uint8_t subreg,
                                      int is_grf)
{
    /* DirectOperand (Vol 2d, p.366): RegNum [13:6], SubRegNum
     * [5:1], RegFile [0]. */
    return (uint16_t)(((uint16_t)reg << 6) |
                      ((uint16_t)(subreg & 0x1Fu) << 1) |
                      (is_grf ? 1u : 0u));
}

uint8_t vbs_alchemist_opcode(vbs_alchemist_op_t kind, int *ok)
{
    if (ok != NULL)
        *ok = 0;
    if (kind < 0 || kind >= VBS_ALCHEMIST_OP_COUNT)
        return 0;
    if (!VBS_ALCHEMIST_OPCODE_TABLE[kind].sourced)
        return 0; /* UNSOURCED — never emit */
    if (ok != NULL)
        *ok = 1;
    return VBS_ALCHEMIST_OPCODE_TABLE[kind].code;
}

void vbs_alchemist_serialize(const vbs_alchemist_inst_t *inst,
                             uint8_t out[16])
{
    /* DW0 first, each dword little-endian. Byte 0 of the
     * instruction word holds the opcode (Vol 9, p.469). */
    for (unsigned i = 0; i < 4; i++) {
        uint32_t w = inst->dw[i];
        out[4 * i + 0] = (uint8_t)(w & 0xFFu);
        out[4 * i + 1] = (uint8_t)((w >> 8) & 0xFFu);
        out[4 * i + 2] = (uint8_t)((w >> 16) & 0xFFu);
        out[4 * i + 3] = (uint8_t)((w >> 24) & 0xFFu);
    }
}

uint32_t vbs_alchemist_desc_build(uint32_t msg_len, uint32_t resp_len,
                                  uint32_t header, uint32_t msg_type,
                                  uint32_t simd_mode, uint32_t cmask,
                                  uint32_t bti)
{
    /* desc layout (digest §6.6): [28:25] Message Length, [24:20]
     * Response Length, [19] Header Present, [18:14] Message Type,
     * [13:12] SIMD Mode, [11:8] Channel Mask, [7:0] BTI.
     * Bits [31:29] are 0 (packed-payload flags must be 0 for the
     * untyped messages — Vol 2a, p.1340). */
    return ((msg_len & 0xFu) << 25) | ((resp_len & 0x1Fu) << 20) |
           ((header & 0x1u) << 19) | ((msg_type & 0x1Fu) << 14) |
           ((simd_mode & 0x3u) << 12) | ((cmask & 0xFu) << 8) |
           (bti & 0xFFu);
}

uint32_t vbs_alchemist_exdesc_build(uint32_t ext_msg_len)
{
    /* ex_desc: Extended Message Length [10:6]; all other bits 0
     * for the v1 messages (digest §9 item 3 — UNSOURCED beyond
     * this field; hardware run must confirm 0 is correct). */
    return (ext_msg_len & 0x1Fu) << 6;
}

/* Validate one source operand's region against the ExecSize and
 * the Vol 9 p.387 / Vol 2b region rules. Returns 0 if legal. */
static int check_src_region(vbs_alchemist_operand_t src,
                            unsigned channels)
{
    unsigned w, v, h, nb, rows, span;

    if (src.reg > 127 || src.subreg > 31)
        return -1;
    nb = type_bytes(src.type);
    if (nb == 0)
        return -1;
    w = width_elems(src.width);
    v = vstride_elems(src.vstride);
    h = hstride_elems(src.hstride);
    if (w == 0 || w > channels)
        return -1; /* Width must be <= ExecSize (Vol 2b, p.78) */
    if (channels % w != 0)
        return -1;
    rows = channels / w;
    /* Span in elements from the first to the last element touched:
     * (rows-1) row steps of VertStride, plus one row of Width
     * elements at HorzStride. Must fit in 64 bytes (two GRFs). */
    span = ((rows - 1) * v + (w - 1) * h + 1) * nb;
    if (span > 64)
        return -1;
    return 0;
}

/* Validate a destination operand: ExecSize * element size <= 64
 * (Vol 9, p.387) and the destination span fits in 64 bytes. */
static int check_dst_region(vbs_alchemist_operand_t dst,
                            unsigned channels)
{
    unsigned nb, h, span;

    if (dst.reg > 127 || dst.subreg > 31)
        return -1;
    nb = type_bytes(dst.type);
    if (nb == 0)
        return -1;
    h = hstride_elems(dst.hstride);
    if (channels * nb > 64)
        return -1; /* ExecSize * n <= 64 (Vol 9, p.387) */
    span = ((channels - 1) * h + 1) * nb;
    if (span > 64)
        return -1;
    return 0;
}

/* Shared binary-ALU encoder. kind selects the opcode; src1 may be
 * NULL (mov: no src1 fields, DW3 = 0); is_imm selects the
 * immediate-src1 form (DW3 = imm, Src1.DataType = imm_type). */
/* Store a fully assembled instruction's four dwords and report
 * success. Every emitter ends here. */
static int finish_inst(vbs_alchemist_inst_t *inst, uint32_t dw0,
                       uint32_t dw1, uint32_t dw2, uint32_t dw3) {
    inst->dw[0] = dw0;
    inst->dw[1] = dw1;
    inst->dw[2] = dw2;
    inst->dw[3] = dw3;
    return 0;
}

static int emit_alu(vbs_alchemist_inst_t *inst, vbs_alchemist_op_t kind,
                    uint8_t exec_enc, vbs_alchemist_operand_t dst,
                    vbs_alchemist_operand_t src0,
                    const vbs_alchemist_operand_t *src1, int is_imm,
                    uint32_t imm, uint8_t imm_type)
{
    int ok = 0;
    uint8_t opcode;
    unsigned channels, nb_max;
    uint16_t dst_op, src0_op, src1_op = 0;
    uint32_t dw0, dw1, dw2, dw3;

    if (inst == NULL)
        return -1;
    opcode = vbs_alchemist_opcode(kind, &ok);
    if (!ok)
        return -1; /* UNSOURCED opcode — never emit */
    channels = exec_channels(exec_enc);
    if (channels == 0)
        return -1;

    /* ExecSize * n <= 64, where n is the largest element size of
     * any source or destination operand (Vol 9, p.387). */
    nb_max = type_bytes(dst.type);
    if (type_bytes(src0.type) > nb_max)
        nb_max = type_bytes(src0.type);
    if (src1 != NULL && type_bytes(src1->type) > nb_max)
        nb_max = type_bytes(src1->type);
    if (nb_max == 0 || channels * nb_max > 64)
        return -1;

    if (check_dst_region(dst, channels) != 0)
        return -1;
    if (check_src_region(src0, channels) != 0)
        return -1;
    if (src1 != NULL && check_src_region(*src1, channels) != 0)
        return -1;
    if (is_imm && type_bytes(imm_type) == 0)
        return -1;

    dst_op = vbs_alchemist_direct_operand(dst.reg, dst.subreg, 1);
    src0_op = vbs_alchemist_direct_operand(src0.reg, src0.subreg, 1);
    if (src1 != NULL)
        src1_op = vbs_alchemist_direct_operand(src1->reg, src1->subreg, 1);

    /* DW0: opcode [6:0], ExecSize [18:16]. Everything else 0:
     * MaskCtrl, CmptCtrl (native 128-bit), PredCtrl, ChanOff, and
     * header bits [15:7] (the Access Mode bit is UNSOURCED —
     * left 0, Align1 only; see the opcodes header). */
    dw0 = (uint32_t)opcode | ((uint32_t)exec_enc << 16);

    /* DW1: Dst.Operand [63:50], Dst.HorzStride [49:48],
     * Src1.IsImm [47], Src0.IsImm [46], Src0.Mod [45:44] = 0,
     * Src0.DataType [43:40], Dst.DataType [39:36], Dst.AddrMode
     * [35] = 0 (Direct), Saturate/AccWrCtrl/AtomicCtrl = 0. */
    dw1 = ((uint32_t)dst_op << 18) |
          ((uint32_t)(dst.hstride & 0x3u) << 16) |
          ((uint32_t)(is_imm ? 1u : 0u) << 15) |
          ((uint32_t)(src0.type & 0xFu) << 8) |
          ((uint32_t)(dst.type & 0xFu) << 4);
    /* Note: for mov, bit 47 (Src1.IsImm) is Reserved/MBZ
     * (Vol 2a, p.1113) — is_imm is 0 for mov, so it stays 0. */

    /* DW2: CondCtrl [95:92] = 0, Src1.DataType [91:88] (0 for mov:
     * no field there in the register form — Vol 2a, p.1112),
     * Src0.VertStride [87:84], Src0.Width [83:81], Src0.AddrMode
     * [80] = 0 (Direct), Src0.Operand [79:66], Src0.HorzStride
     * [65:64]. */
    dw2 = ((uint32_t)((src1 != NULL ? src1->type : imm_type) & 0xFu) << 24) |
          ((uint32_t)(src0.vstride & 0xFu) << 20) |
          ((uint32_t)(src0.width & 0x7u) << 17) |
          ((uint32_t)src0_op << 2) |
          (uint32_t)(src0.hstride & 0x3u);
    if (src1 == NULL && !is_imm)
        dw2 &= 0xF0FFFFFFu; /* mov: clear Src1.DataType field */

    /* DW3: register src1 — Src1.Mod [121:120] = 0, Src1.VertStride
     * [119:116], Src1.Width [115:113], Src1.AddrMode [112] = 0,
     * Src1.Operand [111:98], Src1.HorzStride [97:96], bits
     * [127:122] MBZ. Immediate src1: DW3 = the immediate value.
     * mov: DW3 = 0 (no src1 fields exist — Vol 2a, p.1112). */
    if (is_imm) {
        dw3 = imm;
    } else if (src1 != NULL) {
        dw3 = ((uint32_t)(src1->vstride & 0xFu) << 20) |
              ((uint32_t)(src1->width & 0x7u) << 17) |
              ((uint32_t)src1_op << 2) |
              (uint32_t)(src1->hstride & 0x3u);
    } else {
        dw3 = 0;
    }

    return finish_inst(inst, dw0, dw1, dw2, dw3);
}

int vbs_alchemist_emit_add(vbs_alchemist_inst_t *inst, uint8_t exec_enc,
                           vbs_alchemist_operand_t dst,
                           vbs_alchemist_operand_t src0,
                           vbs_alchemist_operand_t src1)
{
    return emit_alu(inst, VBS_ALCHEMIST_OP_ADD, exec_enc, dst, src0,
                    &src1, 0, 0, 0);
}

int vbs_alchemist_emit_mov(vbs_alchemist_inst_t *inst, uint8_t exec_enc,
                           vbs_alchemist_operand_t dst,
                           vbs_alchemist_operand_t src0)
{
    return emit_alu(inst, VBS_ALCHEMIST_OP_MOV, exec_enc, dst, src0,
                    NULL, 0, 0, 0);
}

int vbs_alchemist_emit_shl_imm(vbs_alchemist_inst_t *inst,
                               uint8_t exec_enc,
                               vbs_alchemist_operand_t dst,
                               vbs_alchemist_operand_t src0,
                               uint32_t imm, uint8_t imm_type)
{
    return emit_alu(inst, VBS_ALCHEMIST_OP_SHL, exec_enc, dst, src0,
                    NULL, 1, imm, imm_type);
}

int vbs_alchemist_emit_send(vbs_alchemist_inst_t *inst, uint8_t exec_enc,
                            uint8_t sfid, uint8_t dst_reg,
                            uint8_t src0_reg, uint8_t src1_reg,
                            uint32_t desc, uint32_t ex_desc, int eot)
{
    int ok = 0;
    uint8_t opcode;
    uint32_t msg_len, resp_len, ext_len;
    uint32_t dst_regnum, dst_regfile, src0_regfile, src1_regnum,
        src1_regfile;
    uint32_t dw0, dw1, dw2, dw3;

    if (inst == NULL)
        return -1;
    opcode = vbs_alchemist_opcode(VBS_ALCHEMIST_OP_SEND, &ok);
    if (!ok)
        return -1;
    if (exec_channels(exec_enc) == 0 || sfid > 15)
        return -1;
    if (src0_reg > 127)
        return -1; /* src0 is always a GRF for v1 messages */
    if (dst_reg != VBS_ALCHEMIST_REG_NULL && dst_reg > 127)
        return -1;
    if (src1_reg != VBS_ALCHEMIST_REG_NULL && src1_reg > 127)
        return -1;

    /* Lengths come from desc/ex_desc (single source of truth). */
    msg_len = (desc >> 25) & 0xFu;
    resp_len = (desc >> 20) & 0x1Fu;
    ext_len = (ex_desc >> 6) & 0x1Fu;

    /* Sourced send rules (Vol 2a, pp.1253-1255). */
    if (dst_reg == VBS_ALCHEMIST_REG_NULL && resp_len != 0)
        return -1; /* null dst requires Response Length 0 */
    if (dst_reg != VBS_ALCHEMIST_REG_NULL && resp_len == 0)
        return -1; /* GRF dst requires a nonzero Response Length */
    if (src1_reg == VBS_ALCHEMIST_REG_NULL && ext_len != 0)
        return -1; /* null src1 requires Extended Msg Length 0 */
    if (src1_reg != VBS_ALCHEMIST_REG_NULL && ext_len == 0)
        return -1; /* GRF src1 requires a nonzero Ext Msg Length */
    if (eot) {
        /* "The posted destination operand must be null." (p.1254)
         * "An EOT send must use register space r112-r127 for
         * sources." (p.1255) */
        if (dst_reg != VBS_ALCHEMIST_REG_NULL)
            return -1;
        if (src0_reg < 112 || src0_reg > 127)
            return -1;
    }

    if (dst_reg == VBS_ALCHEMIST_REG_NULL) {
        dst_regnum = 0;
        dst_regfile = 0; /* null = ARF type 0000b (Vol 9, p.348) */
    } else {
        dst_regnum = dst_reg;
        dst_regfile = 1;
    }
    src0_regfile = 1;
    if (src1_reg == VBS_ALCHEMIST_REG_NULL) {
        src1_regnum = 0;
        src1_regfile = 0;
    } else {
        src1_regnum = src1_reg;
        src1_regfile = 1;
    }

    /* DW0: opcode [6:0] (0x31), ExecSize [18:16]. CmptCtrl = 0
     * (sends are never compacted — Vol 9, p.456). */
    dw0 = (uint32_t)opcode | ((uint32_t)exec_enc << 16);

    /* DW1: Dst.RegNum [63:56], Dst.Length = desc[24:20] [55:51],
     * Dst.RegFile [50], ExDesc.IsReg [49] = 0, Desc.IsReg [48] = 0,
     * ExDesc[23:11] [47:35], EOT [34], FusionCtrl [33] = 0,
     * AtomicCtrl [32] = 0. */
    dw1 = (dst_regnum << 24) | (resp_len << 19) |
          (dst_regfile << 18) | (((ex_desc >> 11) & 0x1FFFu) << 3) |
          ((uint32_t)(eot ? 1u : 0u) << 2);

    /* DW2: SFID [95:92], Desc[10:0] [91:81], Reserved [80] = 0,
     * Src0.RegNum [79:72], Desc[29] [71], Src0.Length =
     * desc[28:25] [70:67], Src0.RegFile [66], ExDesc[25:24]
     * [65:64]. */
    dw2 = ((uint32_t)sfid << 28) | ((desc & 0x7FFu) << 17) |
          ((uint32_t)src0_reg << 8) | (((desc >> 29) & 0x1u) << 7) |
          (msg_len << 3) | (src0_regfile << 2) |
          ((ex_desc >> 24) & 0x3u);

    /* DW3: ExDesc[31:28] [127:124], Desc[31:30] [123:122],
     * Desc[19:11] [121:113], Reserved [112] = 0, Src1.RegNum
     * [111:104], Src1.Length = ex_desc[10:6] [103:99],
     * Src1.RegFile [98], ExDesc[27:26] [97:96]. */
    dw3 = ((ex_desc >> 28) << 28) | (((desc >> 30) & 0x3u) << 26) |
          (((desc >> 11) & 0x1FFu) << 17) |
          (src1_regnum << 8) | (ext_len << 3) |
          (src1_regfile << 2) | ((ex_desc >> 26) & 0x3u);

    return finish_inst(inst, dw0, dw1, dw2, dw3);
}

int vbs_alchemist_emit_nop(vbs_alchemist_inst_t *inst)
{
    int ok = 0;
    uint8_t opcode;

    if (inst == NULL)
        return -1;
    opcode = vbs_alchemist_opcode(VBS_ALCHEMIST_OP_NOP, &ok);
    if (!ok)
        return -1;
    memset(inst, 0, sizeof(*inst));
    inst->dw[0] = opcode;
    return 0;
}

int vbs_alchemist_list_line(char *buf, size_t len, uint32_t addr,
                            const vbs_alchemist_inst_t *inst,
                            const char *mnemonic, const char *operands,
                            const char *citation)
{
    if (buf == NULL || len == 0 || inst == NULL || mnemonic == NULL ||
        operands == NULL || citation == NULL)
        return -1;
    /* Encoding printed DW3..DW0 so the line reads like the PRM bit
     * tables (bits [127:96] first). */
    return snprintf(buf, len,
                    "0x%04x  %08x %08x %08x %08x  %-8s %-24s ; %s",
                    addr, inst->dw[3], inst->dw[2], inst->dw[1],
                    inst->dw[0], mnemonic, operands, citation);
}
