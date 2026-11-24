/* vbs_turing_sass.c — minimal SASS assembler for Turing (sm_75).
 *
 * Original Verum code. Every encoding fact cites the spec digest
 * (SPEC-DIGEST.md §5) and/or the opcode table
 * (vbs_turing_sass_opcodes.h). Opcode values are looked up from
 * that table, never hardcoded here.
 *
 * Cross-checks used while writing this file (all from NVIDIA's
 * own published nvdisasm output in the CUDA Binary Utilities
 * doc, S5 — the sm_70 sample in the 12.4.1 archive and the
 * sm_100 sample in the current doc):
 *   LDG.E.SYS  -> lo bits 0-40 = 0x1EE900 (.E, .32, .SYS,
 *                  strong 1, ldgp PT, cache 1)
 *   STG.E.SYS  -> lo bits 0-40 = 0x10E900 (same, no ldgp)
 *   IADD3      -> lo bits 0-40 carry nopd0/nopd1/nops0/nops1
 *   IMAD(.WIDE)-> lo bits 0-40 carry nopd0/nops0, type .S32
 *   EXIT       -> lo bits 0-40 = 0x03800000 (cp = PT)
 *   LDC.64     -> lo bits 0-40 = 0xA00, LDC -> 0x800
 *   MOV const  -> lo bits 0-40 = 0xF00 (special rule)
 * The unit tests assert the emitters reproduce these values.
 */

#include "vbs_turing_sass.h"

/* ------------------------------------------------------------------ */
/* Bit positions (instruction-bit numbers; hi word = bits 64-127,    */
/* so a hi-word shift is the instruction bit minus 64).               */
/* Digest §5.2; S11.                                                   */
/* ------------------------------------------------------------------ */

#define OPCODE_SHIFT   0u   /* hi bits 0-11  = inst bits 64-75 */
#define PRED_SHIFT     12u  /* hi bits 12-15 = inst bits 76-79 */
#define RD_SHIFT       16u  /* hi bits 16-23 = inst bits 80-87 */
#define RS0_SHIFT      24u  /* hi bits 24-31 = inst bits 88-95 */
#define RS1_SHIFT      32u  /* hi bits 32-63 = inst bits 96-127 */
#define IS0W24_SHIFT   40u  /* hi bits 40-63 = inst bits 104-127 */
#define RS2_SHIFT      0u   /* lo bits 0-7 */
#define CTRL_SHIFT     41u  /* lo bits 41-63 */

/* ICR nibble lives at instruction bits 72-75 = hi bits 8-11. */
#define ICR_SHIFT      8u
#define ICR_MASK       0xF00u

/* Constant-operand encoding (S11 GetC): the value is shifted
 * left 6, then placed at the operand shift. cs1 at inst bit 96
 * (hi bit 32); bank (c34) at inst bit 112 (hi bit 48). */
#define CS1_SHIFT      32u
#define C34_SHIFT      48u

uint64_t vbs_turing_ctrl_pack(vbs_turing_ctrl_t ctrl) {
    /* Digest §5.2 / S11 ReadCtrl: stall | yield<<4 | wrtdb<<5 |
     * readb<<8 | watdb<<11. Fields are masked to their widths. */
    uint64_t v = 0;
    v |= (uint64_t)(ctrl.stall & 0xFu);
    v |= (uint64_t)(ctrl.yield & 0x1u) << 4;
    v |= (uint64_t)(ctrl.wrtdb & 0x7u) << 5;
    v |= (uint64_t)(ctrl.readb & 0x7u) << 8;
    v |= (uint64_t)(ctrl.watdb & 0x1Fu) << 11;
    return v;
}

uint16_t vbs_turing_opcode(vbs_turing_op_t kind, int *ok) {
    if (kind < 0 || kind >= VBS_TURING_OP_COUNT) {
        if (ok) *ok = 0;
        return 0;
    }
    if (!VBS_TURING_OPCODE_TABLE[kind].sourced) {
        /* UNSOURCED entry: never emit a guessed opcode. */
        if (ok) *ok = 0;
        return 0;
    }
    if (ok) *ok = 1;
    return VBS_TURING_OPCODE_TABLE[kind].code;
}

void vbs_turing_sass_serialize(const vbs_turing_inst_t *inst,
                               uint8_t out[16]) {
    /* Instruction word first, then control word, each
     * little-endian (see header). */
    uint64_t hi = inst->hi;
    uint64_t lo = inst->lo;
    for (int i = 0; i < 8; i++) {
        out[i] = (uint8_t)(hi & 0xFFu);
        hi >>= 8;
        out[8 + i] = (uint8_t)(lo & 0xFFu);
        lo >>= 8;
    }
}

/* ------------------------------------------------------------------ */
/* Internal builders                                                   */
/* ------------------------------------------------------------------ */

/* Start an instruction: opcode + predicate in the hi word,
 * control code in the lo word. */
static vbs_turing_inst_t begin_inst(vbs_turing_op_t kind,
                                    unsigned pred,
                                    vbs_turing_ctrl_t ctrl) {
    int ok = 0;
    uint16_t code = vbs_turing_opcode(kind, &ok);
    vbs_turing_inst_t in;
    /* The table lookup cannot fail for the kinds the emitters
     * use (all sourced); a failure would be a programming error,
     * so emit a NOP-shaped zero rather than a guessed opcode. */
    in.hi = (uint64_t)(ok ? code : 0);
    in.hi |= (uint64_t)(pred & 0xFu) << PRED_SHIFT;
    in.lo = vbs_turing_ctrl_pack(ctrl) << CTRL_SHIFT;
    return in;
}

/* Apply an ICR value: replace the opcode's top nibble
 * (inst bits 72-75) with the operand-type selector. S11 clears
 * only bit 73 and ORs; masking the whole nibble is the clean
 * equivalent and agrees on every cross-checked encoding
 * (LDG 0x381+0x9 -> 0x981, MOV 0x202+0xa -> 0xa02,
 * IMAD.WIDE 0x225+0x8 -> 0x825). */
static void apply_icr(vbs_turing_inst_t *in, unsigned icr) {
    in->hi = (in->hi & ~(uint64_t)ICR_MASK) |
             ((uint64_t)(icr & 0xFu) << ICR_SHIFT);
}

/* Constant-buffer operand: (value << 6) at the operand shift. */
static uint64_t const_bits(unsigned value, unsigned shift) {
    return ((uint64_t)value << 6) << shift;
}

/* --- Emitters ------------------------------------------------------ */

vbs_turing_inst_t vbs_turing_emit_ldc(unsigned rd, unsigned bank,
                                      unsigned offset, int is64,
                                      unsigned pred,
                                      vbs_turing_ctrl_t ctrl) {
    vbs_turing_inst_t in = begin_inst(VBS_TURING_OP_LDC, pred, ctrl);
    /* rd at 80; special rule: 0xFF at rs0 (bits 88-95). */
    in.hi |= (uint64_t)(rd & 0xFFu) << RD_SHIFT;
    in.hi |= (uint64_t)0xFFu << RS0_SHIFT;
    in.hi |= const_bits(offset, CS1_SHIFT);
    in.hi |= const_bits(bank, C34_SHIFT);
    /* Type: .64 = 5, .32 = 4 at bits 9-11. */
    in.lo |= (uint64_t)(is64 ? VBS_TURING_TYPE_64
                             : VBS_TURING_TYPE_32)
             << VBS_TURING_TYPE_SHIFT;
    return in;
}

vbs_turing_inst_t vbs_turing_emit_s2r(unsigned rd, unsigned sr,
                                      unsigned pred,
                                      vbs_turing_ctrl_t ctrl) {
    vbs_turing_inst_t in = begin_inst(VBS_TURING_OP_S2R, pred, ctrl);
    in.hi |= (uint64_t)(rd & 0xFFu) << RD_SHIFT;
    /* Special-register selector at lo bits 8-16. */
    in.lo |= (uint64_t)(sr & 0x1FFu) << VBS_TURING_SR_SHIFT;
    return in;
}

/* Fill the IMAD operands shared by the register and wide-immediate
 * forms: rd, rs0, rs2, the .S32 type, and the no-predicate
 * fillers. The caller sets rs1 (register or immediate) and the
 * ICR bit; all fields are disjoint, so the order does not matter. */
static void imad_operands(vbs_turing_inst_t *in, unsigned rd,
                          unsigned rs0, unsigned rs2) {
    in->hi |= (uint64_t)(rd & 0xFFu) << RD_SHIFT;
    in->hi |= (uint64_t)(rs0 & 0xFFu) << RS0_SHIFT;
    in->lo |= (uint64_t)(rs2 & 0xFFu) << RS2_SHIFT;
    /* Type .S32 (default) at bit 9; no-predicate fillers. */
    in->lo |= (uint64_t)VBS_TURING_INT_TYPE_S32
              << VBS_TURING_INT_TYPE_SHIFT;
    in->lo |= VBS_TURING_NOPD0 | VBS_TURING_NOPS0;
}

/* The memory-operation flags LDG and STG share: .E, type .32,
 * scope .SYS, strong default, cache default. LDG adds its ldgp
 * predicate bit after the call. */
static void mem_op_flags(vbs_turing_inst_t *in) {
    in->lo |= 1u << VBS_TURING_E_BIT;
    in->lo |= (uint64_t)VBS_TURING_TYPE_32 << VBS_TURING_TYPE_SHIFT;
    in->lo |= (uint64_t)VBS_TURING_SCOPE_SYS
              << VBS_TURING_SCOPE_SHIFT;
    in->lo |= (uint64_t)VBS_TURING_STRONG_DEFAULT
              << VBS_TURING_STRONG_SHIFT;
    in->lo |= (uint64_t)VBS_TURING_CACHE_DEFAULT
              << VBS_TURING_CACHE_SHIFT;
}

vbs_turing_inst_t vbs_turing_emit_imad(unsigned rd, unsigned rs0,
                                       unsigned rs1, unsigned rs2,
                                       unsigned pred,
                                       vbs_turing_ctrl_t ctrl) {
    vbs_turing_inst_t in = begin_inst(VBS_TURING_OP_IMAD, pred, ctrl);
    imad_operands(&in, rd, rs0, rs2);
    in.hi |= (uint64_t)(rs1 & 0xFFu) << RS1_SHIFT;
    return in;
}

vbs_turing_inst_t vbs_turing_emit_imad_wide_imm(
        unsigned rd, unsigned rs0, uint32_t imm, unsigned rs2,
        unsigned pred, vbs_turing_ctrl_t ctrl) {
    vbs_turing_inst_t in = begin_inst(VBS_TURING_OP_IMAD_WIDE,
                                      pred, ctrl);
    /* Immediate at rs1 -> ICR 0x8 (S5 sm_100: 0x225 -> 0x825). */
    apply_icr(&in, VBS_TURING_ICR_IS1);
    imad_operands(&in, rd, rs0, rs2);
    in.hi |= (uint64_t)imm << RS1_SHIFT;
    return in;
}

vbs_turing_inst_t vbs_turing_emit_iadd3(unsigned rd, unsigned rs0,
                                        unsigned rs1, unsigned rs2,
                                        unsigned pred,
                                        vbs_turing_ctrl_t ctrl) {
    vbs_turing_inst_t in = begin_inst(VBS_TURING_OP_IADD3, pred, ctrl);
    in.hi |= (uint64_t)(rd & 0xFFu) << RD_SHIFT;
    in.hi |= (uint64_t)(rs0 & 0xFFu) << RS0_SHIFT;
    in.hi |= (uint64_t)(rs1 & 0xFFu) << RS1_SHIFT;
    in.lo |= (uint64_t)(rs2 & 0xFFu) << RS2_SHIFT;
    in.lo |= VBS_TURING_NOPD0 | VBS_TURING_NOPD1 |
             VBS_TURING_NOPS0 | VBS_TURING_NOPS1;
    return in;
}

vbs_turing_inst_t vbs_turing_emit_iadd3_imm(
        unsigned rd, unsigned rs0, uint32_t imm, unsigned rs2,
        unsigned pred, vbs_turing_ctrl_t ctrl) {
    vbs_turing_inst_t in = begin_inst(VBS_TURING_OP_IADD3, pred, ctrl);
    /* Immediate at rs1 -> ICR 0x8 (digest §5.3 worked example:
     * IADD3 R0, R1, 0x10, R2). */
    apply_icr(&in, VBS_TURING_ICR_IS1);
    in.hi |= (uint64_t)(rd & 0xFFu) << RD_SHIFT;
    in.hi |= (uint64_t)(rs0 & 0xFFu) << RS0_SHIFT;
    in.hi |= (uint64_t)imm << RS1_SHIFT;
    in.lo |= (uint64_t)(rs2 & 0xFFu) << RS2_SHIFT;
    in.lo |= VBS_TURING_NOPD0 | VBS_TURING_NOPD1 |
             VBS_TURING_NOPS0 | VBS_TURING_NOPS1;
    return in;
}

vbs_turing_inst_t vbs_turing_emit_ldg(unsigned rd, unsigned rs0,
                                      uint32_t offset24,
                                      unsigned pred,
                                      vbs_turing_ctrl_t ctrl) {
    vbs_turing_inst_t in = begin_inst(VBS_TURING_OP_LDG, pred, ctrl);
    in.hi |= (uint64_t)(rd & 0xFFu) << RD_SHIFT;
    in.hi |= (uint64_t)(rs0 & 0xFFu) << RS0_SHIFT;
    in.hi |= (uint64_t)(offset24 & 0xFFFFFFu) << IS0W24_SHIFT;
    /* .E | type .32 | scope .SYS | strong default |
     * ldgp PT | cache default. Reproduces the S5 sm_70
     * LDG.E.SYS lower word (bits 0-40 = 0x1EE900). */
    mem_op_flags(&in);
    in.lo |= (uint64_t)VBS_TURING_PRED_PT << 17; /* ldgp */
    return in;
}

vbs_turing_inst_t vbs_turing_emit_stg(unsigned rs0, unsigned rs1,
                                      uint32_t offset24,
                                      unsigned pred,
                                      vbs_turing_ctrl_t ctrl) {
    vbs_turing_inst_t in = begin_inst(VBS_TURING_OP_STG, pred, ctrl);
    /* No rd for a store; address in rs0, data in rs1. */
    in.hi |= (uint64_t)(rs0 & 0xFFu) << RS0_SHIFT;
    in.hi |= (uint64_t)(rs1 & 0xFFu) << RS1_SHIFT;
    in.hi |= (uint64_t)(offset24 & 0xFFFFFFu) << IS0W24_SHIFT;
    /* Same flags as LDG except no ldgp. Reproduces the S5
     * sm_70 STG.E.SYS lower word (bits 0-40 = 0x10E900). */
    mem_op_flags(&in);
    return in;
}

vbs_turing_inst_t vbs_turing_emit_fadd(unsigned rd, unsigned rs0,
                                       unsigned rs1, unsigned pred,
                                       vbs_turing_ctrl_t ctrl) {
    vbs_turing_inst_t in = begin_inst(VBS_TURING_OP_FADD, pred, ctrl);
    in.hi |= (uint64_t)(rd & 0xFFu) << RD_SHIFT;
    in.hi |= (uint64_t)(rs0 & 0xFFu) << RS0_SHIFT;
    in.hi |= (uint64_t)(rs1 & 0xFFu) << RS1_SHIFT;
    /* No flags, no rs2 (S5 sm_100 FADD lower word = 0). */
    return in;
}

vbs_turing_inst_t vbs_turing_emit_mov_reg(unsigned rd, unsigned rs1,
                                          unsigned pred,
                                          vbs_turing_ctrl_t ctrl) {
    vbs_turing_inst_t in = begin_inst(VBS_TURING_OP_MOV, pred, ctrl);
    in.hi |= (uint64_t)(rd & 0xFFu) << RD_SHIFT;
    in.hi |= (uint64_t)(rs1 & 0xFFu) << RS1_SHIFT;
    /* Special rule: MOV ORs 0xF00 (digest §5.3). */
    in.lo |= 0xF00u;
    return in;
}

vbs_turing_inst_t vbs_turing_emit_mov_const(unsigned rd, unsigned bank,
                                            unsigned offset,
                                            unsigned pred,
                                            vbs_turing_ctrl_t ctrl) {
    vbs_turing_inst_t in = begin_inst(VBS_TURING_OP_MOV, pred, ctrl);
    /* Constant at rs1 -> ICR 0xa (S5 sm_70 MOV R3,c[0x0][0x164]
     * -> opcode field 0xa02). */
    apply_icr(&in, VBS_TURING_ICR_CS1);
    in.hi |= (uint64_t)(rd & 0xFFu) << RD_SHIFT;
    in.hi |= const_bits(offset, CS1_SHIFT);
    in.hi |= const_bits(bank, C34_SHIFT);
    in.lo |= 0xF00u;
    return in;
}

vbs_turing_inst_t vbs_turing_emit_isetp_ge_u32(
        unsigned pdst, unsigned rs0, unsigned rs1,
        unsigned pred, vbs_turing_ctrl_t ctrl) {
    vbs_turing_inst_t in = begin_inst(VBS_TURING_OP_ISETP, pred, ctrl);
    in.hi |= (uint64_t)(rs0 & 0xFFu) << RS0_SHIFT;
    in.hi |= (uint64_t)(rs1 & 0xFFu) << RS1_SHIFT;
    /* Special rule: ISETP ORs 0x7 << 4. Compare .GE at bits
     * 12-14, type .U32 (0, no bits), boolOp .AND (0, no bits).
     * Destinations: pd0 = pdst (bits 17-19), pd1 = PT
     * (bits 20-22); source predicate ps0 = PT (bits 23-25). */
    in.lo |= (uint64_t)0x7u << 4;
    in.lo |= (uint64_t)VBS_TURING_CMP_GE << VBS_TURING_CMP_SHIFT;
    in.lo |= (uint64_t)(pdst & 0x7u) << 17;
    in.lo |= (uint64_t)VBS_TURING_PRED_PT << 20;
    in.lo |= (uint64_t)VBS_TURING_PRED_PT << 23;
    return in;
}

vbs_turing_inst_t vbs_turing_emit_bar_sync(unsigned bar_id,
                                           unsigned pred,
                                           vbs_turing_ctrl_t ctrl) {
    vbs_turing_inst_t in = begin_inst(VBS_TURING_OP_BAR, pred, ctrl);
    /* .SYNC flag = 0; barrier id at inst bits 118-125
     * (hi bits 54-61). */
    in.hi |= (uint64_t)(bar_id & 0xFFu) << 54;
    return in;
}

vbs_turing_inst_t vbs_turing_emit_exit(unsigned pred,
                                       vbs_turing_ctrl_t ctrl) {
    vbs_turing_inst_t in = begin_inst(VBS_TURING_OP_EXIT, pred, ctrl);
    /* Control predicate cp = PT at bits 23-25 (S5 sm_70/sm_100
     * EXIT lower word = 0x03800000). */
    in.lo |= (uint64_t)VBS_TURING_PRED_PT << 23;
    return in;
}

vbs_turing_inst_t vbs_turing_emit_bra(int32_t offset, unsigned pred,
                                      vbs_turing_ctrl_t ctrl) {
    vbs_turing_inst_t in = begin_inst(VBS_TURING_OP_BRA, pred, ctrl);
    in.hi |= (uint64_t)(uint32_t)offset << RS1_SHIFT;
    /* Negative offsets set bits 0-17 to 0x3FFFF (S11 is1neg
     * flag; S5 sm_70 BRA 0xd0 lower word = 0x0383FFFF). */
    if (offset < 0)
        in.lo |= 0x3FFFFu;
    in.lo |= (uint64_t)VBS_TURING_PRED_PT << 23; /* cp = PT */
    return in;
}

vbs_turing_inst_t vbs_turing_emit_nop(unsigned pred,
                                      vbs_turing_ctrl_t ctrl) {
    return begin_inst(VBS_TURING_OP_NOP, pred, ctrl);
}
