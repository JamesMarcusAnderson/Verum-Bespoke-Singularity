/* vbs_turing_sass.h — minimal SASS assembler for Turing (sm_75).
 *
 * Emits 128-bit Volta/Turing SASS instructions from structured
 * descriptions: one function per instruction kind, bit-field
 * encoding per the ISA facts in vbs_turing_sass_opcodes.h. The
 * encoding is programmatic and testable — no magic byte blobs.
 *
 * All code here is original Verum work. Opcode values come from
 * the table in vbs_turing_sass_opcodes.h (documented hardware
 * facts, cited per entry); nothing is transcribed from any other
 * assembler.
 *
 * Instruction layout (digest §5.2; S11):
 *   hi word (bits 64-127): opcode[64-75], predicate[76-79],
 *       rd[80-87], rs0[88-95], rs1/is1/cs1[96-127].
 *   lo word (bits 0-63): rs2[0-7], instruction flags[8-40],
 *       control code[41-63].
 * In memory the hi word is stored first (little-endian u64),
 * then the lo word (see vbs_turing_sass_serialize).
 *
 * Registers: pass 0-254 for R0-R254, 255 for RZ (the zero
 * register, encoded 0xFF). Uniform registers are not used by
 * the emitters in this file (the kernel is uniform-free).
 *
 * Predicate: pass 7 for PT (always true, the usual case),
 * 0-6 for P0-P6, or 8-15 for the negated form (!P0-!P6, !PT).
 */

#ifndef VBS_TURING_SASS_H
#define VBS_TURING_SASS_H

#include <stddef.h>
#include <stdint.h>

#include "vbs_turing_sass_opcodes.h"

/* One 128-bit instruction: hi = instruction word (bits 64-127),
 * lo = control word (bits 0-63). */
typedef struct {
    uint64_t hi;
    uint64_t lo;
} vbs_turing_inst_t;

/* Scheduling control code (bits 41-63 of the lo word).
 *   stall  0-15: stall cycles before the next instruction issues.
 *   yield  0 = yield the warp slot, 1 = no yield.
 *   wrtdb  0-6 = dependency barrier this instruction writes,
 *          7 = none.
 *   readb  0-6 = dependency barrier this instruction reads,
 *          7 = none.
 *   watdb  wait-barrier bitmask (bit i = wait for barrier i);
 *          0 = wait for nothing.
 * Digest §5.2; S11 ReadCtrl layout (stall | yield<<4 | wrtdb<<5
 * | readb<<8 | watdb<<11, placed at bit 41). */
typedef struct {
    unsigned stall;
    unsigned yield;
    unsigned wrtdb;
    unsigned readb;
    unsigned watdb;
} vbs_turing_ctrl_t;

/* Default control: stall 1, no yield, no barriers, no wait.
 * (Stall 1 is the minimum pipeline stall; digest §5.2 / S11.) */
#define VBS_TURING_CTRL_DEFAULT ((vbs_turing_ctrl_t){ 1, 1, 7, 7, 0 })

/* Pack a control struct into the 23-bit control field (before
 * the <<41 shift into the lo word). */
uint64_t vbs_turing_ctrl_pack(vbs_turing_ctrl_t ctrl);

/* Look an opcode up in the table by kind. Returns 0 and sets
 * *ok = 0 when the entry is UNSOURCED (never emit a guessed
 * opcode). */
uint16_t vbs_turing_opcode(vbs_turing_op_t kind, int *ok);

/* Serialize one instruction to 16 bytes: hi word first, then lo
 * word, each little-endian (turingas cubin.py GenerateText;
 * NVIDIA forum: the control word follows the instruction word). */
void vbs_turing_sass_serialize(const vbs_turing_inst_t *inst,
                               uint8_t out[16]);

/* --- Emitters: one function per instruction kind. ---------------- */
/* Each takes the instruction predicate (7 = PT) and a control
 * code. Every encoding fact cites digest §5 / the opcode table. */

/* LDC[.64] rd, c[bank][offset] — load from a constant buffer.
 * is64: 1 for a 64-bit load (register pair rd:rd+1), 0 for 32-bit.
 * Digest §5.3 (LDC special rule 0xFF<<88); S5 sm_100 LDC.64. */
vbs_turing_inst_t vbs_turing_emit_ldc(unsigned rd, unsigned bank,
                                      unsigned offset, int is64,
                                      unsigned pred,
                                      vbs_turing_ctrl_t ctrl);

/* S2R rd, SR_x — read a special register (TID/CTAID/LANEID).
 * sr is one of VBS_TURING_SR_* (bits 8-16 of the lo word). */
vbs_turing_inst_t vbs_turing_emit_s2r(unsigned rd, unsigned sr,
                                      unsigned pred,
                                      vbs_turing_ctrl_t ctrl);

/* IMAD rd, rs0, rs1, rs2 — 32-bit integer multiply-add,
 * rd = rs0 * rs1 + rs2 (all-register form, .S32 default type). */
vbs_turing_inst_t vbs_turing_emit_imad(unsigned rd, unsigned rs0,
                                       unsigned rs1, unsigned rs2,
                                       unsigned pred,
                                       vbs_turing_ctrl_t ctrl);

/* IMAD.WIDE rd, rs0, imm, rs2 — 64-bit multiply-add with a
 * 32-bit immediate multiplier: rd:rd+1 = rs0 * imm + rs2:rs2+1
 * (rs2 is a 64-bit register pair). ICR 0x8 (immediate at rs1). */
vbs_turing_inst_t vbs_turing_emit_imad_wide_imm(
        unsigned rd, unsigned rs0, uint32_t imm, unsigned rs2,
        unsigned pred, vbs_turing_ctrl_t ctrl);

/* IADD3 rd, rs0, rs1, rs2 — 32-bit integer 3-input add. */
vbs_turing_inst_t vbs_turing_emit_iadd3(unsigned rd, unsigned rs0,
                                        unsigned rs1, unsigned rs2,
                                        unsigned pred,
                                        vbs_turing_ctrl_t ctrl);

/* IADD3 rd, rs0, imm, rs2 — immediate at rs1 (ICR 0x8).
 * Digest §5.3 worked example. */
vbs_turing_inst_t vbs_turing_emit_iadd3_imm(
        unsigned rd, unsigned rs0, uint32_t imm, unsigned rs2,
        unsigned pred, vbs_turing_ctrl_t ctrl);

/* LDG.E.32.SYS rd, [rs0+offset] — 32-bit global load.
 * Default strength/cache, system scope, PT load-predicate.
 * The address is a 64-bit register pair (rs0:rs0+1); the
 * optional 24-bit immediate offset is added by the hardware. */
vbs_turing_inst_t vbs_turing_emit_ldg(unsigned rd, unsigned rs0,
                                      uint32_t offset24,
                                      unsigned pred,
                                      vbs_turing_ctrl_t ctrl);

/* STG.E.32.SYS [rs0+offset], rs1 — 32-bit global store. */
vbs_turing_inst_t vbs_turing_emit_stg(unsigned rs0, unsigned rs1,
                                      uint32_t offset24,
                                      unsigned pred,
                                      vbs_turing_ctrl_t ctrl);

/* FADD rd, rs0, rs1 — IEEE-754 single-precision add. */
vbs_turing_inst_t vbs_turing_emit_fadd(unsigned rd, unsigned rs0,
                                       unsigned rs1, unsigned pred,
                                       vbs_turing_ctrl_t ctrl);

/* MOV rd, rs1 — register copy (special rule: OR 0xF00). */
vbs_turing_inst_t vbs_turing_emit_mov_reg(unsigned rd, unsigned rs1,
                                          unsigned pred,
                                          vbs_turing_ctrl_t ctrl);

/* MOV rd, c[bank][offset] — constant load through MOV
 * (ICR 0xa for the constant source; S5 sm_70 sample). */
vbs_turing_inst_t vbs_turing_emit_mov_const(unsigned rd, unsigned bank,
                                            unsigned offset,
                                            unsigned pred,
                                            vbs_turing_ctrl_t ctrl);

/* ISETP.GE.U32.AND pdst, PT, rs0, rs1, PT — unsigned compare:
 * P(pdst) = (rs0 >= rs1). The second destination and the source
 * predicate are both PT (compare result used as-is). */
vbs_turing_inst_t vbs_turing_emit_isetp_ge_u32(
        unsigned pdst, unsigned rs0, unsigned rs1,
        unsigned pred, vbs_turing_ctrl_t ctrl);

/* BAR.SYNC id — CTA-wide barrier arrival + wait. Thread-count
 * form is UNSOURCED (see opcode-table note); this emits the
 * id-only form ptxas uses for bar.sync 0. */
vbs_turing_inst_t vbs_turing_emit_bar_sync(unsigned bar_id,
                                           unsigned pred,
                                           vbs_turing_ctrl_t ctrl);

/* EXIT — terminate the thread. pred = 7 for unconditional exit,
 * 0-6 for @P exit (the bounds-check early exit). */
vbs_turing_inst_t vbs_turing_emit_exit(unsigned pred,
                                       vbs_turing_ctrl_t ctrl);

/* BRA offset — branch to (pc + 16 + offset), offset in bytes.
 * The caller computes the offset from the target address. */
vbs_turing_inst_t vbs_turing_emit_bra(int32_t offset, unsigned pred,
                                      vbs_turing_ctrl_t ctrl);

/* NOP — no operation. */
vbs_turing_inst_t vbs_turing_emit_nop(unsigned pred,
                                      vbs_turing_ctrl_t ctrl);

#endif /* VBS_TURING_SASS_H */
