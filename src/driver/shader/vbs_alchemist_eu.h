/* vbs_alchemist_eu.h — minimal EU assembler for Intel Arc Alchemist
 * (Xe-HPG / DG2).
 *
 * Emits 128-bit EU instructions from structured descriptions: one
 * function per instruction kind, bit-field encoding per the ISA
 * facts in vbs_alchemist_eu_opcodes.h. The encoding is programmatic
 * and testable — no magic byte blobs.
 *
 * All code here is original Verum work. Opcode values come from the
 * table in vbs_alchemist_eu_opcodes.h (documented hardware facts,
 * cited per entry); nothing is transcribed from any other assembler.
 *
 * Instruction layout (digest §6.4; Vol 2a pp.219-224 for `add`,
 * pp.1111-1115 for `mov`, pp.1275-1279 for `shl`, pp.1253-1260 for
 * `send`):
 *   DW0  bits [31:0]:   opcode [6:0], ExecSize [18:16], header
 *                       [15:0] (Access Mode bit UNSOURCED — left 0).
 *   DW1  bits [63:32]:  Dst operand/type, src IsImm flags, src0 type.
 *   DW2  bits [95:64]:  src0 region, src1 type (ALU) / SFID + desc
 *                       low bits + src0 (send).
 *   DW3  bits [127:96]: src1 region or 32-bit immediate (ALU) /
 *                       desc high bits + src1 (send).
 * In memory DW0 comes first, each dword little-endian (byte 0 of
 * the instruction word holds the opcode — Vol 9, p.469).
 *
 * Registers: pass 0-127 for GRF r0-r127, VBS_ALCHEMIST_REG_NULL
 * (255) for the null register (send dst/src1 only). The null
 * register is ARF type 0000b, RegNum 0 (Vol 9, p.348).
 *
 * ExecSize is passed as the ENCODING (VBS_ALCHEMIST_EXECSIZE_*),
 * not the channel count, so the call sites read exactly what the
 * hardware sees.
 */

#ifndef VBS_ALCHEMIST_EU_H
#define VBS_ALCHEMIST_EU_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "vbs_alchemist_eu_opcodes.h"

/* One 128-bit instruction: dw[0] = bits [31:0] ... dw[3] = bits
 * [127:96]. Serialized DW0-first, little-endian per dword. */
typedef struct {
    uint32_t dw[4];
} vbs_alchemist_inst_t;

/* Null-register sentinel for send dst/src1 (GRF numbers are 0-127). */
#define VBS_ALCHEMIST_REG_NULL 255u

/* A register operand: GRF number, byte subregister, data type, and
 * Align1 region (VertStride/Width/HorzStride ENCODINGS, per the
 * VBS_ALCHEMIST_REGION_* presets in the opcodes header). */
typedef struct {
    uint8_t reg;     /* GRF number 0-127 */
    uint8_t subreg;  /* byte offset within the GRF, 0-31 */
    uint8_t type;    /* RegDataType (VBS_ALCHEMIST_TYPE_*) */
    uint8_t vstride; /* VertStride encoding (Vol 2b, p.126) */
    uint8_t width;   /* Width encoding (Vol 2b, p.127) */
    uint8_t hstride; /* HorzStride encoding (Vol 2b, p.82) */
} vbs_alchemist_operand_t;

/* Build an operand in one expression. */
static inline vbs_alchemist_operand_t
vbs_alchemist_op(uint8_t reg, uint8_t subreg, uint8_t type,
                 uint8_t vstride, uint8_t width, uint8_t hstride)
{
    vbs_alchemist_operand_t o;
    o.reg = reg;
    o.subreg = subreg;
    o.type = type;
    o.vstride = vstride;
    o.width = width;
    o.hstride = hstride;
    return o;
}

/* Destination operand: only reg/subreg/type/hstride are encoded
 * (destinations have no VertStride/Width fields). */
static inline vbs_alchemist_operand_t
vbs_alchemist_dst(uint8_t reg, uint8_t subreg, uint8_t type,
                  uint8_t hstride)
{
    return vbs_alchemist_op(reg, subreg, type, 0, 0, hstride);
}

/* Pack a DirectOperand (14 bits; Vol 2d, p.366): RegNum [13:6],
 * SubRegNum [5:1], RegFile [0] (1 = GRF, 0 = ARF). */
uint16_t vbs_alchemist_direct_operand(uint8_t reg, uint8_t subreg,
                                      int is_grf);

/* Look an opcode up in the table by kind. Returns 0 and sets *ok = 0
 * when the entry is UNSOURCED (never emit a guessed opcode). */
uint8_t vbs_alchemist_opcode(vbs_alchemist_op_t kind, int *ok);

/* Serialize one instruction to 16 bytes: DW0 first, each dword
 * little-endian (byte 0 holds the opcode — Vol 9, p.469). */
void vbs_alchemist_serialize(const vbs_alchemist_inst_t *inst,
                             uint8_t out[16]);

/* Build a message descriptor (desc, 32 bits) from its fields
 * (digest §6.6): Message Length [28:25], Response Length [24:20],
 * Header Present [19], Message Type [18:14], SIMD Mode [13:12],
 * Channel Mask [11:8], Binding Table Index [7:0]. Bits [31:29] are
 * 0 (packed-payload flags must be 0 — Vol 2a p.1340). */
uint32_t vbs_alchemist_desc_build(uint32_t msg_len, uint32_t resp_len,
                                  uint32_t header, uint32_t msg_type,
                                  uint32_t simd_mode, uint32_t cmask,
                                  uint32_t bti);

/* Build an extended message descriptor (ex_desc, 32 bits) from its
 * fields: Extended Message Length [10:6]; all other bits 0 for the
 * v1 messages (digest §9 item 3 — ex_desc low-bit semantics for
 * dataport messages are UNSOURCED beyond the Extended Message
 * Length; the first hardware run must confirm 0 is correct). */
uint32_t vbs_alchemist_exdesc_build(uint32_t ext_msg_len);

/* --- Emitters: one function per instruction kind. ----------------- */
/* Each returns 0 on success, -1 if an opcode lookup failed (an
 * UNSOURCED opcode is never emitted) or an operand is out of range.
 * All reserved/MBZ fields are left 0; all modifiers (SrcMod,
 * Saturate, PredCtrl, CondCtrl, ChanOff) are 0 (none). */

/* add (exec) dst src0 src1 — component-wise add, register sources.
 * Vol 2a, pp.219-224; digest §6.4. */
int vbs_alchemist_emit_add(vbs_alchemist_inst_t *inst, uint8_t exec_enc,
                           vbs_alchemist_operand_t dst,
                           vbs_alchemist_operand_t src0,
                           vbs_alchemist_operand_t src1);

/* mov (exec) dst src0 — move with optional type conversion; src1
 * fields are MBZ (bit 47 Reserved for mov, Vol 2a p.1113; DW3 = 0
 * for a register source). Vol 2a, pp.1111-1115; digest §6.4. */
int vbs_alchemist_emit_mov(vbs_alchemist_inst_t *inst, uint8_t exec_enc,
                           vbs_alchemist_operand_t dst,
                           vbs_alchemist_operand_t src0);

/* shl (exec) dst src0 imm32 — shift left by an immediate count.
 * Src1.IsImm = 1, DW3 = the immediate, Src1.DataType = imm_type.
 * Vol 2a, pp.1275-1279; digest §6.4. */
int vbs_alchemist_emit_shl_imm(vbs_alchemist_inst_t *inst,
                               uint8_t exec_enc,
                               vbs_alchemist_operand_t dst,
                               vbs_alchemist_operand_t src0,
                               uint32_t imm, uint8_t imm_type);

/* send (exec) dst src0 src1 — message to a shared function.
 * Vol 2a, pp.1253-1260; digest §6.5. desc/ex_desc are the full
 * 32-bit descriptors (built with vbs_alchemist_desc_build /
 * vbs_alchemist_exdesc_build); the emitter scatters their bits
 * into the instruction and derives the Length fields from them
 * (Dst.Length = desc[24:20], Src0.Length = desc[28:25],
 * Src1.Length = ex_desc[10:6]) so desc is the single source of
 * truth. dst_reg / src1_reg take VBS_ALCHEMIST_REG_NULL for the
 * null register (loads: src1 null; stores and EOT: dst null).
 * eot = 1 sets instruction bit 34 (EOT) — only for the terminating
 * send, which must use r112-r127 sources (Vol 2a, p.1255).
 *
 * ExecSize for send is DERIVED, not directly stated on the send
 * page: the send format is `send.<sfid> (exec_size)` and WrEn
 * (from ExecSize) is forwarded as the message channel enables
 * (Vol 2a, p.1254), so a SIMD16 message uses ExecSize 16 (all 16
 * lanes enabled) and the EOT gateway message uses ExecSize 1 (no
 * data channels). Confirm on the hardware run. */
int vbs_alchemist_emit_send(vbs_alchemist_inst_t *inst, uint8_t exec_enc,
                            uint8_t sfid, uint8_t dst_reg,
                            uint8_t src0_reg, uint8_t src1_reg,
                            uint32_t desc, uint32_t ex_desc, int eot);

/* nop — no operation (opcode 0x60, all other fields 0). */
int vbs_alchemist_emit_nop(vbs_alchemist_inst_t *inst);

/* --- Listing ------------------------------------------------------- */
/* Format one disassembly-style listing line:
 *   0xADDR  DW3 DW2 DW1 DW0  mnemonic operands  ; citation
 * The encoding is printed most-significant dword first (DW3..DW0)
 * so the line reads like the PRM bit tables (bits [127:96] first).
 * Returns the number of characters written (excluding NUL). */
int vbs_alchemist_list_line(char *buf, size_t len, uint32_t addr,
                            const vbs_alchemist_inst_t *inst,
                            const char *mnemonic, const char *operands,
                            const char *citation);

#endif /* VBS_ALCHEMIST_EU_H */
