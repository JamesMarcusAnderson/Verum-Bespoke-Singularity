/* vbs_rdna2_asm.h — minimal RDNA 2 (gfx1030) assembler for Verum.
 *
 * Emits RDNA 2 instructions from structured descriptions: one
 * function per instruction kind, bit-field encoding per the ISA
 * facts in vbs_rdna2_isa.h. The encoding is programmatic and
 * testable — no magic byte blobs. Opcode numbers come only from
 * the opcode table (vbs_rdna2_opcode); this file contains no
 * opcode literals.
 *
 * All code here is original Verum work. Instruction formats and
 * field positions come from the AMD RDNA 2 ISA Reference Guide
 * via SPEC-DIGEST.md §5.2 (cited per emitter); nothing is
 * transcribed from Mesa, ROCm, LLVM's AMDGPU backend, or any
 * other assembler.
 *
 * UNSOURCED item (digest §10 item 5): the ADDTID instructions
 * (GLOBAL_LOAD/STORE_DWORD_ADDTID) supply no VGPR address —
 * "No VGPR address is supplied in this instruction"
 * ([RDNA2-ISA] §12.8) — but no public document states what
 * value the unused ADDR field must hold. The emitters below
 * write 0, the conservative choice (a field the hardware is
 * told to ignore should be zero). DERIVATION PROCEDURE for the
 * hardware run: dispatch the vecadd kernel with ADDR = 0; if
 * the dispatch faults or returns wrong data, vary the field
 * (0x7F, then the VDST VGPR number) and record which value the
 * hardware accepts. See the report.
 *
 * Instruction representation: RDNA 2 instructions are 32 or
 * 64 bits. vbs_rdna2_inst_t holds both dwords; dw1 is only
 * meaningful when is64 is set. Serialization is little-endian,
 * dw0 first.
 *
 * Register arguments: pass plain register numbers (SGPR 0-105,
 * VGPR 0-255). VOP source operands take operand CODES
 * (VBS_RDNA2_SRC_* from vbs_rdna2_isa.h), because a VOP source
 * may be an SGPR, a VGPR, a special register, or an inline
 * constant — the caller builds the code with the SRC macros.
 *
 * SMEM alignment rule ([RDNA2-ISA] §7.1, p.51; digest §5.2):
 * "Reads of two Dwords must have an even SDST-sgpr. Reads of
 * four or more Dwords must have their DST-gpr aligned to a
 * multiple of 4." The SMEM emitter enforces this and returns
 * the all-zero instruction (an invalid encoding, never a valid
 * one) when the caller violates it. The all-zero instruction
 * is also the return value for any other invalid input, so a
 * caller that checks for it can never silently emit a wrong
 * instruction.
 */

#ifndef VBS_RDNA2_ASM_H
#define VBS_RDNA2_ASM_H

#include <stddef.h>
#include <stdint.h>

#include "vbs_rdna2_isa.h"

/* One RDNA 2 instruction: dw0 always present, dw1 only when
 * is64 is set (64-bit formats: SMEM, VOP3A, VOP3B, FLAT). */
typedef struct {
    uint32_t dw0;
    uint32_t dw1;
    int is64;
} vbs_rdna2_inst_t;

/* The all-zero instruction: returned by emitters on invalid
 * input (only the SMEM emitter and the generic VOP3A emitter
 * can fail — see their docs). Callers that need to distinguish
 * failure check the return against this value before emitting
 * further instructions; the vecadd kernel never hits the
 * failure path (its inputs are fixed and valid). */
#define VBS_RDNA2_INST_INVALID ((vbs_rdna2_inst_t){ 0, 0, 0 })

/* Size of an instruction in bytes: 4 or 8. */
static inline size_t vbs_rdna2_inst_size(const vbs_rdna2_inst_t *inst) {
    return inst->is64 ? 8u : 4u;
}

/* Serialize one instruction to 4 or 8 bytes, little-endian,
 * dw0 first. Returns the byte count written. */
size_t vbs_rdna2_inst_serialize(const vbs_rdna2_inst_t *inst,
                                uint8_t *out);

/* --- SOPP emitters ----------------------------------------------------
 * SOPP (32-bit) [RDNA2-ISA] §13.1.5, p.245; digest §5.2:
 *   SIMM16 [15:0], OP [22:16], ENCODING [31:23] = 0x17F. */

/* S_ENDPGM — terminate the wavefront. "S_ENDPGM implicitly
 * executes S_WAITCNT 0 first" [RDNA2-ISA] §13.1.5. SIMM16 = 0. */
vbs_rdna2_inst_t vbs_rdna2_emit_s_endpgm(void);

/* S_BARRIER — synchronize the waves of a thread group. SIMM16 = 0. */
vbs_rdna2_inst_t vbs_rdna2_emit_s_barrier(void);

/* S_WAITCNT — wait until the outstanding-operation counters are
 * at or below the given levels. SIMM16 fields [RDNA2-ISA]
 * §13.1.5 / §12.2: vmcnt {SIMM16[15:14], SIMM16[3:0]} (6 bits),
 * expcnt SIMM16[6:4] (3 bits), lgkmcnt SIMM16[13:8] (6 bits).
 * S_WAITCNT 0 (all zero) waits for everything. Note: VMCNT
 * counts vector-memory LOADS only; vector stores are tracked by
 * the separate VS_CNT counter ([RDNA2-ISA] §4.4), so a waitcnt
 * does not by itself prove a store reached L2 — the wave's
 * retirement at S_ENDPGM is what drains stores. */
vbs_rdna2_inst_t vbs_rdna2_emit_s_waitcnt(unsigned vmcnt,
                                          unsigned expcnt,
                                          unsigned lgkmcnt);

/* --- SMEM emitters ----------------------------------------------------
 * SMEM (64-bit) [RDNA2-ISA] §13.2.1, p.246; §7.1, p.51; digest
 * §5.2:
 *   DW0: SBASE [5:0] (SGPR-pair number — LSB omitted, so the
 *        pair is s[SBASE*2 : SBASE*2+1]), SDATA [12:6], DLC [14],
 *        GLC [16], OP [25:18], ENCODING [31:26] = 0x3D.
 *   DW1: OFFSET [52:32] (signed 21-bit byte offset), SOFFSET
 *        [63:57] (SGPR supplying an unsigned byte offset;
 *        125 = NULL disables it).
 *   Address = SGPR[base] + inst_offset + { M0 or SGPR[offset]
 *        or zero } [RDNA2-ISA] §7.2.1, p.52. */

/* S_LOAD_DWORD* — load 1, 2, 4, 8 or 16 dwords from scalar
 * memory into SGPRs starting at sdata.
 *   sdata: first destination SGPR (alignment rule above: even
 *     for 2 dwords, multiple of 4 for 4+ dwords).
 *   sbase_pair: SBASE field = sbase_sgpr / 2 (the pair number).
 *     Pass the PAIR number, e.g. 1 for s[2:3].
 *   offset: signed 21-bit immediate byte offset.
 *   soffset: SGPR number supplying an extra byte offset, or
 *     VBS_RDNA2_SRC_NULL (125) to disable.
 *   dwords: 1, 2, 4, 8 or 16.
 * Returns VBS_RDNA2_INST_INVALID when dwords is not one of
 * those values or the alignment rule is violated. */
vbs_rdna2_inst_t vbs_rdna2_emit_s_load_dwordx(unsigned sdata,
                                              unsigned sbase_pair,
                                              int32_t offset,
                                              unsigned soffset,
                                              unsigned dwords);

/* Convenience wrapper: S_LOAD_DWORDX2 with SOFFSET = NULL. */
vbs_rdna2_inst_t vbs_rdna2_emit_s_load_dwordx2(unsigned sdata,
                                               unsigned sbase_pair,
                                               int32_t offset);

/* --- VOP1 emitter -----------------------------------------------------
 * VOP1 (32-bit) [RDNA2-ISA] §13.3.2, pp.249-250; digest §5.2:
 *   SRC0 [8:0], OP [16:9], VDST [24:17], ENCODING [31:25] = 0x3F. */

/* V_MOV_B32 vdst, src0 — copy src0 (operand code) to a VGPR. */
vbs_rdna2_inst_t vbs_rdna2_emit_v_mov_b32(unsigned vdst,
                                          unsigned src0);

/* --- VOP2 emitter -----------------------------------------------------
 * VOP2 (32-bit) [RDNA2-ISA] §13.3.1, pp.247-248; digest §5.2:
 *   SRC0 [8:0], VSRC1 [16:9], VDST [24:17], OP [30:25],
 *   ENCODING [31] = 0. */

/* V_ADD_NC_U32 vdst, src0, vsrc1 — 32-bit add, no carry in/out.
 * src0 is an operand code; vsrc1 is a plain VGPR number. */
vbs_rdna2_inst_t vbs_rdna2_emit_v_add_nc_u32(unsigned vdst,
                                             unsigned src0,
                                             unsigned vsrc1);

/* --- VOP3A emitter ----------------------------------------------------
 * VOP3A (64-bit) [RDNA2-ISA] §13.3.4, p.257; digest §5.2:
 *   DW0: VDST [7:0], ABS [10:8], OPSEL [14:11], CLMP [15],
 *        OP [25:16], ENCODING [31:26] = 0x35.
 *   DW1: SRC0 [40:32], SRC1 [49:41], SRC2 [58:50], OMOD [60:59],
 *        NEG [63:61].
 * ABS/OPSEL/CLMP/OMOD/NEG are all zero here (plain integer ops
 * need no modifiers). */

/* Generic VOP3A emitter for any opcode-table entry whose format
 * is VOP3A (native VOP3A ops and VOP1/VOP2 re-encodes). src0/1/2
 * are operand codes. Returns VBS_RDNA2_INST_INVALID when the
 * kind is not a VOP3A entry. */
vbs_rdna2_inst_t vbs_rdna2_emit_vop3a(vbs_rdna2_op_t kind,
                                      unsigned vdst,
                                      unsigned src0,
                                      unsigned src1,
                                      unsigned src2);

/* --- VOP3B emitter ----------------------------------------------------
 * VOP3B (64-bit) [RDNA2-ISA] §13.3.5, pp.260-261; digest §5.2:
 *   DW0: VDST [7:0], SDST [14:8] (scalar destination — VCC for
 *        the carry), CLMP [15], OP [25:16], ENCODING [31:26]
 *        = 0x35.
 *   DW1: SRC0 [40:32], SRC1 [49:41], SRC2 [58:50].
 * "In VOP3 the VCC destination may be an arbitrary SGPR-pair."
 * [RDNA2-ISA] §12.10. */

/* V_ADD_CO_U32 vdst, sdst, src0, src1, src2 — D = S0 + S1 with
 * carry-out to SDST; src2 is the carry-in (usually VCC).
 *   sdst: scalar destination SGPR number (VCC = 106, the low
 *     SGPR of the VCC pair, is the usual choice).
 *   src0/src1/src2: operand codes. */
vbs_rdna2_inst_t vbs_rdna2_emit_v_add_co_u32(unsigned vdst,
                                             unsigned sdst,
                                             unsigned src0,
                                             unsigned src1,
                                             unsigned src2);

/* --- FLAT / GLOBAL emitters -------------------------------------------
 * FLAT (64-bit) [RDNA2-ISA] §13.8.1-2, pp.279-281; §9.3-9.4;
 * digest §5.2:
 *   DW0: OFFSET [11:0] (12-bit signed immediate), DLC [12],
 *        LDS [13], SEG [15:14] (0=flat, 1=scratch, 2=global),
 *        GLC [16], SLC [17], OP [24:18], ENCODING [31:26] = 0x37.
 *   DW1: ADDR [39:32] (VGPR holding the address/offset),
 *        DATA [47:40] (source VGPR for stores), SADDR [54:48]
 *        (SGPR base; 0x7F/NULL disables), VDST [63:56]
 *        (destination VGPR for loads).
 * Global addressing modes [RDNA2-ISA] §9.3.1, §9.4:
 *   (a) VGPR supplies the address (SADDR == NULL).
 *   (b) SGPR (64-bit) supplies a base, VGPR (32-bit) an offset.
 * SADDR is the SGPR NUMBER of the base pair's low dword
 * ([RDNA2-ISA] §13.8.1: "Scalar SGPR which provides an address
 * or offset"; a 64-bit value lives in an even-aligned pair with
 * the LSB in SGPR[n], §3.6.3) — e.g. s[2:3] -> SADDR = 2.
 * Unlike SMEM's SBASE, SADDR is NOT divided by two. */

/* GLOBAL_LOAD_DWORD vdst, [saddr + vgpr_offset + offset] —
 * mode (b) when saddr != NULL, mode (a) when saddr is
 * VBS_RDNA2_SADDR_NULL (then addr_vgpr holds the full address).
 *   offset: 12-bit signed immediate byte offset. */
vbs_rdna2_inst_t vbs_rdna2_emit_global_load_dword(
        unsigned vdst, unsigned addr_vgpr, unsigned saddr,
        int16_t offset);

/* GLOBAL_LOAD_DWORD_ADDTID vdst, s[saddr:saddr+1] — the
 * simplest load for a vector kernel: no VGPR address is
 * supplied; memory_Addr = sgpr_addr(64) + inst_offset(12) +
 * tid*4 [RDNA2-ISA] §12.8. ADDR field = 0 (UNSOURCED, see the
 * file header). */
vbs_rdna2_inst_t vbs_rdna2_emit_global_load_dword_addtid(
        unsigned vdst, unsigned saddr, int16_t offset);

/* GLOBAL_STORE_DWORD [saddr + vgpr_offset + offset], vdata —
 * the store half of GLOBAL_LOAD_DWORD. */
vbs_rdna2_inst_t vbs_rdna2_emit_global_store_dword(
        unsigned vdata, unsigned addr_vgpr, unsigned saddr,
        int16_t offset);

/* GLOBAL_STORE_DWORD_ADDTID vdata, s[saddr:saddr+1] — the
 * store half of GLOBAL_LOAD_DWORD_ADDTID. ADDR field = 0
 * (UNSOURCED, see the file header). */
vbs_rdna2_inst_t vbs_rdna2_emit_global_store_dword_addtid(
        unsigned vdata, unsigned saddr, int16_t offset);

#endif /* VBS_RDNA2_ASM_H */
