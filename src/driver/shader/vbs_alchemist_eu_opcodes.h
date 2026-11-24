/* vbs_alchemist_eu_opcodes.h — EU opcode table for Intel Arc Alchemist
 * (Xe-HPG / DG2).
 *
 * THE opcode table for Verum's Intel shader path. This file is the
 * ONLY place opcode values live: the assembler (vbs_alchemist_eu.c)
 * looks every opcode up here, so any entry can be corrected from
 * hardware results by editing this file alone — no assembler logic
 * changes, no kernel changes.
 *
 * Every entry carries its sourcing status in its note:
 *   [S]  sourced: value from the Intel ACM PRM (cited per entry).
 *   [U]  UNSOURCED — no usable value known; the note gives the exact
 *        derivation procedure. Never invent a value for these.
 *
 * Source document for all citations below: the Intel spec digest at
 * ~/workspace/verum-driver-2026-11-17/specs/intel/SPEC-DIGEST.md
 * ("the digest"), which cites the Intel Arc A-Series / Data Center
 * GPU Flex Series PRM (March 2023, Rev 1.0/2.0), Linux v7.3.5
 * headers, and the Xe-HPG white paper. "Vol 2a p.NNN" means the ACM
 * PRM Vol 2a (Command Reference: Instructions); "Vol 9 p.NNN" the
 * ACM PRM Vol 9 (Render Engine); "Vol 2b p.NNN" the ACM PRM Vol 2b
 * (Enumerations); "Vol 2d p.NNN" the ACM PRM Vol 2d (Structures).
 * The digest section is given alongside each PRM citation so any
 * fact can be re-checked in seconds.
 *
 * Format facts (digest §6.1, §6.4, §6.5):
 *   - Every EU instruction is 128 bits = four 32-bit dwords.
 *     DW0 = bits [31:0] (opcode + ExecSize + header), DW1 = bits
 *     [63:32] (destination + src0 type info), DW2 = bits [95:64]
 *     (src0 region + src1 type), DW3 = bits [127:96] (src1 region
 *     or immediate). In memory DW0 comes first, each dword
 *     little-endian: byte 0 of the instruction word holds the
 *     opcode (Vol 9, p.469; digest §6.1).
 *   - Opcode: 7 bits, byte 0 bits [6:0]; bit 7 reserved.
 *   - CmptCtrl (bit 29) = 0: native 128-bit format, no compaction
 *     (Vol 9, p.456; sends are never compacted).
 *   - Header bits [15:7] are left 0. The Access Mode bit (Align1
 *     vs Align16) lives somewhere in the header but its bit number
 *     is UNSOURCED (digest §6.1, §9 item 1) — see
 *     VBS_ALCHEMIST_ACCESS_MODE_NOTE below. All v1 instructions
 *     are non-IEEE-macro instructions, which "must use Align1
 *     accessmode" (Vol 9, p.387), and all v1 regioning is Align1.
 *
 * Originality: the VALUES in this table are documented hardware
 * facts (opcode numbers published in Intel's public PRM, studied
 * as data). The table layout, comments, and all code around it are
 * original Verum work. The code is our own — nothing here comes
 * from Mesa, Intel's compute runtime, or any other implementation.
 */

#ifndef VBS_ALCHEMIST_EU_OPCODES_H
#define VBS_ALCHEMIST_EU_OPCODES_H

#include <stdint.h>

/* ------------------------------------------------------------------ */
/* One table entry                                                     */
/* ------------------------------------------------------------------ */

typedef struct {
    const char *name;   /* EU mnemonic, e.g. "add" */
    uint8_t     code;   /* 7-bit opcode, placed at instruction bits [6:0] */
    uint8_t     sourced;/* 1 = opcode value sourced (see note);
                           0 = UNSOURCED, never emit (lookup refuses) */
    const char *note;   /* source citation, or the derivation
                           procedure when UNSOURCED */
} vbs_alchemist_opcode_t;

/* Instruction kinds — indexes into VBS_ALCHEMIST_OPCODE_TABLE. */
typedef enum {
    VBS_ALCHEMIST_OP_ADD = 0,
    VBS_ALCHEMIST_OP_MUL,
    VBS_ALCHEMIST_OP_MOV,
    VBS_ALCHEMIST_OP_MOV_ALT,   /* 0x01: UNSOURCED purpose, see note */
    VBS_ALCHEMIST_OP_SHL,
    VBS_ALCHEMIST_OP_SEND,
    VBS_ALCHEMIST_OP_SENDC,
    VBS_ALCHEMIST_OP_NOP,
    VBS_ALCHEMIST_OP_AND,
    VBS_ALCHEMIST_OP_OR,
    VBS_ALCHEMIST_OP_XOR,
    VBS_ALCHEMIST_OP_CMP,
    VBS_ALCHEMIST_OP_COUNT
} vbs_alchemist_op_t;

/* ------------------------------------------------------------------ */
/* THE opcode table.                                                   */
/*                                                                     */
/* [S] = sourced from the ACM PRM Instruction Set Summary tables       */
/* (Vol 9, pp.470-476) and the per-opcode group tables; [U] =          */
/* UNSOURCED (see note). "S5" is not used here; every citation is a    */
/* PRM volume + page and the digest section that extracted it.         */
/* ------------------------------------------------------------------ */

static const vbs_alchemist_opcode_t VBS_ALCHEMIST_OPCODE_TABLE[] = {
    { "add",    0x40, 1, "[S] Parallel Arithmetic, 2 src. Vol 9, pp.470-476 (Instruction Set Summary) and pp.473-476; binary format Vol 2a, pp.219-224 ('Addition'). Digest §6.3, §6.4." },
    { "mul",    0x41, 1, "[S] Parallel Arithmetic, 2 src. Vol 9, pp.470-476. Digest §6.3." },
    { "mov",    0x61, 1, "[S] Move and Logic, 1 src. Vol 9, p.470 (Move and Logic table); binary format Vol 2a, pp.1111-1115 ('Move'). Digest §6.3, §6.4." },
    { "mov?",   0x01, 0, "[U] UNSOURCED: the Instruction Set Summary table also lists 'mov' at 0x01, but the extracted PRM text never explains what the 0x01 encoding is for (digest §6.3, §9 item 2). The value 0x01 is recorded here only so the entry is visible; the assembler MUST refuse to emit it. DERIVE: find the 0x01 entry in the Vol 9 summary table context (pp.470-476) and the Vol 2a 'Move' pages; if no use is documented, drop this entry. Use 0x61 for all moves." },
    { "shl",    0x69, 1, "[S] Move and Logic, 2 src. Vol 9, pp.470-476; binary format Vol 2a, pp.1275-1279 ('Shift Left'), same 128-bit layout as 'add' (verified against the shl page itself). Digest §6.3, §6.4." },
    { "send",   0x31, 1, "[S] Miscellaneous, 2 src. Vol 9, pp.470-476; binary format Vol 2a, pp.1253-1260 ('Send Message'). Digest §6.3, §6.5." },
    { "sendc",  0x32, 1, "[S] Miscellaneous, 2 src (conditional send). Vol 9, pp.470-476. Digest §6.3." },
    { "nop",    0x60, 1, "[S] Move and Logic, 0 src. Vol 9, pp.470-476. Digest §6.3." },
    { "and",    0x65, 1, "[S] Move and Logic. Vol 9, pp.470-476. Digest §6.3." },
    { "or",     0x66, 1, "[S] Move and Logic. Vol 9, pp.470-476. Digest §6.3." },
    { "xor",    0x67, 1, "[S] Move and Logic. Vol 9, pp.470-476. Digest §6.3." },
    { "cmp",    0x70, 1, "[S] Move and Logic, 2 src. Vol 9, pp.470-476. Digest §6.3." },
};

/* ------------------------------------------------------------------ */
/* UNSOURCED note: the Access Mode bit (digest §6.1, §9 item 1).       */
/*                                                                     */
/* The PRM states there are two register access modes "controlled by   */
/* a single bit instruction subfield called Access Mode" (Vol 9,       */
/* p.386) but the extracted PRM text never states which instruction    */
/* bit it is, and the per-instruction pages group bits [15:0] as       */
/* "Header" without further breakdown (checked: Vol 2a 'Addition'      */
/* p.219, 'Move' p.1111, 'Send Message' p.1253, 'Shift Left' p.1275;   */
/* Vol 9 pp.385-387, 456-476 — no bit number anywhere).               */
/*                                                                     */
/* v1 position: use Align1 regioning everywhere (the documented mode   */
/* for non-IEEE-macro instructions, Vol 9 p.387) with header bits      */
/* [15:7] = 0, per the digest §6.1 workaround. Do NOT use Align16      */
/* anywhere until the bit is found.                                    */
/*                                                                     */
/* DERIVE: (1) re-extract the Vol 9 "Instruction Fields" section       */
/* tables from the PDF with a table-aware extractor — pdftotext may    */
/* have dropped the header sub-field table; (2) check the Vol 2a      */
/* per-instruction pages' "Header" format definition in Vol 2d;       */
/* (3) on hardware, a kernel whose first instruction faults with an    */
/* illegal-instruction error points here first: try the candidate bit  */
/* values one at a time with a single-instruction probe kernel.        */
/* ------------------------------------------------------------------ */
#define VBS_ALCHEMIST_ACCESS_MODE_NOTE \
    "Access Mode bit location UNSOURCED (digest §6.1, §9 item 1); " \
    "header bits [15:7] = 0, Align1 regioning only; see header note"

/* ------------------------------------------------------------------ */
/* Derived-but-unconfirmed note: add with :uq operands.                */
/*                                                                     */
/* The digest §6.9 prescribes `add` with `:uq` for the base+offset     */
/* add, citing the Vol 2a p.220 'Src Types Dst Types' table. The       */
/* extracted table on p.220 shows only *B,*W,*D / F / HF / BF rows —   */
/* no Q/UQ row. But the same extraction drops Q/UQ rows from tables    */
/* where the PROSE confirms Q support (shl: "The operation uses       */
/* QWord mode if src0 or dst has the Q or UQ type", Vol 2a p.1275;    */
/* mov: "no direct conversion from B/UB to Q/UQ ... Use ... a DWord   */
/* intermediate", Vol 2a p.1111 — implying DWord<->Q/UQ conversion    */
/* IS direct). Q is a documented execution data type (Vol 9, p.387:   */
/* "Q Quadword. 64-bit signed integer") with a mixed-width rule ("If   */
/* source operands have different integer widths, use the widest      */
/* width specified"). So add :uq is DERIVED, not directly confirmed:   */
/* the type-table omission is weak evidence against it, and the        */
/* execution-type rules support it. The first hardware run must       */
/* confirm it (a fault on the first :uq add points here; the fallback  */
/* is 32-bit-half adds with addc carry, Vol 2a p.230).                 */
/* ------------------------------------------------------------------ */
#define VBS_ALCHEMIST_ADD_UQ_NOTE \
    "add :uq derived (digest §6.9; Vol 9 p.387 execution types); " \
    "confirm on hardware; fallback is addc (Vol 2a p.230)"

/* ------------------------------------------------------------------ */
/* ExecSize encodings (instruction bits [18:16]; Vol 2b, p.78;        */
/* digest §6.1). For F32 the widest single instruction is SIMD16; a   */
/* SIMD32 thread executes F32 math as two 16-wide halves.             */
/*                                                                     */
/* HARD RULE (Vol 9, p.387, Register Region Restrictions): for ALU     */
/* instructions, ExecSize * (largest operand element size in bytes)    */
/* must be <= 64. So :uq (8-byte) ALU ops are SIMD8 at most — a        */
/* SIMD16 :uq add/mov is ILLEGAL. The kernel splits all :uq work into  */
/* two SIMD8 halves.                                                   */
/* ------------------------------------------------------------------ */
#define VBS_ALCHEMIST_EXECSIZE_1  0u   /* 1 channel  */
#define VBS_ALCHEMIST_EXECSIZE_2  1u   /* 2 channels */
#define VBS_ALCHEMIST_EXECSIZE_4  2u   /* 4 channels */
#define VBS_ALCHEMIST_EXECSIZE_8  3u   /* 8 channels */
#define VBS_ALCHEMIST_EXECSIZE_16 4u   /* 16 channels (widest for F32) */
#define VBS_ALCHEMIST_EXECSIZE_32 5u   /* 32 channels (1/2-byte types only) */

/* ------------------------------------------------------------------ */
/* RegDataType (4 bits; Vol 2b, p.93; digest §6.2).                    */
/* ------------------------------------------------------------------ */
#define VBS_ALCHEMIST_TYPE_UB 0u
#define VBS_ALCHEMIST_TYPE_UW 1u
#define VBS_ALCHEMIST_TYPE_UD 2u
#define VBS_ALCHEMIST_TYPE_UQ 3u
#define VBS_ALCHEMIST_TYPE_B  4u
#define VBS_ALCHEMIST_TYPE_W  5u
#define VBS_ALCHEMIST_TYPE_D  6u
#define VBS_ALCHEMIST_TYPE_Q  7u
#define VBS_ALCHEMIST_TYPE_HF 9u
#define VBS_ALCHEMIST_TYPE_F  10u
#define VBS_ALCHEMIST_TYPE_DF 11u
#define VBS_ALCHEMIST_TYPE_BF 13u

/* ------------------------------------------------------------------ */
/* Region encodings (Align1; Vol 2b, pp.82, 126-127; digest §6.2).    */
/* VertStride: 0=0, 1=1, 2=2, 3=4, 4=8, 5=16, 6=32 elements.           */
/* Width:      0=1, 1=2, 2=4, 3=8, 4=16 elements.                      */
/* HorzStride: 0=0, 1=1, 2=2, 3=4 elements.                            */
/*                                                                     */
/* Presets below are (VertStride, Width, HorzStride) encodings. The    */
/* <8;8,1> F32 region and the <0;1,0> scalar-broadcast region are      */
/* confirmed by the Vol 9 "SIMD16 Mode of Operation" examples (pp.339- */
/* 340): `add (16) r18<1>:f r4<8;8,1>:f r14<8;8,1>:f` and               */
/* `add (16) r18<1>:f r2.1<0;1,0>:f r14<8;8,1>:f` — this resolves      */
/* digest §9 item 11 (regioning cross-check) for these two regions.    */
/* The <4;4,1> UQ region and <16;16,1> UW region follow the enum      */
/* enum definitions the same way (rows of one GRF each); they follow   */
/* the same rule (a row never crosses a GRF boundary) and are flagged  */
/* for the hardware run to confirm.                                    */
/* ------------------------------------------------------------------ */
/* <8;8,1>: contiguous SIMD8/SIMD16 F32 or UD — two rows of 8. */
#define VBS_ALCHEMIST_REGION_F32_V  4u
#define VBS_ALCHEMIST_REGION_F32_W  3u
#define VBS_ALCHEMIST_REGION_F32_H  1u
/* <4;4,1>: contiguous SIMD8 UQ — two rows of 4 (one GRF each). */
#define VBS_ALCHEMIST_REGION_UQ8_V  3u
#define VBS_ALCHEMIST_REGION_UQ8_W  2u
#define VBS_ALCHEMIST_REGION_UQ8_H  1u
/* <16;16,1>: contiguous SIMD16 UW — one row of 16 (one GRF). */
#define VBS_ALCHEMIST_REGION_UW16_V 5u
#define VBS_ALCHEMIST_REGION_UW16_W 4u
#define VBS_ALCHEMIST_REGION_UW16_H 1u
/* <0;1,0>: scalar broadcast — every lane reads the same element. */
#define VBS_ALCHEMIST_REGION_SCALAR_V 0u
#define VBS_ALCHEMIST_REGION_SCALAR_W 0u
#define VBS_ALCHEMIST_REGION_SCALAR_H 0u

/* ------------------------------------------------------------------ */
/* Shared Function IDs (SFID; 4 bits; Vol 2b, p.106; digest §6.5).    */
/* ------------------------------------------------------------------ */
#define VBS_ALCHEMIST_SFID_NULL   0u
#define VBS_ALCHEMIST_SFID_UGML   1u
#define VBS_ALCHEMIST_SFID_SAMPLER 2u
#define VBS_ALCHEMIST_SFID_GATEWAY 3u   /* Message Gateway; EOT lives here */
#define VBS_ALCHEMIST_SFID_DP_DC2 4u
#define VBS_ALCHEMIST_SFID_DP_RC  5u
#define VBS_ALCHEMIST_SFID_URB    6u
#define VBS_ALCHEMIST_SFID_BTD    7u
#define VBS_ALCHEMIST_SFID_RTA    8u
#define VBS_ALCHEMIST_SFID_DP_DCRO 9u
#define VBS_ALCHEMIST_SFID_DP_DC0 10u
#define VBS_ALCHEMIST_SFID_PI     11u
#define VBS_ALCHEMIST_SFID_DP_DC1 12u  /* Data Cache Data Port 1: A64 msgs */
#define VBS_ALCHEMIST_SFID_TGM    13u
#define VBS_ALCHEMIST_SFID_SLM    14u
#define VBS_ALCHEMIST_SFID_UGM    15u

/* ------------------------------------------------------------------ */
/* Message descriptor (desc) field values for vector-add (digest      */
/* §6.6, §6.8; MSD pages in Vol 2a). The assembler builds desc from    */
/* these fields (see vbs_alchemist_desc_build); the tests assert the   */
/* built values equal the digest's computed literals below.            */
/* ------------------------------------------------------------------ */
/* A64 Untyped Surface Read, MSD1R_A64_US, Vol 2a p.209, DC1. */
#define VBS_ALCHEMIST_MSG_A64_READ  0x11u  /* Message Type, desc[18:14] */
/* A64 Untyped Surface Write, MSD1W_A64_US, Vol 2a p.211, DC1. */
#define VBS_ALCHEMIST_MSG_A64_WRITE 0x19u  /* Message Type, desc[18:14] */
/* SIMD Mode MDC_SM3 (Vol 2b, p.109; desc[13:12]): 1 = SIMD8, 2 = SIMD16. */
#define VBS_ALCHEMIST_SIMD_MODE_8   1u
#define VBS_ALCHEMIST_SIMD_MODE_16  2u
/* Channel Mask (MDC_CMASK, Vol 2d p.299; inverted, 0 = all channels):
 * 0xE = R only — the value for single-channel F32 traffic. */
#define VBS_ALCHEMIST_CMASK_R       0xEu
/* Binding Table Index 255 = stateless A32/A64 (Vol 9, p.657). */
#define VBS_ALCHEMIST_BTI_STATELESS 0xFFu
/* EOT Subfunction (MSD_EOT, Vol 2a pp.562-563): 0. */

/* Digest §6.8 computed desc literals (the tests check the assembler
 * reproduces these exactly from the fields above). */
#define VBS_ALCHEMIST_DESC_A64_READ_SIMD16  0x08246EFFu
#define VBS_ALCHEMIST_DESC_A64_WRITE_SIMD16 0x08066EFFu
#define VBS_ALCHEMIST_DESC_EOT              0x02000000u
/* ex_desc for the SIMD16 store: Extended Message Length 2 << 6. */
#define VBS_ALCHEMIST_EXDESC_A64_WRITE_SIMD16 0x80u

#endif /* VBS_ALCHEMIST_EU_OPCODES_H */
