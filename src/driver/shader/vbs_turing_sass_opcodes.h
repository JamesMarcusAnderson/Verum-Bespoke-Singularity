/* vbs_turing_sass_opcodes.h — SASS opcode table for Turing (sm_75).
 *
 * THE opcode table for Verum's NVIDIA shader path. This file is the
 * ONLY place opcode values live: the assembler (vbs_turing_sass.c)
 * looks every opcode up here, so any entry can be corrected from
 * hardware results by editing this file alone — no assembler logic
 * changes, no kernel changes.
 *
 * Every entry carries its sourcing status in its note:
 *   [X2]  cross-checked: turingas opcode table AND NVIDIA's own
 *         published nvdisasm output (CUDA Binary Utilities doc, S5).
 *   [T1]  turingas opcode table only — no published nvdisasm hex
 *         found for it yet; needs the hardware-validation run.
 *   [U]   UNSOURCED — no value known; the note gives the exact
 *         derivation procedure. Never invent a value for these.
 *
 * Format facts (digest §5.2; S11 DeepWiki "Instruction Encoding
 * Format" + "Grammar and Instruction Encoding"):
 *   - Every SASS instruction is 128 bits = two 64-bit words. In
 *     memory the instruction word (bits 64-127: opcode, registers,
 *     predicate) comes FIRST, then the control word (bits 0-63:
 *     flags, third register, control code) — turingas cubin.py
 *     GenerateText packs (code>>64, code), and the NVIDIA forum
 *     notes the control word "follows after every instruction"
 *     on Volta/Ampere.
 *   - Opcode: 12 bits at instruction bits 64-75. The top nibble
 *     (bits 72-75) is the ICR (operand-type) selector: when an
 *     instruction takes an immediate, constant, or uniform operand
 *     in the rs1/rs2 position, bit 73 is cleared and bits 72-75
 *     are replaced with the ICR value for that operand kind
 *     (see VBS_TURING_ICR_* below). Plain register forms keep the
 *     base code as listed.
 *   - Predicate: 4 bits at 76-79 (P0-P6 = 0-6, PT = 7, +8 negated).
 *   - Registers: rd bits 80-87, rs0 bits 88-95, rs1 bits 96-103,
 *     rs2 bits 0-7. R0-R254 by number; RZ = 0xFF; UR0-UR61 by
 *     number; URZ = 0x3F.
 *   - Control code: bits 41-63 (stall 41-44, yield 45, write
 *     barrier 46-48, read barrier 49-51, wait mask 52-56, reuse
 *     flags 58-60). 7 = "no barrier" for wrtdb/readb.
 *
 * Originality: the VALUES in this table are documented hardware
 * facts (opcode numbers published in NVIDIA's own documentation
 * samples and in the public turingas assembler's opcode table,
 * studied as data). The table layout, comments, and all code
 * around it are original Verum work. The code is our own — nothing
 * here comes from turingas, nvdisasm, Mesa, or any other implementation.
 */

#ifndef VBS_TURING_SASS_OPCODES_H
#define VBS_TURING_SASS_OPCODES_H

#include <stdint.h>

/* ------------------------------------------------------------------ */
/* One table entry                                                     */
/* ------------------------------------------------------------------ */

typedef struct {
    const char *name;   /* SASS mnemonic, e.g. "LDG" */
    uint16_t    code;   /* 12-bit base opcode, placed at bits 64-75 */
    uint8_t     sourced;/* 1 = opcode value sourced (see note);
                           0 = UNSOURCED, code value 0 (unset) */
    const char *note;   /* source citation, or the derivation
                           procedure when UNSOURCED */
} vbs_turing_opcode_t;

/* Instruction kinds — indexes into VBS_TURING_OPCODE_TABLE. */
typedef enum {
    VBS_TURING_OP_MOV = 0,
    VBS_TURING_OP_LDG,
    VBS_TURING_OP_STG,
    VBS_TURING_OP_LDS,
    VBS_TURING_OP_STS,
    VBS_TURING_OP_LDSM,
    VBS_TURING_OP_LDGSTS,
    VBS_TURING_OP_LDGDEPBAR,
    VBS_TURING_OP_DEPBAR,
    VBS_TURING_OP_LDC,
    VBS_TURING_OP_UMOV,
    VBS_TURING_OP_USHF,
    VBS_TURING_OP_UIADD3,
    VBS_TURING_OP_ULEA,
    VBS_TURING_OP_ULDC,
    VBS_TURING_OP_IADD3,
    VBS_TURING_OP_IMUL,       /* UNSOURCED, see note */
    VBS_TURING_OP_LEA,
    VBS_TURING_OP_IMAD,
    VBS_TURING_OP_IMAD_WIDE,
    VBS_TURING_OP_ISETP,
    VBS_TURING_OP_LOP3,
    VBS_TURING_OP_SHF,
    VBS_TURING_OP_PRMT,
    VBS_TURING_OP_SEL,
    VBS_TURING_OP_SGXT,
    VBS_TURING_OP_IABS,
    VBS_TURING_OP_IMNMX,
    VBS_TURING_OP_HADD2,
    VBS_TURING_OP_HFMA2,
    VBS_TURING_OP_HMUL2,
    VBS_TURING_OP_FFMA,
    VBS_TURING_OP_MUFU,
    VBS_TURING_OP_FADD,
    VBS_TURING_OP_FMUL,
    VBS_TURING_OP_FMNMX,
    VBS_TURING_OP_HMMA,
    VBS_TURING_OP_IMMA,
    VBS_TURING_OP_BMMA,
    VBS_TURING_OP_BRA,
    VBS_TURING_OP_BSSY,
    VBS_TURING_OP_BSYNC,
    VBS_TURING_OP_BMOV,
    VBS_TURING_OP_EXIT,
    VBS_TURING_OP_CS2R,
    VBS_TURING_OP_S2R,
    VBS_TURING_OP_S2UR,
    VBS_TURING_OP_NOP,
    VBS_TURING_OP_YIELD,
    VBS_TURING_OP_BAR,
    VBS_TURING_OP_P2R,
    VBS_TURING_OP_R2P,
    VBS_TURING_OP_SHFL,
    VBS_TURING_OP_F2F,
    VBS_TURING_OP_F2FP,
    VBS_TURING_OP_F2I,
    VBS_TURING_OP_I2F,
    VBS_TURING_OP_ATOMS,
    VBS_TURING_OP_ATOMG,
    VBS_TURING_OP_RED,
    VBS_TURING_OP_COUNT
} vbs_turing_op_t;

/* ------------------------------------------------------------------ */
/* THE TABLE — the only place opcode values live.                      */
/*                                                                     */
/* Source key: [X2] turingas table + NVIDIA-published nvdisasm hex;    */
/* [T1] turingas table only; [U] UNSOURCED (see note). "S5" is the     */
/* CUDA Binary Utilities doc (specs/cuda-binary-utilities.txt, and     */
/* the 12.4.1 archive page for the sm_70 sample); "S11" is the         */
/* turingas opcode table (grammar.py `grammar` dict, studied as        */
/* data). "Digest" is SPEC-DIGEST.md.                                  */
/* ------------------------------------------------------------------ */

static const vbs_turing_opcode_t VBS_TURING_OPCODE_TABLE[VBS_TURING_OP_COUNT] = {
    /* name       code    src  note */
    { "MOV",      0x202,  1, "[X2] S11; S5 sm_70 MOV R3,c[0x0][0x164] -> 0x7a02 (0x202 + ICR 0xa for the constant source)" },
    { "LDG",      0x381,  1, "[X2] Digest §5.3 worked example; S5 sm_70 LDG.E.SYS -> 0x7381, sm_100 LDG.E desc -> 0x7981 (0x381 + ICR 0x9)" },
    { "STG",      0x386,  1, "[X2] S11; S5 sm_70 STG.E.SYS -> 0x7386, sm_100 STG.E desc -> 0x7986 (0x386 + ICR 0x9)" },
    { "LDS",      0x984,  1, "[T1] S11 only" },
    { "STS",      0x388,  1, "[T1] S11 only" },
    { "LDSM",     0x83b,  1, "[T1] S11 only; turingas marks LDSM TODO for the .ZFILL/UR-offset forms" },
    { "LDGSTS",   0xfae,  1, "[T1] S11 only (Ampere-primary; Turing support in turingas is partial)" },
    { "LDGDEPBAR",0x9af,  1, "[T1] S11 only" },
    { "DEPBAR",   0x91a,  1, "[T1] S11 only; sm_75+ per the flag table" },
    { "LDC",      0xb82,  1, "[X2] S11; S5 sm_100 LDC/LDC.64 -> 0x7b82" },
    { "UMOV",     0x282,  1, "[T1] S11 only" },
    { "USHF",     0x299,  1, "[T1] S11 only" },
    { "UIADD3",   0x290,  1, "[T1] S11 only" },
    { "ULEA",     0x291,  1, "[T1] S11 only" },
    { "ULDC",     0xab9,  1, "[T1] S11 only (uniform-path constant load; NOT sm_100's LDCU 0x7ac, which has no Turing source)" },
    { "IADD3",    0x210,  1, "[X2] Digest §5.3 worked example; S5 sm_70 IADD3 R9,R2,R5,RZ -> 0x7210" },
    { "IMUL",     0x000,  0, "[U] UNSOURCED: the turingas table holds 0x000 as a stub for IMUL and no published nvdisasm hex for a Turing IMUL was found. Integer multiply on Turing is normally emitted as IMAD, so IMUL may have no distinct Turing encoding. DERIVE: ptxas -arch=sm_75 a kernel with mul.lo.u32 whose operands defeat IMAD fusion, nvdisasm --print-code, read bits 64-75 of the multiply instruction; if ptxas never emits IMUL, drop this entry." },
    { "LEA",      0x211,  1, "[T1] S11 only" },
    { "IMAD",     0x224,  1, "[X2] S11; S5 sm_70 IMAD.MOV.U32 -> 0x7624 (0x224 + ICR 0x6), sm_100 IMAD R9,R0,UR6,R9 -> 0x7c24 (0x224 + ICR 0xc)" },
    { "IMAD.WIDE",0x225,  1, "[X2] S11; S5 sm_100 IMAD.WIDE R2,R9,0x4,R2 -> 0x7825 (0x225 + ICR 0x8 for the immediate)" },
    { "ISETP",    0x20c,  1, "[T1] S11 only; flags (cmp/boolOp/type) from the S11 flag table. No published nvdisasm hex found — validate on hardware (see REPORT)." },
    { "LOP3",     0x212,  1, "[T1] S11 only" },
    { "SHF",      0x219,  1, "[T1] S11 only" },
    { "PRMT",     0x216,  1, "[T1] S11 only" },
    { "SEL",      0x207,  1, "[T1] S11 only" },
    { "SGXT",     0x21a,  1, "[T1] S11 only" },
    { "IABS",     0x213,  1, "[T1] S11 only" },
    { "IMNMX",    0x217,  1, "[T1] S11 only" },
    { "HADD2",    0x230,  1, "[T1] S11 only" },
    { "HFMA2",    0x231,  1, "[T1] S11 only" },
    { "HMUL2",    0x232,  1, "[T1] S11 only" },
    { "FFMA",     0x223,  1, "[X2] S11 DeepWiki opcode table; same FP-pipe family as FADD 0x221 (sm_100)" },
    { "MUFU",     0x308,  1, "[T1] S11 only" },
    { "FADD",     0x221,  1, "[X2] S11; S5 sm_100 FADD R9,R2,R5 -> 0x7221 (digest §10 item 1, resolved)" },
    { "FMUL",     0x220,  1, "[T1] S11 only" },
    { "FMNMX",    0x209,  1, "[T1] S11 only" },
    { "HMMA",     0x23c,  1, "[X2] Digest §5.3 worked example (0x23c << 64)" },
    { "IMMA",     0x237,  1, "[T1] S11 only" },
    { "BMMA",     0x23d,  1, "[T1] S11 only" },
    { "BRA",      0x947,  1, "[X2] S11 DeepWiki opcode table; S5 sm_70 BRA 0xd0 -> 0x7947, sm_100 BRA 0x110 -> 0x7947" },
    { "BSSY",     0x945,  1, "[T1] S11 only" },
    { "BSYNC",    0x941,  1, "[T1] S11 only" },
    { "BMOV",     0x355,  1, "[T1] S11 only" },
    { "EXIT",     0x94d,  1, "[X2] S11; S5 sm_70 EXIT -> 0x794d, sm_100 EXIT -> 0x794d (digest §10 item 1, resolved)" },
    { "CS2R",     0x805,  1, "[T1] S11 only" },
    { "S2R",      0x919,  1, "[X2] S11; S5 sm_100 S2R R9,SR_TID.X -> 0x7919 (digest §10 item 1, resolved)" },
    { "S2UR",     0x9c3,  1, "[X2] S11; S5 sm_100 S2UR UR6,SR_CTAID.X -> 0x79c3" },
    { "NOP",      0x918,  1, "[X2] S11; S5 sm_70/sm_100 NOP -> 0x7918" },
    { "YIELD",    0x946,  1, "[T1] S11 only" },
    { "BAR",      0xb1d,  1, "[T1] S11 only (digest §10 item 1, opcode resolved; the BAR.SYNC thread-count field, if any, is still UNSOURCED — see VBS_TURING_BAR_NOTE). No published nvdisasm hex found — validate on hardware." },
    { "P2R",      0x803,  1, "[T1] S11 only" },
    { "R2P",      0x804,  1, "[T1] S11 only" },
    { "SHFL",     0x389,  1, "[X2] S11; S5 sm_70 SHFL.IDX -> 0xf389 (opcode 0x389, predicate !PT)" },
    { "F2F",      0x304,  1, "[T1] S11 only; turingas lists type conversions as TODO, opcode from the table" },
    { "F2FP",     0x23e,  1, "[T1] S11 only" },
    { "F2I",      0x305,  1, "[T1] S11 only" },
    { "I2F",      0x306,  1, "[T1] S11 only" },
    { "ATOMS",    0x38d,  1, "[T1] S11 only" },
    { "ATOMG",    0x3a9,  1, "[T1] S11 only" },
    { "RED",      0x98e,  1, "[T1] S11 only" },
};

/* BAR.SYNC thread-count note (UNSOURCED, digest §10 item 1 follow-up):
 * the turingas table encodes BAR.SYNC with only a barrier id (ibar at
 * bits 118-125) and no thread-count field, and no published nvdisasm
 * hex for a BAR.SYNC with an explicit thread count was found. PTX
 * bar.sync takes an optional thread count; whether SASS carries it
 * (and where) is unknown. DERIVE: ptxas -arch=sm_75 a kernel with
 * "bar.sync 0, 128;" (count != CTA size), nvdisasm --print-code, and
 * compare against "bar.sync 0;" — any differing bits outside the
 * barrier id are the count field. Until then the assembler emits
 * BAR.SYNC with the id only (the form ptxas emits for bar.sync 0). */
#define VBS_TURING_BAR_NOTE "BAR.SYNC thread-count field UNSOURCED; see header note"

/* S2R special-register values that are UNSOURCED: the turingas flag
 * table lists only LANEID/TID.X/Y/Z/CTAID.X/Y/Z (see
 * VBS_TURING_SR_* below). SR_NTID.X is NOT in it. The Fermi doc
 * (S6 envytools hw/graph/fermi/cuda/isa.rst) lists $sr41 as ntid.x
 * on Fermi, but the Volta/Turing S2R sr encoding for NTID is not
 * published anywhere found. DERIVE: ptxas -arch=sm_75 a kernel
 * reading %ntid.x, nvdisasm --print-code, read the sr field
 * (bits 8-16 of the S2R control word). The vecadd kernel avoids
 * this entirely: the driver writes blockDim.x into c[0] and the
 * kernel loads it with LDC. */
#define VBS_TURING_SR_NTID_X_UNSOURCED 1

/* ------------------------------------------------------------------ */
/* Flag values — bit-field data for the instruction modifiers.         */
/* Each value is the field VALUE; the assembler shifts it into       */
/* place. Sources: S11 flag table (grammar.py flag_str), cross-      */
/* checked against S5 published nvdisasm hex where noted.            */
/* ------------------------------------------------------------------ */

/* Load/store data type, bits 9-11 (LDG/STG/LDS/STS/LDC/ULDC).
 * Digest §5.3 gave .64 = 5; the .32 = 4 value (digest §10 item 1)
 * comes from the S11 flag table and is confirmed by the S5 sm_70
 * LDG.E.SYS/STG.E.SYS lower words (type bits read back as 4). */
#define VBS_TURING_TYPE_U8   0u
#define VBS_TURING_TYPE_S8   1u
#define VBS_TURING_TYPE_U16  2u
#define VBS_TURING_TYPE_S16  3u
#define VBS_TURING_TYPE_32   4u
#define VBS_TURING_TYPE_64   5u
#define VBS_TURING_TYPE_128  6u
#define VBS_TURING_TYPE_SHIFT 9u

/* Memory strength, bits 15-16 (LDG/STG/ATOMS/ATOMG/RED).
 * Default 1; .STRONG = 2. Confirmed by the S5 sm_70 LDG/STG lower
 * words (strong bits read back as 1 for plain .SYS accesses). */
#define VBS_TURING_STRONG_CONSTANT 0u
#define VBS_TURING_STRONG_DEFAULT  1u
#define VBS_TURING_STRONG_STRONG   2u
#define VBS_TURING_STRONG_SHIFT    15u

/* Cache policy, bits 20-22 (LDG/STG/RED). Default 1; .LU = 3.
 * Confirmed by the S5 sm_70 LDG/STG lower words (cache = 1). */
#define VBS_TURING_CACHE_EF       0u
#define VBS_TURING_CACHE_DEFAULT  1u
#define VBS_TURING_CACHE_LU       3u
#define VBS_TURING_CACHE_SHIFT    20u

/* Scope, bits 13-14 (LDG/STG/ATOMG/RED). Default .SYS = 3.
 * Confirmed by the S5 sm_70 LDG.E.SYS/STG.E.SYS lower words. */
#define VBS_TURING_SCOPE_CTA  0u
#define VBS_TURING_SCOPE_GPU  2u
#define VBS_TURING_SCOPE_SYS  3u
#define VBS_TURING_SCOPE_SHIFT 13u

/* .E bit (LDG/STG/RED), bit 8. */
#define VBS_TURING_E_BIT 8u
/* .U bit (LDG/LDS), bit 12. Set only for the uniform-descriptor
 * load form; plain LDG.E leaves it clear (S5 sm_70 sample). */
#define VBS_TURING_U_BIT 12u
/* 64-bit-address bit (LDG/RED desc form only), bit 26. Plain
 * [R] loads/stores leave it clear even though the address is a
 * 64-bit register pair (S5 sm_70 LDG.E.SYS lower word: bit 26 = 0). */
#define VBS_TURING_OFF64_BIT 26u
/* Uniform-operand-present bit, bit 27 (set when any UR operand
 * is present). */
#define VBS_TURING_UNIFORM_BIT 27u

/* S2R/S2UR special-register selector, bits 8-16. Values from the
 * S11 flag table; TID.X = 33 confirmed by the S5 sm_100 S2R
 * (lower-word sr field reads 0x21), CTAID.X = 37 confirmed by
 * the S5 sm_100 S2UR (sr field reads 0x25). */
#define VBS_TURING_SR_LANEID   0u
#define VBS_TURING_SR_TID_X    33u
#define VBS_TURING_SR_TID_Y    34u
#define VBS_TURING_SR_TID_Z    35u
#define VBS_TURING_SR_CTAID_X  37u
#define VBS_TURING_SR_CTAID_Y  38u
#define VBS_TURING_SR_CTAID_Z  39u
#define VBS_TURING_SR_SHIFT    8u

/* ICR (operand-type selector) values, replacing bits 72-75 when
 * the instruction takes a non-register operand in the rs1/rs2
 * position (S11 icr_dict + DeepWiki ICR table). */
#define VBS_TURING_ICR_BASE    0x0u  /* register (no ICR change) */
#define VBS_TURING_ICR_IS2     0x4u  /* immediate at src2 */
#define VBS_TURING_ICR_FS2     0x4u  /* float immediate at src2 */
#define VBS_TURING_ICR_CS2     0x6u  /* constant at src2 */
#define VBS_TURING_ICR_CS1ADD  0x6u  /* constant (add form) at src1 */
#define VBS_TURING_ICR_IS1     0x8u  /* immediate at src1 */
#define VBS_TURING_ICR_FS1     0x8u  /* float immediate at src1 */
#define VBS_TURING_ICR_UOFF    0x9u  /* uniform-register offset */
#define VBS_TURING_ICR_CS1     0xau  /* constant at src1 */
#define VBS_TURING_ICR_US1     0xcu  /* uniform register at src1 */
#define VBS_TURING_ICR_US2E1   0xeu  /* uniform register, IMAD form */

/* ISETP comparison, bits 12-14 (S11 flag table). */
#define VBS_TURING_CMP_LT 1u
#define VBS_TURING_CMP_EQ 2u
#define VBS_TURING_CMP_LE 3u
#define VBS_TURING_CMP_GT 4u
#define VBS_TURING_CMP_NE 5u
#define VBS_TURING_CMP_GE 6u
#define VBS_TURING_CMP_SHIFT 12u

/* ISETP/IMAD integer type: .U32 = 0, .S32 = 1 at bit 9 (.S32 is
 * the default for IMAD; ISETP.U32 is used for unsigned index
 * compares). */
#define VBS_TURING_INT_TYPE_SHIFT 9u
#define VBS_TURING_INT_TYPE_U32 0u
#define VBS_TURING_INT_TYPE_S32 1u

/* BAR flags (S11 flag table): .SYNC = 0, .ARV = 1<<13,
 * .RED.POPC = 1<<14, .SYNCALL = 1<<15. Barrier id at bits 118-125. */
#define VBS_TURING_BAR_SYNC        0u
#define VBS_TURING_BAR_ARV         (1u << 13)
#define VBS_TURING_BAR_RED_POPC    (1u << 14)
#define VBS_TURING_BAR_SYNCALL     (1u << 15)
#define VBS_TURING_BAR_ID_SHIFT    118u

/* "No predicate" filler bits some instructions require (S11 flag
 * table nopd0/nopd1/nops0/nops1 defaults). These are not optional:
 * the S5 sm_70 IADD3/IMAD lower words carry exactly these values
 * (nopd0 = 7 at bits 17-19, nopd1 = 7 at bits 20-22, nops0 = 15
 * at bits 23-26, nops1 = 15 at bits 13-16). */
#define VBS_TURING_NOPD0 (7u << 17)
#define VBS_TURING_NOPD1 (7u << 20)
#define VBS_TURING_NOPS0 (15u << 23)
#define VBS_TURING_NOPS1 (15u << 13)

/* Predicate field: PT (always true) = 7 at bits 76-79, and the
 * load-predicate (ldgp) / control-predicate (cp) slots at bits
 * 17-19 / 23-25 default to PT as well. */
#define VBS_TURING_PRED_PT 7u
#define VBS_TURING_PRED_SHIFT 76u

#endif /* VBS_TURING_SASS_OPCODES_H */
