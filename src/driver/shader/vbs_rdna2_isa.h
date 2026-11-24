/* vbs_rdna2_isa.h — RDNA 2 (gfx1030) opcode table for the Verum assembler.
 *
 * THE OPCODE TABLE lives in this file and nowhere else. Every opcode
 * number the assembler emits comes from VBS_RDNA2_OP_TABLE below; the
 * assembler logic in vbs_rdna2_asm.c never contains a literal opcode.
 * If a hardware run shows an entry is wrong, correct the one table row
 * and every emitter picks it up.
 *
 * Every entry cites its source: "digest §x" = SPEC-DIGEST.md in
 * ~/workspace/verum-driver-2026-11-17/specs/amd/ (worker D), which
 * cites the AMD RDNA 2 ISA Reference Guide ("[RDNA2-ISA]", 30-Nov-2020)
 * section and page; "ISA-XML" = AMD's machine-readable RDNA 2 ISA
 * (amdgpu_isa_rdna2.xml, GPUOpen, MIT). All opcodes in this table were
 * cross-checked against the ISA-XML on 2026-11-17.
 *
 * All code here is original Verum work. Opcode values are documented
 * hardware facts; nothing is transcribed from Mesa, ROCm, LLVM's
 * AMDGPU backend, or any other implementation.
 *
 * Format encoding tags (the ENCODING field of each format), all from
 * [RDNA2-ISA] §13:
 *   SOPP  ENCODING [31:23] = 0x17F  (§13.1.5, p.245)
 *   SMEM  ENCODING [31:26] = 0x3D   (§13.2.1, p.246)
 *   VOP1  ENCODING [31:25] = 0x3F   (§13.3.2, pp.249-250)
 *   VOP2  ENCODING [31]    = 0      (§13.3.1, pp.247-248)
 *   VOP3A ENCODING [31:26] = 0x35   (§13.3.4, p.257)
 *   VOP3B ENCODING [31:26] = 0x35   (§13.3.5, pp.260-261)
 *   FLAT  ENCODING [31:26] = 0x37   (§13.8.1-2, pp.279-281)
 *
 * VOP3 re-encoding rule (sourced from ISA-XML, three data points):
 * a VOP2 instruction re-encoded in VOP3A keeps its semantics with
 * opcode = VOP2 opcode + 256: V_ADD_F32 3 -> 259, V_MUL_F32 8 -> 264,
 * V_ADD_NC_U32 37 -> 293. VOP1 maps differently (V_MOV_B32 1 -> 385,
 * i.e. +384, one ISA-XML data point). The VOP3A rows below carry the
 * VOP3 opcode directly, so no emitter has to apply the rule.
 */

#ifndef VBS_RDNA2_ISA_H
#define VBS_RDNA2_ISA_H

#include <stddef.h>
#include <stdint.h>

/* --- Instruction formats ------------------------------------------- */
typedef enum {
    VBS_RDNA2_FMT_SOPP = 0,  /* 32-bit scalar op (S_ENDPGM, S_WAITCNT) */
    VBS_RDNA2_FMT_SMEM,      /* 64-bit scalar memory (S_LOAD_DWORD*) */
    VBS_RDNA2_FMT_VOP1,      /* 32-bit vector, 1 source (V_MOV_B32) */
    VBS_RDNA2_FMT_VOP2,      /* 32-bit vector, 2 sources (V_ADD_NC_U32) */
    VBS_RDNA2_FMT_VOP3A,     /* 64-bit vector, 3 sources (V_MAD_*, VOP2 re-encodes) */
    VBS_RDNA2_FMT_VOP3B,     /* 64-bit vector + scalar dest (V_ADD_CO_U32) */
    VBS_RDNA2_FMT_FLAT       /* 64-bit flat/global/scratch memory */
} vbs_rdna2_fmt_t;

/* Format encoding tags (see the file header for citations). */
#define VBS_RDNA2_ENC_SOPP 0x17Fu
#define VBS_RDNA2_ENC_SMEM 0x3Du
#define VBS_RDNA2_ENC_VOP1 0x3Fu
#define VBS_RDNA2_ENC_VOP2 0x0u   /* single bit 31, always 0 */
#define VBS_RDNA2_ENC_VOP3 0x35u  /* shared by VOP3A and VOP3B */
#define VBS_RDNA2_ENC_FLAT 0x37u

/* --- Operand codes (VOP SRC0/SRC1/SRC2 fields) ----------------------
 * [RDNA2-ISA] §13.3.1 / §13.3.4: 0-105 SGPR, 106 VCC_LO, 107 VCC_HI,
 * 124 M0, 125 NULL, 128 constant 0, 129-192 integer 1..64, 193-208
 * integer -1..-16, 255 literal (32-bit constant follows), 256-511
 * VGPR 0-255. Digest §5.2. */
#define VBS_RDNA2_SRC_SGPR(n) ((unsigned)(n))            /* SGPR n, 0-105 */
#define VBS_RDNA2_SRC_VGPR(n) (256u + (unsigned)(n))     /* VGPR n, 0-255 */
#define VBS_RDNA2_SRC_VCC_LO  106u
#define VBS_RDNA2_SRC_VCC_HI  107u
#define VBS_RDNA2_SRC_M0      124u
#define VBS_RDNA2_SRC_NULL    125u
#define VBS_RDNA2_SRC_EXEC_LO 126u
#define VBS_RDNA2_SRC_EXEC_HI 127u
#define VBS_RDNA2_SRC_CONST0  128u
#define VBS_RDNA2_SRC_INT(n)  (128u + (unsigned)(n))     /* int 1..64 */
#define VBS_RDNA2_SRC_NEGINT(n) (192u + (unsigned)(n))   /* int -1..-16 */
#define VBS_RDNA2_SRC_LITERAL 255u

/* FLAT SADDR value that disables the SGPR base (mode (a): the VGPR
 * supplies the whole address). [RDNA2-ISA] §9.1 / §13.8.1. */
#define VBS_RDNA2_SADDR_NULL 0x7Fu

/* FLAT SEG field values. [RDNA2-ISA] §9.1: 0=flat, 1=scratch,
 * 2=global, 3=reserved. */
#define VBS_RDNA2_SEG_FLAT    0u
#define VBS_RDNA2_SEG_SCRATCH 1u
#define VBS_RDNA2_SEG_GLOBAL  2u

/* --- Opcode kinds ---------------------------------------------------- */
/* One kind per (instruction, format) pair the assembler can emit.
 * Kinds ending in _VOP3 are VOP2/VOP1 instructions re-encoded in
 * VOP3A (see the +256/+384 rule in the file header). */
typedef enum {
    /* SOPP */
    VBS_RDNA2_OP_S_ENDPGM = 0,
    VBS_RDNA2_OP_S_BARRIER,
    VBS_RDNA2_OP_S_WAITCNT,
    /* SMEM */
    VBS_RDNA2_OP_S_LOAD_DWORD,
    VBS_RDNA2_OP_S_LOAD_DWORDX2,
    VBS_RDNA2_OP_S_LOAD_DWORDX4,
    VBS_RDNA2_OP_S_LOAD_DWORDX8,
    VBS_RDNA2_OP_S_LOAD_DWORDX16,
    /* VOP1 */
    VBS_RDNA2_OP_V_MOV_B32,
    /* VOP2 */
    VBS_RDNA2_OP_V_ADD_F32,
    VBS_RDNA2_OP_V_MUL_F32,
    VBS_RDNA2_OP_V_ADD_NC_U32,
    VBS_RDNA2_OP_V_ADD_CO_CI_U32,
    /* VOP3A (native) */
    VBS_RDNA2_OP_V_MAD_I32_I24,
    VBS_RDNA2_OP_V_BFE_U32,
    VBS_RDNA2_OP_V_BFI_B32,
    /* VOP3A (VOP1/VOP2 re-encodes) */
    VBS_RDNA2_OP_V_MOV_B32_VOP3,
    VBS_RDNA2_OP_V_ADD_F32_VOP3,
    VBS_RDNA2_OP_V_MUL_F32_VOP3,
    VBS_RDNA2_OP_V_ADD_NC_U32_VOP3,
    /* VOP3B */
    VBS_RDNA2_OP_V_ADD_CO_U32,
    /* FLAT / GLOBAL */
    VBS_RDNA2_OP_GLOBAL_LOAD_DWORD,
    VBS_RDNA2_OP_GLOBAL_LOAD_DWORDX2,
    VBS_RDNA2_OP_GLOBAL_LOAD_DWORD_ADDTID,
    VBS_RDNA2_OP_GLOBAL_STORE_DWORD,
    VBS_RDNA2_OP_GLOBAL_STORE_DWORDX2,
    VBS_RDNA2_OP_GLOBAL_STORE_DWORDX4,
    VBS_RDNA2_OP_GLOBAL_STORE_DWORD_ADDTID,
    VBS_RDNA2_OP_COUNT
} vbs_rdna2_op_t;

/* One row of the opcode table. `sourced` is 1 when the opcode value
 * comes from a cited public document; it is 0 only for entries whose
 * value is a guess — there are none in this table (every opcode here
 * is sourced; the single UNSOURCED item in the shader work is the
 * ADDTID instruction's unused ADDR field value, which is not an
 * opcode — see vbs_rdna2_asm.h). */
typedef struct {
    vbs_rdna2_op_t kind;
    const char *name;       /* ISA mnemonic, e.g. "V_ADD_NC_U32" */
    vbs_rdna2_fmt_t fmt;
    uint16_t opcode;        /* opcode number, per the citation */
    int sourced;            /* 1 = cited below; 0 = UNSOURCED (none) */
    const char *citation;   /* document + section for this opcode */
} vbs_rdna2_op_entry_t;

/* =====================================================================
 * THE OPCODE TABLE — the single source of opcode numbers.
 * Correct an entry here and every emitter follows. Do not put opcode
 * literals anywhere else.
 * ===================================================================== */
static const vbs_rdna2_op_entry_t VBS_RDNA2_OP_TABLE[] = {
    /* SOPP — [RDNA2-ISA] §13.1.5, p.245; digest §5.2. */
    { VBS_RDNA2_OP_S_ENDPGM, "S_ENDPGM", VBS_RDNA2_FMT_SOPP, 1,
      1, "digest §5.2 SOPP; [RDNA2-ISA] §13.1.5; ISA-XML ENC_SOPP op 1" },
    { VBS_RDNA2_OP_S_BARRIER, "S_BARRIER", VBS_RDNA2_FMT_SOPP, 10,
      1, "digest §5.2 SOPP; [RDNA2-ISA] §13.1.5; ISA-XML ENC_SOPP op 10" },
    { VBS_RDNA2_OP_S_WAITCNT, "S_WAITCNT", VBS_RDNA2_FMT_SOPP, 12,
      1, "digest §5.2 SOPP; [RDNA2-ISA] §13.1.5; ISA-XML ENC_SOPP op 12" },
    /* SMEM — [RDNA2-ISA] §13.2.1; digest §5.2. */
    { VBS_RDNA2_OP_S_LOAD_DWORD, "S_LOAD_DWORD", VBS_RDNA2_FMT_SMEM, 0,
      1, "digest §5.2 SMEM; [RDNA2-ISA] §13.2.1; ISA-XML ENC_SMEM op 0" },
    { VBS_RDNA2_OP_S_LOAD_DWORDX2, "S_LOAD_DWORDX2", VBS_RDNA2_FMT_SMEM, 1,
      1, "digest §5.2 SMEM; [RDNA2-ISA] §13.2.1; ISA-XML ENC_SMEM op 1" },
    { VBS_RDNA2_OP_S_LOAD_DWORDX4, "S_LOAD_DWORDX4", VBS_RDNA2_FMT_SMEM, 2,
      1, "digest §5.2 SMEM; [RDNA2-ISA] §13.2.1" },
    { VBS_RDNA2_OP_S_LOAD_DWORDX8, "S_LOAD_DWORDX8", VBS_RDNA2_FMT_SMEM, 3,
      1, "digest §5.2 SMEM; [RDNA2-ISA] §13.2.1" },
    { VBS_RDNA2_OP_S_LOAD_DWORDX16, "S_LOAD_DWORDX16", VBS_RDNA2_FMT_SMEM, 4,
      1, "digest §5.2 SMEM; [RDNA2-ISA] §13.2.1" },
    /* VOP1 — [RDNA2-ISA] §13.3.2; digest §5.2. */
    { VBS_RDNA2_OP_V_MOV_B32, "V_MOV_B32", VBS_RDNA2_FMT_VOP1, 1,
      1, "digest §5.2 VOP1; [RDNA2-ISA] §13.3.2; ISA-XML ENC_VOP1 op 1" },
    /* VOP2 — [RDNA2-ISA] §13.3.1; digest §5.2. */
    { VBS_RDNA2_OP_V_ADD_F32, "V_ADD_F32", VBS_RDNA2_FMT_VOP2, 3,
      1, "[RDNA2-ISA] §13.3.1 VOP2 table; ISA-XML ENC_VOP2 op 3" },
    { VBS_RDNA2_OP_V_MUL_F32, "V_MUL_F32", VBS_RDNA2_FMT_VOP2, 8,
      1, "[RDNA2-ISA] §13.3.1 VOP2 table; ISA-XML ENC_VOP2 op 8" },
    { VBS_RDNA2_OP_V_ADD_NC_U32, "V_ADD_NC_U32", VBS_RDNA2_FMT_VOP2, 37,
      1, "digest §5.2 VOP2; [RDNA2-ISA] §13.3.1; ISA-XML ENC_VOP2 op 37" },
    { VBS_RDNA2_OP_V_ADD_CO_CI_U32, "V_ADD_CO_CI_U32", VBS_RDNA2_FMT_VOP2, 40,
      1, "digest §5.2 VOP2; [RDNA2-ISA] §13.3.1; ISA-XML ENC_VOP2 op 40" },
    /* VOP3A native — [RDNA2-ISA] §13.3.4 Table 83; digest §5.2. */
    { VBS_RDNA2_OP_V_MAD_I32_I24, "V_MAD_I32_I24", VBS_RDNA2_FMT_VOP3A, 322,
      1, "[RDNA2-ISA] §13.3.4 Table 83; ISA-XML ENC_VOP3 op 322" },
    { VBS_RDNA2_OP_V_BFE_U32, "V_BFE_U32", VBS_RDNA2_FMT_VOP3A, 328,
      1, "[RDNA2-ISA] §13.3.4 Table 83; ISA-XML ENC_VOP3 op 328" },
    { VBS_RDNA2_OP_V_BFI_B32, "V_BFI_B32", VBS_RDNA2_FMT_VOP3A, 330,
      1, "[RDNA2-ISA] §13.3.4 Table 83; ISA-XML ENC_VOP3 op 330" },
    /* VOP3A re-encodes — ISA-XML ENC_VOP3 entries (VOP2 +256, VOP1 +384). */
    { VBS_RDNA2_OP_V_MOV_B32_VOP3, "V_MOV_B32 (VOP3A)", VBS_RDNA2_FMT_VOP3A, 385,
      1, "ISA-XML ENC_VOP3 op 385 for V_MOV_B32 (VOP1 op 1 + 384)" },
    { VBS_RDNA2_OP_V_ADD_F32_VOP3, "V_ADD_F32 (VOP3A)", VBS_RDNA2_FMT_VOP3A, 259,
      1, "ISA-XML ENC_VOP3 op 259 for V_ADD_F32 (VOP2 op 3 + 256)" },
    { VBS_RDNA2_OP_V_MUL_F32_VOP3, "V_MUL_F32 (VOP3A)", VBS_RDNA2_FMT_VOP3A, 264,
      1, "ISA-XML ENC_VOP3 op 264 for V_MUL_F32 (VOP2 op 8 + 256)" },
    { VBS_RDNA2_OP_V_ADD_NC_U32_VOP3, "V_ADD_NC_U32 (VOP3A)", VBS_RDNA2_FMT_VOP3A, 293,
      1, "ISA-XML ENC_VOP3 op 293 for V_ADD_NC_U32 (VOP2 op 37 + 256)" },
    /* VOP3B — [RDNA2-ISA] §13.3.5; digest §5.2. */
    { VBS_RDNA2_OP_V_ADD_CO_U32, "V_ADD_CO_U32", VBS_RDNA2_FMT_VOP3B, 783,
      1, "digest §5.2 VOP3B; [RDNA2-ISA] §13.3.5; ISA-XML VOP3_SDST_ENC op 783" },
    /* FLAT / GLOBAL — [RDNA2-ISA] §12.8 / §13.8 Table 103; digest §5.2. */
    { VBS_RDNA2_OP_GLOBAL_LOAD_DWORD, "GLOBAL_LOAD_DWORD", VBS_RDNA2_FMT_FLAT, 12,
      1, "digest §5.2 FLAT; [RDNA2-ISA] §13.8 Table 103; ISA-XML ENC_FLAT_GLBL op 12" },
    { VBS_RDNA2_OP_GLOBAL_LOAD_DWORDX2, "GLOBAL_LOAD_DWORDX2", VBS_RDNA2_FMT_FLAT, 13,
      1, "digest §5.2 FLAT; [RDNA2-ISA] §13.8 Table 103" },
    { VBS_RDNA2_OP_GLOBAL_LOAD_DWORD_ADDTID, "GLOBAL_LOAD_DWORD_ADDTID", VBS_RDNA2_FMT_FLAT, 22,
      1, "digest §5.2 FLAT; [RDNA2-ISA] §12.8 + §13.8 Table 103; ISA-XML ENC_FLAT_GLBL op 22" },
    { VBS_RDNA2_OP_GLOBAL_STORE_DWORD, "GLOBAL_STORE_DWORD", VBS_RDNA2_FMT_FLAT, 28,
      1, "digest §5.2 FLAT; [RDNA2-ISA] §13.8 Table 103; ISA-XML ENC_FLAT_GLBL op 28" },
    { VBS_RDNA2_OP_GLOBAL_STORE_DWORDX2, "GLOBAL_STORE_DWORDX2", VBS_RDNA2_FMT_FLAT, 29,
      1, "digest §5.2 FLAT; [RDNA2-ISA] §13.8 Table 103" },
    { VBS_RDNA2_OP_GLOBAL_STORE_DWORDX4, "GLOBAL_STORE_DWORDX4", VBS_RDNA2_FMT_FLAT, 30,
      1, "digest §5.2 FLAT; [RDNA2-ISA] §13.8 Table 103" },
    { VBS_RDNA2_OP_GLOBAL_STORE_DWORD_ADDTID, "GLOBAL_STORE_DWORD_ADDTID", VBS_RDNA2_FMT_FLAT, 23,
      1, "digest §5.2 FLAT; [RDNA2-ISA] §12.8 + §13.8 Table 103; ISA-XML ENC_FLAT_GLBL op 23" },
};

/* Number of rows in the opcode table. */
#define VBS_RDNA2_OP_TABLE_COUNT \
    (sizeof(VBS_RDNA2_OP_TABLE) / sizeof(VBS_RDNA2_OP_TABLE[0]))

/* Look an opcode-table entry up by kind. Returns NULL for an
 * out-of-range kind. The emitters use this, never a literal. */
static inline const vbs_rdna2_op_entry_t *vbs_rdna2_op_lookup(
        vbs_rdna2_op_t kind) {
    if ((int)kind < 0 || (unsigned)kind >= VBS_RDNA2_OP_TABLE_COUNT)
        return NULL;
    /* Table rows are stored in kind order; verify rather than trust. */
    if (VBS_RDNA2_OP_TABLE[kind].kind != kind)
        return NULL;
    return &VBS_RDNA2_OP_TABLE[kind];
}

/* Opcode number for a kind, or 0 with *ok = 0 when the kind is
 * unknown or the entry is UNSOURCED (never emit a guessed opcode). */
static inline uint16_t vbs_rdna2_opcode(vbs_rdna2_op_t kind, int *ok) {
    const vbs_rdna2_op_entry_t *e = vbs_rdna2_op_lookup(kind);
    if (!e || !e->sourced) {
        if (ok)
            *ok = 0;
        return 0;
    }
    if (ok)
        *ok = 1;
    return e->opcode;
}

#endif /* VBS_RDNA2_ISA_H */
