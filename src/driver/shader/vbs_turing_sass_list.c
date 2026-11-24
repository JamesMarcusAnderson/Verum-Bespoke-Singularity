/* vbs_turing_sass_list.c — listing generator for the vecadd kernel.
 *
 * Assembles the kernel with the real assembler and prints one
 * line per instruction: byte address, 128-bit encoding (hi word
 * then lo word, the order nvdisasm prints them), mnemonic,
 * operands, and the spec citation for the encoding.
 *
 * The description table below MUST match vbs_turing_vecadd_build
 * instruction for instruction; the unit tests assert the counts
 * agree and that every emitted instruction's bit fields match
 * the description. Usage: ./vbs_turing_sass_list
 * (stdout is the listing; redirect it to vecadd-listing.txt).
 */

#include <stdio.h>

#include "vbs_turing_vecadd.h"

/* One row per kernel instruction: mnemonic, operands, citation.
 * Citations: "digest §x" = SPEC-DIGEST.md; "S5" = CUDA Binary
 * Utilities doc (NVIDIA-published nvdisasm output); "S11" =
 * turingas opcode/flag tables (studied as hardware facts);
 * "S1" = NVIDIA open-gpu-doc clc5c0qmd.h. */
static const struct {
    const char *text;       /* "MNEMONIC operands" */
    const char *citation;
} LINES[VBS_TURING_VECADD_NUM_INSTS] = {
    { "LDC.64 R0, c[0x0][0x00]",
      "opcode 0xb82 [X2] S11+S5 sm_100; .64 type 5<<9; 0xFF<<88 rule (digest §5.3)" },
    { "LDC.64 R2, c[0x0][0x08]",
      "opcode 0xb82 [X2] S11+S5 sm_100; .64 type 5<<9; 0xFF<<88 rule (digest §5.3)" },
    { "LDC.64 R4, c[0x0][0x10]",
      "opcode 0xb82 [X2] S11+S5 sm_100; .64 type 5<<9; 0xFF<<88 rule (digest §5.3)" },
    { "LDC R8, c[0x0][0x1c]",
      "opcode 0xb82 [X2] S11+S5 sm_100; .32 type 4<<9 (digest §10 item 1, resolved)" },
    { "LDC R10, c[0x0][0x18]",
      "opcode 0xb82 [X2] S11+S5 sm_100; .32 type 4<<9 (digest §10 item 1, resolved)" },
    { "S2R R6, SR_TID.X",
      "opcode 0x919 [X2] S11+S5 sm_100; sr SR_TID.X = 33<<8 (S11 flag table, S5-confirmed)" },
    { "S2R R7, SR_CTAID.X",
      "opcode 0x919 [X2] S11+S5 sm_100; sr SR_CTAID.X = 37<<8 (S11 flag table, S5-confirmed via S2UR)" },
    { "IMAD R9, R7, R8, R6",
      "opcode 0x224 [X2] S11+S5 sm_70/sm_100; .S32 default 1<<9; nopd0/nops0 fillers" },
    { "ISETP.GE.U32.AND P0, PT, R9, R10, PT",
      "opcode 0x20c [T1] S11 only; .GE 6<<12; 0x7<<4 rule; no nvdisasm hex found — validate on hardware" },
    { "@P0 EXIT",
      "opcode 0x94d [X2] S11+S5 sm_70/sm_100; pred P0 at bits 76-79; cp PT at bits 23-25" },
    { "IMAD.WIDE R0, R9, 0x4, R0",
      "opcode 0x225 + ICR 0x8 -> 0x825 [X2] S11+S5 sm_100; .S32 default; nopd0/nops0 fillers" },
    { "IMAD.WIDE R2, R9, 0x4, R2",
      "opcode 0x225 + ICR 0x8 -> 0x825 [X2] S11+S5 sm_100; .S32 default; nopd0/nops0 fillers" },
    { "IMAD.WIDE R4, R9, 0x4, R4",
      "opcode 0x225 + ICR 0x8 -> 0x825 [X2] S11+S5 sm_100; .S32 default; nopd0/nops0 fillers" },
    { "LDG.E R11, [R0]",
      "opcode 0x381 [X2] digest §5.3 + S5 sm_70; .E 1<<8, .32 4<<9, .SYS 3<<13, strong 1, cache 1, ldgp PT" },
    { "LDG.E R12, [R2]",
      "opcode 0x381 [X2] digest §5.3 + S5 sm_70; .E 1<<8, .32 4<<9, .SYS 3<<13, strong 1, cache 1, ldgp PT" },
    { "FADD R13, R11, R12",
      "opcode 0x221 [X2] S11+S5 sm_100 (digest §10 item 1, resolved)" },
    { "STG.E [R4], R13",
      "opcode 0x386 [X2] S11+S5 sm_70/sm_100; .E 1<<8, .32 4<<9, .SYS 3<<13, strong 1, cache 1" },
    { "EXIT",
      "opcode 0x94d [X2] S11+S5 sm_70/sm_100; cp PT at bits 23-25 (lo bits 0-40 = 0x03800000)" },
};

int main(void) {
    vbs_turing_inst_t insts[VBS_TURING_VECADD_NUM_INSTS];
    size_t n = vbs_turing_vecadd_build(insts,
                                       VBS_TURING_VECADD_NUM_INSTS);
    if (n != VBS_TURING_VECADD_NUM_INSTS) {
        fprintf(stderr, "vecadd build failed (got %zu insts)\n", n);
        return 1;
    }

    printf("Verum vecadd — Turing (sm_75) vector-add kernel\n");
    printf("c[i] = a[i] + b[i], one float per thread, bounds-checked\n");
    printf("18 instructions, 288 bytes, 14 registers (R0-R13)\n");
    printf("No shader program header: first instruction at offset 0\n");
    printf("  (digest §5.5; all published cuobjdump samples start at /*0000*/)\n");
    printf("Params in c[0]: a @0x00 (u64), b @0x08 (u64), c @0x10 (u64),\n");
    printf("  n @0x18 (u32), ntid.x @0x1c (u32). CB0 binding: 256 bytes.\n");
    printf("\n");
    printf("addr  encoding (hi word, then lo word)              instruction\n");
    printf("----  ------------------------------------------  -----------\n");
    for (size_t i = 0; i < n; i++) {
        printf("/*%04zx*/  0x%016llx 0x%016llx  %s\n",
               i * 16,
               (unsigned long long)insts[i].hi,
               (unsigned long long)insts[i].lo,
               LINES[i].text);
        printf("         ; %s\n", LINES[i].citation);
    }
    printf("\n");
    printf("Control-code key: stall:yield:wrtdb:readb:wait-mask live in\n");
    printf("lo-word bits 41-63. Barrier plan: 0=LDC results (reused by\n");
    printf("LDG), 1=S2R results, 2=idx, 3=P0, 4=addresses. Only\n");
    printf("barriers 0-4 are waited on (digest §5.2 wait mask = bits\n");
    printf("52-56), so the kernel is correct whether the mask is 5 or\n");
    printf("6 bits wide.\n");
    return 0;
}
