/* vbs_rdna2_vecadd_list.c — listing generator for the vecadd kernel.
 *
 * Assembles the kernel with the real assembler and prints one
 * line per instruction: byte address, encoding (DW0, then DW1
 * for 64-bit instructions), mnemonic, operands, and the spec
 * citation for the encoding. Then prints the kernel descriptor
 * and the ELF code-object layout.
 *
 * The description table below MUST match vbs_rdna2_vecadd_build
 * instruction for instruction; the unit tests assert the counts
 * agree and that every emitted instruction's bit fields match
 * the description. Usage: ./vbs_rdna2_vecadd_list
 * (stdout is the listing; redirect it to
 * vbs_rdna2_vecadd_listing.txt).
 */

#include <stdio.h>

#include "vbs_rdna2_vecadd.h"

/* One row per kernel instruction: mnemonic, operands, citation.
 * Citations: "digest §x" = SPEC-DIGEST.md (worker D);
 * "[RDNA2-ISA]" = AMD RDNA 2 ISA Reference Guide; "ISA-XML" =
 * AMD machine-readable RDNA 2 ISA. */
static const struct {
    const char *text;       /* "MNEMONIC operands" */
    const char *citation;
} LINES[VBS_RDNA2_VECADD_NUM_INSTS] = {
    { "GLOBAL_LOAD_DWORD_ADDTID v0, s[0:1]",
      "op 22, ENC 0x37, SEG=2 (digest §5.2 FLAT; [RDNA2-ISA] §12.8, §13.8; ISA-XML ENC_FLAT_GLBL op 22); ADDR=0 UNSOURCED" },
    { "GLOBAL_LOAD_DWORD_ADDTID v1, s[2:3]",
      "op 22, ENC 0x37, SEG=2 (digest §5.2 FLAT; [RDNA2-ISA] §12.8, §13.8; ISA-XML ENC_FLAT_GLBL op 22); ADDR=0 UNSOURCED" },
    { "S_WAITCNT vmcnt(0) expcnt(0) lgkmcnt(0)",
      "SOPP op 12, SIMM16=0 (digest §5.2 SOPP; [RDNA2-ISA] §13.1.5; ISA-XML ENC_SOPP op 12)" },
    { "V_ADD_NC_U32 v2, v0, v1",
      "VOP2 op 37 (digest §5.2 VOP2 + §5.3 oracle 0x4A040300; [RDNA2-ISA] §13.3.1; ISA-XML ENC_VOP2 op 37)" },
    { "GLOBAL_STORE_DWORD_ADDTID v2, s[4:5]",
      "op 23, ENC 0x37, SEG=2 (digest §5.2 FLAT; [RDNA2-ISA] §12.8, §13.8; ISA-XML ENC_FLAT_GLBL op 23); ADDR=0 UNSOURCED" },
    { "S_WAITCNT vmcnt(0) expcnt(0) lgkmcnt(0)",
      "SOPP op 12, SIMM16=0 (digest §5.2 SOPP; [RDNA2-ISA] §13.1.5; ISA-XML ENC_SOPP op 12)" },
    { "S_ENDPGM",
      "SOPP op 1 (digest §5.2 SOPP + §5.3 oracle 0xBF810000; [RDNA2-ISA] §13.1.5; ISA-XML ENC_SOPP op 1)" },
};

int main(void) {
    vbs_rdna2_inst_t insts[VBS_RDNA2_VECADD_NUM_INSTS];
    uint8_t desc[VBS_RDNA2_DESC_SIZE];
    uint8_t code[VBS_RDNA2_VECADD_CODE_SIZE];
    vbs_rdna2_vecadd_dispatch_t disp;
    size_t n, off, i;

    n = vbs_rdna2_vecadd_build(insts, VBS_RDNA2_VECADD_NUM_INSTS);
    vbs_rdna2_vecadd_descriptor(desc);
    vbs_rdna2_vecadd_serialize(code);
    vbs_rdna2_vecadd_dispatch(0x10000000ull, 0x20000000ull,
                              0x30000000ull, 1024, &disp);

    printf("Verum RDNA 2 (gfx1030) vector-add kernel — c[i] = a[i] + b[i]\n");
    printf("wave64, 64 threads/group, USER_SGPR = 6 (s0-s5 = A/B/C VAs)\n");
    printf("VGPRs v0-v2, no LDS, no scratch. 7 instructions, 40 bytes.\n");
    printf("\n");
    printf("addr  encoding             instruction\n");
    printf("----  -------------------  -----------\n");
    off = 0;
    for (i = 0; i < n; i++) {
        if (insts[i].is64)
            printf("0x%02zx  %08X %08X  %s\n",
                   off, insts[i].dw0, insts[i].dw1, LINES[i].text);
        else
            printf("0x%02zx  %08X           %s\n",
                   off, insts[i].dw0, LINES[i].text);
        printf("      ; %s\n", LINES[i].citation);
        off += vbs_rdna2_inst_size(&insts[i]);
    }

    printf("\nraw code (%u bytes):\n  ", VBS_RDNA2_VECADD_CODE_SIZE);
    for (i = 0; i < VBS_RDNA2_VECADD_CODE_SIZE; i++)
        printf("%02X%s", code[i],
               (i + 1) % 16 == 0 ? "\n  " : " ");
    printf("\n");

    printf("\nkernel descriptor (64 bytes, [AMDGPUUsage] V3 layout):\n");
    printf("  GROUP_SEGMENT_FIXED_SIZE   = 0x%08X (no LDS)\n",
           (unsigned)desc[0] | ((unsigned)desc[1] << 8) |
           ((unsigned)desc[2] << 16) | ((unsigned)desc[3] << 24));
    printf("  PRIVATE_SEGMENT_FIXED_SIZE = 0x%08X (no scratch)\n",
           (unsigned)desc[4] | ((unsigned)desc[5] << 8) |
           ((unsigned)desc[6] << 16) | ((unsigned)desc[7] << 24));
    printf("  KERNARG_SIZE               = 0x%08X (args via USER_DATA)\n",
           (unsigned)desc[8] | ((unsigned)desc[9] << 8) |
           ((unsigned)desc[10] << 16) | ((unsigned)desc[11] << 24));
    printf("  KERNEL_CODE_ENTRY_BYTE_OFFSET = %u\n",
           VBS_RDNA2_VECADD_ENTRY_OFFSET);
    printf("  COMPUTE_PGM_RSRC3          = 0x%08X\n",
           VBS_RDNA2_VECADD_PGM_RSRC3);
    printf("  COMPUTE_PGM_RSRC1          = 0x%08X (VGPRS field 0: 3 VGPRs, wave64)\n",
           VBS_RDNA2_VECADD_PGM_RSRC1);
    printf("  COMPUTE_PGM_RSRC2          = 0x%08X (USER_SGPR 6)\n",
           VBS_RDNA2_VECADD_PGM_RSRC2);
    printf("  KERNEL_CODE_PROPERTIES     = 0x%08X (wave64, no descriptor SGPR setup)\n",
           0u);

    printf("\nELF code object (v5, ET_DYN, %u bytes):\n",
           VBS_RDNA2_ELF_SIZE);
    printf("  .rodata @ 0x%03X: descriptor (64 bytes, 64-aligned)\n",
           VBS_RDNA2_ELF_RODATA_OFF);
    printf("  .text   @ 0x%03X: code (40 bytes, 256-aligned entry)\n",
           VBS_RDNA2_ELF_TEXT_OFF);
    printf("  symbols: %s (STT_FUNC, .text), %s (STT_OBJECT, .rodata)\n",
           VBS_RDNA2_VECADD_SYM, VBS_RDNA2_VECADD_SYM_KD);
    printf("  e_machine = EM_AMDGPU (224), e_flags = 0x%03X (gfx1030)\n",
           VBS_RDNA2_ELF_FLAGS_GFX1030);

    printf("\nexample dispatch (n = 1024, A=0x10000000 B=0x20000000 C=0x30000000):\n");
    printf("  NUM_THREAD_X/Y/Z = %u/%u/%u, DIM_X/Y/Z = %u/%u/%u, initiator = 0x%X\n",
           disp.num_thread_x, disp.num_thread_y, disp.num_thread_z,
           disp.dim_x, disp.dim_y, disp.dim_z, disp.initiator);
    printf("  USER_DATA_0..5 = %08X %08X %08X %08X %08X %08X\n",
           disp.user_data[0], disp.user_data[1], disp.user_data[2],
           disp.user_data[3], disp.user_data[4], disp.user_data[5]);
    printf("  PGM_RSRC1/2/3 = %08X %08X %08X\n",
           disp.pgm_rsrc1, disp.pgm_rsrc2, disp.pgm_rsrc3);

    printf("\nNOTE: no bounds check — buffers must hold ceil(n/64)*64 elements.\n");
    printf("NOTE: ADDTID ADDR field = 0 is UNSOURCED (digest §10 item 5);\n");
    printf("      the hardware run must confirm it (see amd-shader-REPORT.md).\n");
    return 0;
}
