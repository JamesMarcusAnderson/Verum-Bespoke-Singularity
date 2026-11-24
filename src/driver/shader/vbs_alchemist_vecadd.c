/* vbs_alchemist_vecadd.c — minimal Alchemist (Xe-HPG) vector-add kernel.
 *
 * Follows the digest §6.9 7-step kernel sequence, with one
 * correction: the digest sketch does the 64-bit address math in
 * SIMD16, but Vol 9, p.387 forbids it (ExecSize * largest element
 * size <= 64, so SIMD16 :uq is illegal). All :uq work is split
 * into two SIMD8 halves. See vbs_alchemist_vecadd.h for the full
 * register map and the reasoning.
 *
 * All code here is original Verum work. The instruction encodings
 * come from the assembler (vbs_alchemist_eu.c) driven by the
 * documented ISA facts; nothing is transcribed from Mesa, Intel's
 * compute runtime, or any other implementation.
 */

#include "vbs_alchemist_vecadd.h"

#include <string.h>

/* Shorthand operand builders (regions per the opcodes header). */
#define OP_F32(r) \
    vbs_alchemist_op((r), 0, VBS_ALCHEMIST_TYPE_F, \
                     VBS_ALCHEMIST_REGION_F32_V, \
                     VBS_ALCHEMIST_REGION_F32_W, \
                     VBS_ALCHEMIST_REGION_F32_H)
#define OP_UD(r) \
    vbs_alchemist_op((r), 0, VBS_ALCHEMIST_TYPE_UD, \
                     VBS_ALCHEMIST_REGION_F32_V, \
                     VBS_ALCHEMIST_REGION_F32_W, \
                     VBS_ALCHEMIST_REGION_F32_H)
#define OP_UQ8(r) \
    vbs_alchemist_op((r), 0, VBS_ALCHEMIST_TYPE_UQ, \
                     VBS_ALCHEMIST_REGION_UQ8_V, \
                     VBS_ALCHEMIST_REGION_UQ8_W, \
                     VBS_ALCHEMIST_REGION_UQ8_H)
#define OP_UW16(r) \
    vbs_alchemist_op((r), 0, VBS_ALCHEMIST_TYPE_UW, \
                     VBS_ALCHEMIST_REGION_UW16_V, \
                     VBS_ALCHEMIST_REGION_UW16_W, \
                     VBS_ALCHEMIST_REGION_UW16_H)
#define OP_SCALAR_UQ(r, sub) \
    vbs_alchemist_op((r), (sub), VBS_ALCHEMIST_TYPE_UQ, \
                     VBS_ALCHEMIST_REGION_SCALAR_V, \
                     VBS_ALCHEMIST_REGION_SCALAR_W, \
                     VBS_ALCHEMIST_REGION_SCALAR_H)
#define DST_F32(r) \
    vbs_alchemist_dst((r), 0, VBS_ALCHEMIST_TYPE_F, 1)
#define DST_UD(r) \
    vbs_alchemist_dst((r), 0, VBS_ALCHEMIST_TYPE_UD, 1)
#define DST_UQ(r) \
    vbs_alchemist_dst((r), 0, VBS_ALCHEMIST_TYPE_UQ, 1)

/* Emit the 4-instruction block that builds one buffer's 64-bit
 * addresses: broadcast the base (from R2 at byte offset
 * base_subreg) into the temp r9-r12 in two SIMD8 halves, then add
 * the per-lane UQ offsets (r5-r8) into addr_reg..addr_reg+3.
 * `n` is the running instruction index. */
static int emit_addr_block(vbs_alchemist_inst_t *out, unsigned *n,
                           uint8_t base_subreg, uint8_t addr_reg)
{
    /* mov (8) r9 :uq = r2.base_subreg<0;1,0>:uq — broadcast, lo half */
    if (vbs_alchemist_emit_mov(&out[(*n)++], VBS_ALCHEMIST_EXECSIZE_8,
                               DST_UQ(VBS_ALCHEMIST_VECADD_R_BASE),
                               OP_SCALAR_UQ(VBS_ALCHEMIST_VECADD_R_INLINE,
                                            base_subreg)) != 0)
        return -1;
    /* mov (8) r11 :uq = r2.base_subreg<0;1,0>:uq — broadcast, hi half */
    if (vbs_alchemist_emit_mov(&out[(*n)++], VBS_ALCHEMIST_EXECSIZE_8,
                               DST_UQ(VBS_ALCHEMIST_VECADD_R_BASE + 2),
                               OP_SCALAR_UQ(VBS_ALCHEMIST_VECADD_R_INLINE,
                                            base_subreg)) != 0)
        return -1;
    /* add (8) addr :uq = off_uq_lo + base_lo */
    if (vbs_alchemist_emit_add(&out[(*n)++], VBS_ALCHEMIST_EXECSIZE_8,
                               DST_UQ(addr_reg),
                               OP_UQ8(VBS_ALCHEMIST_VECADD_R_OFF_UQ),
                               OP_UQ8(VBS_ALCHEMIST_VECADD_R_BASE)) != 0)
        return -1;
    /* add (8) addr+2 :uq = off_uq_hi + base_hi */
    if (vbs_alchemist_emit_add(&out[(*n)++], VBS_ALCHEMIST_EXECSIZE_8,
                               DST_UQ(addr_reg + 2),
                               OP_UQ8(VBS_ALCHEMIST_VECADD_R_OFF_UQ + 2),
                               OP_UQ8(VBS_ALCHEMIST_VECADD_R_BASE + 2)) != 0)
        return -1;
    return 0;
}

int vbs_alchemist_vecadd_insts(vbs_alchemist_inst_t out[VBS_ALCHEMIST_VECADD_NUM_INSTS])
{
    unsigned n = 0;
    uint32_t desc_read, desc_write, exdesc_write, desc_eot;

    if (out == NULL)
        return -1;

    /* Message descriptors, built from fields (digest §6.6, §6.8).
     * The tests assert these equal the digest's computed literals
     * (0x08246EFF / 0x08066EFF / 0x02000000 / 0x80). */
    desc_read = vbs_alchemist_desc_build(
        4, 2, 0, VBS_ALCHEMIST_MSG_A64_READ,
        VBS_ALCHEMIST_SIMD_MODE_16, VBS_ALCHEMIST_CMASK_R,
        VBS_ALCHEMIST_BTI_STATELESS);
    desc_write = vbs_alchemist_desc_build(
        4, 0, 0, VBS_ALCHEMIST_MSG_A64_WRITE,
        VBS_ALCHEMIST_SIMD_MODE_16, VBS_ALCHEMIST_CMASK_R,
        VBS_ALCHEMIST_BTI_STATELESS);
    exdesc_write = vbs_alchemist_exdesc_build(2);
    desc_eot = vbs_alchemist_desc_build(1, 0, 0, 0, 0, 0, 0);

    /* -- Step 1 (digest §6.9): per-lane byte offsets and 64-bit
     * addresses. -- */

    /* 0: mov (16) r3:ud = r1:uw — local ID U16 -> U32. */
    if (vbs_alchemist_emit_mov(&out[n++], VBS_ALCHEMIST_EXECSIZE_16,
                               DST_UD(VBS_ALCHEMIST_VECADD_R_OFF),
                               OP_UW16(VBS_ALCHEMIST_VECADD_R_LOCALID)) != 0)
        return -1;
    /* 1: shl (16) r3:ud = r3:ud, 2 — byte offset = id * 4. */
    if (vbs_alchemist_emit_shl_imm(&out[n++], VBS_ALCHEMIST_EXECSIZE_16,
                                   DST_UD(VBS_ALCHEMIST_VECADD_R_OFF),
                                   OP_UD(VBS_ALCHEMIST_VECADD_R_OFF),
                                   2, VBS_ALCHEMIST_TYPE_UD) != 0)
        return -1;
    /* 2: mov (8) r5:uq = r3:ud — offsets, lanes 0-7, to UQ. */
    if (vbs_alchemist_emit_mov(&out[n++], VBS_ALCHEMIST_EXECSIZE_8,
                               DST_UQ(VBS_ALCHEMIST_VECADD_R_OFF_UQ),
                               OP_UD(VBS_ALCHEMIST_VECADD_R_OFF)) != 0)
        return -1;
    /* 3: mov (8) r7:uq = r4:ud — offsets, lanes 8-15, to UQ. */
    if (vbs_alchemist_emit_mov(&out[n++], VBS_ALCHEMIST_EXECSIZE_8,
                               DST_UQ(VBS_ALCHEMIST_VECADD_R_OFF_UQ + 2),
                               OP_UD(VBS_ALCHEMIST_VECADD_R_OFF + 1)) != 0)
        return -1;
    /* 4-7: A addresses (base at R2 bytes 0-7). */
    if (emit_addr_block(out, &n, 0, VBS_ALCHEMIST_VECADD_R_ADDR_A) != 0)
        return -1;
    /* 8-11: B addresses (base at R2 bytes 8-15). */
    if (emit_addr_block(out, &n, 8, VBS_ALCHEMIST_VECADD_R_ADDR_B) != 0)
        return -1;
    /* 12-15: C addresses (base at R2 bytes 16-23). */
    if (emit_addr_block(out, &n, 16, VBS_ALCHEMIST_VECADD_R_ADDR_C) != 0)
        return -1;

    /* -- Step 2: load A (SIMD16 A64 untyped read, DC1). -- */
    if (vbs_alchemist_emit_send(&out[n++], VBS_ALCHEMIST_EXECSIZE_16,
                                VBS_ALCHEMIST_SFID_DP_DC1,
                                VBS_ALCHEMIST_VECADD_R_DATA_A,
                                VBS_ALCHEMIST_VECADD_R_ADDR_A,
                                VBS_ALCHEMIST_REG_NULL,
                                desc_read, 0, 0) != 0)
        return -1;
    /* -- Step 3: load B. -- */
    if (vbs_alchemist_emit_send(&out[n++], VBS_ALCHEMIST_EXECSIZE_16,
                                VBS_ALCHEMIST_SFID_DP_DC1,
                                VBS_ALCHEMIST_VECADD_R_DATA_B,
                                VBS_ALCHEMIST_VECADD_R_ADDR_B,
                                VBS_ALCHEMIST_REG_NULL,
                                desc_read, 0, 0) != 0)
        return -1;
    /* -- Step 4: add — one SIMD16 F32 add covers all 16 lanes. -- */
    if (vbs_alchemist_emit_add(&out[n++], VBS_ALCHEMIST_EXECSIZE_16,
                               DST_F32(VBS_ALCHEMIST_VECADD_R_DATA_C),
                               OP_F32(VBS_ALCHEMIST_VECADD_R_DATA_A),
                               OP_F32(VBS_ALCHEMIST_VECADD_R_DATA_B)) != 0)
        return -1;
    /* -- Step 5: store C (SIMD16 A64 untyped write, DC1). -- */
    if (vbs_alchemist_emit_send(&out[n++], VBS_ALCHEMIST_EXECSIZE_16,
                                VBS_ALCHEMIST_SFID_DP_DC1,
                                VBS_ALCHEMIST_REG_NULL,
                                VBS_ALCHEMIST_VECADD_R_ADDR_C,
                                VBS_ALCHEMIST_VECADD_R_DATA_C,
                                desc_write, exdesc_write, 0) != 0)
        return -1;
    /* -- Step 6: no fence instruction — the walker's PostSync /
     * trailing PIPE_CONTROL orders the stores (see the header). --
     *
     * -- Step 7: exit — EOT send to the Gateway, src0 = r112. -- */
    if (vbs_alchemist_emit_send(&out[n++], VBS_ALCHEMIST_EXECSIZE_1,
                                VBS_ALCHEMIST_SFID_GATEWAY,
                                VBS_ALCHEMIST_REG_NULL,
                                VBS_ALCHEMIST_VECADD_R_EOT,
                                VBS_ALCHEMIST_REG_NULL,
                                desc_eot, 0, 1) != 0)
        return -1;

    if (n != VBS_ALCHEMIST_VECADD_NUM_INSTS)
        return -1; /* internal count mismatch */
    return 0;
}

int vbs_alchemist_vecadd_build(uint8_t out[VBS_ALCHEMIST_VECADD_CODE_SIZE])
{
    vbs_alchemist_inst_t insts[VBS_ALCHEMIST_VECADD_NUM_INSTS];
    unsigned i;

    if (out == NULL)
        return -1;
    if (vbs_alchemist_vecadd_insts(insts) != 0)
        return -1;
    for (i = 0; i < VBS_ALCHEMIST_VECADD_NUM_INSTS; i++)
        vbs_alchemist_serialize(&insts[i], &out[16 * i]);
    return 0;
}

/* Listing text: one (mnemonic, operands, citation) row per
 * instruction, in assembly order. The ENCODING printed on each
 * line comes from the assembler (never from this table), so the
 * listing can never disagree with the binary. */
static const struct {
    const char *mnemonic;
    const char *operands;
    const char *citation;
} LIST_ROWS[VBS_ALCHEMIST_VECADD_NUM_INSTS] = {
    { "mov (16)",  "r3.0<1>:ud r1.0<16;16,1>:uw",
      "Vol 2a p.1111 (mov); digest §6.4, §6.9 step 1" },
    { "shl (16)",  "r3.0<1>:ud r3.0<8;8,1>:ud 2:ud",
      "Vol 2a p.1275 (shl); digest §6.4, §6.9 step 1" },
    { "mov (8)",   "r5.0<1>:uq r3.0<8;8,1>:ud",
      "Vol 2a p.1111 (mov); digest §6.9 step 1; Vol 9 p.387 (SIMD8 :uq)" },
    { "mov (8)",   "r7.0<1>:uq r4.0<8;8,1>:ud",
      "Vol 2a p.1111 (mov); digest §6.9 step 1; Vol 9 p.387 (SIMD8 :uq)" },
    { "mov (8)",   "r9.0<1>:uq r2.0<0;1,0>:uq",
      "Vol 2a p.1111 (mov); Vol 9 p.339 (<0;1,0> broadcast); digest §6.9 step 1" },
    { "mov (8)",   "r11.0<1>:uq r2.0<0;1,0>:uq",
      "Vol 2a p.1111 (mov); Vol 9 p.339 (<0;1,0> broadcast); digest §6.9 step 1" },
    { "add (8)",   "r13.0<1>:uq r5.0<4;4,1>:uq r9.0<4;4,1>:uq",
      "Vol 2a p.219 (add); digest §6.9 step 1; DERIVED :uq, see VBS_ALCHEMIST_ADD_UQ_NOTE" },
    { "add (8)",   "r15.0<1>:uq r7.0<4;4,1>:uq r11.0<4;4,1>:uq",
      "Vol 2a p.219 (add); digest §6.9 step 1; DERIVED :uq, see VBS_ALCHEMIST_ADD_UQ_NOTE" },
    { "mov (8)",   "r9.0<1>:uq r2.8<0;1,0>:uq",
      "Vol 2a p.1111 (mov); Vol 9 p.339 (<0;1,0> broadcast); digest §6.9 step 1" },
    { "mov (8)",   "r11.0<1>:uq r2.8<0;1,0>:uq",
      "Vol 2a p.1111 (mov); Vol 9 p.339 (<0;1,0> broadcast); digest §6.9 step 1" },
    { "add (8)",   "r17.0<1>:uq r5.0<4;4,1>:uq r9.0<4;4,1>:uq",
      "Vol 2a p.219 (add); digest §6.9 step 1; DERIVED :uq, see VBS_ALCHEMIST_ADD_UQ_NOTE" },
    { "add (8)",   "r19.0<1>:uq r7.0<4;4,1>:uq r11.0<4;4,1>:uq",
      "Vol 2a p.219 (add); digest §6.9 step 1; DERIVED :uq, see VBS_ALCHEMIST_ADD_UQ_NOTE" },
    { "mov (8)",   "r9.0<1>:uq r2.16<0;1,0>:uq",
      "Vol 2a p.1111 (mov); Vol 9 p.339 (<0;1,0> broadcast); digest §6.9 step 1" },
    { "mov (8)",   "r11.0<1>:uq r2.16<0;1,0>:uq",
      "Vol 2a p.1111 (mov); Vol 9 p.339 (<0;1,0> broadcast); digest §6.9 step 1" },
    { "add (8)",   "r21.0<1>:uq r5.0<4;4,1>:uq r9.0<4;4,1>:uq",
      "Vol 2a p.219 (add); digest §6.9 step 1; DERIVED :uq, see VBS_ALCHEMIST_ADD_UQ_NOTE" },
    { "add (8)",   "r23.0<1>:uq r7.0<4;4,1>:uq r11.0<4;4,1>:uq",
      "Vol 2a p.219 (add); digest §6.9 step 1; DERIVED :uq, see VBS_ALCHEMIST_ADD_UQ_NOTE" },
    { "send (16)", "r25:dc1 r13 null 0x0 0x08246EFF",
      "Vol 2a p.1253 (send), p.209 (MSD1R_A64_US); digest §6.5, §6.8, §6.9 step 2" },
    { "send (16)", "r27:dc1 r17 null 0x0 0x08246EFF",
      "Vol 2a p.1253 (send), p.209 (MSD1R_A64_US); digest §6.5, §6.8, §6.9 step 3" },
    { "add (16)",  "r29.0<1>:f r25.0<8;8,1>:f r27.0<8;8,1>:f",
      "Vol 2a p.219 (add); Vol 9 p.339 (<8;8,1>); digest §6.9 step 4" },
    { "send (16)", "null:dc1 r21 r29 0x80 0x08066EFF",
      "Vol 2a p.1253 (send), p.211 (MSD1W_A64_US); digest §6.5, §6.8, §6.9 step 5" },
    { "send (1)",  "null:gtwy r112 null 0x0 0x02000000 EOT",
      "Vol 2a p.1253 (send), pp.562-563 (MSD_EOT); digest §6.5, §6.7, §6.9 step 7" },
};

void vbs_alchemist_vecadd_list(FILE *out)
{
    vbs_alchemist_inst_t insts[VBS_ALCHEMIST_VECADD_NUM_INSTS];
    char line[512];
    unsigned i;

    if (out == NULL)
        return;
    if (vbs_alchemist_vecadd_insts(insts) != 0) {
        fprintf(out, "ERROR: kernel assembly failed\n");
        return;
    }
    fprintf(out,
            "Verum Alchemist (Xe-HPG) vector-add kernel — c[i] = a[i] + b[i]\n"
            "21 instructions x 16 bytes = 336 bytes, 64-byte aligned, no header\n"
            "Payload: R0 = thread payload, R1 = local X IDs (U16/lane),\n"
            "         R2 = inline data (A/B/C 64-bit base addresses)\n"
            "addr    encoding (DW3 DW2 DW1 DW0)            instruction\n"
            "-------- ------------------------------------ ---------------------------\n");
    for (i = 0; i < VBS_ALCHEMIST_VECADD_NUM_INSTS; i++) {
        if (vbs_alchemist_list_line(line, sizeof(line), 16 * i, &insts[i],
                                    LIST_ROWS[i].mnemonic,
                                    LIST_ROWS[i].operands,
                                    LIST_ROWS[i].citation) < 0)
            continue;
        fprintf(out, "%s\n", line);
    }
}

uint32_t vbs_alchemist_vecadd_walker_dw6(uint32_t threads_per_group)
{
    /* Local X count = T * 16 work items; DW6[9:0] = count - 1.
     * Y/Z maxima are 0 (their Emit bits are clear — Vol 2a, p.489:
     * "When an enable bit is not set, its corresponding Local
     * Maximum value in DW6 must be 0"). */
    if (threads_per_group < 1 || threads_per_group > 64)
        return 0;
    return (threads_per_group * 16u - 1u) & 0x3FFu;
}

int vbs_alchemist_vecadd_idd(uint32_t idd[8], uint64_t kernel_start_offset,
                             uint32_t threads_per_group)
{
    if (idd == NULL)
        return -1;
    if (threads_per_group < 1 || threads_per_group > 64)
        return -1;
    memset(idd, 0, 8 * sizeof(uint32_t));
    /* DW0-1: Kernel Start Pointer, byte offset from the Instruction
     * Base Address, 64-byte aligned (low 6 bits masked off). */
    idd[0] = (uint32_t)(kernel_start_offset & 0xFFFFFFC0ULL);
    idd[1] = (uint32_t)(kernel_start_offset >> 32);
    /* DW2: 0 (IEEE FP, FTZ denorms, no exceptions).
     * DW3: 0 (no samplers).
     * DW4: 0 (no binding table — A64 stateless, BTI 255). */
    /* DW5: Number of Threads in GPGPU Thread Group [9:0]; SLM 0,
     * barriers 0, TG dispatch size 8 (default), rounding RTNE. */
    idd[5] = threads_per_group & 0x3FFu;
    /* DW6-7: 0. */
    return 0;
}
