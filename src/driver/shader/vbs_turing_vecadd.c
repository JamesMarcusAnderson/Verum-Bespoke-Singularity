/* vbs_turing_vecadd.c — minimal Turing (sm_75) vector-add kernel.
 *
 * Instruction list (18 instructions, addresses 0x000-0x110):
 *
 *   addr  instruction                          dependency plan
 *   ----  -----------                          ---------------
 *   0x000 LDC.64 R0, c[0][0x00]   ; a           wrtdb 0
 *   0x010 LDC.64 R2, c[0][0x08]   ; b           wrtdb 0
 *   0x020 LDC.64 R4, c[0][0x10]   ; c           wrtdb 0
 *   0x030 LDC    R8, c[0][0x1c]   ; ntid.x      wrtdb 0
 *   0x040 LDC    R10, c[0][0x18]  ; n           wrtdb 0
 *   0x050 S2R    R6, SR_TID.X                   wrtdb 1
 *   0x060 S2R    R7, SR_CTAID.X                 wrtdb 1
 *   0x070 IMAD   R9, R7, R8, R6   ; idx         watdb 0b011, wrtdb 2
 *   0x080 ISETP.GE.U32.AND P0, PT, R9, R10, PT  watdb 0b101, wrtdb 3
 *   0x090 @P0 EXIT                              watdb 0b1000, stall 5
 *   0x0a0 IMAD.WIDE R0, R9, 0x4, R0 ; a addr    watdb 0b101, wrtdb 4
 *   0x0b0 IMAD.WIDE R2, R9, 0x4, R2 ; b addr    watdb 0b101, wrtdb 4
 *   0x0c0 IMAD.WIDE R4, R9, 0x4, R4 ; c addr    watdb 0b101, wrtdb 4
 *   0x0d0 LDG.E  R11, [R0]                      watdb 0b10000, wrtdb 0
 *   0x0e0 LDG.E  R12, [R2]                      watdb 0b10000, wrtdb 0
 *   0x0f0 FADD   R13, R11, R12                  watdb 0b00001, stall 5, Y
 *   0x100 STG.E  [R4], R13                      stall 2
 *   0x110 EXIT                                  stall 5
 *
 * Dependency-barrier plan (barriers 0-4 only; the wait mask is
 * documented as bits 52-56 = barriers 0-4, digest §5.2 — the
 * kernel never waits on barrier 5, so it is correct whether the
 * mask is 5 or 6 bits wide):
 *   barrier 0: the 5 LDC results (pointers, ntid, n). Reused by
 *     the 2 LDGs after every barrier-0 consumer has waited —
 *     the same reuse pattern ptxas emits (S5 sm_100 sample:
 *     LDC.64 R2 and both LDGs write barrier 2).
 *   barrier 1: the 2 S2R results (tid, ctaid).
 *   barrier 2: IMAD result (idx).
 *   barrier 3: ISETP result (P0).
 *   barrier 4: the 3 IMAD.WIDE results (addresses).
 * Short ALU latencies are covered by stalls instead of barriers
 * (FADD stall 5 covers the FADD->STG dependency, exactly as in
 * the S5 sm_100 sample; @P0 EXIT carries both a wait and a
 * stall of 5 so the ISETP predicate is ready even if the wait
 * mask were ignored).
 *
 * No bar.sync: the kernel has no shared memory and no
 * cross-thread communication, and neither published ptxas
 * sample emits one for this kernel shape. The BAR.SYNC
 * emitter exists for future shared-memory kernels.
 */

#include "vbs_turing_vecadd.h"

size_t vbs_turing_vecadd_build(vbs_turing_inst_t *out,
                               size_t capacity) {
    if (!out || capacity < VBS_TURING_VECADD_NUM_INSTS)
        return 0;

    const unsigned PT = VBS_TURING_PRED_PT;
    size_t i = 0;

    /* Shorthand: ctrl(stall, yield, wrtdb, readb, watdb). */
#define CTRL(s, y, w, r, m) ((vbs_turing_ctrl_t){ (s), (y), (w), (r), (m) })

    /* 0x000: load the kernel arguments from c[0]. */
    out[i++] = vbs_turing_emit_ldc(VBS_TURING_VECADD_R_A_LO, 0,
                                   VBS_TURING_VECADD_CB_A_OFF, 1,
                                   PT, CTRL(1, 1, 0, 7, 0));
    out[i++] = vbs_turing_emit_ldc(VBS_TURING_VECADD_R_B_LO, 0,
                                   VBS_TURING_VECADD_CB_B_OFF, 1,
                                   PT, CTRL(1, 1, 0, 7, 0));
    out[i++] = vbs_turing_emit_ldc(VBS_TURING_VECADD_R_C_LO, 0,
                                   VBS_TURING_VECADD_CB_C_OFF, 1,
                                   PT, CTRL(1, 1, 0, 7, 0));
    out[i++] = vbs_turing_emit_ldc(VBS_TURING_VECADD_R_NTID, 0,
                                   VBS_TURING_VECADD_CB_NTID_OFF, 0,
                                   PT, CTRL(1, 1, 0, 7, 0));
    out[i++] = vbs_turing_emit_ldc(VBS_TURING_VECADD_R_N, 0,
                                   VBS_TURING_VECADD_CB_N_OFF, 0,
                                   PT, CTRL(1, 1, 0, 7, 0));

    /* 0x050: thread/block indices from special registers. */
    out[i++] = vbs_turing_emit_s2r(VBS_TURING_VECADD_R_TID,
                                   VBS_TURING_SR_TID_X,
                                   PT, CTRL(1, 1, 1, 7, 0));
    out[i++] = vbs_turing_emit_s2r(VBS_TURING_VECADD_R_CTAID,
                                   VBS_TURING_SR_CTAID_X,
                                   PT, CTRL(1, 1, 1, 7, 0));

    /* 0x070: idx = ctaid.x * ntid.x + tid.x. */
    out[i++] = vbs_turing_emit_imad(VBS_TURING_VECADD_R_IDX,
                                    VBS_TURING_VECADD_R_CTAID,
                                    VBS_TURING_VECADD_R_NTID,
                                    VBS_TURING_VECADD_R_TID,
                                    PT, CTRL(1, 1, 2, 7, 0x3));

    /* 0x080: P0 = (idx >= n), unsigned. */
    out[i++] = vbs_turing_emit_isetp_ge_u32(0,
                                            VBS_TURING_VECADD_R_IDX,
                                            VBS_TURING_VECADD_R_N,
                                            PT, CTRL(1, 1, 3, 7, 0x5));

    /* 0x090: out-of-range threads exit here. */
    out[i++] = vbs_turing_emit_exit(0, CTRL(5, 1, 7, 7, 0x8));

    /* 0x0a0: byte addresses = base + idx * 4 (in place). */
    out[i++] = vbs_turing_emit_imad_wide_imm(
            VBS_TURING_VECADD_R_A_LO, VBS_TURING_VECADD_R_IDX, 4,
            VBS_TURING_VECADD_R_A_LO,
            PT, CTRL(1, 1, 4, 7, 0x5));
    out[i++] = vbs_turing_emit_imad_wide_imm(
            VBS_TURING_VECADD_R_B_LO, VBS_TURING_VECADD_R_IDX, 4,
            VBS_TURING_VECADD_R_B_LO,
            PT, CTRL(1, 1, 4, 7, 0x5));
    out[i++] = vbs_turing_emit_imad_wide_imm(
            VBS_TURING_VECADD_R_C_LO, VBS_TURING_VECADD_R_IDX, 4,
            VBS_TURING_VECADD_R_C_LO,
            PT, CTRL(1, 1, 4, 7, 0x5));

    /* 0x0d0: load the operands. */
    out[i++] = vbs_turing_emit_ldg(VBS_TURING_VECADD_R_AVAL,
                                   VBS_TURING_VECADD_R_A_LO, 0,
                                   PT, CTRL(1, 1, 0, 7, 0x10));
    out[i++] = vbs_turing_emit_ldg(VBS_TURING_VECADD_R_BVAL,
                                   VBS_TURING_VECADD_R_B_LO, 0,
                                   PT, CTRL(1, 1, 0, 7, 0x10));

    /* 0x0f0: the add. Yield while waiting on the loads (matches
     * the S5 sm_100 sample's FADD control). */
    out[i++] = vbs_turing_emit_fadd(VBS_TURING_VECADD_R_SUM,
                                    VBS_TURING_VECADD_R_AVAL,
                                    VBS_TURING_VECADD_R_BVAL,
                                    PT, CTRL(5, 0, 7, 7, 0x1));

    /* 0x100: store the result. */
    out[i++] = vbs_turing_emit_stg(VBS_TURING_VECADD_R_C_LO,
                                   VBS_TURING_VECADD_R_SUM, 0,
                                   PT, CTRL(2, 1, 7, 7, 0));

    /* 0x110: done. (Stall 5 matches the published ptxas EXIT.) */
    out[i++] = vbs_turing_emit_exit(PT, CTRL(5, 1, 7, 7, 0));

#undef CTRL
    return i;  /* 18 */
}

void vbs_turing_vecadd_serialize(
        uint8_t out[VBS_TURING_VECADD_CODE_SIZE]) {
    vbs_turing_inst_t insts[VBS_TURING_VECADD_NUM_INSTS];
    size_t n = vbs_turing_vecadd_build(insts,
                                       VBS_TURING_VECADD_NUM_INSTS);
    for (size_t i = 0; i < n; i++)
        vbs_turing_sass_serialize(&insts[i], out + i * 16);
}
