/* test_turing_sass.c — unit tests for the Turing SASS assembler,
 * the vecadd kernel, and the QMD fields the kernel needs.
 *
 * Runs on a machine with NO GPU: everything here is encoding
 * logic, checked against documented facts.
 *
 * The strongest checks are the GOLDEN tests: the emitters must
 * reproduce NVIDIA's own published nvdisasm output bit-for-bit
 * (CUDA Binary Utilities doc, S5 — the sm_70 sample in the
 * 12.4.1 archive and the sm_100 sample in the current doc).
 * If an emitter's bit fields were wrong, these would fail.
 *
 * QMD checks use the sibling Turing driver's builder
 * (../vbs_drv_turing_enc.c) — this file does not duplicate it;
 * it verifies the builder produces what THIS kernel needs.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "vbs_turing_sass.h"
#include "vbs_turing_vecadd.h"
#include "vbs_drv_turing.h"

static int failures = 0;
static int checks = 0;

#define CHECK(cond, msg) do { \
    checks++; \
    if (!(cond)) { \
        failures++; \
        printf("FAIL: %s (line %d)\n", (msg), __LINE__); \
    } \
} while (0)

#define CHECK_EQ_U64(got, want, msg) do { \
    checks++; \
    if ((uint64_t)(got) != (uint64_t)(want)) { \
        failures++; \
        printf("FAIL: %s (line %d): got 0x%016llx, want 0x%016llx\n", \
               (msg), __LINE__, \
               (unsigned long long)(uint64_t)(got), \
               (unsigned long long)(uint64_t)(want)); \
    } \
} while (0)

#define CHECK_EQ_U32(got, want, msg) do { \
    checks++; \
    if ((uint32_t)(got) != (uint32_t)(want)) { \
        failures++; \
        printf("FAIL: %s (line %d): got 0x%08x, want 0x%08x\n", \
               (msg), __LINE__, (unsigned)(got), (unsigned)(want)); \
    } \
} while (0)

#define PT 7u
#define CTRL(s, y, w, r, m) ((vbs_turing_ctrl_t){ (s), (y), (w), (r), (m) })

/* Field readers for the hi/lo words. */
static unsigned hi_opcode(vbs_turing_inst_t in) {
    return (unsigned)(in.hi & 0xFFFu);
}
static unsigned hi_pred(vbs_turing_inst_t in) {
    return (unsigned)((in.hi >> 12) & 0xFu);
}
static unsigned hi_rd(vbs_turing_inst_t in) {
    return (unsigned)((in.hi >> 16) & 0xFFu);
}
static unsigned hi_rs0(vbs_turing_inst_t in) {
    return (unsigned)((in.hi >> 24) & 0xFFu);
}
static unsigned hi_rs1(vbs_turing_inst_t in) {
    return (unsigned)((in.hi >> 32) & 0xFFu);
}
static uint64_t lo_flags(vbs_turing_inst_t in) {
    return in.lo & 0x1FFFFFFFFFFull; /* bits 0-40 */
}
static uint64_t lo_ctrl(vbs_turing_inst_t in) {
    return (in.lo >> 41) & 0x7FFFFFull; /* bits 41-63 */
}

/* ------------------------------------------------------------------ */
/* Opcode table                                                        */
/* ------------------------------------------------------------------ */

static void test_opcode_table(void) {
    /* Every sourced entry has a nonzero code; the table size
     * matches the enum. */
    for (int i = 0; i < VBS_TURING_OP_COUNT; i++) {
        const vbs_turing_opcode_t *e = &VBS_TURING_OPCODE_TABLE[i];
        CHECK(e->name != NULL && e->name[0] != '\0',
              "opcode entry has a name");
        CHECK(e->note != NULL && e->note[0] != '\0',
              "opcode entry has a source note");
        if (e->sourced)
            CHECK(e->code != 0, "sourced opcode is nonzero");
    }

    /* Spot-check the kernel's opcodes against the table. */
    int ok = 0;
    CHECK_EQ_U32(vbs_turing_opcode(VBS_TURING_OP_LDG, &ok), 0x381,
                 "LDG opcode 0x381");
    CHECK(ok, "LDG lookup ok");
    CHECK_EQ_U32(vbs_turing_opcode(VBS_TURING_OP_STG, &ok), 0x386,
                 "STG opcode 0x386");
    CHECK_EQ_U32(vbs_turing_opcode(VBS_TURING_OP_FADD, &ok), 0x221,
                 "FADD opcode 0x221");
    CHECK_EQ_U32(vbs_turing_opcode(VBS_TURING_OP_EXIT, &ok), 0x94d,
                 "EXIT opcode 0x94d");
    CHECK_EQ_U32(vbs_turing_opcode(VBS_TURING_OP_S2R, &ok), 0x919,
                 "S2R opcode 0x919");
    CHECK_EQ_U32(vbs_turing_opcode(VBS_TURING_OP_LDC, &ok), 0xb82,
                 "LDC opcode 0xb82");
    CHECK_EQ_U32(vbs_turing_opcode(VBS_TURING_OP_BAR, &ok), 0xb1d,
                 "BAR opcode 0xb1d");
    CHECK_EQ_U32(vbs_turing_opcode(VBS_TURING_OP_IMAD_WIDE, &ok), 0x225,
                 "IMAD.WIDE opcode 0x225");

    /* UNSOURCED entries must refuse to emit. */
    CHECK_EQ_U32(vbs_turing_opcode(VBS_TURING_OP_IMUL, &ok), 0,
                 "IMUL (unsourced) has no opcode");
    CHECK(!ok, "IMUL lookup reports not-ok");
    CHECK_EQ_U32(vbs_turing_opcode(VBS_TURING_OP_COUNT, &ok), 0,
                 "out-of-range kind has no opcode");
    CHECK(!ok, "out-of-range lookup reports not-ok");
    CHECK_EQ_U32(vbs_turing_opcode((vbs_turing_op_t)-1, &ok), 0,
                 "negative kind has no opcode");
    CHECK(!ok, "negative lookup reports not-ok");
}

/* ------------------------------------------------------------------ */
/* Control-code packing                                                */
/* ------------------------------------------------------------------ */

static void test_ctrl_pack(void) {
    /* The FADD control from the S5 sm_100 sample decodes to
     * stall 5, yield 0, wrtdb 7, readb 7, watdb 4 = 0x27E5. */
    CHECK_EQ_U64(vbs_turing_ctrl_pack(CTRL(5, 0, 7, 7, 4)), 0x27E5ull,
                 "ctrl pack: FADD sample 0x27E5");
    /* Default control: stall 1, no yield, no barriers = 0x7F1. */
    CHECK_EQ_U64(vbs_turing_ctrl_pack(VBS_TURING_CTRL_DEFAULT), 0x7F1ull,
                 "ctrl pack: default 0x7F1");
    /* Field isolation: each field lands in its own bits. */
    CHECK_EQ_U64(vbs_turing_ctrl_pack(CTRL(0xF, 0, 0, 0, 0)), 0xFull,
                 "ctrl pack: stall field");
    CHECK_EQ_U64(vbs_turing_ctrl_pack(CTRL(0, 1, 7, 7, 0)), 0x7F0ull,
                 "ctrl pack: yield+barrier fields");
    CHECK_EQ_U64(vbs_turing_ctrl_pack(CTRL(0, 1, 2, 7, 0)), 0x750ull,
                 "ctrl pack: wrtdb 2");
    CHECK_EQ_U64(vbs_turing_ctrl_pack(CTRL(0, 1, 7, 7, 0x10)), 0x87F0ull,
                 "ctrl pack: watdb bit 4");
    /* Over-wide inputs are masked, never spill into neighbours.
     * Max packed value: stall 0xF | yield | wrtdb 7 | readb 7 |
     * watdb 0x1F (5-bit mask) = 0xFFFF. */
    CHECK_EQ_U64(vbs_turing_ctrl_pack(CTRL(0xFF, 0xFF, 0xFF, 0xFF, 0xFF)),
                 0xFFFFull, "ctrl pack: inputs masked to field widths");
}

/* ------------------------------------------------------------------ */
/* Golden tests — reproduce NVIDIA's published nvdisasm hex exactly.   */
/* S5 = CUDA Binary Utilities doc. sm_70 values are from the 12.4.1    */
/* archive sample; sm_100 values from the current doc's sample.        */
/* ------------------------------------------------------------------ */

static void test_golden_sm70(void) {
    vbs_turing_inst_t in;

    /* LDG.E.SYS R2, [R2] (sm_70). */
    in = vbs_turing_emit_ldg(2, 2, 0, PT, CTRL(4, 1, 2, 7, 0));
    CHECK_EQ_U64(in.hi, 0x0000000002027381ull, "golden LDG hi");
    CHECK_EQ_U64(in.lo, 0x000ea800001ee900ull, "golden LDG lo");

    /* STG.E.SYS [R6], R9 (sm_70). */
    in = vbs_turing_emit_stg(6, 9, 0, PT, CTRL(1, 1, 7, 7, 0));
    CHECK_EQ_U64(in.hi, 0x0000000906007386ull, "golden STG hi");
    CHECK_EQ_U64(in.lo, 0x000fe2000010e900ull, "golden STG lo");

    /* IADD3 R9, R2, R5, RZ (sm_70). */
    in = vbs_turing_emit_iadd3(9, 2, 5, 255, PT, CTRL(8, 0, 7, 7, 4));
    CHECK_EQ_U64(in.hi, 0x0000000502097210ull, "golden IADD3 hi");
    CHECK_EQ_U64(in.lo, 0x004fd00007ffe0ffull, "golden IADD3 lo");

    /* MOV R3, c[0x0][0x164] (sm_70). */
    in = vbs_turing_emit_mov_const(3, 0, 0x164, PT, CTRL(1, 1, 7, 7, 0));
    CHECK_EQ_U64(in.hi, 0x0000590000037a02ull, "golden MOV-const hi");
    CHECK_EQ_U64(in.lo, 0x000fe20000000f00ull, "golden MOV-const lo");

    /* EXIT (sm_70 and sm_100 agree). */
    in = vbs_turing_emit_exit(PT, CTRL(5, 1, 7, 7, 0));
    CHECK_EQ_U64(in.hi, 0x000000000000794dull, "golden EXIT hi");
    CHECK_EQ_U64(in.lo, 0x000fea0003800000ull, "golden EXIT lo");

    /* BRA 0xd0 at pc 0xd0: offset -16 (sm_70). */
    in = vbs_turing_emit_bra(-16, PT, CTRL(0, 0, 7, 7, 0));
    CHECK_EQ_U64(in.hi, 0xfffffff000007947ull, "golden BRA hi");
    CHECK_EQ_U64(in.lo, 0x000fc0000383ffffull, "golden BRA lo");

    /* NOP (sm_70 and sm_100 agree). */
    in = vbs_turing_emit_nop(PT, CTRL(0, 0, 7, 7, 0));
    CHECK_EQ_U64(in.hi, 0x0000000000007918ull, "golden NOP hi");
    CHECK_EQ_U64(in.lo, 0x000fc00000000000ull, "golden NOP lo");
}

static void test_golden_sm100(void) {
    vbs_turing_inst_t in;

    /* LDC R1, c[0x0][0x37c] (sm_100). */
    in = vbs_turing_emit_ldc(1, 0, 0x37c, 0, PT, CTRL(1, 1, 7, 7, 0));
    CHECK_EQ_U64(in.hi, 0x0000df00ff017b82ull, "golden LDC hi");
    CHECK_EQ_U64(in.lo, 0x000fe20000000800ull, "golden LDC lo");

    /* LDC.64 R2, c[0x0][0x380] (sm_100). */
    in = vbs_turing_emit_ldc(2, 0, 0x380, 1, PT, CTRL(8, 1, 2, 7, 0));
    CHECK_EQ_U64(in.hi, 0x0000e000ff027b82ull, "golden LDC.64 hi");
    CHECK_EQ_U64(in.lo, 0x000eb00000000a00ull, "golden LDC.64 lo");

    /* S2R R9, SR_TID.X (sm_100). */
    in = vbs_turing_emit_s2r(9, VBS_TURING_SR_TID_X, PT,
                             CTRL(7, 1, 0, 7, 0));
    CHECK_EQ_U64(in.hi, 0x0000000000097919ull, "golden S2R hi");
    CHECK_EQ_U64(in.lo, 0x000e2e0000002100ull, "golden S2R lo");

    /* IMAD.WIDE R2, R9, 0x4, R2 (sm_100). */
    in = vbs_turing_emit_imad_wide_imm(2, 9, 4, 2, PT,
                                       CTRL(6, 0, 7, 7, 4));
    CHECK_EQ_U64(in.hi, 0x0000000409027825ull, "golden IMAD.WIDE hi");
    CHECK_EQ_U64(in.lo, 0x004fcc00078e0202ull, "golden IMAD.WIDE lo");

    /* FADD R9, R2, R5 (sm_100). */
    in = vbs_turing_emit_fadd(9, 2, 5, PT, CTRL(5, 0, 7, 7, 4));
    CHECK_EQ_U64(in.hi, 0x0000000502097221ull, "golden FADD hi");
    CHECK_EQ_U64(in.lo, 0x004fca0000000000ull, "golden FADD lo");
}

/* ------------------------------------------------------------------ */
/* Per-instruction field checks (no published golden available).       */
/* ------------------------------------------------------------------ */

static void test_emit_fields(void) {
    vbs_turing_inst_t in;

    /* IADD3 with immediate: digest §5.3 worked example
     * (IADD3 R0, R1, 0x10, R2) — opcode becomes 0x810 via ICR. */
    in = vbs_turing_emit_iadd3_imm(0, 1, 0x10, 2, PT,
                                   VBS_TURING_CTRL_DEFAULT);
    CHECK_EQ_U32(hi_opcode(in), 0x810, "IADD3-imm opcode 0x810 (ICR 8)");
    CHECK_EQ_U32(hi_rd(in), 0, "IADD3-imm rd");
    CHECK_EQ_U32(hi_rs0(in), 1, "IADD3-imm rs0");
    CHECK_EQ_U64(in.hi >> 32, 0x10ull, "IADD3-imm immediate at rs1");
    CHECK_EQ_U32((unsigned)(in.lo & 0xFFu), 2, "IADD3-imm rs2");
    CHECK_EQ_U64(lo_flags(in) & 0x7FFFFF8ull,
                 (uint64_t)(VBS_TURING_NOPD0 | VBS_TURING_NOPD1 |
                            VBS_TURING_NOPS0 | VBS_TURING_NOPS1),
                 "IADD3-imm predicate fillers");

    /* IMAD all-register form: opcode stays 0x224 (no ICR). */
    in = vbs_turing_emit_imad(9, 7, 8, 6, PT, VBS_TURING_CTRL_DEFAULT);
    CHECK_EQ_U32(hi_opcode(in), 0x224, "IMAD opcode 0x224");
    CHECK_EQ_U32(hi_rd(in), 9, "IMAD rd");
    CHECK_EQ_U32(hi_rs0(in), 7, "IMAD rs0");
    CHECK_EQ_U32(hi_rs1(in), 8, "IMAD rs1");
    CHECK_EQ_U32((unsigned)(in.lo & 0xFFu), 6, "IMAD rs2");
    CHECK_EQ_U64(lo_flags(in) & 0x7FFFFF8ull,
                 (uint64_t)(0x200u | VBS_TURING_NOPD0 |
                            VBS_TURING_NOPS0),
                 "IMAD type .S32 + fillers");

    /* LDG with a 24-bit immediate offset (digest §5.3 form
     * [R2+0x100]). */
    in = vbs_turing_emit_ldg(4, 2, 0x100, PT, VBS_TURING_CTRL_DEFAULT);
    CHECK_EQ_U32(hi_opcode(in), 0x381, "LDG+off opcode");
    CHECK_EQ_U64(in.hi >> 40, 0x100ull, "LDG+off is0w24 at bits 104-127");
    CHECK_EQ_U64(lo_flags(in), 0x1EE900ull, "LDG+off flag bits");

    /* STG flag bits (no ldgp) and store-data register. */
    in = vbs_turing_emit_stg(4, 13, 0, PT, VBS_TURING_CTRL_DEFAULT);
    CHECK_EQ_U32(hi_opcode(in), 0x386, "STG opcode");
    CHECK_EQ_U32(hi_rs0(in), 4, "STG address register");
    CHECK_EQ_U32(hi_rs1(in), 13, "STG data register");
    CHECK_EQ_U64(lo_flags(in), 0x10E900ull, "STG flag bits");

    /* ISETP.GE.U32.AND P0, PT, R9, R10, PT. */
    in = vbs_turing_emit_isetp_ge_u32(0, 9, 10, PT,
                                      VBS_TURING_CTRL_DEFAULT);
    CHECK_EQ_U32(hi_opcode(in), 0x20c, "ISETP opcode 0x20c");
    CHECK_EQ_U32(hi_rs0(in), 9, "ISETP rs0");
    CHECK_EQ_U32(hi_rs1(in), 10, "ISETP rs1");
    CHECK_EQ_U64(lo_flags(in),
                 (0x7ull << 4) | (6ull << 12) | (0ull << 17) |
                     (7ull << 20) | (7ull << 23),
                 "ISETP flags: 0x7<<4, .GE, pd0 P0, pd1 PT, ps0 PT");

    /* Predicated EXIT: predicate field P0, cp still PT. */
    in = vbs_turing_emit_exit(0, VBS_TURING_CTRL_DEFAULT);
    CHECK_EQ_U32(hi_opcode(in), 0x94d, "@P0 EXIT opcode");
    CHECK_EQ_U32(hi_pred(in), 0, "@P0 EXIT predicate field");
    CHECK_EQ_U64(lo_flags(in), 0x03800000ull, "@P0 EXIT cp PT");

    /* BAR.SYNC: opcode + barrier id at bits 118-125. */
    in = vbs_turing_emit_bar_sync(0, PT, VBS_TURING_CTRL_DEFAULT);
    CHECK_EQ_U32(hi_opcode(in), 0xb1d, "BAR opcode 0xb1d");
    CHECK_EQ_U64(in.hi >> 54, 0ull, "BAR.SYNC id 0");
    in = vbs_turing_emit_bar_sync(3, PT, VBS_TURING_CTRL_DEFAULT);
    CHECK_EQ_U64(in.hi >> 54, 3ull, "BAR.SYNC id 3");

    /* MOV register copy: opcode + 0xF00 rule. */
    in = vbs_turing_emit_mov_reg(5, 7, PT, VBS_TURING_CTRL_DEFAULT);
    CHECK_EQ_U32(hi_opcode(in), 0x202, "MOV-reg opcode");
    CHECK_EQ_U32(hi_rd(in), 5, "MOV-reg rd");
    CHECK_EQ_U32(hi_rs1(in), 7, "MOV-reg rs1");
    CHECK_EQ_U64(lo_flags(in), 0xF00ull, "MOV-reg 0xF00 rule");

    /* S2R CTAID.X selector value. */
    in = vbs_turing_emit_s2r(7, VBS_TURING_SR_CTAID_X, PT,
                             VBS_TURING_CTRL_DEFAULT);
    CHECK_EQ_U32(hi_opcode(in), 0x919, "S2R CTAID opcode");
    CHECK_EQ_U64((in.lo >> 8) & 0x1FFu, 37u, "S2R SR_CTAID.X = 37");

    /* Serialization: hi word first, then lo, little-endian. */
    {
        vbs_turing_inst_t t;
        t.hi = 0x0102030405060708ull;
        t.lo = 0x1112131415161718ull;
        uint8_t buf[16];
        vbs_turing_sass_serialize(&t, buf);
        static const uint8_t want[16] = {
            0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01,
            0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11
        };
        CHECK(memcmp(buf, want, 16) == 0,
              "serialize: hi word first, lo second, LE");
    }
}

/* ------------------------------------------------------------------ */
/* Kernel binary layout                                                */
/* ------------------------------------------------------------------ */

static void test_kernel_layout(void) {
    vbs_turing_inst_t insts[VBS_TURING_VECADD_NUM_INSTS];
    size_t n;

    CHECK_EQ_U32(vbs_turing_vecadd_build(NULL, 18), 0,
                 "kernel build rejects NULL out");
    CHECK_EQ_U32(vbs_turing_vecadd_build(insts, 17), 0,
                 "kernel build rejects short capacity");
    n = vbs_turing_vecadd_build(insts, VBS_TURING_VECADD_NUM_INSTS);
    CHECK_EQ_U32(n, 18, "kernel has 18 instructions");
    CHECK_EQ_U32(VBS_TURING_VECADD_CODE_SIZE, 288,
                 "kernel code is 288 bytes");
    CHECK(VBS_TURING_VECADD_CODE_SIZE % 16 == 0,
          "kernel code size is 16-byte aligned");

    /* First instruction is the first LDC.64, last is EXIT. */
    CHECK_EQ_U32(hi_opcode(insts[0]), 0xb82,
                 "kernel[0] is LDC.64 (no SPH: code starts at 0)");
    CHECK_EQ_U32(hi_opcode(insts[17]), 0x94d,
                 "kernel[17] is EXIT");
    CHECK_EQ_U32(hi_pred(insts[17]), PT, "final EXIT is unpredicated");
    CHECK_EQ_U32(hi_pred(insts[9]), 0,
                 "kernel[9] is the predicated @P0 EXIT");

    /* The core sequence is present in order. */
    CHECK_EQ_U32(hi_opcode(insts[7]), 0x224, "kernel[7] IMAD");
    CHECK_EQ_U32(hi_opcode(insts[8]), 0x20c, "kernel[8] ISETP");
    CHECK_EQ_U32(hi_opcode(insts[13]), 0x381, "kernel[13] LDG");
    CHECK_EQ_U32(hi_opcode(insts[14]), 0x381, "kernel[14] LDG");
    CHECK_EQ_U32(hi_opcode(insts[15]), 0x221, "kernel[15] FADD");
    CHECK_EQ_U32(hi_opcode(insts[16]), 0x386, "kernel[16] STG");

    /* FADD reads the two loaded values, writes the sum. */
    CHECK_EQ_U32(hi_rd(insts[15]), 13, "FADD rd = R13 (sum)");
    CHECK_EQ_U32(hi_rs0(insts[15]), 11, "FADD rs0 = R11 (a[i])");
    CHECK_EQ_U32(hi_rs1(insts[15]), 12, "FADD rs1 = R12 (b[i])");

    /* Barrier plan from vecadd.c: (wrtdb, watdb) per instruction. */
    static const struct { unsigned w, m; } plan[18] = {
        { 0, 0x00 }, { 0, 0x00 }, { 0, 0x00 }, { 0, 0x00 },
        { 0, 0x00 }, { 1, 0x00 }, { 1, 0x00 }, { 2, 0x03 },
        { 3, 0x05 }, { 7, 0x08 }, { 4, 0x05 }, { 4, 0x05 },
        { 4, 0x05 }, { 0, 0x10 }, { 0, 0x10 }, { 7, 0x01 },
        { 7, 0x00 }, { 7, 0x00 },
    };
    for (size_t i = 0; i < 18; i++) {
        uint64_t c = lo_ctrl(insts[i]);
        unsigned w = (unsigned)((c >> 5) & 0x7u);
        unsigned m = (unsigned)((c >> 11) & 0x1Fu);
        CHECK(w == plan[i].w, "kernel barrier plan: wrtdb");
        CHECK(m == plan[i].m, "kernel barrier plan: watdb");
        /* No instruction waits on barrier 5+ (mask is 5 bits). */
        CHECK(m <= 0x1F, "kernel wait mask fits 5 bits");
    }
    /* FADD yields while waiting on the loads; its stall is 5. */
    {
        uint64_t c = lo_ctrl(insts[15]);
        CHECK_EQ_U32(c & 0xFu, 5, "FADD stall 5 (matches ptxas)");
        CHECK_EQ_U32((c >> 4) & 0x1u, 0, "FADD yields");
    }

    /* Serialization: 288 bytes, first instruction's hi word at
     * offset 0, lo word at offset 8. */
    {
        uint8_t code[VBS_TURING_VECADD_CODE_SIZE];
        vbs_turing_vecadd_serialize(code);
        uint64_t hi0 = 0, lo0 = 0;
        for (int i = 0; i < 8; i++) {
            hi0 |= (uint64_t)code[i] << (8 * i);
            lo0 |= (uint64_t)code[8 + i] << (8 * i);
        }
        CHECK_EQ_U64(hi0, insts[0].hi, "serialized inst 0 hi word");
        CHECK_EQ_U64(lo0, insts[0].lo, "serialized inst 0 lo word");
        /* Last instruction (EXIT) at offset 272. */
        uint64_t hi17 = 0;
        for (int i = 0; i < 8; i++)
            hi17 |= (uint64_t)code[272 + i] << (8 * i);
        CHECK_EQ_U64(hi17, insts[17].hi, "serialized inst 17 hi word");
    }

    /* Parameter struct layout matches the CB offsets. */
    CHECK_EQ_U32(offsetof(vbs_turing_vecadd_params_t, a), 0x00,
                 "param a at 0x00");
    CHECK_EQ_U32(offsetof(vbs_turing_vecadd_params_t, b), 0x08,
                 "param b at 0x08");
    CHECK_EQ_U32(offsetof(vbs_turing_vecadd_params_t, c), 0x10,
                 "param c at 0x10");
    CHECK_EQ_U32(offsetof(vbs_turing_vecadd_params_t, n), 0x18,
                 "param n at 0x18");
    CHECK_EQ_U32(offsetof(vbs_turing_vecadd_params_t, ntid), 0x1c,
                 "param ntid at 0x1c");
    CHECK_EQ_U32(sizeof(vbs_turing_vecadd_params_t), 32,
                 "param struct is 32 bytes");
    CHECK_EQ_U32(VBS_TURING_VECADD_CB_SIZE, 256,
                 "CB0 binding is 256 bytes");
    CHECK_EQ_U32(VBS_TURING_VECADD_NUM_REGS, 14,
                 "kernel uses 14 registers");
}

/* ------------------------------------------------------------------ */
/* QMD fields for this kernel (via the sibling Turing builder).        */
/* Launch: code VA 0x100000000 (288 bytes), grid 1024, block 256,    */
/* 14 regs, 0 shared, 0 barriers, CB0 VA 0x200000000 (256 B),        */
/* RELEASE0 VA 0x300000000 payload 0xC0FFEE. Field positions are     */
/* digest §4.1/§4.2 (S1 clc5c0qmd.h), verified against the header.    */
/* ------------------------------------------------------------------ */

static void test_qmd_for_vecadd(void) {
    vbs_drv_buffer_t code = { 0 };
    code.gpu_va = 0x100000000ULL;
    code.size = 4096;
    vbs_drv_buffer_t cb0 = { 0 };
    cb0.gpu_va = 0x200000000ULL;
    cb0.size = VBS_TURING_VECADD_CB_SIZE;
    vbs_drv_buffer_t sem = { 0 };
    sem.gpu_va = 0x300000000ULL;
    sem.size = 4096;

    vbs_drv_launch_desc_t desc = { 0 };
    desc.code = &code;
    desc.code_offset = 0;
    desc.code_size = VBS_TURING_VECADD_CODE_SIZE;
    desc.register_count = VBS_TURING_VECADD_NUM_REGS;
    desc.shared_mem_bytes = 0;
    desc.barrier_count = 0;
    desc.grid_x = 1024; desc.grid_y = 1; desc.grid_z = 1;
    desc.block_x = 256; desc.block_y = 1; desc.block_z = 1;
    desc.const_bufs[0] = &cb0;
    desc.const_buf_count = 1;
    desc.sem_buf = &sem;
    desc.sem_offset = 0;
    desc.sem_payload = 0xC0FFEE;
    desc.use_semaphore = true;

    vbs_drv_turing_qmd_t qmd;
    CHECK(vbs_drv_turing_qmd_build(&desc, &qmd) == VBS_DRV_OK,
          "qmd_build succeeds for the vecadd launch");

    /* Versions: word 18 low byte 0x23 (v2.3), word 31 top 0x75. */
    CHECK_EQ_U32(qmd.w[18], 0x01000023u,
                 "QMD word 18: blockDim.x 256 + QMD v2.3");
    CHECK_EQ_U32(qmd.w[31], 0x75000000u,
                 "QMD word 31: SASS_VERSION 0x75, prefetch size 0");

    /* Program address + prefetch (code VA 0x100000000). */
    CHECK_EQ_U32(qmd.w[48], 0x00000000u, "QMD word 48: prog addr low");
    CHECK_EQ_U32(qmd.w[49], 0x00000001u, "QMD word 49: prog addr high");
    CHECK_EQ_U32(qmd.w[8], 0x01000000u, "QMD word 8: prefetch addr>>8");

    /* Grid / block. */
    CHECK_EQ_U32(qmd.w[12], 1024u, "QMD word 12: gridDim.x");
    CHECK_EQ_U32(qmd.w[13], 1u, "QMD word 13: gridDim.y");
    CHECK_EQ_U32(qmd.w[14], 1u, "QMD word 14: gridDim.z");
    CHECK_EQ_U32(qmd.w[19], 0x00010001u, "QMD word 19: blockDim.y/z");

    /* Resources: 14 registers, 0 shared, 0 barriers. */
    CHECK_EQ_U32(qmd.w[17], 0u, "QMD word 17: shared 0");
    CHECK_EQ_U32(qmd.w[20], 0x00000E01u,
                 "QMD word 20: CB0 valid + REGISTER_COUNT_V 14");
    CHECK_EQ_U32(qmd.w[30], 0x0E000000u, "QMD word 30: REGISTER_COUNT 14");
    CHECK_EQ_U32(qmd.w[29], 0u, "QMD word 29: BARRIER_COUNT 0");

    /* CB0: VA 0x200000000, 256 bytes, invalidate set. */
    CHECK_EQ_U32(qmd.w[32], 0x00000000u, "QMD word 32: CB0 addr low");
    CHECK_EQ_U32(qmd.w[33], 0x00840002u,
                 "QMD word 33: CB0 addr high + invalidate + size>>4");

    /* Word 4: SM global caching on (bit 6, builder default) +
     * RELEASE0 enable (bit 10); IS_QUEUE (bit 8) clear. */
    CHECK_EQ_U32(qmd.w[4], 0x00000440u,
                 "QMD word 4: caching + release0 enable");

    /* Word 5: instruction + constant-cache invalidate bits. */
    CHECK_EQ_U32(qmd.w[5], 0xC0000000u, "QMD word 5: invalidate bits");

    /* RELEASE0: VA 0x300000000, payload, plain write
     * (reduction off, structure size 0 = FOUR_WORDS per S1). */
    CHECK_EQ_U32(qmd.w[23], 0x00000000u, "QMD word 23: release addr low");
    CHECK_EQ_U32(qmd.w[24], 0x00000003u, "QMD word 24: release addr high");
    CHECK_EQ_U32(qmd.w[25], 0x00C0FFEEu, "QMD word 25: release payload");

    /* A bad launch description must fail, never half-build. */
    {
        vbs_drv_launch_desc_t bad = desc;
        bad.register_count = 0;
        vbs_drv_turing_qmd_t q2;
        CHECK(vbs_drv_turing_qmd_build(&bad, &q2) != VBS_DRV_OK,
              "qmd_build rejects register_count 0");
    }
    {
        vbs_drv_launch_desc_t bad = desc;
        bad.code_size = 0;
        vbs_drv_turing_qmd_t q2;
        CHECK(vbs_drv_turing_qmd_build(&bad, &q2) != VBS_DRV_OK,
              "qmd_build rejects code_size 0");
    }
}

int main(void) {
    test_opcode_table();
    test_ctrl_pack();
    test_golden_sm70();
    test_golden_sm100();
    test_emit_fields();
    test_kernel_layout();
    test_qmd_for_vecadd();

    printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
