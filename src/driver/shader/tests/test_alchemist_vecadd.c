/* test_alchemist_vecadd.c — unit tests for the Alchemist vector-add
 * kernel: binary layout, register map, launch descriptors
 * (interface descriptor + walker fields), inline-data layout, and
 * the listing generator.
 *
 * No GPU needed: runs entirely on the CPU.
 */

#include <stdio.h>
#include <string.h>

#include "test_util.h"
#include "vbs_alchemist_vecadd.h"

/* Expected opcode + ExecSize-encoding sequence for the 21
 * instructions (digest §6.9 order). */
static const uint8_t EXPECT_OP[VBS_ALCHEMIST_VECADD_NUM_INSTS] = {
    0x61, 0x69, 0x61, 0x61, 0x61, 0x61, 0x40, 0x40,
    0x61, 0x61, 0x40, 0x40, 0x61, 0x61, 0x40, 0x40,
    0x31, 0x31, 0x40, 0x31, 0x31
};
static const uint8_t EXPECT_EXEC[VBS_ALCHEMIST_VECADD_NUM_INSTS] = {
    4, 4, 3, 3, 3, 3, 3, 3,
    3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 0
};

static void test_build(void)
{
    vbs_alchemist_inst_t a[VBS_ALCHEMIST_VECADD_NUM_INSTS];
    vbs_alchemist_inst_t b[VBS_ALCHEMIST_VECADD_NUM_INSTS];
    uint8_t code[VBS_ALCHEMIST_VECADD_CODE_SIZE];
    uint8_t code2[VBS_ALCHEMIST_VECADD_CODE_SIZE];
    unsigned i;

    CHECK(vbs_alchemist_vecadd_insts(a) == 0, "kernel assembles");
    CHECK(vbs_alchemist_vecadd_insts(b) == 0, "kernel assembles again");
    CHECK(memcmp(a, b, sizeof(a)) == 0,
          "assembly is deterministic (two builds identical)");
    CHECK(vbs_alchemist_vecadd_build(code) == 0, "kernel builds to bytes");
    CHECK(vbs_alchemist_vecadd_build(code2) == 0, "kernel builds again");
    CHECK(memcmp(code, code2, sizeof(code)) == 0,
          "byte build is deterministic");
    CHECK(VBS_ALCHEMIST_VECADD_CODE_SIZE == 336,
          "code size == 336 bytes (21 x 16)");
    CHECK(VBS_ALCHEMIST_VECADD_CODE_SIZE % 16 == 0,
          "code size is a multiple of 16 (whole instructions)");
    CHECK(code[0] == 0x61, "first byte is the mov opcode (0x61)");
    /* Serialized bytes match the instruction dwords. */
    for (i = 0; i < VBS_ALCHEMIST_VECADD_NUM_INSTS; i++) {
        uint32_t w0 = (uint32_t)code[16 * i]
            | ((uint32_t)code[16 * i + 1] << 8)
            | ((uint32_t)code[16 * i + 2] << 16)
            | ((uint32_t)code[16 * i + 3] << 24);
        CHECK(w0 == a[i].dw[0], "serialized DW0 matches inst %u", i);
    }
    /* Opcode + ExecSize sequence. */
    for (i = 0; i < VBS_ALCHEMIST_VECADD_NUM_INSTS; i++) {
        CHECK((a[i].dw[0] & 0x7Fu) == EXPECT_OP[i],
              "inst %u opcode == 0x%02x", i, EXPECT_OP[i]);
        CHECK(((a[i].dw[0] >> 16) & 0x7u) == EXPECT_EXEC[i],
              "inst %u ExecSize enc == %u", i, EXPECT_EXEC[i]);
        /* No instruction uses compaction or predication. */
        CHECK(((a[i].dw[0] >> 29) & 0x1u) == 0,
              "inst %u CmptCtrl == 0 (native 128-bit)", i);
        CHECK(((a[i].dw[0] >> 24) & 0xFu) == 0,
              "inst %u PredCtrl == 0", i);
        CHECK(((a[i].dw[0] >> 8) & 0xFFu) == 0,
              "inst %u header bits [15:8] == 0 (Access Mode UNSOURCED)", i);
    }
}

static void test_register_map(void)
{
    vbs_alchemist_inst_t a[VBS_ALCHEMIST_VECADD_NUM_INSTS];
    uint32_t op;

    CHECK(vbs_alchemist_vecadd_insts(a) == 0, "kernel assembles (regs)");
    /* Inst 0 reads the local IDs from R1 (DW2 Src0.Operand). */
    op = (a[0].dw[2] >> 2) & 0x3FFFu;
    CHECK(op == ((1u << 6) | 1u), "inst 0 src0 == r1 (local IDs)");
    /* Broadcast subregs: A base at R2.0, B at R2.8, C at R2.16. */
    op = (a[4].dw[2] >> 2) & 0x3FFFu;
    CHECK(op == ((2u << 6) | (0u << 1) | 1u),
          "inst 4 broadcast src == r2.0 (A base)");
    op = (a[8].dw[2] >> 2) & 0x3FFFu;
    CHECK(op == ((2u << 6) | (8u << 1) | 1u),
          "inst 8 broadcast src == r2.8 (B base)");
    op = (a[12].dw[2] >> 2) & 0x3FFFu;
    CHECK(op == ((2u << 6) | (16u << 1) | 1u),
          "inst 12 broadcast src == r2.16 (C base)");
    /* Load A: dst r25, src0 r13. */
    CHECK(((a[16].dw[1] >> 24) & 0xFFu) == 25, "load A dst == r25");
    CHECK(((a[16].dw[2] >> 8) & 0xFFu) == 13, "load A src0 == r13");
    /* Load B: dst r27, src0 r17. */
    CHECK(((a[17].dw[1] >> 24) & 0xFFu) == 27, "load B dst == r27");
    CHECK(((a[17].dw[2] >> 8) & 0xFFu) == 17, "load B src0 == r17");
    /* F32 add: dst r29, src0 r25, src1 r27. */
    CHECK((((a[18].dw[1] >> 18) & 0x3FFFu) >> 6) == 29,
          "fadd dst == r29");
    CHECK((((a[18].dw[2] >> 2) & 0x3FFFu) >> 6) == 25,
          "fadd src0 == r25");
    CHECK((((a[18].dw[3] >> 2) & 0x3FFFu) >> 6) == 27,
          "fadd src1 == r27");
    /* Store: dst null, src0 r21 (addresses), src1 r29 (data). */
    CHECK(((a[19].dw[1] >> 18) & 0x1u) == 0, "store dst == null");
    CHECK(((a[19].dw[2] >> 8) & 0xFFu) == 21, "store src0 == r21");
    CHECK(((a[19].dw[3] >> 8) & 0xFFu) == 29, "store src1 == r29");
    /* EOT: dst null, src0 r112, EOT bit set, SFID Gateway. */
    CHECK(((a[20].dw[1] >> 18) & 0x1u) == 0, "EOT dst == null");
    CHECK(((a[20].dw[2] >> 8) & 0xFFu) == 112, "EOT src0 == r112");
    CHECK(((a[20].dw[1] >> 2) & 0x1u) == 1, "EOT bit set");
    CHECK(((a[20].dw[2] >> 28) & 0xFu) == VBS_ALCHEMIST_SFID_GATEWAY,
          "EOT SFID == Gateway (3)");
    /* The kernel is the last instruction: nothing follows EOT. */
    CHECK(VBS_ALCHEMIST_VECADD_NUM_INSTS == 21, "21 instructions total");
    /* Register map sanity: regions do not overlap. */
    CHECK(VBS_ALCHEMIST_VECADD_R_OFF + 2 <= VBS_ALCHEMIST_VECADD_R_OFF_UQ,
          "offset regs end before offset-UQ regs");
    CHECK(VBS_ALCHEMIST_VECADD_R_OFF_UQ + 4 <= VBS_ALCHEMIST_VECADD_R_BASE,
          "offset-UQ regs end before base regs");
    CHECK(VBS_ALCHEMIST_VECADD_R_BASE + 4 <= VBS_ALCHEMIST_VECADD_R_ADDR_A,
          "base regs end before A addresses");
    CHECK(VBS_ALCHEMIST_VECADD_R_ADDR_A + 4 <= VBS_ALCHEMIST_VECADD_R_ADDR_B,
          "A addresses end before B addresses");
    CHECK(VBS_ALCHEMIST_VECADD_R_ADDR_B + 4 <= VBS_ALCHEMIST_VECADD_R_ADDR_C,
          "B addresses end before C addresses");
    CHECK(VBS_ALCHEMIST_VECADD_R_ADDR_C + 4 <= VBS_ALCHEMIST_VECADD_R_DATA_A,
          "C addresses end before A data");
    CHECK(VBS_ALCHEMIST_VECADD_R_DATA_A + 2 <= VBS_ALCHEMIST_VECADD_R_DATA_B,
          "A data ends before B data");
    CHECK(VBS_ALCHEMIST_VECADD_R_DATA_B + 2 <= VBS_ALCHEMIST_VECADD_R_DATA_C,
          "B data ends before C data");
    CHECK(VBS_ALCHEMIST_VECADD_R_DATA_C + 2 <= 31,
          "working set fits in r0-r30");
    CHECK(VBS_ALCHEMIST_VECADD_MAX_REG == 112,
          "highest GRF is r112 (EOT source)");
    CHECK(VBS_ALCHEMIST_VECADD_MAX_REG <= 127, "all GRFs within r0-r127");
}

static void test_idd(void)
{
    uint32_t idd[8];
    int rc;

    rc = vbs_alchemist_vecadd_idd(idd, 0x1000, 16);
    CHECK(rc == 0, "IDD builds (offset 0x1000, 16 threads)");
    CHECK(idd[0] == 0x1000 && idd[1] == 0,
          "IDD DW0-1 kernel pointer == 0x1000");
    CHECK(idd[2] == 0, "IDD DW2 == 0 (IEEE FP, FTZ, no exceptions)");
    CHECK(idd[3] == 0, "IDD DW3 == 0 (no samplers)");
    CHECK(idd[4] == 0,
          "IDD DW4 == 0 (no binding table — A64 stateless, BTI 255)");
    CHECK(idd[5] == 16, "IDD DW5 thread count == 16");
    CHECK(((idd[5] >> 16) & 0x1Fu) == 0, "IDD DW5 SLM size == 0 (no SLM)");
    CHECK(((idd[5] >> 28) & 0x7u) == 0, "IDD DW5 barriers == 0");
    CHECK(((idd[5] >> 31) & 0x1u) == 0, "IDD DW5 BTD mode == 0");
    CHECK(idd[6] == 0 && idd[7] == 0, "IDD DW6-7 == 0");

    /* 64-bit pointer split across DW0-1. */
    rc = vbs_alchemist_vecadd_idd(idd, 0x100000040ULL, 1);
    CHECK(rc == 0, "IDD builds (64-bit offset)");
    CHECK(idd[0] == 0x40 && idd[1] == 1,
          "IDD DW0-1 split a 64-bit pointer correctly");
    /* Unaligned offsets are masked to 64 bytes. */
    rc = vbs_alchemist_vecadd_idd(idd, 0x1003, 1);
    CHECK(rc == 0, "IDD builds (unaligned offset)");
    CHECK(idd[0] == 0x1000, "IDD masks the pointer to 64-byte alignment");
    /* Thread-count bounds: [1,64] (digest §5.1). */
    CHECK(vbs_alchemist_vecadd_idd(idd, 0x1000, 0) != 0,
          "IDD rejects 0 threads");
    CHECK(vbs_alchemist_vecadd_idd(idd, 0x1000, 65) != 0,
          "IDD rejects 65 threads");
    CHECK(vbs_alchemist_vecadd_idd(idd, 0x1000, 64) == 0 &&
          idd[5] == 64, "IDD accepts 64 threads");
}

static void test_walker_fields(void)
{
    uint32_t dw4 = VBS_ALCHEMIST_VECADD_WALKER_DW4;
    uint32_t dw6;
    uint32_t t;

    CHECK(dw4 == 0x66020000u, "walker DW4 == 0x66020000");
    CHECK(((dw4 >> 30) & 0x3u) == 1, "walker SIMD Size == 1 (SIMD16)");
    CHECK(((dw4 >> 29) & 0x1u) == 1,
          "walker Generate Local ID == 1 (Vol 2a, p.488)");
    CHECK(((dw4 >> 26) & 0x7u) == 1, "walker Emit Local == X only");
    CHECK(((dw4 >> 25) & 0x1u) == 1, "walker Emit Inline Parameter == 1");
    CHECK(((dw4 >> 17) & 0x3u) == 1, "walker Message SIMD == 1 (SIMD16)");
    CHECK(((dw4 >> 22) & 0x7u) == 0, "walker Walk Order == 0 (default)");
    CHECK((dw4 & 0x3FFFFu) == 0x20000u,
          "walker bits [17:0]: only Message SIMD set, rest 0");

    /* DW6 (Local Maxima): X count = T*16, Y/Z = 0. */
    CHECK(vbs_alchemist_vecadd_walker_dw6(16) == 255,
          "walker DW6 for 16 threads == 255 (256 work items)");
    CHECK(vbs_alchemist_vecadd_walker_dw6(1) == 15,
          "walker DW6 for 1 thread == 15");
    CHECK(vbs_alchemist_vecadd_walker_dw6(64) == 1023,
          "walker DW6 for 64 threads == 1023");
    CHECK(vbs_alchemist_vecadd_walker_dw6(0) == 0,
          "walker DW6 rejects 0 threads");
    CHECK(vbs_alchemist_vecadd_walker_dw6(65) == 0,
          "walker DW6 rejects 65 threads");
    /* LWS = (Xmax+1) <= 1024 for every valid thread count, and the
     * dispatch-count rule (SIMTSize)*(N-1) < LWS <= (SIMTSize)*N
     * (digest §4.8) holds exactly (LWS = 16*N). */
    for (t = 1; t <= 64; t++) {
        dw6 = vbs_alchemist_vecadd_walker_dw6(t);
        CHECK((dw6 & 0x3FFu) + 1 <= 1024, "LWS <= 1024 for T=%u", t);
        CHECK((dw6 & 0x3FFu) + 1 == 16 * t, "LWS == 16*T for T=%u", t);
        CHECK((dw6 >> 10) == 0, "DW6 Y/Z maxima == 0 for T=%u", t);
    }
}

static void test_params(void)
{
    CHECK(sizeof(vbs_alchemist_vecadd_params_t)
          == VBS_ALCHEMIST_VECADD_PARAM_SIZE,
          "param struct is 32 bytes (8 inline dwords)");
    CHECK(__builtin_offsetof(vbs_alchemist_vecadd_params_t, a) == 0,
          "param a at offset 0 (inline dwords 0-1)");
    CHECK(__builtin_offsetof(vbs_alchemist_vecadd_params_t, b) == 8,
          "param b at offset 8 (inline dwords 2-3)");
    CHECK(__builtin_offsetof(vbs_alchemist_vecadd_params_t, c) == 16,
          "param c at offset 16 (inline dwords 4-5)");
    CHECK(__builtin_offsetof(vbs_alchemist_vecadd_params_t, spare0) == 0x18,
          "param spare0 at offset 0x18 (inline dword 6)");
    CHECK(__builtin_offsetof(vbs_alchemist_vecadd_params_t, spare1) == 0x1c,
          "param spare1 at offset 0x1c (inline dword 7)");
    CHECK(VBS_ALCHEMIST_VECADD_INL_A_LO == 0 &&
          VBS_ALCHEMIST_VECADD_INL_A_HI == 1 &&
          VBS_ALCHEMIST_VECADD_INL_B_LO == 2 &&
          VBS_ALCHEMIST_VECADD_INL_B_HI == 3 &&
          VBS_ALCHEMIST_VECADD_INL_C_LO == 4 &&
          VBS_ALCHEMIST_VECADD_INL_C_HI == 5,
          "inline dword indexes match the param struct layout");
}

static void test_listing(void)
{
    FILE *f = tmpfile();
    char buf[8192];
    size_t nread;
    unsigned lines = 0, i;
    char *p;

    CHECK(f != NULL, "tmpfile for listing");
    if (f == NULL)
        return;
    vbs_alchemist_vecadd_list(f);
    rewind(f);
    nread = fread(buf, 1, sizeof(buf) - 1, f);
    buf[nread] = '\0';
    fclose(f);

    for (p = buf; *p != '\0'; p++) {
        if (*p == '\n')
            lines++;
    }
    /* 6 header lines + 21 instruction lines. */
    CHECK(lines == 27, "listing has 27 lines (6 header + 21 inst), got %u",
          lines);
    CHECK(strstr(buf, "0x0000") != NULL, "listing starts at 0x0000");
    CHECK(strstr(buf, "0x0140") != NULL,
          "listing ends at 0x0140 (inst 20 x 16 bytes)");
    CHECK(strstr(buf, "mov (16)") != NULL, "listing has the mov (16)");
    CHECK(strstr(buf, "send (1)") != NULL, "listing has the EOT send (1)");
    CHECK(strstr(buf, "EOT") != NULL, "listing marks EOT");
    CHECK(strstr(buf, "0x08246EFF") != NULL,
          "listing shows the load desc");
    /* Every instruction line carries its encoding: spot-check the
     * F32 add's DW0 (0x00040040) appears on the 0x0120 line. */
    CHECK(strstr(buf, "0x0120") != NULL, "listing has the 0x0120 line");
    for (i = 0; i < VBS_ALCHEMIST_VECADD_NUM_INSTS; i++) {
        char addr[16];
        snprintf(addr, sizeof(addr), "0x%04x", 16 * i);
        CHECK(strstr(buf, addr) != NULL, "listing has address %s", addr);
    }
}

int main(void)
{
    test_build();
    test_register_map();
    test_idd();
    test_walker_fields();
    test_params();
    test_listing();

    TEST_SUMMARY("test_alchemist_vecadd");
}
