/* test_alchemist_eu.c — unit tests for the Alchemist EU assembler.
 *
 * Per-instruction encoding checks: every emitted instruction's bit
 * fields are extracted straight from the dwords and checked against
 * the ISA docs (digest §6.4, §6.5, §6.8), independently of the
 * assembler's internals. Golden dword values were cross-checked
 * against a second, independent implementation of the same field
 * tables (see the report); the field-extraction checks below are
 * the primary evidence, the goldens are a second layer.
 *
 * No GPU needed: runs entirely on the CPU.
 */

#include <stdio.h>
#include <string.h>

#include "test_util.h"
#include "vbs_alchemist_eu.h"

/* Field extractors — raw bit reads, independent of the assembler. */
static uint32_t dw0op(vbs_alchemist_inst_t *i) { return i->dw[0] & 0x7Fu; }
static uint32_t dw0exec(vbs_alchemist_inst_t *i)
{ return (i->dw[0] >> 16) & 0x7u; }

/* Reconstitute a send's desc from its scattered instruction bits
 * (digest §6.5 table) and compare with the intended value. */
static uint32_t send_desc(vbs_alchemist_inst_t *i)
{
    uint32_t d = 0;
    d |= ((i->dw[3] >> 26) & 0x3u) << 30;  /* Desc[31:30] */
    d |= ((i->dw[2] >> 7) & 0x1u) << 29;   /* Desc[29] */
    d |= ((i->dw[2] >> 3) & 0xFu) << 25;   /* Desc[28:25] MsgLen */
    d |= ((i->dw[1] >> 19) & 0x1Fu) << 20; /* Desc[24:20] RespLen */
    d |= ((i->dw[3] >> 17) & 0x1FFu) << 11;/* Desc[19:11] */
    d |= ((i->dw[2] >> 17) & 0x7FFu);      /* Desc[10:0] */
    return d;
}

/* Reconstitute a send's ex_desc from its scattered bits. */
static uint32_t send_exdesc(vbs_alchemist_inst_t *i)
{
    uint32_t e = 0;
    e |= ((i->dw[3] >> 28) & 0xFu) << 28;  /* ExDesc[31:28] */
    e |= ((i->dw[3] >> 0) & 0x3u) << 26;   /* ExDesc[27:26] */
    e |= ((i->dw[2] >> 0) & 0x3u) << 24;   /* ExDesc[25:24] */
    e |= ((i->dw[1] >> 3) & 0x1FFFu) << 11;/* ExDesc[23:11] */
    e |= ((i->dw[3] >> 3) & 0x1Fu) << 6;   /* ExDesc[10:6] ExtLen */
    return e;
}

/* The standard f32 add used by several cases: add r29 = r25 + r27
 * (<8;8,1> regions). Returns the emit result. */
static int emit_add_f32(vbs_alchemist_inst_t *i, uint8_t exec_enc) {
    return vbs_alchemist_emit_add(
        i, exec_enc,
        vbs_alchemist_dst(29, 0, VBS_ALCHEMIST_TYPE_F, 1),
        vbs_alchemist_op(25, 0, VBS_ALCHEMIST_TYPE_F,
                         VBS_ALCHEMIST_REGION_F32_V,
                         VBS_ALCHEMIST_REGION_F32_W,
                         VBS_ALCHEMIST_REGION_F32_H),
        vbs_alchemist_op(27, 0, VBS_ALCHEMIST_TYPE_F,
                         VBS_ALCHEMIST_REGION_F32_V,
                         VBS_ALCHEMIST_REGION_F32_W,
                         VBS_ALCHEMIST_REGION_F32_H));
}

/* The standard :uq add used by several cases: add r13 = r5 + r9
 * (UQ8 regions). Returns the emit result. */
static int emit_add_uq(vbs_alchemist_inst_t *i, uint8_t exec_enc) {
    return vbs_alchemist_emit_add(
        i, exec_enc,
        vbs_alchemist_dst(13, 0, VBS_ALCHEMIST_TYPE_UQ, 1),
        vbs_alchemist_op(5, 0, VBS_ALCHEMIST_TYPE_UQ,
                         VBS_ALCHEMIST_REGION_UQ8_V,
                         VBS_ALCHEMIST_REGION_UQ8_W,
                         VBS_ALCHEMIST_REGION_UQ8_H),
        vbs_alchemist_op(9, 0, VBS_ALCHEMIST_TYPE_UQ,
                         VBS_ALCHEMIST_REGION_UQ8_V,
                         VBS_ALCHEMIST_REGION_UQ8_W,
                         VBS_ALCHEMIST_REGION_UQ8_H));
}

static void test_opcode_table(void)
{
    int ok = 0;

    CHECK(vbs_alchemist_opcode(VBS_ALCHEMIST_OP_ADD, &ok) == 0x40 && ok,
          "add opcode 0x40 (Vol 9, pp.470-476)");
    CHECK(vbs_alchemist_opcode(VBS_ALCHEMIST_OP_MUL, &ok) == 0x41 && ok,
          "mul opcode 0x41");
    CHECK(vbs_alchemist_opcode(VBS_ALCHEMIST_OP_MOV, &ok) == 0x61 && ok,
          "mov opcode 0x61");
    CHECK(vbs_alchemist_opcode(VBS_ALCHEMIST_OP_SHL, &ok) == 0x69 && ok,
          "shl opcode 0x69");
    CHECK(vbs_alchemist_opcode(VBS_ALCHEMIST_OP_SEND, &ok) == 0x31 && ok,
          "send opcode 0x31");
    CHECK(vbs_alchemist_opcode(VBS_ALCHEMIST_OP_SENDC, &ok) == 0x32 && ok,
          "sendc opcode 0x32");
    CHECK(vbs_alchemist_opcode(VBS_ALCHEMIST_OP_NOP, &ok) == 0x60 && ok,
          "nop opcode 0x60");
    CHECK(vbs_alchemist_opcode(VBS_ALCHEMIST_OP_AND, &ok) == 0x65 && ok,
          "and opcode 0x65");
    CHECK(vbs_alchemist_opcode(VBS_ALCHEMIST_OP_OR, &ok) == 0x66 && ok,
          "or opcode 0x66");
    CHECK(vbs_alchemist_opcode(VBS_ALCHEMIST_OP_XOR, &ok) == 0x67 && ok,
          "xor opcode 0x67");
    CHECK(vbs_alchemist_opcode(VBS_ALCHEMIST_OP_CMP, &ok) == 0x70 && ok,
          "cmp opcode 0x70");
    /* UNSOURCED entries must be refused, never emitted. */
    ok = 1;
    CHECK(vbs_alchemist_opcode(VBS_ALCHEMIST_OP_MOV_ALT, &ok) == 0 &&
          !ok, "mov 0x01 UNSOURCED: lookup refuses (digest §9 item 2)");
    ok = 1;
    CHECK(vbs_alchemist_opcode(VBS_ALCHEMIST_OP_COUNT, &ok) == 0 && !ok,
          "out-of-range kind refused");
    /* Table size matches the enum. */
    CHECK(sizeof(VBS_ALCHEMIST_OPCODE_TABLE) / sizeof(VBS_ALCHEMIST_OPCODE_TABLE[0])
          == (size_t)VBS_ALCHEMIST_OP_COUNT,
          "opcode table size matches enum count");
}

static void test_direct_operand(void)
{
    CHECK(vbs_alchemist_direct_operand(3, 0, 1) == 0x00C1,
          "DirectOperand r3.0 GRF == 0xC1 (Vol 2d, p.366)");
    CHECK(vbs_alchemist_direct_operand(2, 8, 1) == 0x0091,
          "DirectOperand r2.8 GRF == 0x91");
    CHECK(vbs_alchemist_direct_operand(127, 31, 1) == 0x1FFF,
          "DirectOperand r127.31 GRF == 0x1FFF");
    CHECK(vbs_alchemist_direct_operand(0, 0, 0) == 0x0000,
          "null register (ARF type 0) == 0 (Vol 9, p.348)");
}

static void test_desc_build(void)
{
    /* Digest §6.8 computed values, rebuilt from fields. */
    CHECK(vbs_alchemist_desc_build(4, 2, 0, VBS_ALCHEMIST_MSG_A64_READ,
                                   VBS_ALCHEMIST_SIMD_MODE_16,
                                   VBS_ALCHEMIST_CMASK_R,
                                   VBS_ALCHEMIST_BTI_STATELESS)
          == VBS_ALCHEMIST_DESC_A64_READ_SIMD16,
          "A64 read SIMD16 desc == 0x08246EFF (digest §6.8)");
    CHECK(vbs_alchemist_desc_build(4, 0, 0, VBS_ALCHEMIST_MSG_A64_WRITE,
                                   VBS_ALCHEMIST_SIMD_MODE_16,
                                   VBS_ALCHEMIST_CMASK_R,
                                   VBS_ALCHEMIST_BTI_STATELESS)
          == VBS_ALCHEMIST_DESC_A64_WRITE_SIMD16,
          "A64 write SIMD16 desc == 0x08066EFF (digest §6.8)");
    CHECK(vbs_alchemist_desc_build(2, 1, 0, VBS_ALCHEMIST_MSG_A64_READ,
                                   VBS_ALCHEMIST_SIMD_MODE_8,
                                   VBS_ALCHEMIST_CMASK_R,
                                   VBS_ALCHEMIST_BTI_STATELESS)
          == 0x04145EFFu,
          "A64 read SIMD8 desc == 0x04145EFF (digest §6.8)");
    CHECK(vbs_alchemist_desc_build(1, 0, 0, 0, 0, 0, 0)
          == VBS_ALCHEMIST_DESC_EOT,
          "EOT desc == 0x02000000 (digest §6.7)");
    CHECK(vbs_alchemist_exdesc_build(2)
          == VBS_ALCHEMIST_EXDESC_A64_WRITE_SIMD16,
          "store ex_desc == 0x80 (ExtMsgLen 2, digest §6.8)");
    CHECK(vbs_alchemist_exdesc_build(0) == 0, "ex_desc 0 for loads/EOT");
}

static void test_add_f32(void)
{
    vbs_alchemist_inst_t i;
    int rc;

    /* add (16) r29.0<1>:f r25.0<8;8,1>:f r27.0<8;8,1>:f */
    rc = emit_add_f32(&i, VBS_ALCHEMIST_EXECSIZE_16);
    CHECK(rc == 0, "add f32 emits");
    CHECK(dw0op(&i) == 0x40, "add opcode bits [6:0] == 0x40");
    CHECK(dw0exec(&i) == VBS_ALCHEMIST_EXECSIZE_16,
          "add ExecSize bits [18:16] == 4 (SIMD16)");
    CHECK(i.dw[0] == 0x00040040u,
          "add DW0 golden 0x00040040 (header bits [15:7] = 0)");
    /* DW1: Dst r29, HorzStride 1, types :f/:f, IsImm 0. */
    CHECK(((i.dw[1] >> 18) & 0x3FFFu) == 0x0741,
          "add Dst.Operand == r29 GRF");
    CHECK(((i.dw[1] >> 16) & 0x3u) == 1, "add Dst.HorzStride == 1");
    CHECK(((i.dw[1] >> 15) & 0x1u) == 0, "add Src1.IsImm == 0");
    CHECK(((i.dw[1] >> 14) & 0x1u) == 0, "add Src0.IsImm == 0");
    CHECK(((i.dw[1] >> 8) & 0xFu) == VBS_ALCHEMIST_TYPE_F,
          "add Src0.DataType == :f (10)");
    CHECK(((i.dw[1] >> 4) & 0xFu) == VBS_ALCHEMIST_TYPE_F,
          "add Dst.DataType == :f (10)");
    CHECK(i.dw[1] == 0x1D050AA0u, "add DW1 golden 0x1D050AA0");
    /* DW2: Src1 type :f, Src0 region <8;8,1>, Src0 r25. */
    CHECK(((i.dw[2] >> 24) & 0xFu) == VBS_ALCHEMIST_TYPE_F,
          "add Src1.DataType == :f");
    CHECK(((i.dw[2] >> 20) & 0xFu) == 4, "add Src0.VertStride == 4 (8 el)");
    CHECK(((i.dw[2] >> 17) & 0x7u) == 3, "add Src0.Width == 3 (8 el)");
    CHECK(((i.dw[2] >> 2) & 0x3FFFu) == 0x0641,
          "add Src0.Operand == r25 GRF");
    CHECK((i.dw[2] & 0x3u) == 1, "add Src0.HorzStride == 1");
    CHECK(i.dw[2] == 0x0A461905u, "add DW2 golden 0x0A461905");
    /* DW3: Src1 region <8;8,1>, Src1 r27, reserved bits 0. */
    CHECK(((i.dw[3] >> 20) & 0xFu) == 4, "add Src1.VertStride == 4");
    CHECK(((i.dw[3] >> 17) & 0x7u) == 3, "add Src1.Width == 3");
    CHECK(((i.dw[3] >> 2) & 0x3FFFu) == 0x06C1,
          "add Src1.Operand == r27 GRF");
    CHECK((i.dw[3] & 0x3u) == 1, "add Src1.HorzStride == 1");
    CHECK((i.dw[3] >> 26) == 0,
          "add DW3 reserved bits [127:122] == 0 (DW3 bits [31:26])");
    CHECK(i.dw[3] == 0x00461B05u, "add DW3 golden 0x00461B05");
}

static void test_mov(void)
{
    vbs_alchemist_inst_t i;
    int rc;

    /* mov (16) r3.0<1>:ud r1.0<16;16,1>:uw */
    rc = vbs_alchemist_emit_mov(
        &i, VBS_ALCHEMIST_EXECSIZE_16,
        vbs_alchemist_dst(3, 0, VBS_ALCHEMIST_TYPE_UD, 1),
        vbs_alchemist_op(1, 0, VBS_ALCHEMIST_TYPE_UW,
                         VBS_ALCHEMIST_REGION_UW16_V,
                         VBS_ALCHEMIST_REGION_UW16_W,
                         VBS_ALCHEMIST_REGION_UW16_H));
    CHECK(rc == 0, "mov uw->ud emits");
    CHECK(dw0op(&i) == 0x61, "mov opcode == 0x61");
    CHECK(i.dw[0] == 0x00040061u, "mov DW0 golden 0x00040061");
    CHECK(i.dw[3] == 0, "mov DW3 == 0 (no src1 fields, Vol 2a p.1112)");
    CHECK(((i.dw[1] >> 15) & 0x1u) == 0,
          "mov bit 47 Reserved == 0 (Vol 2a, p.1113)");
    CHECK(((i.dw[2] >> 24) & 0xFu) == 0,
          "mov DW2 bits [91:88] == 0 (no Src1.DataType field)");
    CHECK(((i.dw[1] >> 8) & 0xFu) == VBS_ALCHEMIST_TYPE_UW,
          "mov Src0.DataType == :uw (1)");
    CHECK(((i.dw[1] >> 4) & 0xFu) == VBS_ALCHEMIST_TYPE_UD,
          "mov Dst.DataType == :ud (2)");
    CHECK(((i.dw[2] >> 20) & 0xFu) == 5,
          "mov Src0.VertStride == 5 (16 el)");
    CHECK(((i.dw[2] >> 17) & 0x7u) == 4, "mov Src0.Width == 4 (16 el)");
    CHECK(i.dw[1] == 0x03050120u, "mov DW1 golden 0x03050120");
    CHECK(i.dw[2] == 0x00580105u, "mov DW2 golden 0x00580105");

    /* mov (8) r5:uq = r3:ud — the SIMD8 :uq conversion. */
    rc = vbs_alchemist_emit_mov(
        &i, VBS_ALCHEMIST_EXECSIZE_8,
        vbs_alchemist_dst(5, 0, VBS_ALCHEMIST_TYPE_UQ, 1),
        vbs_alchemist_op(3, 0, VBS_ALCHEMIST_TYPE_UD,
                         VBS_ALCHEMIST_REGION_F32_V,
                         VBS_ALCHEMIST_REGION_F32_W,
                         VBS_ALCHEMIST_REGION_F32_H));
    CHECK(rc == 0, "mov ud->uq SIMD8 emits");
    CHECK(i.dw[0] == 0x00030061u, "mov-uq DW0 golden 0x00030061");
    CHECK(i.dw[1] == 0x05050230u, "mov-uq DW1 golden 0x05050230");
    CHECK(i.dw[2] == 0x00460305u, "mov-uq DW2 golden 0x00460305");

    /* mov (8) r9:uq = r2.0<0;1,0>:uq — scalar broadcast. */
    rc = vbs_alchemist_emit_mov(
        &i, VBS_ALCHEMIST_EXECSIZE_8,
        vbs_alchemist_dst(9, 0, VBS_ALCHEMIST_TYPE_UQ, 1),
        vbs_alchemist_op(2, 0, VBS_ALCHEMIST_TYPE_UQ,
                         VBS_ALCHEMIST_REGION_SCALAR_V,
                         VBS_ALCHEMIST_REGION_SCALAR_W,
                         VBS_ALCHEMIST_REGION_SCALAR_H));
    CHECK(rc == 0, "mov broadcast emits");
    CHECK(((i.dw[2] >> 20) & 0xFu) == 0, "broadcast VertStride == 0");
    CHECK(((i.dw[2] >> 17) & 0x7u) == 0, "broadcast Width == 0 (1 el)");
    CHECK((i.dw[2] & 0x3u) == 0, "broadcast HorzStride == 0");
    CHECK(i.dw[2] == 0x00000204u, "mov-bcast DW2 golden 0x00000204");
}

static void test_shl_imm(void)
{
    vbs_alchemist_inst_t i;
    int rc;

    /* shl (16) r3.0<1>:ud r3.0<8;8,1>:ud 2:ud */
    rc = vbs_alchemist_emit_shl_imm(
        &i, VBS_ALCHEMIST_EXECSIZE_16,
        vbs_alchemist_dst(3, 0, VBS_ALCHEMIST_TYPE_UD, 1),
        vbs_alchemist_op(3, 0, VBS_ALCHEMIST_TYPE_UD,
                         VBS_ALCHEMIST_REGION_F32_V,
                         VBS_ALCHEMIST_REGION_F32_W,
                         VBS_ALCHEMIST_REGION_F32_H),
        2, VBS_ALCHEMIST_TYPE_UD);
    CHECK(rc == 0, "shl imm emits");
    CHECK(dw0op(&i) == 0x69, "shl opcode == 0x69");
    CHECK(i.dw[3] == 2, "shl DW3 == immediate 2 (Src1.ImmValue)");
    CHECK(((i.dw[1] >> 15) & 0x1u) == 1, "shl Src1.IsImm == 1");
    CHECK(((i.dw[2] >> 24) & 0xFu) == VBS_ALCHEMIST_TYPE_UD,
          "shl Src1.DataType (ImmDataType) == :ud");
    CHECK(i.dw[0] == 0x00040069u, "shl DW0 golden 0x00040069");
    CHECK(i.dw[1] == 0x03058220u, "shl DW1 golden 0x03058220");
    CHECK(i.dw[2] == 0x02460305u, "shl DW2 golden 0x02460305");
}

static void test_add_uq(void)
{
    vbs_alchemist_inst_t i;
    int rc;

    /* add (8) r13.0<1>:uq r5.0<4;4,1>:uq r9.0<4;4,1>:uq */
    rc = emit_add_uq(&i, VBS_ALCHEMIST_EXECSIZE_8);
    CHECK(rc == 0, "add :uq SIMD8 emits");
    CHECK(((i.dw[2] >> 20) & 0xFu) == 3, "add-uq VertStride == 3 (4 el)");
    CHECK(((i.dw[2] >> 17) & 0x7u) == 2, "add-uq Width == 2 (4 el)");
    CHECK(i.dw[0] == 0x00030040u, "add-uq DW0 golden 0x00030040");
    CHECK(i.dw[1] == 0x0D050330u, "add-uq DW1 golden 0x0D050330");
    CHECK(i.dw[2] == 0x03340505u, "add-uq DW2 golden 0x03340505");
    CHECK(i.dw[3] == 0x00340905u, "add-uq DW3 golden 0x00340905");
}

static void test_send_load(void)
{
    vbs_alchemist_inst_t i;
    int rc;

    /* send (16) r25:dc1 r13, null, desc 0x08246EFF, ex_desc 0 */
    rc = vbs_alchemist_emit_send(&i, VBS_ALCHEMIST_EXECSIZE_16,
                                 VBS_ALCHEMIST_SFID_DP_DC1, 25, 13,
                                 VBS_ALCHEMIST_REG_NULL,
                                 VBS_ALCHEMIST_DESC_A64_READ_SIMD16, 0, 0);
    CHECK(rc == 0, "send load emits");
    CHECK(dw0op(&i) == 0x31, "send opcode == 0x31");
    CHECK(dw0exec(&i) == VBS_ALCHEMIST_EXECSIZE_16,
          "send load ExecSize == 16 (DERIVED: WrEn covers 16 lanes)");
    CHECK(i.dw[0] == 0x00040031u, "send load DW0 golden 0x00040031");
    CHECK(((i.dw[2] >> 28) & 0xFu) == VBS_ALCHEMIST_SFID_DP_DC1,
          "send SFID == 12 (DC1)");
    CHECK(send_desc(&i) == VBS_ALCHEMIST_DESC_A64_READ_SIMD16,
          "send load desc reconstituted == 0x08246EFF");
    CHECK(send_exdesc(&i) == 0, "send load ex_desc reconstituted == 0");
    CHECK(((i.dw[1] >> 24) & 0xFFu) == 25, "send load Dst.RegNum == 25");
    CHECK(((i.dw[1] >> 19) & 0x1Fu) == 2,
          "send load Dst.Length (RespLen) == 2");
    CHECK(((i.dw[1] >> 18) & 0x1u) == 1, "send load Dst.RegFile == GRF");
    CHECK(((i.dw[2] >> 8) & 0xFFu) == 13, "send load Src0.RegNum == 13");
    CHECK(((i.dw[2] >> 3) & 0xFu) == 4,
          "send load Src0.Length (MsgLen) == 4");
    CHECK(((i.dw[2] >> 2) & 0x1u) == 1, "send load Src0.RegFile == GRF");
    CHECK(((i.dw[3] >> 8) & 0xFFu) == 0, "send load Src1.RegNum == 0 (null)");
    CHECK(((i.dw[3] >> 3) & 0x1Fu) == 0, "send load Src1.Length == 0");
    CHECK(((i.dw[3] >> 2) & 0x1u) == 0, "send load Src1.RegFile == null");
    CHECK(((i.dw[1] >> 2) & 0x1u) == 0, "send load EOT == 0");
    CHECK(((i.dw[1] >> 0) & 0x3u) == 0,
          "send load FusionCtrl/AtomicCtrl == 0");
    CHECK(((i.dw[2] >> 16) & 0x1u) == 0, "send load DW2 bit 80 == 0");
    CHECK(((i.dw[3] >> 16) & 0x1u) == 0, "send load DW3 bit 112 == 0");
    CHECK(i.dw[1] == 0x19140000u, "send load DW1 golden 0x19140000");
    CHECK(i.dw[2] == 0xCDFE0D24u, "send load DW2 golden 0xCDFE0D24");
    CHECK(i.dw[3] == 0x011A0000u, "send load DW3 golden 0x011A0000");
    /* The desc fields themselves, from the digest §6.8 breakdown. */
    CHECK(((VBS_ALCHEMIST_DESC_A64_READ_SIMD16 >> 25) & 0xFu) == 4,
          "read desc MsgLen == 4 (SIMD16 x 8B addresses = 4 GRFs)");
    CHECK(((VBS_ALCHEMIST_DESC_A64_READ_SIMD16 >> 20) & 0x1Fu) == 2,
          "read desc RespLen == 2 (16 floats = 2 GRFs)");
    CHECK(((VBS_ALCHEMIST_DESC_A64_READ_SIMD16 >> 19) & 0x1u) == 0,
          "read desc Header == 0 (forbidden, MSD1R_A64_US)");
    CHECK(((VBS_ALCHEMIST_DESC_A64_READ_SIMD16 >> 14) & 0x1Fu) == 0x11,
          "read desc MsgType == 0x11");
    CHECK(((VBS_ALCHEMIST_DESC_A64_READ_SIMD16 >> 12) & 0x3u) == 2,
          "read desc SIMD Mode == 2 (SIMD16)");
    CHECK(((VBS_ALCHEMIST_DESC_A64_READ_SIMD16 >> 8) & 0xFu) == 0xE,
          "read desc Channel Mask == 0xE (R only)");
    CHECK((VBS_ALCHEMIST_DESC_A64_READ_SIMD16 & 0xFFu) == 0xFF,
          "read desc BTI == 255 (stateless A64)");
}

static void test_send_store(void)
{
    vbs_alchemist_inst_t i;
    int rc;

    /* send (16) null:dc1 r21, r29, desc 0x08066EFF, ex_desc 0x80 */
    rc = vbs_alchemist_emit_send(&i, VBS_ALCHEMIST_EXECSIZE_16,
                                 VBS_ALCHEMIST_SFID_DP_DC1,
                                 VBS_ALCHEMIST_REG_NULL, 21, 29,
                                 VBS_ALCHEMIST_DESC_A64_WRITE_SIMD16,
                                 VBS_ALCHEMIST_EXDESC_A64_WRITE_SIMD16, 0);
    CHECK(rc == 0, "send store emits");
    CHECK(send_desc(&i) == VBS_ALCHEMIST_DESC_A64_WRITE_SIMD16,
          "send store desc reconstituted == 0x08066EFF");
    CHECK(send_exdesc(&i) == VBS_ALCHEMIST_EXDESC_A64_WRITE_SIMD16,
          "send store ex_desc reconstituted == 0x80");
    CHECK(((i.dw[1] >> 24) & 0xFFu) == 0, "send store Dst.RegNum == 0 (null)");
    CHECK(((i.dw[1] >> 19) & 0x1Fu) == 0, "send store RespLen == 0");
    CHECK(((i.dw[1] >> 18) & 0x1u) == 0, "send store Dst.RegFile == null");
    CHECK(((i.dw[2] >> 8) & 0xFFu) == 21, "send store Src0.RegNum == 21");
    CHECK(((i.dw[2] >> 3) & 0xFu) == 4, "send store MsgLen == 4");
    CHECK(((i.dw[3] >> 8) & 0xFFu) == 29, "send store Src1.RegNum == 29");
    CHECK(((i.dw[3] >> 3) & 0x1Fu) == 2,
          "send store Src1.Length (ExtMsgLen) == 2");
    CHECK(((i.dw[3] >> 2) & 0x1u) == 1, "send store Src1.RegFile == GRF");
    CHECK(((i.dw[1] >> 2) & 0x1u) == 0, "send store EOT == 0");
    CHECK(i.dw[1] == 0x00000000u, "send store DW1 golden 0x00000000");
    CHECK(i.dw[2] == 0xCDFE1524u, "send store DW2 golden 0xCDFE1524");
    CHECK(i.dw[3] == 0x019A1D14u, "send store DW3 golden 0x019A1D14");
}

static void test_send_eot(void)
{
    vbs_alchemist_inst_t i;
    int rc;

    /* send (1) null:gtwy r112, null, desc 0x02000000, EOT */
    rc = vbs_alchemist_emit_send(&i, VBS_ALCHEMIST_EXECSIZE_1,
                                 VBS_ALCHEMIST_SFID_GATEWAY,
                                 VBS_ALCHEMIST_REG_NULL, 112,
                                 VBS_ALCHEMIST_REG_NULL,
                                 VBS_ALCHEMIST_DESC_EOT, 0, 1);
    CHECK(rc == 0, "send EOT emits");
    CHECK(dw0exec(&i) == VBS_ALCHEMIST_EXECSIZE_1,
          "send EOT ExecSize == 1 (DERIVED: no data channels)");
    CHECK(i.dw[0] == 0x00000031u, "send EOT DW0 golden 0x00000031");
    CHECK(((i.dw[2] >> 28) & 0xFu) == VBS_ALCHEMIST_SFID_GATEWAY,
          "send EOT SFID == 3 (Gateway)");
    CHECK(send_desc(&i) == VBS_ALCHEMIST_DESC_EOT,
          "send EOT desc reconstituted == 0x02000000");
    CHECK(((i.dw[1] >> 2) & 0x1u) == 1, "send EOT bit (bit 34) == 1");
    CHECK(((i.dw[2] >> 8) & 0xFFu) == 112, "send EOT Src0 == r112");
    CHECK(((i.dw[2] >> 3) & 0xFu) == 1, "send EOT MsgLen == 1");
    CHECK(i.dw[1] == 0x00000004u, "send EOT DW1 golden 0x00000004");
    CHECK(i.dw[2] == 0x3000700Cu, "send EOT DW2 golden 0x3000700C");
    CHECK(i.dw[3] == 0x00000000u, "send EOT DW3 golden 0x00000000");
}

static void test_send_rules(void)
{
    vbs_alchemist_inst_t i;

    /* EOT with a GRF destination: forbidden (Vol 2a, p.1254). */
    CHECK(vbs_alchemist_emit_send(&i, VBS_ALCHEMIST_EXECSIZE_1,
                                  VBS_ALCHEMIST_SFID_GATEWAY, 25, 112,
                                  VBS_ALCHEMIST_REG_NULL,
                                  VBS_ALCHEMIST_DESC_EOT, 0, 1) != 0,
          "EOT with GRF dst refused");
    /* EOT with a source outside r112-r127: forbidden (p.1255). */
    CHECK(vbs_alchemist_emit_send(&i, VBS_ALCHEMIST_EXECSIZE_1,
                                  VBS_ALCHEMIST_SFID_GATEWAY,
                                  VBS_ALCHEMIST_REG_NULL, 50,
                                  VBS_ALCHEMIST_REG_NULL,
                                  VBS_ALCHEMIST_DESC_EOT, 0, 1) != 0,
          "EOT with src0 r50 refused (must be r112-r127)");
    /* Null dst with nonzero Response Length: forbidden. */
    CHECK(vbs_alchemist_emit_send(&i, VBS_ALCHEMIST_EXECSIZE_16,
                                  VBS_ALCHEMIST_SFID_DP_DC1,
                                  VBS_ALCHEMIST_REG_NULL, 13,
                                  VBS_ALCHEMIST_REG_NULL,
                                  VBS_ALCHEMIST_DESC_A64_READ_SIMD16, 0, 0)
          != 0, "null dst + RespLen 2 refused");
    /* Null src1 with nonzero Extended Message Length: forbidden. */
    CHECK(vbs_alchemist_emit_send(&i, VBS_ALCHEMIST_EXECSIZE_16,
                                  VBS_ALCHEMIST_SFID_DP_DC1,
                                  VBS_ALCHEMIST_REG_NULL, 21,
                                  VBS_ALCHEMIST_REG_NULL,
                                  VBS_ALCHEMIST_DESC_A64_WRITE_SIMD16,
                                  VBS_ALCHEMIST_EXDESC_A64_WRITE_SIMD16, 0)
          != 0, "null src1 + ExtMsgLen 2 refused");
}

static void test_alu_rules(void)
{
    vbs_alchemist_inst_t i;

    /* SIMD16 :uq add: ILLEGAL — ExecSize * 8 = 128 > 64
     * (Vol 9, p.387). The assembler must refuse it. */
    CHECK(emit_add_uq(&i, VBS_ALCHEMIST_EXECSIZE_16) != 0,
          "SIMD16 :uq add refused (ExecSize*n <= 64, Vol 9 p.387)");
    /* SIMD16 :uq mov: same rule. */
    CHECK(vbs_alchemist_emit_mov(
              &i, VBS_ALCHEMIST_EXECSIZE_16,
              vbs_alchemist_dst(5, 0, VBS_ALCHEMIST_TYPE_UQ, 1),
              vbs_alchemist_op(3, 0, VBS_ALCHEMIST_TYPE_UD,
                               VBS_ALCHEMIST_REGION_F32_V,
                               VBS_ALCHEMIST_REGION_F32_W,
                               VBS_ALCHEMIST_REGION_F32_H)) != 0,
          "SIMD16 :uq mov refused (ExecSize*n <= 64)");
    /* Width 16 with ExecSize 8: Width must be <= ExecSize. */
    CHECK(vbs_alchemist_emit_mov(
              &i, VBS_ALCHEMIST_EXECSIZE_8,
              vbs_alchemist_dst(3, 0, VBS_ALCHEMIST_TYPE_UD, 1),
              vbs_alchemist_op(1, 0, VBS_ALCHEMIST_TYPE_UW,
                               VBS_ALCHEMIST_REGION_UW16_V,
                               VBS_ALCHEMIST_REGION_UW16_W,
                               VBS_ALCHEMIST_REGION_UW16_H)) != 0,
          "Width 16 > ExecSize 8 refused (Vol 2b, p.78)");
    /* Register out of range. */
    CHECK(vbs_alchemist_emit_mov(
              &i, VBS_ALCHEMIST_EXECSIZE_8,
              vbs_alchemist_dst(200, 0, VBS_ALCHEMIST_TYPE_UD, 1),
              vbs_alchemist_op(1, 0, VBS_ALCHEMIST_TYPE_UW,
                               VBS_ALCHEMIST_REGION_F32_V,
                               VBS_ALCHEMIST_REGION_F32_W,
                               VBS_ALCHEMIST_REGION_F32_H)) != 0,
          "GRF 200 refused (r0-r127 only)");
}

static void test_nop_serialize(void)
{
    vbs_alchemist_inst_t i;
    uint8_t bytes[16];
    int rc;

    rc = vbs_alchemist_emit_nop(&i);
    CHECK(rc == 0, "nop emits");
    CHECK(i.dw[0] == 0x60 && i.dw[1] == 0 && i.dw[2] == 0 && i.dw[3] == 0,
          "nop == opcode 0x60, all else 0");

    /* Serialization: DW0 first, little-endian; byte 0 is the
     * opcode (Vol 9, p.469). */
    rc = emit_add_f32(&i, VBS_ALCHEMIST_EXECSIZE_16);
    CHECK(rc == 0, "add for serialize emits");
    vbs_alchemist_serialize(&i, bytes);
    CHECK(bytes[0] == 0x40 && bytes[1] == 0x00 && bytes[2] == 0x04 &&
          bytes[3] == 0x00,
          "serialize: bytes 0-3 are DW0 little-endian, opcode first");
    CHECK(bytes[4] == 0xA0 && bytes[5] == 0x0A && bytes[6] == 0x05 &&
          bytes[7] == 0x1D,
          "serialize: bytes 4-7 are DW1 little-endian (0x1D050AA0)");
    CHECK(bytes[15] == 0x00 && bytes[14] == 0x46,
          "serialize: bytes 12-15 are DW3 little-endian (0x00461B05)");
}

int main(void)
{
    test_opcode_table();
    test_direct_operand();
    test_desc_build();
    test_add_f32();
    test_mov();
    test_shl_imm();
    test_add_uq();
    test_send_load();
    test_send_store();
    test_send_eot();
    test_send_rules();
    test_alu_rules();
    test_nop_serialize();

    TEST_SUMMARY("test_alchemist_eu");
}
