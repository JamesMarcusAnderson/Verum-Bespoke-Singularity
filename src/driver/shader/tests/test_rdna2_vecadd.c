/* test_rdna2_vecadd.c — RDNA 2 vector-add kernel tests.
 *
 * Checks the assembled kernel (exact encodings), the raw code
 * bytes, the kernel descriptor, the ELF code object (header,
 * program headers, section headers, symbols, alignments), the
 * PGM_RSRC field helpers, and the dispatch setup (thread-group
 * dims, USER_DATA -> SGPR mapping).
 *
 * No GPU needed: pure encoding and layout checks.
 */

#include <stdint.h>
#include <string.h>

#include "test_util.h"
#include "vbs_rdna2_vecadd.h"

/* Little-endian readers for the ELF checks. */
static uint16_t get16(const uint8_t *p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t get32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t get64(const uint8_t *p) {
    return (uint64_t)get32(p) | ((uint64_t)get32(p + 4) << 32);
}

/* Expected kernel: one row per instruction (digest §5.4). */
static const struct {
    uint32_t dw0;
    uint32_t dw1;
    int is64;
} EXPECTED[VBS_RDNA2_VECADD_NUM_INSTS] = {
    { 0xDC588000u, 0x00000000u, 1 },  /* load A: v0, s[0:1] */
    { 0xDC588000u, 0x01020000u, 1 },  /* load B: v1, s[2:3] */
    { 0xBF8C0000u, 0x00000000u, 0 },  /* s_waitcnt 0 */
    { 0x4A040300u, 0x00000000u, 0 },  /* v_add_nc_u32 v2, v0, v1 */
    { 0xDC5C8000u, 0x00040200u, 1 },  /* store C: v2, s[4:5] */
    { 0xBF8C0000u, 0x00000000u, 0 },  /* s_waitcnt 0 */
    { 0xBF810000u, 0x00000000u, 0 },  /* s_endpgm */
};

static const uint8_t EXPECTED_CODE[VBS_RDNA2_VECADD_CODE_SIZE] = {
    0x00, 0x80, 0x58, 0xDC, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x80, 0x58, 0xDC, 0x00, 0x00, 0x02, 0x01,
    0x00, 0x00, 0x8C, 0xBF, 0x00, 0x03, 0x04, 0x4A,
    0x00, 0x80, 0x5C, 0xDC, 0x00, 0x02, 0x04, 0x00,
    0x00, 0x00, 0x8C, 0xBF, 0x00, 0x00, 0x81, 0xBF,
};

static void test_kernel_build(void) {
    vbs_rdna2_inst_t insts[VBS_RDNA2_VECADD_NUM_INSTS];
    uint8_t code[VBS_RDNA2_VECADD_CODE_SIZE];
    size_t n, i, total = 0;
    static const size_t SIZES[7] = { 8, 8, 4, 4, 8, 4, 4 };

    CHECK(vbs_rdna2_vecadd_build(NULL, 7) == 0, "build NULL rejected");
    CHECK(vbs_rdna2_vecadd_build(insts, 6) == 0,
          "build with capacity 6 rejected");

    n = vbs_rdna2_vecadd_build(insts, VBS_RDNA2_VECADD_NUM_INSTS);
    CHECK(n == VBS_RDNA2_VECADD_NUM_INSTS, "build returns 7");
    for (i = 0; i < n; i++) {
        CHECK_EQ_U32(insts[i].dw0, EXPECTED[i].dw0,
                     "kernel inst dw0");
        CHECK_EQ_U32(insts[i].dw1, EXPECTED[i].dw1,
                     "kernel inst dw1");
        CHECK(insts[i].is64 == EXPECTED[i].is64,
              "kernel inst width");
        CHECK(vbs_rdna2_inst_size(&insts[i]) == SIZES[i],
              "kernel inst size");
        total += vbs_rdna2_inst_size(&insts[i]);
    }
    CHECK(total == VBS_RDNA2_VECADD_CODE_SIZE,
          "kernel total size 40 bytes");

    /* The add is the digest §5.3 oracle, unchanged. */
    CHECK_EQ_U32(insts[3].dw0, 0x4A040300u, "v_add oracle in kernel");
    /* First and last instructions. */
    CHECK_EQ_U32(insts[0].dw0, 0xDC588000u, "first inst is the A load");
    CHECK_EQ_U32(insts[6].dw0, 0xBF810000u, "last inst is s_endpgm");

    /* Raw code bytes match the expected stream exactly. */
    vbs_rdna2_vecadd_serialize(code);
    CHECK(memcmp(code, EXPECTED_CODE, VBS_RDNA2_VECADD_CODE_SIZE) == 0,
          "raw code bytes match");
}

static void test_descriptor(void) {
    uint8_t desc[VBS_RDNA2_DESC_SIZE];
    size_t i;

    vbs_rdna2_vecadd_descriptor(desc);
    CHECK_EQ_U32(get32(desc + VBS_RDNA2_DESC_GROUP_SEGMENT_FIXED_SIZE),
                 0u, "desc GROUP_SEGMENT_FIXED_SIZE = 0 (no LDS)");
    CHECK_EQ_U32(get32(desc + VBS_RDNA2_DESC_PRIVATE_SEGMENT_FIXED_SIZE),
                 0u, "desc PRIVATE_SEGMENT_FIXED_SIZE = 0 (no scratch)");
    CHECK_EQ_U32(get32(desc + VBS_RDNA2_DESC_KERNARG_SIZE), 0u,
                 "desc KERNARG_SIZE = 0 (args via USER_DATA)");
    CHECK_EQ_U32(get32(desc + 0x0C), 0u, "desc reserved @0x0C = 0");
    CHECK_EQ_U64(get64(desc + VBS_RDNA2_DESC_CODE_ENTRY_BYTE_OFFSET),
                 256u, "desc KERNEL_CODE_ENTRY_BYTE_OFFSET = 256");
    for (i = 0x18; i < 0x2C; i++)
        CHECK(desc[i] == 0, "desc reserved @0x18-0x2B = 0");
    CHECK_EQ_U32(get32(desc + VBS_RDNA2_DESC_PGM_RSRC3), 0u,
                 "desc PGM_RSRC3 = 0");
    CHECK_EQ_U32(get32(desc + VBS_RDNA2_DESC_PGM_RSRC1), 0u,
                 "desc PGM_RSRC1 = 0 (VGPRS field 0, 3 VGPRs wave64)");
    CHECK_EQ_U32(get32(desc + VBS_RDNA2_DESC_PGM_RSRC2), 0xCu,
                 "desc PGM_RSRC2 = 0xC (USER_SGPR 6)");
    CHECK_EQ_U32(get32(desc + VBS_RDNA2_DESC_CODE_PROPERTIES), 0u,
                 "desc KERNEL_CODE_PROPERTIES = 0 (wave64)");
    CHECK_EQ_U32(get32(desc + 0x3C), 0u, "desc reserved @0x3C = 0");

    /* PGM_RSRC2 bit fields: USER_SGPR [5:1] = 6, SCRATCH_EN = 0,
     * LDS_SIZE [23:15] = 0 (digest §4.4). */
    CHECK_EQ_U32((get32(desc + VBS_RDNA2_DESC_PGM_RSRC2) >> 1) & 0x1Fu,
                 6u, "desc USER_SGPR field = 6");
    CHECK_EQ_U32(get32(desc + VBS_RDNA2_DESC_PGM_RSRC2) & 1u, 0u,
                 "desc SCRATCH_EN = 0");
    CHECK_EQ_U32((get32(desc + VBS_RDNA2_DESC_PGM_RSRC2) >> 15) & 0x1FFu,
                 0u, "desc LDS_SIZE field = 0");
}

static void test_pgm_rsrc_helpers(void) {
    /* VGPRS field, wave64: max(0, ceil(v/4) - 1). */
    CHECK_EQ_U32(vbs_rdna2_vgprs_field(0, 1), 0u, "vgprs 0 wave64");
    CHECK_EQ_U32(vbs_rdna2_vgprs_field(1, 1), 0u, "vgprs 1 wave64");
    CHECK_EQ_U32(vbs_rdna2_vgprs_field(3, 1), 0u,
                 "vgprs 3 wave64 (this kernel)");
    CHECK_EQ_U32(vbs_rdna2_vgprs_field(4, 1), 0u, "vgprs 4 wave64");
    CHECK_EQ_U32(vbs_rdna2_vgprs_field(5, 1), 1u, "vgprs 5 wave64");
    CHECK_EQ_U32(vbs_rdna2_vgprs_field(8, 1), 1u, "vgprs 8 wave64");
    CHECK_EQ_U32(vbs_rdna2_vgprs_field(9, 1), 2u, "vgprs 9 wave64");
    /* VGPRS field, wave32: max(0, ceil(v/8) - 1). */
    CHECK_EQ_U32(vbs_rdna2_vgprs_field(8, 0), 0u, "vgprs 8 wave32");
    CHECK_EQ_U32(vbs_rdna2_vgprs_field(9, 0), 1u, "vgprs 9 wave32");
    CHECK_EQ_U32(vbs_rdna2_vgprs_field(16, 0), 1u, "vgprs 16 wave32");
    CHECK_EQ_U32(vbs_rdna2_vgprs_field(17, 0), 2u, "vgprs 17 wave32");
    /* LDS_SIZE field: roundup(bytes / 512). */
    CHECK_EQ_U32(vbs_rdna2_lds_size_field(0), 0u, "lds 0 bytes");
    CHECK_EQ_U32(vbs_rdna2_lds_size_field(1), 1u, "lds 1 byte");
    CHECK_EQ_U32(vbs_rdna2_lds_size_field(512), 1u, "lds 512 bytes");
    CHECK_EQ_U32(vbs_rdna2_lds_size_field(513), 2u, "lds 513 bytes");
    CHECK_EQ_U32(vbs_rdna2_lds_size_field(1024), 2u, "lds 1024 bytes");
}

/* Section-header field offsets (Elf64_Shdr). */
#define SH_NAME 0
#define SH_TYPE 4
#define SH_FLAGS 8
#define SH_ADDR 16
#define SH_OFFSET 24
#define SH_SIZE 32
#define SH_LINK 40
#define SH_INFO 44
#define SH_ALIGN 48
#define SH_ENTSIZE 56

static void test_elf(void) {
    static uint8_t elf[VBS_RDNA2_ELF_SIZE];
    uint8_t desc[VBS_RDNA2_DESC_SIZE];
    uint8_t code[VBS_RDNA2_VECADD_CODE_SIZE];
    size_t n;
    uint64_t shoff;
    const uint8_t *sh;
    const uint8_t *shstr;
    const uint8_t *strtab;
    const uint8_t *symtab;

    CHECK(vbs_rdna2_vecadd_elf(NULL, VBS_RDNA2_ELF_SIZE) == 0,
          "elf NULL rejected");
    CHECK(vbs_rdna2_vecadd_elf(elf, VBS_RDNA2_ELF_SIZE - 1) == 0,
          "elf short capacity rejected");

    n = vbs_rdna2_vecadd_elf(elf, sizeof(elf));
    CHECK(n == VBS_RDNA2_ELF_SIZE, "elf size 0x448");

    /* ELF header ([AMDGPUUsage] "AMDGPU ELF Header"). */
    CHECK(elf[0] == 0x7F && elf[1] == 'E' && elf[2] == 'L' &&
          elf[3] == 'F', "elf magic");
    CHECK_EQ_U32(elf[4], 2u, "elf EI_CLASS = ELFCLASS64");
    CHECK_EQ_U32(elf[5], 1u, "elf EI_DATA = ELFDATA2LSB");
    CHECK_EQ_U32(elf[6], 1u, "elf EI_VERSION");
    CHECK_EQ_U32(elf[7], 64u, "elf EI_OSABI = HSA");
    CHECK_EQ_U32(elf[8], 3u, "elf EI_ABIVERSION = V5");
    CHECK_EQ_U32(get16(elf + 16), 3u, "elf e_type = ET_DYN");
    CHECK_EQ_U32(get16(elf + 18), 224u, "elf e_machine = EM_AMDGPU");
    CHECK_EQ_U32(get32(elf + 20), 1u, "elf e_version");
    CHECK_EQ_U64(get64(elf + 24), 0u, "elf e_entry = 0");
    CHECK_EQ_U64(get64(elf + 32), 64u, "elf e_phoff = 64");
    shoff = get64(elf + 40);
    CHECK_EQ_U64(shoff, 0x2C8u, "elf e_shoff");
    CHECK_EQ_U32(get32(elf + 48), 0x036u, "elf e_flags = gfx1030");
    CHECK_EQ_U32(get16(elf + 52), 64u, "elf e_ehsize");
    CHECK_EQ_U32(get16(elf + 54), 56u, "elf e_phentsize");
    CHECK_EQ_U32(get16(elf + 56), 2u, "elf e_phnum");
    CHECK_EQ_U32(get16(elf + 58), 64u, "elf e_shentsize");
    CHECK_EQ_U32(get16(elf + 60), 6u, "elf e_shnum");
    CHECK_EQ_U32(get16(elf + 62), 5u, "elf e_shstrndx");

    /* Program headers: PT_LOAD .rodata (R), PT_LOAD .text (R+X).
     * Elf64_Phdr: type @0, flags @4, offset @8, vaddr @16,
     * paddr @24, filesz @32, memsz @40, align @48. */
    CHECK_EQ_U32(get32(elf + 64), 1u, "phdr0 type PT_LOAD");
    CHECK_EQ_U32(get32(elf + 64 + 4), 4u, "phdr0 flags R");
    CHECK_EQ_U64(get64(elf + 64 + 8), 0x100u, "phdr0 offset");
    CHECK_EQ_U64(get64(elf + 64 + 16), 0x100u, "phdr0 vaddr");
    CHECK_EQ_U64(get64(elf + 64 + 32), 64u, "phdr0 filesz");
    CHECK_EQ_U64(get64(elf + 64 + 48), 0x1000u, "phdr0 align");
    CHECK_EQ_U32(get32(elf + 120), 1u, "phdr1 type PT_LOAD");
    CHECK_EQ_U32(get32(elf + 120 + 4), 5u, "phdr1 flags R+X");
    CHECK_EQ_U64(get64(elf + 120 + 8), 0x200u, "phdr1 offset");
    CHECK_EQ_U64(get64(elf + 120 + 16), 0x200u, "phdr1 vaddr");
    CHECK_EQ_U64(get64(elf + 120 + 32), 40u, "phdr1 filesz");
    /* p_vaddr == p_offset (mod p_align) for both segments. */
    CHECK((get64(elf + 64 + 16) - get64(elf + 64 + 8)) % 0x1000 == 0,
          "phdr0 vaddr congruent to offset");
    CHECK((get64(elf + 120 + 16) - get64(elf + 120 + 8)) % 0x1000 == 0,
          "phdr1 vaddr congruent to offset");

    /* Section headers. */
    sh = elf + shoff;
    shstr = elf + get64(sh + 5 * 64 + SH_OFFSET);
    CHECK(get32(sh + 0 * 64 + SH_TYPE) == 0, "shdr0 null type");
    CHECK(get64(sh + 0 * 64 + SH_SIZE) == 0, "shdr0 null size");

    /* .text (index 1). */
    CHECK(strcmp((const char *)shstr +
                 get32(sh + 1 * 64 + SH_NAME), ".text") == 0,
          ".text name");
    CHECK_EQ_U32(get32(sh + 1 * 64 + SH_TYPE), 1u, ".text PROGBITS");
    CHECK_EQ_U64(get64(sh + 1 * 64 + SH_FLAGS), 0x6u,
                 ".text ALLOC|EXECINSTR");
    CHECK_EQ_U64(get64(sh + 1 * 64 + SH_ADDR), 0x200u, ".text addr");
    CHECK_EQ_U64(get64(sh + 1 * 64 + SH_OFFSET), 0x200u, ".text offset");
    CHECK_EQ_U64(get64(sh + 1 * 64 + SH_SIZE), 40u, ".text size");
    CHECK_EQ_U64(get64(sh + 1 * 64 + SH_ALIGN), 256u,
                 ".text align 256 (code entry alignment)");

    /* .rodata (index 2). */
    CHECK(strcmp((const char *)shstr +
                 get32(sh + 2 * 64 + SH_NAME), ".rodata") == 0,
          ".rodata name");
    CHECK_EQ_U32(get32(sh + 2 * 64 + SH_TYPE), 1u, ".rodata PROGBITS");
    CHECK_EQ_U64(get64(sh + 2 * 64 + SH_FLAGS), 0x2u, ".rodata ALLOC");
    CHECK_EQ_U64(get64(sh + 2 * 64 + SH_ADDR), 0x100u, ".rodata addr");
    CHECK_EQ_U64(get64(sh + 2 * 64 + SH_OFFSET), 0x100u,
                 ".rodata offset");
    CHECK_EQ_U64(get64(sh + 2 * 64 + SH_SIZE), 64u, ".rodata size");
    CHECK_EQ_U64(get64(sh + 2 * 64 + SH_ALIGN), 64u,
                 ".rodata align 64 (descriptor alignment)");

    /* .symtab (index 3) and .strtab (index 4). */
    CHECK_EQ_U32(get32(sh + 3 * 64 + SH_TYPE), 2u, ".symtab SYMTAB");
    CHECK_EQ_U64(get64(sh + 3 * 64 + SH_SIZE), 72u, ".symtab size");
    CHECK_EQ_U32(get32(sh + 3 * 64 + SH_LINK), 4u, ".symtab link");
    CHECK_EQ_U32(get32(sh + 3 * 64 + SH_INFO), 1u, ".symtab info");
    CHECK_EQ_U64(get64(sh + 3 * 64 + SH_ENTSIZE), 24u,
                 ".symtab entsize");
    CHECK_EQ_U32(get32(sh + 4 * 64 + SH_TYPE), 3u, ".strtab STRTAB");
    CHECK_EQ_U64(get64(sh + 4 * 64 + SH_SIZE), 38u, ".strtab size");
    CHECK_EQ_U32(get32(sh + 5 * 64 + SH_TYPE), 3u, ".shstrtab STRTAB");
    CHECK_EQ_U64(get64(sh + 5 * 64 + SH_SIZE), 41u, ".shstrtab size");

    /* Descriptor bytes in .rodata match the descriptor builder,
     * and code bytes in .text match the raw serializer. */
    vbs_rdna2_vecadd_descriptor(desc);
    vbs_rdna2_vecadd_serialize(code);
    CHECK(memcmp(elf + 0x100, desc, 64) == 0,
          ".rodata holds the descriptor");
    CHECK(memcmp(elf + 0x200, code, 40) == 0,
          ".text holds the code");

    /* KERNEL_CODE_ENTRY_BYTE_OFFSET == text vaddr - rodata vaddr. */
    CHECK_EQ_U64(get64(elf + 0x100 +
                       VBS_RDNA2_DESC_CODE_ENTRY_BYTE_OFFSET),
                 get64(sh + 1 * 64 + SH_ADDR) -
                         get64(sh + 2 * 64 + SH_ADDR),
                 "entry offset == text addr - descriptor addr");
    CHECK_EQ_U64(get64(elf + 0x100 +
                       VBS_RDNA2_DESC_CODE_ENTRY_BYTE_OFFSET),
                 256u, "entry offset = 256");
    /* Alignment: descriptor 64-aligned, code entry 256-aligned. */
    CHECK(get64(sh + 2 * 64 + SH_ADDR) % 64 == 0,
          "descriptor 64-byte aligned");
    CHECK(get64(sh + 1 * 64 + SH_ADDR) % 256 == 0,
          "code entry 256-byte aligned");

    /* Symbols: null, .kd (STT_OBJECT, .rodata), kernel
     * (STT_FUNC, .text). Elf64_Sym: name @0, info @4, other @5,
     * shndx @6, value @8, size @16. */
    symtab = elf + get64(sh + 3 * 64 + SH_OFFSET);
    strtab = elf + get64(sh + 4 * 64 + SH_OFFSET);
    CHECK(get32(symtab) == 0 && get64(symtab + 8) == 0,
          "sym0 null");
    CHECK(strcmp((const char *)strtab + get32(symtab + 24),
                 VBS_RDNA2_VECADD_SYM_KD) == 0, "sym1 name .kd");
    CHECK_EQ_U32(symtab[24 + 4], 0x11u, "sym1 info GLOBAL|OBJECT");
    CHECK_EQ_U32(get16(symtab + 24 + 6), 2u, "sym1 shndx .rodata");
    CHECK_EQ_U64(get64(symtab + 24 + 8), 0x100u, "sym1 value");
    CHECK_EQ_U64(get64(symtab + 24 + 16), 64u, "sym1 size");
    CHECK(strcmp((const char *)strtab + get32(symtab + 48),
                 VBS_RDNA2_VECADD_SYM) == 0, "sym2 name kernel");
    CHECK_EQ_U32(symtab[48 + 4], 0x12u, "sym2 info GLOBAL|FUNC");
    CHECK_EQ_U32(get16(symtab + 48 + 6), 1u, "sym2 shndx .text");
    CHECK_EQ_U64(get64(symtab + 48 + 8), 0x200u, "sym2 value");
    CHECK_EQ_U64(get64(symtab + 48 + 16), 40u, "sym2 size");
}

static void test_dispatch(void) {
    vbs_rdna2_vecadd_dispatch_t d;
    static const struct {
        uint32_t n;
        uint32_t dim_x;
    } CASES[] = {
        { 1, 1 }, { 63, 1 }, { 64, 1 }, { 65, 2 },
        { 1024, 16 }, { 4096, 64 }, { 100000, 1563 },
    };
    size_t i;

    for (i = 0; i < sizeof(CASES) / sizeof(CASES[0]); i++) {
        vbs_rdna2_vecadd_dispatch(0, 0, 0, CASES[i].n, &d);
        CHECK_EQ_U32(d.dim_x, CASES[i].dim_x, "dispatch DIM_X");
        CHECK_EQ_U32(d.dim_y, 1u, "dispatch DIM_Y = 1");
        CHECK_EQ_U32(d.dim_z, 1u, "dispatch DIM_Z = 1");
    }

    /* Full setup for the worked example (digest App. A.2 shape). */
    vbs_rdna2_vecadd_dispatch(0x10000000ull, 0x20000000ull,
                              0x30000000ull, 1024, &d);
    CHECK_EQ_U32(d.num_thread_x, 64u, "NUM_THREAD_X = 64");
    CHECK_EQ_U32(d.num_thread_y, 1u, "NUM_THREAD_Y = 1");
    CHECK_EQ_U32(d.num_thread_z, 1u, "NUM_THREAD_Z = 1");
    CHECK_EQ_U32(d.dim_x, 16u, "DIM_X = 16 for n = 1024");
    CHECK_EQ_U32(d.initiator, 0x1u, "initiator COMPUTE_SHADER_EN");
    CHECK_EQ_U32(d.pgm_rsrc1, 0u, "dispatch PGM_RSRC1");
    CHECK_EQ_U32(d.pgm_rsrc2, 0xCu, "dispatch PGM_RSRC2");
    CHECK_EQ_U32(d.pgm_rsrc3, 0u, "dispatch PGM_RSRC3");
    CHECK_EQ_U32(d.user_data[0], 0x10000000u, "USER_DATA_0 = A lo");
    CHECK_EQ_U32(d.user_data[1], 0x00000000u, "USER_DATA_1 = A hi");
    CHECK_EQ_U32(d.user_data[2], 0x20000000u, "USER_DATA_2 = B lo");
    CHECK_EQ_U32(d.user_data[3], 0x00000000u, "USER_DATA_3 = B hi");
    CHECK_EQ_U32(d.user_data[4], 0x30000000u, "USER_DATA_4 = C lo");
    CHECK_EQ_U32(d.user_data[5], 0x00000000u, "USER_DATA_5 = C hi");

    /* 64-bit VA split: low dword first, then high (digest §7.3). */
    vbs_rdna2_vecadd_dispatch(0x123456789ABCDEF0ull,
                              0x0FEDCBA987654321ull,
                              0x00000000FFFFFFFFull, 64, &d);
    CHECK_EQ_U32(d.user_data[0], 0x9ABCDEF0u, "A lo split");
    CHECK_EQ_U32(d.user_data[1], 0x12345678u, "A hi split");
    CHECK_EQ_U32(d.user_data[2], 0x87654321u, "B lo split");
    CHECK_EQ_U32(d.user_data[3], 0x0FEDCBA9u, "B hi split");
    CHECK_EQ_U32(d.user_data[4], 0xFFFFFFFFu, "C lo split");
    CHECK_EQ_U32(d.user_data[5], 0x00000000u, "C hi split");
}

int main(void) {
    test_kernel_build();
    test_descriptor();
    test_pgm_rsrc_helpers();
    test_elf();
    test_dispatch();
    TEST_SUMMARY("test_rdna2_vecadd");
}
