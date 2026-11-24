/* vbs_rdna2_vecadd.c — minimal RDNA 2 (gfx1030) vector-add kernel.
 *
 * Instruction list (7 instructions, 40 bytes, digest §5.4):
 *
 *   addr  instruction                          encoding
 *   ----  -----------                          --------
 *   0x00  global_load_dword_addtid v0, s[0:1]  DC588000 00000000
 *   0x08  global_load_dword_addtid v1, s[2:3]  DC588000 01020000
 *   0x10  s_waitcnt 0                          BF8C0000
 *   0x14  v_add_nc_u32 v2, v0, v1              4A040300
 *   0x18  global_store_dword_addtid v2, s[4:5] DC5C8000 00040200
 *   0x20  s_waitcnt 0                          BF8C0000
 *   0x24  s_endpgm                             BF810000
 *
 * The load/store encodings are the digest §5.3 worked forms
 * with the SADDR/VDST/DATA fields filled in for this kernel's
 * register map (s[0:1] A, s[2:3] B, s[4:5] C; v0 A, v1 B, v2
 * sum). SADDR is the SGPR number of the base pair's low dword
 * ([RDNA2-ISA] §13.8.1), so s[2:3] -> 2 and s[4:5] -> 4.
 *
 * The ELF emitter below builds a code-object-v5 ET_DYN file by
 * hand (explicit little-endian writes, no host-endianness
 * assumptions). Layout is fixed; see the header for the map.
 * All ELF constants are the standard ones (ELFCLASS64,
 * ELFDATA2LSB, ET_DYN, SHT_*, STT_*, PT_LOAD, PF_*) plus the
 * AMDGPU values from [AMDGPUUsage] cited in the header.
 */

#include "vbs_rdna2_vecadd.h"

/* --- little-endian writers (host-independent) ----------------------- */
static void put16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)((v >> 8) & 0xFFu);
}

static void put32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)((v >> 8) & 0xFFu);
    p[2] = (uint8_t)((v >> 16) & 0xFFu);
    p[3] = (uint8_t)((v >> 24) & 0xFFu);
}

static void put64(uint8_t *p, uint64_t v) {
    put32(p, (uint32_t)(v & 0xFFFFFFFFu));
    put32(p + 4, (uint32_t)(v >> 32));
}

/* --- kernel build ---------------------------------------------------- */

size_t vbs_rdna2_vecadd_build(vbs_rdna2_inst_t *out,
                              size_t capacity) {
    if (!out || capacity < VBS_RDNA2_VECADD_NUM_INSTS)
        return 0;

    size_t i = 0;

    /* 0x00: v0 = A[tid]. */
    out[i++] = vbs_rdna2_emit_global_load_dword_addtid(
            VBS_RDNA2_VECADD_V_A, VBS_RDNA2_VECADD_S_A_LO, 0);
    /* 0x08: v1 = B[tid]. */
    out[i++] = vbs_rdna2_emit_global_load_dword_addtid(
            VBS_RDNA2_VECADD_V_B, VBS_RDNA2_VECADD_S_B_LO, 0);
    /* 0x10: wait for both loads. */
    out[i++] = vbs_rdna2_emit_s_waitcnt(0, 0, 0);
    /* 0x14: v2 = v0 + v1. */
    out[i++] = vbs_rdna2_emit_v_add_nc_u32(
            VBS_RDNA2_VECADD_V_SUM,
            VBS_RDNA2_SRC_VGPR(VBS_RDNA2_VECADD_V_A),
            VBS_RDNA2_VECADD_V_B);
    /* 0x18: C[tid] = v2. */
    out[i++] = vbs_rdna2_emit_global_store_dword_addtid(
            VBS_RDNA2_VECADD_V_SUM, VBS_RDNA2_VECADD_S_C_LO, 0);
    /* 0x20: wait (see the header note on vmcnt vs vscnt). */
    out[i++] = vbs_rdna2_emit_s_waitcnt(0, 0, 0);
    /* 0x24: done (implicitly waits first). */
    out[i++] = vbs_rdna2_emit_s_endpgm();

    return i;  /* 7 */
}

void vbs_rdna2_vecadd_serialize(
        uint8_t out[VBS_RDNA2_VECADD_CODE_SIZE]) {
    vbs_rdna2_inst_t insts[VBS_RDNA2_VECADD_NUM_INSTS];
    size_t n = vbs_rdna2_vecadd_build(insts,
                                      VBS_RDNA2_VECADD_NUM_INSTS);
    size_t off = 0;
    for (size_t i = 0; i < n; i++)
        off += vbs_rdna2_inst_serialize(&insts[i], out + off);
}

/* --- kernel descriptor ----------------------------------------------- */

void vbs_rdna2_vecadd_descriptor(uint8_t out[VBS_RDNA2_DESC_SIZE]) {
    for (size_t i = 0; i < VBS_RDNA2_DESC_SIZE; i++)
        out[i] = 0;
    /* Reserved fields stay 0 ("Reserved, must be 0"
     * [AMDGPUUsage]). */
    put32(out + VBS_RDNA2_DESC_GROUP_SEGMENT_FIXED_SIZE, 0);
    put32(out + VBS_RDNA2_DESC_PRIVATE_SEGMENT_FIXED_SIZE, 0);
    put32(out + VBS_RDNA2_DESC_KERNARG_SIZE, 0);
    put64(out + VBS_RDNA2_DESC_CODE_ENTRY_BYTE_OFFSET,
          VBS_RDNA2_VECADD_ENTRY_OFFSET);
    put32(out + VBS_RDNA2_DESC_PGM_RSRC3,
          VBS_RDNA2_VECADD_PGM_RSRC3);
    put32(out + VBS_RDNA2_DESC_PGM_RSRC1,
          VBS_RDNA2_VECADD_PGM_RSRC1);
    put32(out + VBS_RDNA2_DESC_PGM_RSRC2,
          VBS_RDNA2_VECADD_PGM_RSRC2);
    put32(out + VBS_RDNA2_DESC_CODE_PROPERTIES, 0);
}

/* --- ELF code object -------------------------------------------------- */

/* Fixed internal offsets of the non-alloc sections. .text ends at
 * 0x228; .strtab (38 bytes) runs 0x228-0x24E; .symtab is
 * 8-aligned at 0x250 (72 bytes, ends 0x298); .shstrtab (41
 * bytes) runs 0x298-0x2C1; section headers are 8-aligned at
 * 0x2C8 (6 x 64 = 384 bytes, ends 0x448 = VBS_RDNA2_ELF_SIZE). */
#define ELF_STRTAB_OFF 0x228u
#define ELF_SYMTAB_OFF 0x250u
#define ELF_SHSTRTAB_OFF 0x298u
#define ELF_SHOFF 0x2C8u

/* String-table contents (leading NUL, then the names). */
static const char ELF_STRTAB[] =
        "\0" VBS_RDNA2_VECADD_SYM_KD "\0" VBS_RDNA2_VECADD_SYM "\0";
/* Offsets into ELF_STRTAB: .kd at 1, kernel name at 21. */
#define ELF_STR_KD_OFF 1u
#define ELF_STR_SYM_OFF 21u

static const char ELF_SHSTRTAB[] =
        "\0.text\0.rodata\0.symtab\0.strtab\0.shstrtab\0";
/* Offsets into ELF_SHSTRTAB: .text 1, .rodata 7, .symtab 15,
 * .strtab 23, .shstrtab 31. */
#define ELF_SHSTR_TEXT_OFF 1u
#define ELF_SHSTR_RODATA_OFF 7u
#define ELF_SHSTR_SYMTAB_OFF 15u
#define ELF_SHSTR_STRTAB_OFF 23u
#define ELF_SHSTR_SHSTRTAB_OFF 31u

/* Section indices. */
#define ELF_SH_NULL 0u
#define ELF_SH_TEXT 1u
#define ELF_SH_RODATA 2u
#define ELF_SH_SYMTAB 3u
#define ELF_SH_STRTAB 4u
#define ELF_SH_SHSTRTAB 5u

/* ELF constants (standard ELF64; see the header citations). */
#define ELF_ET_DYN 3u
#define ELF_EV_CURRENT 1u
#define ELF_PT_LOAD 1u
#define ELF_PF_X 1u
#define ELF_PF_R 4u
#define ELF_SHT_PROGBITS 1u
#define ELF_SHT_SYMTAB 2u
#define ELF_SHT_STRTAB 3u
#define ELF_SHF_WRITE 1u
#define ELF_SHF_ALLOC 2u
#define ELF_SHF_EXECINSTR 4u
#define ELF_STB_GLOBAL 1u
#define ELF_STT_OBJECT 1u
#define ELF_STT_FUNC 2u

static void emit_phdr(uint8_t *p, uint32_t flags, uint64_t off,
                      uint64_t vaddr, uint64_t size) {
    put32(p + 0, ELF_PT_LOAD);
    put32(p + 4, flags);
    put64(p + 8, off);
    put64(p + 16, vaddr);
    put64(p + 24, vaddr);  /* p_paddr = p_vaddr */
    put64(p + 32, size);
    put64(p + 40, size);
    put64(p + 48, 0x1000);  /* p_align: page */
}

static void emit_shdr(uint8_t *p, uint32_t name, uint32_t type,
                      uint64_t flags, uint64_t addr, uint64_t off,
                      uint64_t size, uint32_t link, uint32_t info,
                      uint64_t align, uint64_t entsize) {
    put32(p + 0, name);
    put32(p + 4, type);
    put64(p + 8, flags);
    put64(p + 16, addr);
    put64(p + 24, off);
    put64(p + 32, size);
    put32(p + 40, link);
    put32(p + 44, info);
    put64(p + 48, align);
    put64(p + 56, entsize);
}

static void emit_sym(uint8_t *p, uint32_t name, uint8_t info,
                     uint16_t shndx, uint64_t value,
                     uint64_t size) {
    put32(p + 0, name);
    p[4] = info;
    p[5] = 0;  /* st_other */
    put16(p + 6, shndx);
    put64(p + 8, value);
    put64(p + 16, size);
}

size_t vbs_rdna2_vecadd_elf(uint8_t *out, size_t capacity) {
    uint8_t desc[VBS_RDNA2_DESC_SIZE];
    uint8_t code[VBS_RDNA2_VECADD_CODE_SIZE];
    size_t i;

    if (!out || capacity < VBS_RDNA2_ELF_SIZE)
        return 0;

    for (i = 0; i < VBS_RDNA2_ELF_SIZE; i++)
        out[i] = 0;

    vbs_rdna2_vecadd_descriptor(desc);
    vbs_rdna2_vecadd_serialize(code);

    /* ELF header (64 bytes). */
    out[0] = 0x7F;
    out[1] = 'E';
    out[2] = 'L';
    out[3] = 'F';
    out[4] = 2;   /* EI_CLASS: ELFCLASS64 */
    out[5] = 1;   /* EI_DATA: ELFDATA2LSB */
    out[6] = 1;   /* EI_VERSION: EV_CURRENT */
    out[7] = (uint8_t)VBS_RDNA2_ELF_OSABI_HSA;
    out[8] = (uint8_t)VBS_RDNA2_ELF_ABIVERSION_V5;
    /* e_ident[9..15] stay 0. */
    put16(out + 16, ELF_ET_DYN);
    put16(out + 18, VBS_RDNA2_ELF_MACHINE);
    put32(out + 20, ELF_EV_CURRENT);
    put64(out + 24, 0);  /* e_entry: 0 [AMDGPUUsage] */
    put64(out + 32, 64); /* e_phoff */
    put64(out + 40, ELF_SHOFF);
    put32(out + 48, VBS_RDNA2_ELF_FLAGS_GFX1030);
    put16(out + 52, 64); /* e_ehsize */
    put16(out + 54, 56); /* e_phentsize */
    put16(out + 56, 2);  /* e_phnum */
    put16(out + 58, 64); /* e_shentsize */
    put16(out + 60, 6);  /* e_shnum */
    put16(out + 62, ELF_SH_SHSTRTAB); /* e_shstrndx */

    /* Program headers: PT_LOAD .rodata (R), PT_LOAD .text (R+X). */
    emit_phdr(out + 64, ELF_PF_R, VBS_RDNA2_ELF_RODATA_OFF,
              VBS_RDNA2_ELF_RODATA_OFF, VBS_RDNA2_DESC_SIZE);
    emit_phdr(out + 64 + 56, ELF_PF_R | ELF_PF_X,
              VBS_RDNA2_ELF_TEXT_OFF, VBS_RDNA2_ELF_TEXT_VADDR,
              VBS_RDNA2_VECADD_CODE_SIZE);

    /* .rodata: the kernel descriptor. */
    for (i = 0; i < VBS_RDNA2_DESC_SIZE; i++)
        out[VBS_RDNA2_ELF_RODATA_OFF + i] = desc[i];

    /* .text: the kernel code. */
    for (i = 0; i < VBS_RDNA2_VECADD_CODE_SIZE; i++)
        out[VBS_RDNA2_ELF_TEXT_OFF + i] = code[i];

    /* .strtab. */
    for (i = 0; i < sizeof(ELF_STRTAB) - 1; i++)
        out[ELF_STRTAB_OFF + i] = (uint8_t)ELF_STRTAB[i];

    /* .symtab: null, .kd (STT_OBJECT), kernel (STT_FUNC). */
    emit_sym(out + ELF_SYMTAB_OFF + 24, ELF_STR_KD_OFF,
             (uint8_t)((ELF_STB_GLOBAL << 4) | ELF_STT_OBJECT),
             ELF_SH_RODATA, VBS_RDNA2_ELF_RODATA_OFF,
             VBS_RDNA2_DESC_SIZE);
    emit_sym(out + ELF_SYMTAB_OFF + 48, ELF_STR_SYM_OFF,
             (uint8_t)((ELF_STB_GLOBAL << 4) | ELF_STT_FUNC),
             ELF_SH_TEXT, VBS_RDNA2_ELF_TEXT_VADDR,
             VBS_RDNA2_VECADD_CODE_SIZE);

    /* .shstrtab. */
    for (i = 0; i < sizeof(ELF_SHSTRTAB) - 1; i++)
        out[ELF_SHSTRTAB_OFF + i] = (uint8_t)ELF_SHSTRTAB[i];

    /* Section headers. Index 0 stays all zero. */
    emit_shdr(out + ELF_SHOFF + 64 * ELF_SH_TEXT,
              ELF_SHSTR_TEXT_OFF, ELF_SHT_PROGBITS,
              ELF_SHF_ALLOC | ELF_SHF_EXECINSTR,
              VBS_RDNA2_ELF_TEXT_VADDR, VBS_RDNA2_ELF_TEXT_OFF,
              VBS_RDNA2_VECADD_CODE_SIZE, 0, 0, 256, 0);
    emit_shdr(out + ELF_SHOFF + 64 * ELF_SH_RODATA,
              ELF_SHSTR_RODATA_OFF, ELF_SHT_PROGBITS,
              ELF_SHF_ALLOC,
              VBS_RDNA2_ELF_RODATA_OFF, VBS_RDNA2_ELF_RODATA_OFF,
              VBS_RDNA2_DESC_SIZE, 0, 0, 64, 0);
    emit_shdr(out + ELF_SHOFF + 64 * ELF_SH_SYMTAB,
              ELF_SHSTR_SYMTAB_OFF, ELF_SHT_SYMTAB,
              0, 0, ELF_SYMTAB_OFF, 72,
              ELF_SH_STRTAB, 1, 8, 24);
    emit_shdr(out + ELF_SHOFF + 64 * ELF_SH_STRTAB,
              ELF_SHSTR_STRTAB_OFF, ELF_SHT_STRTAB,
              0, 0, ELF_STRTAB_OFF, sizeof(ELF_STRTAB) - 1,
              0, 0, 1, 0);
    emit_shdr(out + ELF_SHOFF + 64 * ELF_SH_SHSTRTAB,
              ELF_SHSTR_SHSTRTAB_OFF, ELF_SHT_STRTAB,
              0, 0, ELF_SHSTRTAB_OFF, sizeof(ELF_SHSTRTAB) - 1,
              0, 0, 1, 0);

    return VBS_RDNA2_ELF_SIZE;
}

/* --- dispatch setup --------------------------------------------------- */

void vbs_rdna2_vecadd_dispatch(uint64_t a, uint64_t b, uint64_t c,
                               uint32_t n,
                               vbs_rdna2_vecadd_dispatch_t *out) {
    out->num_thread_x = VBS_RDNA2_VECADD_NUM_THREAD_X;
    out->num_thread_y = VBS_RDNA2_VECADD_NUM_THREAD_Y;
    out->num_thread_z = VBS_RDNA2_VECADD_NUM_THREAD_Z;
    /* DIM_X counts THREAD GROUPS, not threads (digest §2.4). */
    out->dim_x = (n + VBS_RDNA2_VECADD_THREADS_PER_GROUP - 1u) /
                 VBS_RDNA2_VECADD_THREADS_PER_GROUP;
    out->dim_y = 1;
    out->dim_z = 1;
    out->initiator = VBS_RDNA2_VECADD_DISPATCH_INITIATOR;
    out->pgm_rsrc1 = VBS_RDNA2_VECADD_PGM_RSRC1;
    out->pgm_rsrc2 = VBS_RDNA2_VECADD_PGM_RSRC2;
    out->pgm_rsrc3 = VBS_RDNA2_VECADD_PGM_RSRC3;
    /* A 64-bit VA occupies two consecutive USER_DATA registers,
     * low dword first, and lands in an even-aligned SGPR pair
     * (digest §7.3). */
    out->user_data[0] = (uint32_t)(a & 0xFFFFFFFFu);
    out->user_data[1] = (uint32_t)(a >> 32);
    out->user_data[2] = (uint32_t)(b & 0xFFFFFFFFu);
    out->user_data[3] = (uint32_t)(b >> 32);
    out->user_data[4] = (uint32_t)(c & 0xFFFFFFFFu);
    out->user_data[5] = (uint32_t)(c >> 32);
}
