/* vbs_rdna2_vecadd.h — minimal RDNA 2 (gfx1030) vector-add kernel.
 *
 * Computes c[i] = a[i] + b[i] for i in [0, n), one u32 per
 * thread, using the ADDTID addressing forms: the per-thread
 * index comes free from the hardware (memory_Addr =
 * sgpr_addr(64) + inst_offset(12) + tid*4, [RDNA2-ISA] §12.8).
 * Hand-assembled from the documented ISA — no LLVM, no ROCm.
 *
 * Kernel shape (digest §5.4, followed exactly):
 *
 *   addr  instruction                          size
 *   ----  -----------                          ----
 *   0x00  global_load_dword_addtid  v0, s[0:1]  8  ; v0 = A[tid]
 *   0x08  global_load_dword_addtid  v1, s[2:3]  8  ; v1 = B[tid]
 *   0x10  s_waitcnt 0                           4  ; wait for loads
 *   0x14  v_add_nc_u32              v2, v0, v1  4  ; v2 = A + B
 *   0x18  global_store_dword_addtid v2, s[4:5]  8  ; C[tid] = v2
 *   0x20  s_waitcnt 0                           4  ; (see note)
 *   0x24  s_endpgm                              4  ; (waits first)
 *   total: 7 instructions, 40 bytes.
 *
 * Note on the second s_waitcnt: VMCNT counts vector-memory
 * LOADS only ([RDNA2-ISA] §4.4); stores are tracked by the
 * separate VS_CNT counter, so the second waitcnt (vmcnt 0) is
 * already satisfied when it issues — it is the digest's sketch
 * kept verbatim, and the store's completion before wave
 * retirement is the hardware's job at s_endpgm (which
 * implicitly waits first). If the hardware run shows stores
 * racing wave exit, the fix is S_WAITCNT_VSCNT (SOPP op 24),
 * not a second vmcnt wait.
 *
 * BOUNDS: this kernel has NO bounds check — every dispatched
 * thread reads A[tid]/B[tid] and writes C[tid]. The driver
 * must dispatch ceil(n/64) groups of 64 and allocate the
 * buffers for ceil(n/64)*64 elements (padded), or n must be a
 * multiple of 64. A bounds-checked variant (tid compare +
 * exec-mask) is future work.
 *
 * Registers: SGPRs s0-s5 hold the three buffer VAs, loaded by
 * the driver from COMPUTE_USER_DATA_0..5 (USER_SGPR = 6,
 * digest §7.3 / [SI-PG] §1.3.2.1). VGPRs v0-v2 hold the two
 * operands and the sum. No LDS, no scratch, wave64.
 *
 * Binary layout — two views of the same kernel:
 *   (a) RAW CODE: 40 bytes, what the PM4-direct driver uploads
 *       and points COMPUTE_PGM_LO/HI at (digest §6: "The
 *       hardware itself needs only raw code + the PGM_RSRC
 *       registers").
 *   (b) ELF CODE OBJECT (v5, ET_DYN): for ABI compatibility
 *       with standard tooling. Layout (all offsets are file
 *       offsets; vaddr == file offset):
 *         0x000  ELF header (64 bytes)
 *         0x040  2 program headers (PT_LOAD .rodata, PT_LOAD .text)
 *         0x100  .rodata: kernel descriptor (64 bytes, 64-aligned)
 *         0x200  .text: kernel code (40 bytes, 256-aligned entry)
 *         0x228  .strtab, .symtab, .shstrtab, section headers
 *       KERNEL_CODE_ENTRY_BYTE_OFFSET = 0x200 - 0x100 = 256,
 *       matching the documented default of 256
 *       ([AMDGPUUsage] ".rdna2_kernel_code_t" defaults).
 *       Kernel symbols follow [AMDGPUUsage] "Symbols": the
 *       entry point is `vbs_rdna2_vecadd`, STT_FUNC, in .text;
 *       the descriptor is `vbs_rdna2_vecadd.kd`, STT_OBJECT,
 *       in .rodata. There is deliberately NO .note metadata
 *       section in v1: the metadata note is extensible and
 *       optional ("only generated if needed", [AMDGPUUsage]
 *       "AMDGPU ELF Sections"), and a wrong MessagePack blob
 *       would be worse than none. Add it if a loader requires it.
 */

#ifndef VBS_RDNA2_VECADD_H
#define VBS_RDNA2_VECADD_H

#include <stddef.h>
#include <stdint.h>

#include "vbs_rdna2_asm.h"

/* Kernel shape. */
#define VBS_RDNA2_VECADD_NUM_INSTS 7u
#define VBS_RDNA2_VECADD_CODE_SIZE 40u  /* 7 insts: 8+8+4+4+8+4+4 */

/* Register map. */
#define VBS_RDNA2_VECADD_S_A_LO 0u  /* s[0:1] A base VA (USER_DATA_0/1) */
#define VBS_RDNA2_VECADD_S_B_LO 2u  /* s[2:3] B base VA (USER_DATA_2/3) */
#define VBS_RDNA2_VECADD_S_C_LO 4u  /* s[4:5] C base VA (USER_DATA_4/5) */
#define VBS_RDNA2_VECADD_S_COUNT 6u /* USER_SGPR = 6 */
#define VBS_RDNA2_VECADD_V_A    0u  /* v0: A[tid] */
#define VBS_RDNA2_VECADD_V_B    1u  /* v1: B[tid] */
#define VBS_RDNA2_VECADD_V_SUM  2u  /* v2: A[tid] + B[tid] */
#define VBS_RDNA2_VECADD_V_COUNT 3u /* VGPRs used: v0-v2 */

/* Dispatch shape (digest §5.4). */
#define VBS_RDNA2_VECADD_THREADS_PER_GROUP 64u
#define VBS_RDNA2_VECADD_NUM_THREAD_X 64u
#define VBS_RDNA2_VECADD_NUM_THREAD_Y 1u
#define VBS_RDNA2_VECADD_NUM_THREAD_Z 1u
#define VBS_RDNA2_VECADD_DISPATCH_INITIATOR 0x1u  /* COMPUTE_SHADER_EN */

/* PGM_RSRC values for this kernel (digest §4.4).
 * PGM_RSRC1: VGPRS field = max(0, ceil(3/4)-1) = 0 (wave64);
 *   SGPRS field reserved-must-be-0 on GFX10-12; all else 0.
 * PGM_RSRC2: USER_SGPR = 6 at bits [5:1]; SCRATCH_EN = 0,
 *   LDS_SIZE = 0, no TGID/TG_SIZE system SGPRs.
 * PGM_RSRC3: SHARED_VGPR_CNT = 0. */
#define VBS_RDNA2_VECADD_PGM_RSRC1 0x00000000u
#define VBS_RDNA2_VECADD_PGM_RSRC2 0x0000000Cu  /* USER_SGPR 6 << 1 */
#define VBS_RDNA2_VECADD_PGM_RSRC3 0x00000000u

/* Kernel descriptor field offsets (64-byte descriptor,
 * [AMDGPUUsage] "Code Object V3 Kernel Descriptor"). */
#define VBS_RDNA2_DESC_SIZE 64u
#define VBS_RDNA2_DESC_GROUP_SEGMENT_FIXED_SIZE 0x00u
#define VBS_RDNA2_DESC_PRIVATE_SEGMENT_FIXED_SIZE 0x04u
#define VBS_RDNA2_DESC_KERNARG_SIZE 0x08u
#define VBS_RDNA2_DESC_CODE_ENTRY_BYTE_OFFSET 0x10u
#define VBS_RDNA2_DESC_PGM_RSRC3 0x2Cu
#define VBS_RDNA2_DESC_PGM_RSRC1 0x30u
#define VBS_RDNA2_DESC_PGM_RSRC2 0x34u
#define VBS_RDNA2_DESC_CODE_PROPERTIES 0x38u
/* KERNEL_CODE_ENTRY_BYTE_OFFSET for this kernel: the code entry
 * sits 256 bytes past the descriptor base (see the ELF layout
 * above). The entry must be 256-byte aligned [AMDGPUUsage]. */
#define VBS_RDNA2_VECADD_ENTRY_OFFSET 256u

/* ELF code-object layout (v5, ET_DYN). Values from [AMDGPUUsage]
 * "AMDGPU ELF Header": EM_AMDGPU 224, ELFOSABI_AMDGPU_HSA 64,
 * ELFABIVERSION_AMDGPU_HSA_V5 3, EF_AMDGPU_MACH_AMDGCN_GFX1030
 * 0x036. e_type ET_DYN: "The AMD HSA runtime loader requires
 * a ET_DYN code object." */
#define VBS_RDNA2_ELF_MACHINE 224u
#define VBS_RDNA2_ELF_OSABI_HSA 64u
#define VBS_RDNA2_ELF_ABIVERSION_V5 3u
#define VBS_RDNA2_ELF_FLAGS_GFX1030 0x036u
#define VBS_RDNA2_ELF_RODATA_OFF 0x100u  /* descriptor file offset */
#define VBS_RDNA2_ELF_TEXT_OFF 0x200u    /* code file offset */
#define VBS_RDNA2_ELF_TEXT_VADDR 0x200u  /* vaddr == file offset */

/* Total size of the emitted ELF code object (header + phdrs +
 * .rodata + .text + .strtab + .symtab + .shstrtab + shdrs). */
#define VBS_RDNA2_ELF_SIZE 0x448u  /* 1096 bytes */

/* Kernel symbol names. */
#define VBS_RDNA2_VECADD_SYM "vbs_rdna2_vecadd"
#define VBS_RDNA2_VECADD_SYM_KD "vbs_rdna2_vecadd.kd"

/* --- PGM_RSRC field helpers (digest §4.4) --------------------------- */

/* VGPRS field of COMPUTE_PGM_RSRC1, GFX10-12 [AMDGPUUsage]:
 * wave64 -> max(0, ceil(vgprs/4) - 1); wave32 ->
 * max(0, ceil(vgprs/8) - 1). wave32: pass wave64 = 0. */
static inline uint32_t vbs_rdna2_vgprs_field(unsigned vgprs_used,
                                             int wave64) {
    unsigned gran = wave64 ? 4u : 8u;
    unsigned groups = (vgprs_used + gran - 1u) / gran;
    return groups == 0 ? 0u : groups - 1u;
}

/* LDS_SIZE field of COMPUTE_PGM_RSRC2 (GFX7-12) [AMDGPUUsage]:
 * roundup(lds_bytes / 512). */
static inline uint32_t vbs_rdna2_lds_size_field(uint32_t lds_bytes) {
    return (lds_bytes + 511u) / 512u;
}

/* --- Kernel build ---------------------------------------------------- */

/* Assemble the kernel into out[0..6]. capacity must be >= 7.
 * Returns the instruction count (7) or 0 on error (NULL out or
 * capacity too small). */
size_t vbs_rdna2_vecadd_build(vbs_rdna2_inst_t *out,
                              size_t capacity);

/* Serialize the assembled kernel to raw code bytes (40 bytes).
 * This is what the PM4-direct driver uploads; COMPUTE_PGM_LO/HI
 * points at its start. */
void vbs_rdna2_vecadd_serialize(
        uint8_t out[VBS_RDNA2_VECADD_CODE_SIZE]);

/* --- Kernel descriptor ------------------------------------------------ */

/* Fill a 64-byte kernel descriptor for this kernel (all fields
 * per [AMDGPUUsage] "Code Object V3 Kernel Descriptor").
 * GROUP_SEGMENT_FIXED_SIZE = 0 (no LDS),
 * PRIVATE_SEGMENT_FIXED_SIZE = 0 (no scratch), KERNARG_SIZE = 0
 * (arguments arrive via USER_DATA, not a kernarg buffer —
 * digest §6.3, driver-controlled layout), KERNEL_CODE_PROPERTIES
 * = 0 (wave64, no descriptor-driven SGPR setup). */
void vbs_rdna2_vecadd_descriptor(uint8_t out[VBS_RDNA2_DESC_SIZE]);

/* --- ELF code object -------------------------------------------------- */

/* Emit the complete ELF code object (VBS_RDNA2_ELF_SIZE bytes)
 * into out. capacity must be >= VBS_RDNA2_ELF_SIZE. Returns the
 * size written (VBS_RDNA2_ELF_SIZE) or 0 on error (NULL out or
 * capacity too small). */
size_t vbs_rdna2_vecadd_elf(uint8_t *out, size_t capacity);

/* --- Dispatch setup --------------------------------------------------- */

/* Everything the driver needs to dispatch this kernel, in one
 * struct. The driver worker copies these values into its PM4
 * packets (SET_SH_REG for the PGM/RSRC/USER_DATA registers,
 * DISPATCH_DIRECT for the dims — digest §2.4, §4.1, App. A). */
typedef struct {
    uint32_t num_thread_x;  /* COMPUTE_NUM_THREAD_X = 64 */
    uint32_t num_thread_y;  /* COMPUTE_NUM_THREAD_Y = 1 */
    uint32_t num_thread_z;  /* COMPUTE_NUM_THREAD_Z = 1 */
    uint32_t dim_x;         /* DISPATCH_DIRECT DIM_X = ceil(n/64) */
    uint32_t dim_y;         /* DISPATCH_DIRECT DIM_Y = 1 */
    uint32_t dim_z;         /* DISPATCH_DIRECT DIM_Z = 1 */
    uint32_t initiator;     /* DISPATCH_INITIATOR = 0x1 */
    uint32_t pgm_rsrc1;     /* COMPUTE_PGM_RSRC1 */
    uint32_t pgm_rsrc2;     /* COMPUTE_PGM_RSRC2 (USER_SGPR = 6) */
    uint32_t pgm_rsrc3;     /* COMPUTE_PGM_RSRC3 */
    uint32_t user_data[6];  /* COMPUTE_USER_DATA_0..5:
                               A lo, A hi, B lo, B hi, C lo, C hi */
} vbs_rdna2_vecadd_dispatch_t;

/* Fill a dispatch setup for element count n and buffer GPU VAs
 * a, b, c. dim_x = ceil(n/64). The caller must ensure the
 * buffers hold dim_x*64 elements (see the BOUNDS note above). */
void vbs_rdna2_vecadd_dispatch(uint64_t a, uint64_t b, uint64_t c,
                               uint32_t n,
                               vbs_rdna2_vecadd_dispatch_t *out);

#endif /* VBS_RDNA2_VECADD_H */
