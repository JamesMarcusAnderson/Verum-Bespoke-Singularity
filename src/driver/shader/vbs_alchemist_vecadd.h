/* vbs_alchemist_vecadd.h — minimal Alchemist (Xe-HPG) vector-add kernel.
 *
 * Computes c[i] = a[i] + b[i] for F32 arrays, one element per SIMD
 * lane: each SIMD16 thread handles 16 consecutive elements, indexed
 * by its hardware local ID. Hand-assembled from the documented EU
 * ISA encodings — no compiler, no binary container: the driver
 * uploads these raw bytes at a 64-byte-aligned offset and points
 * the INTERFACE_DESCRIPTOR_DATA Kernel Start Pointer at them
 * (digest §6.10).
 *
 * Binary layout (digest §6.10): NO kernel header — the hardware
 * needs only the raw instruction bytes; all launch metadata lives
 * in the COMPUTE_WALKER + interface descriptor. The code is 21
 * instructions x 16 bytes = 336 bytes, 64-byte aligned, first
 * instruction at offset 0.
 *
 * How arguments reach the kernel (digest §5.5, §7.4):
 *   - R0: GPGPU_R0 thread payload (hardware-filled).
 *   - R1: local X IDs, U16 per lane (Emit Local X enabled).
 *   - R2: inline data GRF — the walker's 8 inline dwords (Emit
 *     Inline Parameter enabled): A_lo, A_hi, B_lo, B_hi, C_lo,
 *     C_hi, spare, spare. The three 64-bit buffer base addresses.
 *   - No binding table: all memory traffic is A64 stateless
 *     (BTI 255), so the interface descriptor's Binding Table
 *     Pointer/Count are 0.
 *   - No SLM, no barriers, no scratch (no calls, no spills).
 *
 * Register map (GRFs):
 *   r0        R0 payload (hardware)              r17-r20 B addresses
 *   r1        local X IDs (U16 x16)              r21-r24 C addresses
 *   r2        inline data (bases)                r25-r26 A data (F32)
 *   r3-r4     byte offsets (UD x16)              r27-r28 B data (F32)
 *   r5-r8     offsets as UQ (temp)               r29-r30 C data (F32)
 *   r9-r12    broadcast base UQ (temp, reused)   r112    EOT send src
 * Highest GRF touched: r112 (EOT source; EOT sends must use
 * r112-r127 — Vol 2a, p.1255). Working set r0-r30 (31 GRFs).
 *
 * Why 21 instructions and not 14: the digest §6.9 sketch does the
 * 64-bit address math in SIMD16, but Vol 9, p.387 forbids it —
 * ExecSize * (largest element size) must be <= 64, so a SIMD16
 * :uq add/mov (16 * 8 = 128) is ILLEGAL. All :uq work is split
 * into two SIMD8 halves (lanes 0-7, lanes 8-15). The :uq add
 * itself is DERIVED (see VBS_ALCHEMIST_ADD_UQ_NOTE) and must be
 * confirmed on the hardware run.
 *
 * Write ordering: the kernel issues NO memory fence. The walker's
 * PostSync (Dataport Pipeline Flush + Subslice Cache Flush,
 * digest §5.4) and/or a trailing PIPE_CONTROL (HDC flush +
 * Untyped L1 flush, digest §4.7) order the stores before the
 * driver reads the results — the EOT message's Fence Data Ports
 * field is marked "Ignored" (Vol 2a, p.562), so EOT must NOT be
 * relied on to fence (digest §6.9 step 6).
 */

#ifndef VBS_ALCHEMIST_VECADD_H
#define VBS_ALCHEMIST_VECADD_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "vbs_alchemist_eu.h"

/* Kernel shape. */
#define VBS_ALCHEMIST_VECADD_NUM_INSTS 21u
#define VBS_ALCHEMIST_VECADD_CODE_SIZE \
    (VBS_ALCHEMIST_VECADD_NUM_INSTS * 16u)  /* 336 bytes */
#define VBS_ALCHEMIST_VECADD_ALIGN 64u      /* kernel start alignment */

/* Register map (see the header comment). */
#define VBS_ALCHEMIST_VECADD_R_R0       0u
#define VBS_ALCHEMIST_VECADD_R_LOCALID  1u
#define VBS_ALCHEMIST_VECADD_R_INLINE   2u
#define VBS_ALCHEMIST_VECADD_R_OFF      3u  /* 2 GRFs: r3-r4  */
#define VBS_ALCHEMIST_VECADD_R_OFF_UQ   5u  /* 4 GRFs: r5-r8  */
#define VBS_ALCHEMIST_VECADD_R_BASE     9u  /* 4 GRFs: r9-r12 */
#define VBS_ALCHEMIST_VECADD_R_ADDR_A   13u /* 4 GRFs: r13-r16 */
#define VBS_ALCHEMIST_VECADD_R_ADDR_B   17u /* 4 GRFs: r17-r20 */
#define VBS_ALCHEMIST_VECADD_R_ADDR_C   21u /* 4 GRFs: r21-r24 */
#define VBS_ALCHEMIST_VECADD_R_DATA_A   25u /* 2 GRFs: r25-r26 */
#define VBS_ALCHEMIST_VECADD_R_DATA_B   27u /* 2 GRFs: r27-r28 */
#define VBS_ALCHEMIST_VECADD_R_DATA_C   29u /* 2 GRFs: r29-r30 */
#define VBS_ALCHEMIST_VECADD_R_EOT      112u
#define VBS_ALCHEMIST_VECADD_MAX_REG    112u

/* Inline-data dword indexes (walker DW31-38 -> R2 dwords 0-7). */
#define VBS_ALCHEMIST_VECADD_INL_A_LO 0u
#define VBS_ALCHEMIST_VECADD_INL_A_HI 1u
#define VBS_ALCHEMIST_VECADD_INL_B_LO 2u
#define VBS_ALCHEMIST_VECADD_INL_B_HI 3u
#define VBS_ALCHEMIST_VECADD_INL_C_LO 4u
#define VBS_ALCHEMIST_VECADD_INL_C_HI 5u
#define VBS_ALCHEMIST_VECADD_INL_SPARE0 6u
#define VBS_ALCHEMIST_VECADD_INL_SPARE1 7u

/* Inline parameter block the driver writes into the walker's
 * Inline Data (DW31-38). Field offsets must match the INL_*
 * defines above (checked by tests). */
typedef struct {
    uint64_t a;       /* offset 0x00: base address of A */
    uint64_t b;       /* offset 0x08: base address of B */
    uint64_t c;       /* offset 0x10: base address of C */
    uint32_t spare0;  /* offset 0x18: reserved, 0 */
    uint32_t spare1;  /* offset 0x1c: reserved, 0 */
} vbs_alchemist_vecadd_params_t;
#define VBS_ALCHEMIST_VECADD_PARAM_SIZE 32u  /* 8 dwords */

/* Assemble the kernel into an array of 21 instructions.
 * Returns 0 on success, -1 if any emitter refused (never emits a
 * partial kernel). */
int vbs_alchemist_vecadd_insts(vbs_alchemist_inst_t out[VBS_ALCHEMIST_VECADD_NUM_INSTS]);

/* Assemble the kernel into a flat 336-byte buffer (DW0-first,
 * little-endian per dword). The driver places the buffer at a
 * 64-byte-aligned GPU offset. Returns 0 on success. */
int vbs_alchemist_vecadd_build(uint8_t out[VBS_ALCHEMIST_VECADD_CODE_SIZE]);

/* Print the disassembly-style listing (one line per instruction:
 * address, 128-bit encoding, mnemonic, operands, spec citation)
 * to the given stream. */
void vbs_alchemist_vecadd_list(FILE *out);

/* --- Launch descriptors: what the driver programs around the kernel. --- */

/* COMPUTE_WALKER DW4 for this kernel (digest §4.8; Vol 2a, p.488):
 * SIMD Size [31:30] = 1 (SIMD16); Generate Local ID [29] = 1 (must
 * be set or Emit Local bits are ignored — Vol 2a, p.488); Emit
 * Local [28:26] = X only (bit 26); Emit Inline Parameter [25] = 1;
 * Message SIMD [18:17] = 1 (SIMD16; Vol 2a, p.490: 0 = SIMD8,
 * 1 = SIMD16, 2 = SIMD32; must be <= thread SIMD size).
 * Value: 0x66020000. */
#define VBS_ALCHEMIST_VECADD_WALKER_DW4 0x66020000u

/* COMPUTE_WALKER DW6 (Local Maxima) for a group of
 * threads_per_group SIMD16 threads: Local X Max = T*16 - 1, Y/Z
 * Max = 0 (their Emit bits are clear, so DW6 Y/Z must be 0 —
 * Vol 2a, p.489). threads_per_group should be a power of two so
 * the Local X count (T*16) is a power of two (Generate Local ID
 * restriction — Vol 2a, p.488) and 1 <= T <= 64 (LWS <= 1024 and
 * the interface-descriptor thread-group maximum — digest §4.8,
 * §5.1). Returns 0 on a bad count. */
uint32_t vbs_alchemist_vecadd_walker_dw6(uint32_t threads_per_group);

/* Fill an 8-dword INTERFACE_DESCRIPTOR_DATA (digest §5.1; Vol 2d,
 * pp.714-719) for this kernel:
 *   DW0-1: Kernel Start Pointer = kernel_start_offset (byte offset
 *     from the Instruction Base Address; 64-byte aligned — the low
 *     6 bits are masked off).
 *   DW2: 0 (IEEE FP mode, denorms flush-to-zero, no exceptions).
 *   DW3: 0 (no samplers).
 *   DW4: 0 (no binding table — A64 stateless, BTI 255).
 *   DW5: Number of Threads in GPGPU Thread Group [9:0] =
 *     threads_per_group; SLM size 0, barriers 0, TG dispatch size
 *     8 (default), rounding RTNE (default).
 *   DW6-7: 0.
 * Returns 0 on success, -1 if threads_per_group is out of [1,64]. */
int vbs_alchemist_vecadd_idd(uint32_t idd[8], uint64_t kernel_start_offset,
                             uint32_t threads_per_group);

#endif /* VBS_ALCHEMIST_VECADD_H */
