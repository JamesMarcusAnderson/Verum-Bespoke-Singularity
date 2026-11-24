/* vbs_turing_vecadd.h — minimal Turing (sm_75) vector-add kernel.
 *
 * Computes c[i] = a[i] + b[i] for i in [0, n), one float per
 * thread: thread idx = ctaid.x * ntid.x + tid.x, bounds-checked
 * against n (threads with idx >= n exit early). Hand-assembled
 * from the documented SASS encodings — no ptxas, no cubin
 * container: the driver uploads these raw bytes and points the
 * QMD PROGRAM_ADDRESS at them (digest §5.5).
 *
 * Binary layout (digest §5.5): NO shader program header — the
 * published cuobjdump samples for sm_70/sm_100 show the first
 * instruction at offset 0x0000, and turingas places raw SASS in
 * .text with no SPH. The code is 18 instructions x 16 bytes =
 * 288 bytes, 16-byte aligned, first instruction at offset 0.
 * The SPH question is digest §10 item 3; the hardware run must
 * confirm a raw-SASS launch works (see REPORT).
 *
 * Kernel arguments arrive in constant bank c[0] (digest §6.3):
 * the driver writes the parameter struct below into a GPU
 * buffer and binds it as CB0 in the QMD. The layout is OURS
 * (defined here, not ptxas's): the driver and the kernel agree
 * on it by sharing this header. blockDim.x (ntid.x) is passed
 * in c[0] because the S2R encoding for SR_NTID.X is UNSOURCED
 * (see the opcode table).
 *
 * Registers: R0-R13 (14 total). QMD REGISTER_COUNT = 14.
 */

#ifndef VBS_TURING_VECADD_H
#define VBS_TURING_VECADD_H

#include <stddef.h>
#include <stdint.h>

#include "vbs_turing_sass.h"

/* Kernel shape. */
#define VBS_TURING_VECADD_NUM_INSTS 18u
#define VBS_TURING_VECADD_NUM_REGS  14u
#define VBS_TURING_VECADD_CODE_SIZE \
    (VBS_TURING_VECADD_NUM_INSTS * 16u)  /* 288 bytes */

/* Constant-buffer (c[0]) parameter layout — byte offsets. */
#define VBS_TURING_VECADD_CB_A_OFF    0x00u  /* u64: a pointer */
#define VBS_TURING_VECADD_CB_B_OFF    0x08u  /* u64: b pointer */
#define VBS_TURING_VECADD_CB_C_OFF    0x10u  /* u64: c pointer */
#define VBS_TURING_VECADD_CB_N_OFF    0x18u  /* u32: element count */
#define VBS_TURING_VECADD_CB_NTID_OFF 0x1cu  /* u32: blockDim.x */
/* Bytes the parameter struct occupies; the CB binding itself is
 * 256 bytes (constant spaces bind in multiples of 256 bytes,
 * digest §5.2 / S6 Fermi ISA "Memory"). */
#define VBS_TURING_VECADD_PARAM_SIZE 32u
#define VBS_TURING_VECADD_CB_SIZE    256u

/* Parameter struct the driver writes into CB0. Field offsets
 * must match the CB_*_OFF defines above (checked by tests). */
typedef struct {
    uint64_t a;     /* offset 0x00 */
    uint64_t b;     /* offset 0x08 */
    uint64_t c;     /* offset 0x10 */
    uint32_t n;     /* offset 0x18 */
    uint32_t ntid;  /* offset 0x1c: blockDim.x */
} vbs_turing_vecadd_params_t;

/* Register map (see the .c file for the instruction list). */
#define VBS_TURING_VECADD_R_A_LO     0u   /* R0:R1  a ptr -> a addr */
#define VBS_TURING_VECADD_R_B_LO     2u   /* R2:R3  b ptr -> b addr */
#define VBS_TURING_VECADD_R_C_LO     4u   /* R4:R5  c ptr -> c addr */
#define VBS_TURING_VECADD_R_TID      6u   /* R6     tid.x */
#define VBS_TURING_VECADD_R_CTAID    7u   /* R7     ctaid.x */
#define VBS_TURING_VECADD_R_NTID     8u   /* R8     ntid.x */
#define VBS_TURING_VECADD_R_IDX      9u   /* R9     global index */
#define VBS_TURING_VECADD_R_N        10u  /* R10    n */
#define VBS_TURING_VECADD_R_AVAL     11u  /* R11    a[i] */
#define VBS_TURING_VECADD_R_BVAL     12u  /* R12    b[i] */
#define VBS_TURING_VECADD_R_SUM      13u  /* R13    a[i] + b[i] */

/* Assemble the kernel into out[0..17]. capacity must be >= 18.
 * Returns the instruction count (18) or 0 on error (NULL out
 * or capacity too small). */
size_t vbs_turing_vecadd_build(vbs_turing_inst_t *out,
                               size_t capacity);

/* Serialize the assembled kernel to raw code bytes (288 bytes,
 * 16-byte aligned start, first instruction at offset 0). */
void vbs_turing_vecadd_serialize(uint8_t out[VBS_TURING_VECADD_CODE_SIZE]);

#endif /* VBS_TURING_VECADD_H */
