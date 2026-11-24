/* vbs_drv_turing_enc.c — Turing QMD v2.3 builder + pushbuffer encoder.
 *
 * PURE C11, no ioctls, no platform headers: every function here is
 * testable on a machine with no GPU. All hardware facts come from
 * the project spec digest (SPEC-DIGEST.md) and are cited per fact
 * as digest section + source tag.
 *
 * Originality: written from the digest's documented facts (field
 * positions, method offsets, command formats). No code was copied
 * from any NVIDIA, Nouveau, Mesa, or other implementation.
 */

#include "vbs_drv_turing.h"

#include <string.h>

/* ------------------------------------------------------------------ */
/* Small field helpers                                                 */
/* ------------------------------------------------------------------ */

/* Set bits [hi:lo] of word w to value v (v must fit the field). */
static void set_field(uint32_t *w, unsigned lo, unsigned hi, uint32_t v) {
    uint32_t mask = (hi - lo + 1) >= 32
        ? 0xffffffffu
        : (((1u << (hi - lo + 1)) - 1u) << lo);
    *w = (*w & ~mask) | ((v << lo) & mask);
}

/* ------------------------------------------------------------------ */
/* QMD v2.3 — digest §4, S1 clc5c0qmd.h                                */
/*                                                                     */
/* Field notation in the header is MW(high:low) = absolute bit range  */
/* in the 2048-bit structure; word index = bit / 32.                   */
/* ------------------------------------------------------------------ */

void vbs_drv_turing_qmd_zero(vbs_drv_turing_qmd_t *qmd) {
    if (!qmd) return;
    memset(qmd, 0, sizeof(*qmd));
    /* QMD_VERSION MW(579:576) = word 18 bits 3:0, value 3;      */
    /* QMD_MAJOR_VERSION MW(583:580) = word 18 bits 7:4, value 2 */
    /* (digest §4; S1 clc5c0qmd.h).                               */
    set_field(&qmd->w[18], 0, 3, VBS_DRV_TURING_QMD_VERSION);
    set_field(&qmd->w[18], 4, 7, VBS_DRV_TURING_QMD_MAJOR_VERSION);
    /* SASS_VERSION MW(1023:1016) = word 31 bits 31:24, value    */
    /* 0x75 for Turing (digest §4; S1 + S12).                    */
    set_field(&qmd->w[31], 24, 31, VBS_DRV_TURING_SASS_VERSION);
    /* SM_GLOBAL_CACHING_ENABLE, word 4 bit 6 (digest §4.1):     */
    /* enable SM global caching for the grid.                     */
    set_field(&qmd->w[4], 6, 6, 1);
    /* IS_QUEUE, word 4 bit 8, stays 0: plain grid launch, not    */
    /* queue mode (digest §4.1).                                  */
}

vbs_drv_status_t vbs_drv_turing_qmd_set_program(vbs_drv_turing_qmd_t *qmd,
                                                uint64_t code_va,
                                                uint32_t code_size) {
    if (!qmd || code_va == 0 || code_size == 0)
        return VBS_DRV_ERR_INVALID_ARG;
    if (code_va >= (1ULL << VBS_DRV_TURING_VA_BITS))
        return VBS_DRV_ERR_INVALID_ARG; /* beyond the 47-bit GPU MMU */
    /* PROGRAM_ADDRESS_LOWER word 48; PROGRAM_ADDRESS_UPPER word 49 */
    /* bits 16:0 (digest §4.1).                                    */
    qmd->w[48] = (uint32_t)(code_va & 0xffffffffu);
    set_field(&qmd->w[49], 0, 16, (uint32_t)(code_va >> 32) & 0x1ffffu);
    /* PROGRAM_PREFETCH_ADDR_LOWER_SHIFTED word 8 = code VA >> 8   */
    /* (low part); PROGRAM_PREFETCH_ADDR_UPPER_SHIFTED word 31     */
    /* bits 8:0 = code VA >> 8 (high part) (digest §4.1).          */
    uint64_t shifted = code_va >> 8;
    qmd->w[8] = (uint32_t)(shifted & 0xffffffffu);
    set_field(&qmd->w[31], 0, 8, (uint32_t)(shifted >> 32) & 0x1ffu);
    /* PROGRAM_PREFETCH_SIZE, word 31 bits 17:9: the digest does   */
    /* not state the unit (bytes? 32-byte units?), so the builder  */
    /* leaves it 0 = no prefetch rather than invent a value. The   */
    /* code still runs; it just does not prefetch. See the driver  */
    /* report (unverified item). code_size is validated above but  */
    /* deliberately unused here.                                   */
    (void)code_size;
    return VBS_DRV_OK;
}

vbs_drv_status_t vbs_drv_turing_qmd_set_grid(vbs_drv_turing_qmd_t *qmd,
                                             uint32_t gx, uint32_t gy,
                                             uint32_t gz) {
    if (!qmd || gx == 0 || gy == 0 || gz == 0)
        return VBS_DRV_ERR_INVALID_ARG;
    /* CTA_RASTER_WIDTH word 12; CTA_RASTER_HEIGHT word 13 bits    */
    /* 15:0; CTA_RASTER_DEPTH word 14 bits 15:0 (digest §4.1).     */
    if (gy > 0xffffu || gz > 0xffffu) return VBS_DRV_ERR_INVALID_ARG;
    qmd->w[12] = gx;
    set_field(&qmd->w[13], 0, 15, gy);
    set_field(&qmd->w[14], 0, 15, gz);
    return VBS_DRV_OK;
}

vbs_drv_status_t vbs_drv_turing_qmd_set_block(vbs_drv_turing_qmd_t *qmd,
                                              uint32_t bx, uint32_t by,
                                              uint32_t bz) {
    if (!qmd || bx == 0 || by == 0 || bz == 0)
        return VBS_DRV_ERR_INVALID_ARG;
    /* CTA_THREAD_DIMENSION0 word 18 bits 31:16; DIMENSION1 word   */
    /* 19 bits 15:0; DIMENSION2 word 19 bits 31:16 (digest §4.1).  */
    if (bx > 0xffffu || by > 0xffffu || bz > 0xffffu)
        return VBS_DRV_ERR_INVALID_ARG;
    set_field(&qmd->w[18], 16, 31, bx);
    set_field(&qmd->w[19], 0, 15, by);
    set_field(&qmd->w[19], 16, 31, bz);
    return VBS_DRV_OK;
}

vbs_drv_status_t vbs_drv_turing_qmd_set_resources(vbs_drv_turing_qmd_t *qmd,
                                                  uint32_t regs,
                                                  uint32_t shared_bytes,
                                                  uint32_t barriers) {
    if (!qmd || regs == 0 || regs > 255)
        return VBS_DRV_ERR_INVALID_ARG;
    /* SHARED_MEMORY_SIZE word 17 bits 17:0 (digest §4.1). */
    if (shared_bytes > 0x3ffffu) return VBS_DRV_ERR_INVALID_ARG;
    /* BARRIER_COUNT word 29 bits 31:27 (5 bits). */
    if (barriers > 31) return VBS_DRV_ERR_INVALID_ARG;
    /* REGISTER_COUNT word 30 bits 31:24; REGISTER_COUNT_V word 20 */
    /* bits 16:8, set consistently (digest §4.1).                  */
    set_field(&qmd->w[30], 24, 31, regs);
    set_field(&qmd->w[20], 8, 16, regs);
    set_field(&qmd->w[17], 0, 17, shared_bytes);
    set_field(&qmd->w[29], 27, 31, barriers);
    /* SHADER_LOCAL_MEMORY_LOW_SIZE word 29 bits 23:0 and HIGH     */
    /* word 30 bits 23:0 stay 0: v1 kernels use no per-thread      */
    /* local (spill) memory (digest §4.1).                         */
    return VBS_DRV_OK;
}

vbs_drv_status_t vbs_drv_turing_qmd_set_const_buf(vbs_drv_turing_qmd_t *qmd,
                                                  unsigned idx, uint64_t va,
                                                  uint32_t size_bytes) {
    if (!qmd || idx >= VBS_DRV_MAX_CONST_BUFFERS || va == 0 ||
        size_bytes == 0)
        return VBS_DRV_ERR_INVALID_ARG;
    /* CONSTANT_BUFFER_SIZE_SHIFTED4 is 13 bits of 16-byte units,  */
    /* so size must be a multiple of 16 and at most 128 KiB       */
    /* (digest §4.1).                                              */
    if ((size_bytes & 0xfu) != 0 || size_bytes > (128u * 1024u))
        return VBS_DRV_ERR_INVALID_ARG;
    if (va >= (1ULL << VBS_DRV_TURING_VA_BITS))
        return VBS_DRV_ERR_INVALID_ARG;
    /* CONSTANT_BUFFER_VALID(i) word 20 bit i (digest §4.1). */
    set_field(&qmd->w[20], idx, idx, 1);
    /* ADDR_LOWER(i) word 32+i*2; ADDR_UPPER(i) word 33+i*2 bits   */
    /* 16:0; SIZE_SHIFTED4(i) word 33+i*2 bits 31:19; INVALIDATE   */
    /* (i) word 33+i*2 bit 18 (digest §4.1). The invalidate bit is */
    /* set so the constant cache is refreshed for this launch.     */
    uint32_t lo_word = 32 + idx * 2;
    uint32_t hi_word = 33 + idx * 2;
    qmd->w[lo_word] = (uint32_t)(va & 0xffffffffu);
    set_field(&qmd->w[hi_word], 0, 16, (uint32_t)(va >> 32) & 0x1ffffu);
    set_field(&qmd->w[hi_word], 18, 18, 1);
    set_field(&qmd->w[hi_word], 19, 31, size_bytes >> 4);
    return VBS_DRV_OK;
}

vbs_drv_status_t vbs_drv_turing_qmd_set_release(vbs_drv_turing_qmd_t *qmd,
                                                unsigned which, uint64_t va,
                                                uint32_t payload) {
    if (!qmd || which > 1 || va == 0) return VBS_DRV_ERR_INVALID_ARG;
    if (va >= (1ULL << VBS_DRV_TURING_VA_BITS))
        return VBS_DRV_ERR_INVALID_ARG;
    /* RELEASE0 at words 23-25, RELEASE1 at words 26-28, same     */
    /* layout (digest §4.2): ADDR_LOWER word, ADDR_UPPER next      */
    /* word bits 7:0, PAYLOAD third word. Reduction is disabled    */
    /* (REDUCTION_ENABLE bit 26 = 0), so the op/format fields     */
    /* stay 0 and the hardware writes the plain payload.           */
    /* STRUCTURE_SIZE bit 31 (FOUR_WORDS vs ONE_WORD) is left 0:   */
    /* the digest does not say which value is which, so the        */
    /* builder takes the zero state; confirm on hardware.          */
    uint32_t base = (which == 0) ? 23 : 26;
    qmd->w[base] = (uint32_t)(va & 0xffffffffu);
    set_field(&qmd->w[base + 1], 0, 7, (uint32_t)(va >> 32) & 0xffu);
    qmd->w[base + 2] = payload;
    /* SEMAPHORE_RELEASE_ENABLE0/1, word 4 bits 10/11 (digest      */
    /* §4.2), gate the two release semaphores.                     */
    set_field(&qmd->w[4], 10 + which, 10 + which, 1);
    return VBS_DRV_OK;
}

void vbs_drv_turing_qmd_invalidate_caches(vbs_drv_turing_qmd_t *qmd) {
    if (!qmd) return;
    /* Invalidate bits, word 5 bits 26-31 (digest §4.1). The digest
     * lists the six fields in this order, and the builder assigns
     * them in that order to bits 26..31:
     *   26 INVALIDATE_TEXTURE_HEADER_CACHE
     *   27 INVALIDATE_TEXTURE_SAMPLER_CACHE
     *   28 INVALIDATE_TEXTURE_DATA_CACHE
     *   29 INVALIDATE_SHADER_DATA_CACHE
     *   30 INVALIDATE_INSTRUCTION_CACHE
     *   31 INVALIDATE_SHADER_CONSTANT_CACHE
     * That per-bit order is an inference from the digest's listing
     * order, not a separately sourced fact — confirm on hardware.
     * Only the instruction + constant-cache bits are set: those are
     * the two a compute launch needs after uploading new code.     */
    set_field(&qmd->w[5], 30, 30, 1); /* INVALIDATE_INSTRUCTION_CACHE */
    set_field(&qmd->w[5], 31, 31, 1); /* INVALIDATE_SHADER_CONSTANT_CACHE */
}

vbs_drv_status_t vbs_drv_turing_qmd_build(const vbs_drv_launch_desc_t *desc,
                                          vbs_drv_turing_qmd_t *qmd) {
    if (!qmd) return VBS_DRV_ERR_INVALID_ARG;
    /* Validate the description here as well as in the framework:
     * the encoder is directly testable and must not trust its
     * caller. (The framework performs the same checks on the
     * vbs_drv_launch path.) */
    vbs_drv_status_t vst = vbs_drv_validate_launch_desc(desc);
    if (vst != VBS_DRV_OK) return vst;
    vst = vbs_drv_validate_launch_semaphore(desc);
    if (vst != VBS_DRV_OK) return vst;
    vbs_drv_turing_qmd_zero(qmd);
    vbs_drv_status_t st;
    st = vbs_drv_turing_qmd_set_program(qmd,
                                        desc->code->gpu_va + desc->code_offset,
                                        desc->code_size);
    if (st != VBS_DRV_OK) return st;
    st = vbs_drv_turing_qmd_set_grid(qmd, desc->grid_x, desc->grid_y,
                                     desc->grid_z);
    if (st != VBS_DRV_OK) return st;
    st = vbs_drv_turing_qmd_set_block(qmd, desc->block_x, desc->block_y,
                                      desc->block_z);
    if (st != VBS_DRV_OK) return st;
    st = vbs_drv_turing_qmd_set_resources(qmd, desc->register_count,
                                          desc->shared_mem_bytes,
                                          desc->barrier_count);
    if (st != VBS_DRV_OK) return st;
    for (uint32_t i = 0; i < desc->const_buf_count; i++) {
        st = vbs_drv_turing_qmd_set_const_buf(qmd, i,
                                              desc->const_bufs[i]->gpu_va,
                                              (uint32_t)desc->const_bufs[i]->size);
        if (st != VBS_DRV_OK) return st;
    }
    if (desc->use_semaphore) {
        st = vbs_drv_turing_qmd_set_release(qmd, 0,
                                            desc->sem_buf->gpu_va + desc->sem_offset,
                                            desc->sem_payload);
        if (st != VBS_DRV_OK) return st;
    }
    /* First launch after code upload: invalidate the instruction   */
    /* and constant caches (digest §4.1 + §3.5 step 3). The caller  */
    /* reuses the same QMD buffer across launches; the invalidate   */
    /* bits are harmless on repeat launches.                        */
    vbs_drv_turing_qmd_invalidate_caches(qmd);
    /* CWD_MEMBAR_TYPE (word 11 bits 16:17) and RELEASE_MEMBAR_TYPE */
    /* (word 11 bit 14) stay 0 (FE_NONE / L1_NONE, digest §4.2):   */
    /* v1 relies on the EXEC syncobj wait for the CPU-side barrier, */
    /* not on QMD membars.                                          */
    /* CTA_RASTER_*_RESUME (words 6-7) and HW_ONLY_* (words 50-53) */
    /* stay 0: preemption/resume and hardware-internal use only    */
    /* (digest §4.2).                                               */
    return VBS_DRV_OK;
}

/* ------------------------------------------------------------------ */
/* Pushbuffer encoder — digest §3.3, S6 hw/fifo/dma-pusher.rst        */
/*                                                                     */
/* Old (NV4) format, fully documented in the digest:                  */
/*   increasing:     000CCCCCCCCCCC00SSSMMMMMMMMMMM00                 */
/*   non-increasing: 010CCCCCCCCCCC00SSSMMMMMMMMMMM00                 */
/* C = 11-bit count at bits 28-18, S = subchannel at bits 15-13,      */
/* M = 11-bit method at bits 12-2 (method in x4 address form, so     */
/* M = method >> 2). See the header for the digest's prose/bit-       */
/* string discrepancy on the count field (bits 29-19 vs 28-18): the  */
/* encoder follows the bit-string.                                    */
/* ------------------------------------------------------------------ */

#define VBS_DRV_TURING_PB_OP_INCR    0x00000000u /* bits 31-29 = 000 */
#define VBS_DRV_TURING_PB_OP_NONINCR 0x40000000u /* bits 31-29 = 010 */

static uint32_t pb_header(uint32_t op, uint32_t subchan, uint32_t method,
                          uint32_t count) {
    return op | ((count & 0x7ffu) << 18) | ((subchan & 0x7u) << 13) |
           ((method >> 2) & 0x7ffu);
}

uint32_t vbs_drv_turing_pb_hdr_incr(uint32_t subchan, uint32_t method,
                                    uint32_t count) {
    return pb_header(VBS_DRV_TURING_PB_OP_INCR, subchan, method, count);
}

uint32_t vbs_drv_turing_pb_hdr_nonincr(uint32_t subchan, uint32_t method,
                                       uint32_t count) {
    return pb_header(VBS_DRV_TURING_PB_OP_NONINCR, subchan, method, count);
}

static bool pb_args_ok(uint32_t subchan, uint32_t method, uint32_t count) {
    /* Subchannels are 0-7 (digest §3.2); methods are 0-0x3ffc in   */
    /* x4 form (digest §3.2); count is an 11-bit field.              */
    return subchan <= 7 && (method & 0x3u) == 0 && method <= 0x3ffcu &&
           count >= 1 && count <= 0x7ffu;
}

vbs_drv_status_t vbs_drv_turing_pb_emit_method1(vbs_drv_cmdbuf_t *cb,
                                                uint32_t subchan,
                                                uint32_t method,
                                                uint32_t value) {
    if (!cb || !pb_args_ok(subchan, method, 1))
        return VBS_DRV_ERR_INVALID_ARG;
    vbs_drv_status_t st = vbs_drv_cmdbuf_push(
        cb, vbs_drv_turing_pb_hdr_incr(subchan, method, 1));
    if (st != VBS_DRV_OK) return st;
    return vbs_drv_cmdbuf_push(cb, value);
}

vbs_drv_status_t vbs_drv_turing_pb_emit_methods(vbs_drv_cmdbuf_t *cb,
                                                uint32_t subchan,
                                                uint32_t first_method,
                                                const uint32_t *values,
                                                uint32_t count) {
    if (!cb || !values || !pb_args_ok(subchan, first_method, count))
        return VBS_DRV_ERR_INVALID_ARG;
    /* The run must not walk past the end of the method space. */
    if (first_method + (count - 1) * 4u > 0x3ffcu)
        return VBS_DRV_ERR_INVALID_ARG;
    vbs_drv_status_t st = vbs_drv_cmdbuf_push(
        cb, vbs_drv_turing_pb_hdr_incr(subchan, first_method, count));
    if (st != VBS_DRV_OK) return st;
    return vbs_drv_cmdbuf_push_many(cb, values, count);
}

vbs_drv_status_t vbs_drv_turing_pb_emit_nop(vbs_drv_cmdbuf_t *cb,
                                            uint32_t subchan) {
    /* NO_OPERATION 0x0100 (digest §3.5). Parameter value 0. */
    return vbs_drv_turing_pb_emit_method1(
        cb, subchan, VBS_DRV_TURING_MTHD_NO_OPERATION, 0);
}

vbs_drv_status_t vbs_drv_turing_pb_emit_wait_idle(vbs_drv_cmdbuf_t *cb,
                                                  uint32_t subchan) {
    /* WAIT_FOR_IDLE 0x0110 (digest §3.5). Parameter value 0. */
    return vbs_drv_turing_pb_emit_method1(
        cb, subchan, VBS_DRV_TURING_MTHD_WAIT_FOR_IDLE, 0);
}

vbs_drv_status_t vbs_drv_turing_pb_emit_invalidate_caches(vbs_drv_cmdbuf_t *cb,
                                                          uint32_t subchan) {
    /* INVALIDATE_SHADER_CACHES 0x021c (digest §3.5). Parameter 0. */
    return vbs_drv_turing_pb_emit_method1(
        cb, subchan, VBS_DRV_TURING_MTHD_INVALIDATE_SHADER_CACHES, 0);
}

vbs_drv_status_t vbs_drv_turing_pb_emit_inline_qmd(vbs_drv_cmdbuf_t *cb,
                                                   uint32_t subchan,
                                                   uint64_t qmd_va,
                                                   const vbs_drv_turing_qmd_t *qmd) {
    if (!cb || !qmd || qmd_va == 0) return VBS_DRV_ERR_INVALID_ARG;
    /* SET_INLINE_QMD_ADDRESS_A = upper 32 bits of the >>8-shifted   */
    /* address; SET_INLINE_QMD_ADDRESS_B = lower 32 bits (digest     */
    /* §3.5; S1).                                                     */
    uint64_t shifted = qmd_va >> 8;
    vbs_drv_status_t st = vbs_drv_turing_pb_emit_method1(
        cb, subchan, VBS_DRV_TURING_MTHD_SET_INLINE_QMD_ADDR_A,
        (uint32_t)(shifted >> 32));
    if (st != VBS_DRV_OK) return st;
    st = vbs_drv_turing_pb_emit_method1(
        cb, subchan, VBS_DRV_TURING_MTHD_SET_INLINE_QMD_ADDR_B,
        (uint32_t)(shifted & 0xffffffffu));
    if (st != VBS_DRV_OK) return st;
    /* LOAD_INLINE_QMD_DATA(i) at 0x0320 + i*4 (digest §3.5): 64    */
    /* consecutive methods, so one increasing command streams all   */
    /* 64 QMD words.                                                 */
    return vbs_drv_turing_pb_emit_methods(
        cb, subchan, VBS_DRV_TURING_MTHD_LOAD_INLINE_QMD_DATA_BASE,
        qmd->w, VBS_DRV_TURING_QMD_WORDS);
}

vbs_drv_status_t vbs_drv_turing_pb_emit_launch(vbs_drv_cmdbuf_t *cb,
                                               uint32_t subchan,
                                               uint64_t qmd_va) {
    if (!cb || qmd_va == 0 || qmd_va >= VBS_DRV_TURING_QMD_MAX_VA)
        return VBS_DRV_ERR_INVALID_ARG;
    /* SEND_PCAS_A = QMD address >> 8 (digest §3.5; S1). */
    vbs_drv_status_t st = vbs_drv_turing_pb_emit_method1(
        cb, subchan, VBS_DRV_TURING_MTHD_SEND_PCAS_A,
        (uint32_t)((qmd_va >> 8) & 0xffffffffu));
    if (st != VBS_DRV_OK) return st;
    /* SEND_PCAS_B = FROM/DELTA, 0 for a single launch. */
    st = vbs_drv_turing_pb_emit_method1(
        cb, subchan, VBS_DRV_TURING_MTHD_SEND_PCAS_B, 0);
    if (st != VBS_DRV_OK) return st;
    /* SEND_SIGNALING_PCAS_B = 3 (INVALIDATE + SCHEDULE): the actual */
    /* launch trigger (digest §3.5; S1).                             */
    return vbs_drv_turing_pb_emit_method1(
        cb, subchan, VBS_DRV_TURING_MTHD_SEND_SIGNALING_PCAS_B,
        VBS_DRV_TURING_PCAS_LAUNCH);
}

vbs_drv_status_t vbs_drv_turing_pb_emit_set_qmd_version(vbs_drv_cmdbuf_t *cb,
                                                        uint32_t subchan,
                                                        uint32_t value) {
    /* SET_QMD_VERSION 0x0288 (digest §3.5). The value encoding is   */
    /* not in the digest; the caller supplies it.                    */
    return vbs_drv_turing_pb_emit_method1(
        cb, subchan, VBS_DRV_TURING_MTHD_SET_QMD_VERSION, value);
}

vbs_drv_status_t vbs_drv_turing_pb_emit_report_semaphore(vbs_drv_cmdbuf_t *cb,
                                                         uint32_t subchan,
                                                         uint64_t addr,
                                                         uint32_t payload,
                                                         uint32_t control) {
    if (!cb || addr == 0) return VBS_DRV_ERR_INVALID_ARG;
    /* SET_REPORT_SEMAPHORE_A/B/C/D 0x1b00-0x1b0c (digest §3.5; S1   */
    /* lines 621-660): A/B = address, C = payload, D = control.     */
    /* The four methods are consecutive, so one increasing command  */
    /* emits all of them.                                            */
    uint32_t values[4];
    values[0] = (uint32_t)(addr & 0xffffffffu);
    values[1] = (uint32_t)(addr >> 32);
    values[2] = payload;
    values[3] = control;
    return vbs_drv_turing_pb_emit_methods(
        cb, subchan, VBS_DRV_TURING_MTHD_SET_REPORT_SEMAPHORE_A, values, 4);
}

vbs_drv_status_t vbs_drv_turing_pb_emit_launch_sequence(vbs_drv_cmdbuf_t *cb,
                                                        uint32_t subchan,
                                                        uint64_t qmd_va,
                                                        const vbs_drv_turing_qmd_t *qmd) {
    if (!cb || !qmd) return VBS_DRV_ERR_INVALID_ARG;
    /* Launch sequence, digest §3.5 (driver convention, not        */
    /* hardware law — digest §10 item 8):                           */
    /*  1. (SET_QMD_VERSION skipped: value encoding unsourced)      */
    /*  2. inline-QMD upload                                        */
    /*  3. INVALIDATE_SHADER_CACHES                                 */
    /*  4. SEND_PCAS dispatch (the launch)                          */
    /*  5. (semaphore skipped: EXEC syncobj signals completion)     */
    vbs_drv_status_t st =
        vbs_drv_turing_pb_emit_inline_qmd(cb, subchan, qmd_va, qmd);
    if (st != VBS_DRV_OK) return st;
    st = vbs_drv_turing_pb_emit_invalidate_caches(cb, subchan);
    if (st != VBS_DRV_OK) return st;
    return vbs_drv_turing_pb_emit_launch(cb, subchan, qmd_va);
}
