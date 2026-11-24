/* test_alchemist_enc.c — Alchemist encoder tests (no GPU needed).
 *
 * Every encoded word is checked against the value computed from the
 * Intel spec digest (~/workspace/verum-driver-2026-11-17/specs/intel/
 * SPEC-DIGEST.md): MI_* DW0s, PIPELINE_SELECT, STATE_BASE_ADDRESS,
 * PIPE_CONTROL variants, COMPUTE_WALKER, INTERFACE_DESCRIPTOR_DATA,
 * RENDER_SURFACE_STATE, binding-table entries, and the full launch
 * sequence. Validation failures (bad alignment, oversized blocks,
 * reserved Emit Local shapes) must be rejected, never silently
 * encoded.
 */

#include <stdint.h>
#include <string.h>

#include "test_util.h"
#include "vbs_drv.h"
#include "vbs_drv_alchemist.h"

static void test_dw0_constants(void) {
    /* Digest §4.2/§4.3/§4.4/§4.6/§4.7/§4.8 — the digest's computed
     * DW0 values. */
    CHECK_EQ_U32(VBS_DRV_ALCHEMIST_MI_NOOP, 0x00000000u, "MI_NOOP DW0");
    CHECK_EQ_U32(VBS_DRV_ALCHEMIST_MI_BATCH_BUFFER_END, 0x05000000u,
                 "MI_BATCH_BUFFER_END DW0");
    CHECK_EQ_U32(VBS_DRV_ALCHEMIST_MI_BATCH_BUFFER_START_PPGTT, 0x18800101u,
                 "MI_BATCH_BUFFER_START PPGTT DW0");
    CHECK_EQ_U32(VBS_DRV_ALCHEMIST_MI_STORE_DATA_IMM, 0x10000000u,
                 "MI_STORE_DATA_IMM DW0");
    CHECK_EQ_U32(VBS_DRV_ALCHEMIST_MI_FLUSH_DW, 0x13000000u, "MI_FLUSH_DW DW0");
    CHECK_EQ_U32(VBS_DRV_ALCHEMIST_MI_SEMAPHORE_WAIT, 0x0E000000u,
                 "MI_SEMAPHORE_WAIT DW0");
    CHECK_EQ_U32(VBS_DRV_ALCHEMIST_MI_ATOMIC, 0x17800000u, "MI_ATOMIC DW0");
    CHECK_EQ_U32(VBS_DRV_ALCHEMIST_PIPELINE_SELECT_GPGPU, 0x69040302u,
                 "PIPELINE_SELECT GPGPU DW0");
    CHECK_EQ_U32(VBS_DRV_ALCHEMIST_STATE_BASE_ADDRESS_DW0, 0x61010014u,
                 "STATE_BASE_ADDRESS DW0");
    CHECK_EQ_U32(VBS_DRV_ALCHEMIST_STATE_SIP_DW0, 0x61020001u, "STATE_SIP DW0");
    CHECK_EQ_U32(VBS_DRV_ALCHEMIST_BINDING_TABLE_POOL_ALLOC_DW0, 0x79190002u,
                 "BINDING_TABLE_POOL_ALLOC DW0");
    CHECK_EQ_U32(VBS_DRV_ALCHEMIST_PIPE_CONTROL_DW0, 0x7A000004u,
                 "PIPE_CONTROL DW0");
    CHECK_EQ_U32(VBS_DRV_ALCHEMIST_COMPUTE_WALKER_DW0, 0x72080025u,
                 "COMPUTE_WALKER DW0");
    CHECK(VBS_DRV_ALCHEMIST_COMPUTE_WALKER_WORDS == 39,
          "COMPUTE_WALKER is 39 dwords");
    CHECK(VBS_DRV_ALCHEMIST_STATE_BASE_ADDRESS_WORDS == 22,
          "STATE_BASE_ADDRESS is 22 dwords");
    CHECK(VBS_DRV_ALCHEMIST_PIPE_CONTROL_WORDS == 6,
          "PIPE_CONTROL is 6 dwords");
    CHECK(VBS_DRV_ALCHEMIST_IFACE_DESC_WORDS == 8,
          "INTERFACE_DESCRIPTOR_DATA is 8 dwords");
    CHECK(VBS_DRV_ALCHEMIST_SURFACE_STATE_WORDS == 16,
          "RENDER_SURFACE_STATE is 16 dwords");
}

static void test_mi_emitters(void) {
    vbs_drv_cmdbuf_t cb;
    CHECK(vbs_drv_cmdbuf_init(&cb, 16) == VBS_DRV_OK, "cmdbuf init");
    CHECK(vbs_drv_alchemist_emit_noop(&cb) == VBS_DRV_OK, "emit noop");
    CHECK(vbs_drv_alchemist_emit_pipeline_select_gpgpu(&cb) == VBS_DRV_OK,
          "emit pipeline select");
    CHECK(vbs_drv_alchemist_emit_batch_end(&cb) == VBS_DRV_OK, "emit batch end");
    CHECK(cb.count == 3, "three MI words emitted");
    CHECK_EQ_U32(cb.words[0], 0x00000000u, "noop word");
    CHECK_EQ_U32(cb.words[1], 0x69040302u, "pipeline select word");
    CHECK_EQ_U32(cb.words[2], 0x05000000u, "batch end word");
    CHECK(vbs_drv_alchemist_emit_noop(NULL) == VBS_DRV_ERR_INVALID_ARG,
          "emit noop rejects NULL");
    vbs_drv_cmdbuf_free(&cb);
}

static void test_state_base_address(void) {
    vbs_drv_cmdbuf_t cb;
    CHECK(vbs_drv_cmdbuf_init(&cb, 32) == VBS_DRV_OK, "cmdbuf init");
    /* general = instr = 0x100000000 (the launch choice), surface =
     * 0x200001000. */
    CHECK(vbs_drv_alchemist_emit_state_base_address(
              &cb, 0x100000000ULL, 0x1000, 0x200001000ULL, 0x100000000ULL,
              0x2000) == VBS_DRV_OK,
          "emit state base address");
    CHECK(cb.count == 22, "SBA is 22 words");
    CHECK_EQ_U32(cb.words[0], 0x61010014u, "SBA DW0");
    /* DW1-2: general base 0x1_00000000 + modify enable. */
    CHECK_EQ_U32(cb.words[1], 0x00000001u, "SBA general base low");
    CHECK_EQ_U32(cb.words[2], 0x00000001u, "SBA general base high");
    CHECK_EQ_U32(cb.words[3], 0x00000000u, "SBA DW3 reserved");
    /* DW4-5: surface base 0x2_00001000 + modify enable. */
    CHECK_EQ_U32(cb.words[4], 0x00001001u, "SBA surface base low");
    CHECK_EQ_U32(cb.words[5], 0x00000002u, "SBA surface base high");
    CHECK_EQ_U32(cb.words[6], 0x00000000u, "SBA dynamic base untouched");
    CHECK_EQ_U32(cb.words[7], 0x00000000u, "SBA dynamic base high untouched");
    CHECK_EQ_U32(cb.words[8], 0x00000000u, "SBA indirect base untouched");
    CHECK_EQ_U32(cb.words[9], 0x00000000u, "SBA indirect base high untouched");
    CHECK_EQ_U32(cb.words[10], 0x00000001u, "SBA instr base low");
    CHECK_EQ_U32(cb.words[11], 0x00000001u, "SBA instr base high");
    CHECK_EQ_U32(cb.words[12], 0x00001001u, "SBA general size + modify");
    CHECK_EQ_U32(cb.words[13], 0x00000000u, "SBA dynamic size untouched");
    CHECK_EQ_U32(cb.words[14], 0x00000000u, "SBA indirect size untouched");
    CHECK_EQ_U32(cb.words[15], 0x00002001u, "SBA instr size + modify");
    for (int i = 16; i < 22; i++)
        CHECK_EQ_U32(cb.words[i], 0x00000000u, "SBA bindless untouched");
    /* Validation: unaligned base, zero size, unaligned size. */
    vbs_drv_cmdbuf_reset(&cb);
    CHECK(vbs_drv_alchemist_emit_state_base_address(
              &cb, 0x100000001ULL, 0x1000, 0x200001000ULL, 0x100000000ULL,
              0x2000) == VBS_DRV_ERR_INVALID_ARG,
          "SBA rejects an unaligned general base");
    CHECK(vbs_drv_alchemist_emit_state_base_address(
              &cb, 0x100000000ULL, 0, 0x200001000ULL, 0x100000000ULL,
              0x2000) == VBS_DRV_ERR_INVALID_ARG,
          "SBA rejects a zero general size");
    CHECK(vbs_drv_alchemist_emit_state_base_address(
              &cb, 0x100000000ULL, 0x1000, 0x200001000ULL, 0x100000000ULL,
              0x2001) == VBS_DRV_ERR_INVALID_ARG,
          "SBA rejects an unaligned instr size");
    vbs_drv_cmdbuf_free(&cb);
}

static void test_binding_table_pool(void) {
    vbs_drv_cmdbuf_t cb;
    CHECK(vbs_drv_cmdbuf_init(&cb, 8) == VBS_DRV_OK, "cmdbuf init");
    CHECK(vbs_drv_alchemist_emit_binding_table_pool(&cb, 0x200000000ULL,
                                                0x1000) == VBS_DRV_OK,
          "emit binding table pool");
    CHECK(cb.count == 4, "pool alloc is 4 words");
    CHECK_EQ_U32(cb.words[0], 0x79190002u, "pool DW0");
    CHECK_EQ_U32(cb.words[1], 0x00000000u, "pool base low");
    CHECK_EQ_U32(cb.words[2], 0x00000002u, "pool base high");
    CHECK_EQ_U32(cb.words[3], 0x00001000u, "pool size");
    vbs_drv_cmdbuf_reset(&cb);
    CHECK(vbs_drv_alchemist_emit_binding_table_pool(&cb, 0x200000000ULL, 0) ==
              VBS_DRV_ERR_INVALID_ARG,
          "pool rejects zero size");
    CHECK(vbs_drv_alchemist_emit_binding_table_pool(&cb, 0x200000001ULL,
                                                0x1000) ==
              VBS_DRV_ERR_INVALID_ARG,
          "pool rejects an unaligned base");
    vbs_drv_cmdbuf_free(&cb);
}

static void test_pipe_control(void) {
    vbs_drv_cmdbuf_t cb;
    CHECK(vbs_drv_cmdbuf_init(&cb, 16) == VBS_DRV_OK, "cmdbuf init");
    CHECK(vbs_drv_alchemist_emit_pipe_control_ic_invalidate(&cb) == VBS_DRV_OK,
          "emit leading pipe control");
    CHECK(vbs_drv_alchemist_emit_pipe_control_flush(&cb) == VBS_DRV_OK,
          "emit trailing pipe control");
    CHECK(cb.count == 12, "two pipe controls are 12 words");
    /* Leading: DW0 base, DW1 CS Stall (bit 20) + IC Invalidate
     * (bit 11) = 0x00100800. */
    CHECK_EQ_U32(cb.words[0], 0x7A000004u, "leading PC DW0");
    CHECK_EQ_U32(cb.words[1], 0x00100800u, "leading PC DW1");
    for (int i = 2; i < 6; i++)
        CHECK_EQ_U32(cb.words[i], 0x00000000u, "leading PC tail zero");
    /* Trailing: DW0 base + HDC Flush (bit 9) + Untyped L1 Flush
     * (bit 11) = 0x7A000A04; DW1 CS Stall = 0x00100000. */
    CHECK_EQ_U32(cb.words[6], 0x7A000A04u, "trailing PC DW0");
    CHECK_EQ_U32(cb.words[7], 0x00100000u, "trailing PC DW1");
    for (int i = 8; i < 12; i++)
        CHECK_EQ_U32(cb.words[i], 0x00000000u, "trailing PC tail zero");
    vbs_drv_cmdbuf_free(&cb);
}

static void test_iface_build(void) {
    vbs_drv_alchemist_iface_params_t p;
    vbs_drv_alchemist_iface_desc_t d;
    memset(&p, 0, sizeof(p));
    p.kernel_start_offset = 0x40;
    p.binding_table_offset = 0x20;
    p.slm_bytes = 8192;
    p.barrier_count = 2;
    p.thread_count = 16;
    CHECK(vbs_drv_alchemist_iface_build(&p, &d) == VBS_DRV_OK, "iface build");
    CHECK_EQ_U32(d.w[0], 0x00000040u, "iface kernel start pointer");
    CHECK_EQ_U32(d.w[1], 0x00000000u, "iface DW1 reserved MBZ");
    CHECK_EQ_U32(d.w[2], 0x00000000u, "iface DW2 defaults");
    CHECK_EQ_U32(d.w[3], 0x00000000u, "iface no samplers");
    CHECK_EQ_U32(d.w[4], 0x00000020u, "iface binding table pointer");
    /* DW5: barriers 2 << 28, SLM 8K enum 4 << 16, threads 16. */
    CHECK_EQ_U32(d.w[5], 0x20040010u, "iface DW5");
    CHECK_EQ_U32(d.w[6], 0x00000000u, "iface DW6");
    CHECK_EQ_U32(d.w[7], 0x00000000u, "iface DW7 reserved");

    /* SLM rounding: 0 -> 0, 1 -> 1K, 8193 -> 16K, 64K -> 7. */
    vbs_drv_alchemist_iface_params_t q;
    memset(&q, 0, sizeof(q));
    q.thread_count = 1;
    q.slm_bytes = 0;
    CHECK(vbs_drv_alchemist_iface_build(&q, &d) == VBS_DRV_OK, "iface SLM 0");
    CHECK_EQ_U32(d.w[5], 0x00000001u, "iface default DW5 (1 thread)");
    q.slm_bytes = 1;
    CHECK(vbs_drv_alchemist_iface_build(&q, &d) == VBS_DRV_OK, "iface SLM 1B");
    CHECK_EQ_U32(d.w[5] >> 16 & 0x1fu, 1u, "iface SLM 1B rounds to 1K");
    q.slm_bytes = 8193;
    CHECK(vbs_drv_alchemist_iface_build(&q, &d) == VBS_DRV_OK, "iface SLM 8193");
    CHECK_EQ_U32(d.w[5] >> 16 & 0x1fu, 5u, "iface SLM 8193 rounds to 16K");
    q.slm_bytes = 65536;
    CHECK(vbs_drv_alchemist_iface_build(&q, &d) == VBS_DRV_OK, "iface SLM 64K");
    CHECK_EQ_U32(d.w[5] >> 16 & 0x1fu, 7u, "iface SLM 64K enum 7");

    /* Validation failures. */
    q.slm_bytes = 65537;
    CHECK(vbs_drv_alchemist_iface_build(&q, &d) == VBS_DRV_ERR_INVALID_ARG,
          "iface rejects SLM over 64K");
    q.slm_bytes = 0;
    q.thread_count = 0;
    CHECK(vbs_drv_alchemist_iface_build(&q, &d) == VBS_DRV_ERR_INVALID_ARG,
          "iface rejects 0 threads");
    q.thread_count = 65;
    CHECK(vbs_drv_alchemist_iface_build(&q, &d) == VBS_DRV_ERR_INVALID_ARG,
          "iface rejects 65 threads");
    q.thread_count = 1;
    q.barrier_count = 8;
    CHECK(vbs_drv_alchemist_iface_build(&q, &d) == VBS_DRV_ERR_INVALID_ARG,
          "iface rejects 8 barriers");
    q.barrier_count = 0;
    q.kernel_start_offset = 0x41;
    CHECK(vbs_drv_alchemist_iface_build(&q, &d) == VBS_DRV_ERR_INVALID_ARG,
          "iface rejects an unaligned kernel offset");
    q.kernel_start_offset = 0;
    q.binding_table_offset = 0x21;
    CHECK(vbs_drv_alchemist_iface_build(&q, &d) == VBS_DRV_ERR_INVALID_ARG,
          "iface rejects an unaligned binding-table offset");
    CHECK(vbs_drv_alchemist_iface_build(NULL, &d) == VBS_DRV_ERR_INVALID_ARG,
          "iface rejects NULL params");
}

static void test_surface_and_binding(void) {
    vbs_drv_alchemist_surface_state_t ss;
    CHECK(vbs_drv_alchemist_surface_build(0x12345678000ULL, 4096, &ss) ==
              VBS_DRV_OK,
          "surface build");
    CHECK_EQ_U32(ss.w[0], 0x80000000u, "surface DW0 SURFTYPE_BUFFER");
    CHECK_EQ_U32(ss.w[1], 0x00000000u, "surface DW1 MOCS 0");
    CHECK_EQ_U32(ss.w[2], 0x000003ffu, "surface width 1024 elements - 1");
    for (int i = 3; i < 8; i++)
        CHECK_EQ_U32(ss.w[i], 0x00000000u, "surface DW3-7 zero");
    CHECK_EQ_U32(ss.w[8], 0x45678000u, "surface base low");
    CHECK_EQ_U32(ss.w[9], 0x00000123u, "surface base high");
    for (int i = 10; i < 16; i++)
        CHECK_EQ_U32(ss.w[i], 0x00000000u, "surface DW10-15 zero");
    /* Width clamp: 1 MiB = 262144 elements, field caps at 0x3FFF. */
    CHECK(vbs_drv_alchemist_surface_build(0x100000000ULL, 1024 * 1024, &ss) ==
              VBS_DRV_OK,
          "surface build 1 MiB");
    CHECK_EQ_U32(ss.w[2], 0x00003fffu, "surface width clamps at 0x3FFF");
    CHECK(vbs_drv_alchemist_surface_build(0x100000000ULL, 0, &ss) ==
              VBS_DRV_ERR_INVALID_ARG,
          "surface rejects zero size");
    CHECK(vbs_drv_alchemist_surface_build(0x100000000ULL, 6, &ss) ==
              VBS_DRV_ERR_INVALID_ARG,
          "surface rejects a non-multiple-of-4 size");
    CHECK(vbs_drv_alchemist_surface_build(0x100000002ULL, 4096, &ss) ==
              VBS_DRV_ERR_INVALID_ARG,
          "surface rejects an unaligned base");

    CHECK_EQ_U32(vbs_drv_alchemist_binding_entry(0), 0x00000000u,
                 "binding entry 0");
    CHECK_EQ_U32(vbs_drv_alchemist_binding_entry(0x40), 0x00000040u,
                 "binding entry 0x40");
    CHECK_EQ_U32(vbs_drv_alchemist_binding_entry(0x80), 0x00000080u,
                 "binding entry 0x80");
}

static vbs_drv_alchemist_iface_desc_t make_iface(uint32_t threads) {
    vbs_drv_alchemist_iface_params_t p;
    vbs_drv_alchemist_iface_desc_t d;
    memset(&p, 0, sizeof(p));
    p.kernel_start_offset = 0x40;
    p.thread_count = threads;
    if (vbs_drv_alchemist_iface_build(&p, &d) != VBS_DRV_OK)
        memset(&d, 0, sizeof(d));
    return d;
}

/* Reset the command buffer and refill the walker parameters for
 * the next case. Every case after the first runs on a 1 x 1 x 1
 * grid in Y and Z with block_z 1; only the X grid, the X/Y block,
 * and the interface thread count vary. */
static void reset_walker(vbs_drv_cmdbuf_t *cb,
                         vbs_drv_alchemist_walker_params_t *p,
                         uint32_t grid_x, uint32_t block_x,
                         uint32_t block_y, uint32_t iface_threads) {
    vbs_drv_cmdbuf_reset(cb);
    memset(p, 0, sizeof(*p));
    p->grid_x = grid_x;
    p->grid_y = 1;
    p->grid_z = 1;
    p->block_x = block_x;
    p->block_y = block_y;
    p->block_z = 1;
    p->iface = make_iface(iface_threads);
}

static void test_walker(void) {
    vbs_drv_cmdbuf_t cb;
    CHECK(vbs_drv_cmdbuf_init(&cb, 64) == VBS_DRV_OK, "cmdbuf init");
    vbs_drv_alchemist_walker_params_t p;
    memset(&p, 0, sizeof(p));
    p.grid_x = 3;
    p.grid_y = 2;
    p.grid_z = 1;
    p.block_x = 16;
    p.block_y = 1;
    p.block_z = 1;
    p.iface = make_iface(1);
    CHECK(vbs_drv_alchemist_emit_walker(&cb, &p) == VBS_DRV_OK, "emit walker");
    CHECK(cb.count == 39, "walker is 39 words");
    CHECK_EQ_U32(cb.words[0], 0x72080025u, "walker DW0");
    CHECK_EQ_U32(cb.words[1], 0x00000000u, "walker DW1 reserved");
    CHECK_EQ_U32(cb.words[2], 0x00000000u, "walker DW2 no indirect data");
    CHECK_EQ_U32(cb.words[3], 0x00000000u, "walker DW3");
    /* DW4: SIMD16 (1<<30) | GenLocalID (1<<29) | Emit X (1<<26) |
     * Walk 012 (0) | Msg SIMD16 (1<<17) = 0x64020000. */
    CHECK_EQ_U32(cb.words[4], 0x64020000u, "walker DW4");
    CHECK_EQ_U32(cb.words[5], 0x0000ffffu, "walker exec mask full");
    CHECK_EQ_U32(cb.words[6], 0x0000000fu, "walker local max X=15");
    CHECK_EQ_U32(cb.words[7], 3u, "walker grid X");
    CHECK_EQ_U32(cb.words[8], 2u, "walker grid Y");
    CHECK_EQ_U32(cb.words[9], 1u, "walker grid Z");
    for (int i = 10; i < 18; i++)
        CHECK_EQ_U32(cb.words[i], 0x00000000u, "walker DW10-17 zero");
    CHECK_EQ_U32(cb.words[18], 0x00000040u, "walker iface kernel pointer");
    CHECK_EQ_U32(cb.words[23], 0x00000001u, "walker iface 1 thread");
    for (int i = 26; i < 39; i++)
        CHECK_EQ_U32(cb.words[i], 0x00000000u, "walker postsync/inline zero");

    /* Block (100,1,1): 7 threads, last-thread mask 0xF, walk order
     * 210 (X is the non-power-of-two dimension). */
    reset_walker(&cb, &p, 1, 100, 1, 7);
    CHECK(vbs_drv_alchemist_emit_walker(&cb, &p) == VBS_DRV_OK,
          "emit walker block 100");
    CHECK_EQ_U32(cb.words[4], 0x65420000u, "walker DW4 walk 210");
    CHECK_EQ_U32(cb.words[5], 0x0000000fu, "walker exec mask 0xF");
    CHECK_EQ_U32(cb.words[6], 99u, "walker local max X=99");

    /* Block (16,16,1): 16 threads, Emit XY, DW6 = (15<<10)|15. */
    reset_walker(&cb, &p, 1, 16, 16, 16);
    CHECK(vbs_drv_alchemist_emit_walker(&cb, &p) == VBS_DRV_OK,
          "emit walker block 16x16");
    CHECK_EQ_U32(cb.words[4], 0x6C020000u, "walker DW4 emit XY");
    CHECK_EQ_U32(cb.words[5], 0x0000ffffu, "walker exec mask full 256");
    CHECK_EQ_U32(cb.words[6], 0x00003c0fu, "walker local max 15,15");

    /* Semaphore PostSync: DW26 = flush bits + Write Immediate,
     * DW27-28 = address, DW29-30 = payload. */
    reset_walker(&cb, &p, 1, 16, 1, 1);
    p.use_semaphore = true;
    p.sem_va = 0x100002000ULL;
    p.sem_payload = 0xDEADBEEFu;
    CHECK(vbs_drv_alchemist_emit_walker(&cb, &p) == VBS_DRV_OK,
          "emit walker with semaphore");
    CHECK_EQ_U32(cb.words[26], 0x00001005u, "postsync DW0 flush + write");
    CHECK_EQ_U32(cb.words[27], 0x00002000u, "postsync addr low");
    CHECK_EQ_U32(cb.words[28], 0x00000001u, "postsync addr high");
    CHECK_EQ_U32(cb.words[29], 0xDEADBEEFu, "postsync payload");
    CHECK_EQ_U32(cb.words[30], 0x00000000u, "postsync payload high zero");

    /* Validation failures. */
    reset_walker(&cb, &p, 0, 16, 1, 1);
    CHECK(vbs_drv_alchemist_emit_walker(&cb, &p) == VBS_DRV_ERR_INVALID_ARG,
          "walker rejects grid 0");
    p.grid_x = 1;
    p.block_x = 1025;
    p.iface = make_iface(64);
    CHECK(vbs_drv_alchemist_emit_walker(&cb, &p) == VBS_DRV_ERR_INVALID_ARG,
          "walker rejects LWS over 1024");
    /* (100,10,1): LWS 1000 fits, but two dims are non-power-of-two;
     * iface threads 63 = ceil(1000/16) is consistent, so the pow2
     * rule is what rejects it. */
    p.block_x = 100;
    p.block_y = 10;
    p.iface = make_iface(63);
    CHECK(vbs_drv_alchemist_emit_walker(&cb, &p) == VBS_DRV_ERR_INVALID_ARG,
          "walker rejects two non-power-of-two dims");
    p.block_x = 1;
    p.block_y = 16;
    p.iface = make_iface(1);
    CHECK(vbs_drv_alchemist_emit_walker(&cb, &p) == VBS_DRV_ERR_INVALID_ARG,
          "walker rejects Emit-Y-only block (1,16,1)");
    p.block_x = 16;
    p.block_y = 1;
    p.iface = make_iface(2); /* wrong thread count for LWS 16 */
    CHECK(vbs_drv_alchemist_emit_walker(&cb, &p) == VBS_DRV_ERR_INVALID_ARG,
          "walker rejects an inconsistent iface thread count");
    p.iface = make_iface(1);
    p.use_semaphore = true;
    p.sem_va = 0x1001; /* unaligned */
    CHECK(vbs_drv_alchemist_emit_walker(&cb, &p) == VBS_DRV_ERR_INVALID_ARG,
          "walker rejects an unaligned semaphore address");
    CHECK(vbs_drv_alchemist_emit_walker(&cb, NULL) == VBS_DRV_ERR_INVALID_ARG,
          "walker rejects NULL params");
    vbs_drv_cmdbuf_free(&cb);
}

static void test_launch_sequence(void) {
    vbs_drv_cmdbuf_t cb;
    CHECK(vbs_drv_cmdbuf_init(&cb, 128) == VBS_DRV_OK, "cmdbuf init");
    vbs_drv_alchemist_launch_params_t p;
    memset(&p, 0, sizeof(p));
    p.code_va = 0x100000000ULL;
    p.code_offset = 0;
    p.code_size = 0x1000;
    p.state_va = 0x200000000ULL;
    p.const_count = 2;
    p.grid_x = 4;
    p.grid_y = 1;
    p.grid_z = 1;
    p.block_x = 64;
    p.block_y = 1;
    p.block_z = 1;
    CHECK(vbs_drv_alchemist_emit_launch_sequence(&cb, &p) == VBS_DRV_OK,
          "emit launch sequence");
    /* 22 SBA + 4 pool + 6 PC + 39 walker + 6 PC + 1 end = 78. */
    CHECK(cb.count == 78, "launch sequence is 78 words");
    CHECK((cb.count & 1u) == 0, "launch sequence is a QWord multiple");
    CHECK_EQ_U32(cb.words[0], 0x61010014u, "launch SBA DW0");
    CHECK_EQ_U32(cb.words[4], 0x00001001u, "launch SBA surface base");
    CHECK_EQ_U32(cb.words[22], 0x79190002u, "launch pool DW0");
    CHECK_EQ_U32(cb.words[25], 0x00001000u, "launch pool size");
    CHECK_EQ_U32(cb.words[26], 0x7A000004u, "launch leading PC DW0");
    CHECK_EQ_U32(cb.words[27], 0x00100800u, "launch leading PC DW1");
    CHECK_EQ_U32(cb.words[32], 0x72080025u, "launch walker DW0");
    CHECK_EQ_U32(cb.words[36], 0x64020000u, "launch walker DW4");
    CHECK_EQ_U32(cb.words[37], 0x0000ffffu, "launch walker exec mask");
    CHECK_EQ_U32(cb.words[38], 63u, "launch walker local max 63");
    CHECK_EQ_U32(cb.words[39], 4u, "launch walker grid X");
    CHECK_EQ_U32(cb.words[55], 4u, "launch iface 4 threads");
    CHECK_EQ_U32(cb.words[71], 0x7A000A04u, "launch trailing PC DW0");
    CHECK_EQ_U32(cb.words[72], 0x00100000u, "launch trailing PC DW1");
    CHECK_EQ_U32(cb.words[77], 0x05000000u, "launch batch end");

    /* No constant buffers: the pool alloc is skipped, 74 words. */
    vbs_drv_cmdbuf_reset(&cb);
    p.const_count = 0;
    CHECK(vbs_drv_alchemist_emit_launch_sequence(&cb, &p) == VBS_DRV_OK,
          "emit launch sequence without const bufs");
    CHECK(cb.count == 74, "launch without const bufs is 74 words");
    CHECK_EQ_U32(cb.words[22], 0x7A000004u,
                 "no pool alloc: leading PC follows SBA");

    /* Validation failures. */
    vbs_drv_cmdbuf_reset(&cb);
    p.const_count = 2;
    p.code_offset = 0x41;
    CHECK(vbs_drv_alchemist_emit_launch_sequence(&cb, &p) ==
              VBS_DRV_ERR_INVALID_ARG,
          "launch rejects an unaligned code offset");
    p.code_offset = 0;
    p.slm_bytes = 65537;
    CHECK(vbs_drv_alchemist_emit_launch_sequence(&cb, &p) ==
              VBS_DRV_ERR_INVALID_ARG,
          "launch rejects SLM over 64K");
    p.slm_bytes = 0;
    p.barrier_count = 8;
    CHECK(vbs_drv_alchemist_emit_launch_sequence(&cb, &p) ==
              VBS_DRV_ERR_INVALID_ARG,
          "launch rejects 8 barriers");
    p.barrier_count = 0;
    p.use_semaphore = true;
    p.sem_va = 0x1001;
    CHECK(vbs_drv_alchemist_emit_launch_sequence(&cb, &p) ==
              VBS_DRV_ERR_INVALID_ARG,
          "launch rejects an unaligned semaphore");
    p.use_semaphore = false;
    p.block_x = 1025;
    CHECK(vbs_drv_alchemist_emit_launch_sequence(&cb, &p) ==
              VBS_DRV_ERR_INVALID_ARG,
          "launch rejects LWS over 1024");
    CHECK(vbs_drv_alchemist_emit_launch_sequence(&cb, NULL) ==
              VBS_DRV_ERR_INVALID_ARG,
          "launch rejects NULL params");
    vbs_drv_cmdbuf_free(&cb);
}

int main(void) {
    test_dw0_constants();
    test_mi_emitters();
    test_state_base_address();
    test_binding_table_pool();
    test_pipe_control();
    test_iface_build();
    test_surface_and_binding();
    test_walker();
    test_launch_sequence();
    TEST_SUMMARY("test_alchemist_enc");
}
