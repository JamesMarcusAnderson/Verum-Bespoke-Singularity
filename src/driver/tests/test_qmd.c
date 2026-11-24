/* test_qmd.c — QMD v2.3 encoding tests.
 *
 * Builds a vector-add-style QMD and checks every word the digest
 * defines against the field positions in digest §4.1/§4.2
 * (S1 clc5c0qmd.h). Also checks the builder's validation (bad
 * inputs fail cleanly, never produce a half-built QMD).
 *
 * Launch used for the golden values:
 *   code VA 0x100000000 (4 GiB), 4096 bytes of code
 *   grid 1024 x 1 x 1, block 256 x 1 x 1
 *   32 registers/thread, 0 shared bytes, 1 barrier (bar.sync)
 *   CB0: VA 0x200000000, 256 bytes
 *   RELEASE0: VA 0x300000000, payload 0xC0FFEE
 */

#include <stdint.h>
#include <string.h>

#include "test_drv_helpers.h"
#include "test_util.h"
#include "vbs_drv_turing.h"

static void test_golden_qmd(void) {
    /* Fake buffers: qmd_build only reads gpu_va/size. */
    vbs_drv_buffer_t code, cb0, sem;
    vbs_drv_launch_desc_t desc;
    test_vecadd_launch(&code, &cb0, &sem, &desc, 4096, 32, 1);

    vbs_drv_turing_qmd_t qmd;
    CHECK(vbs_drv_turing_qmd_build(&desc, &qmd) == VBS_DRV_OK,
          "qmd_build succeeds for a valid vector-add launch");

    /* Version fields (digest §4): word 18 low byte 0x23 (version 3
     * + major 2), word 31 top byte 0x75 (SASS_VERSION). */
    CHECK_EQ_U32(qmd.w[18], 0x01000023u,
                 "word 18: blockDim.x 256 + QMD v2.3");
    CHECK_EQ_U32(qmd.w[31], 0x75000000u,
                 "word 31: SASS_VERSION 0x75, prefetch size 0");

    /* Program address (digest §4.1): VA 0x100000000 -> low 0,
     * upper 1. Prefetch address >> 8 -> word 8 = 0x01000000. */
    CHECK_EQ_U32(qmd.w[48], 0x00000000u, "word 48: PROGRAM_ADDRESS_LOWER");
    CHECK_EQ_U32(qmd.w[49], 0x00000001u, "word 49: PROGRAM_ADDRESS_UPPER");
    CHECK_EQ_U32(qmd.w[8], 0x01000000u,
                 "word 8: PROGRAM_PREFETCH_ADDR_LOWER_SHIFTED");

    /* Grid / block (digest §4.1). */
    CHECK_EQ_U32(qmd.w[12], 1024u, "word 12: CTA_RASTER_WIDTH");
    CHECK_EQ_U32(qmd.w[13], 1u, "word 13: CTA_RASTER_HEIGHT");
    CHECK_EQ_U32(qmd.w[14], 1u, "word 14: CTA_RASTER_DEPTH");
    CHECK_EQ_U32(qmd.w[19], 0x00010001u,
                 "word 19: blockDim.y/z = 1/1");

    /* Resources (digest §4.1). */
    CHECK_EQ_U32(qmd.w[17], 0u, "word 17: SHARED_MEMORY_SIZE 0");
    CHECK_EQ_U32(qmd.w[20], 0x00002001u,
                 "word 20: CB0 valid + REGISTER_COUNT_V 32");
    CHECK_EQ_U32(qmd.w[30], 0x20000000u, "word 30: REGISTER_COUNT 32");
    CHECK_EQ_U32(qmd.w[29], 0x08000000u, "word 29: BARRIER_COUNT 1");

    /* CB0 (digest §4.1): addr 0x200000000, size 256 -> shifted 16,
     * invalidate bit set. */
    CHECK_EQ_U32(qmd.w[32], 0x00000000u, "word 32: CB0 ADDR_LOWER");
    CHECK_EQ_U32(qmd.w[33], 0x00840002u,
                 "word 33: CB0 ADDR_UPPER 2 + INVALIDATE + SIZE>>4 16");

    /* RELEASE0 (digest §4.2): addr 0x300000000, payload 0xC0FFEE. */
    CHECK_EQ_U32(qmd.w[23], 0x00000000u, "word 23: RELEASE0_ADDR_LOWER");
    CHECK_EQ_U32(qmd.w[24], 0x00000003u, "word 24: RELEASE0_ADDR_UPPER 3");
    CHECK_EQ_U32(qmd.w[25], 0x00C0FFEEu, "word 25: RELEASE0_PAYLOAD");

    /* Word 4: SM_GLOBAL_CACHING_ENABLE (bit 6) + release enable 0
     * (bit 10). Word 5: invalidate bits 30 + 31. */
    CHECK_EQ_U32(qmd.w[4], 0x00000440u,
                 "word 4: SM caching + RELEASE0 enable");
    CHECK_EQ_U32(qmd.w[5], 0xC0000000u,
                 "word 5: icache + constant-cache invalidate");

    /* Everything else must be zero (resume fields, HW-only fields,
     * unused CBs, RELEASE1, local-memory sizes). */
    static const unsigned zero_words[] = {
        0, 1, 2, 3, 6, 7, 9, 10, 11, 15, 16, 21, 22,
        26, 27, 28, /* RELEASE1 unused */
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, /* CB1-7 */
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
    };
    for (size_t i = 0; i < sizeof(zero_words) / sizeof(zero_words[0]); i++) {
        unsigned w = zero_words[i];
        CHECK(qmd.w[w] == 0, "word %u is zero", w);
    }
}

static void test_qmd_validation(void) {
    vbs_drv_turing_qmd_t qmd;
    vbs_drv_buffer_t code = { 0 };
    code.gpu_va = 0x100000000ULL;
    code.size = 4096;

    /* Setter-level validation. */
    vbs_drv_turing_qmd_zero(&qmd);
    CHECK(vbs_drv_turing_qmd_set_program(&qmd, 0, 4096) ==
              VBS_DRV_ERR_INVALID_ARG,
          "set_program rejects VA 0");
    CHECK(vbs_drv_turing_qmd_set_program(&qmd, 1ULL << 47, 4096) ==
              VBS_DRV_ERR_INVALID_ARG,
          "set_program rejects VA beyond the 47-bit MMU");
    CHECK(vbs_drv_turing_qmd_set_program(&qmd, 0x100000000ULL, 0) ==
              VBS_DRV_ERR_INVALID_ARG,
          "set_program rejects zero code size");
    CHECK(vbs_drv_turing_qmd_set_grid(&qmd, 0, 1, 1) ==
              VBS_DRV_ERR_INVALID_ARG,
          "set_grid rejects grid.x 0");
    CHECK(vbs_drv_turing_qmd_set_grid(&qmd, 1, 0x10000, 1) ==
              VBS_DRV_ERR_INVALID_ARG,
          "set_grid rejects grid.y above the 16-bit field");
    CHECK(vbs_drv_turing_qmd_set_block(&qmd, 0, 1, 1) ==
              VBS_DRV_ERR_INVALID_ARG,
          "set_block rejects block.x 0");
    CHECK(vbs_drv_turing_qmd_set_resources(&qmd, 0, 0, 0) ==
              VBS_DRV_ERR_INVALID_ARG,
          "set_resources rejects 0 registers");
    CHECK(vbs_drv_turing_qmd_set_resources(&qmd, 256, 0, 0) ==
              VBS_DRV_ERR_INVALID_ARG,
          "set_resources rejects 256 registers (8-bit field)");
    CHECK(vbs_drv_turing_qmd_set_resources(&qmd, 32, 0x40000, 0) ==
              VBS_DRV_ERR_INVALID_ARG,
          "set_resources rejects shared memory above the 18-bit field");
    CHECK(vbs_drv_turing_qmd_set_resources(&qmd, 32, 0, 32) ==
              VBS_DRV_ERR_INVALID_ARG,
          "set_resources rejects 32 barriers (5-bit field)");
    CHECK(vbs_drv_turing_qmd_set_const_buf(&qmd, 8, 0x200000000ULL, 256) ==
              VBS_DRV_ERR_INVALID_ARG,
          "set_const_buf rejects index 8");
    CHECK(vbs_drv_turing_qmd_set_const_buf(&qmd, 0, 0x200000000ULL, 255) ==
              VBS_DRV_ERR_INVALID_ARG,
          "set_const_buf rejects size not a multiple of 16");
    CHECK(vbs_drv_turing_qmd_set_const_buf(&qmd, 0, 0x200000000ULL,
                                           128 * 1024 + 16) ==
              VBS_DRV_ERR_INVALID_ARG,
          "set_const_buf rejects size above 128 KiB");
    CHECK(vbs_drv_turing_qmd_set_release(&qmd, 2, 0x300000000ULL, 1) ==
              VBS_DRV_ERR_INVALID_ARG,
          "set_release rejects index 2");
    CHECK(vbs_drv_turing_qmd_set_release(&qmd, 1, 0x300000000ULL, 0x42) ==
              VBS_DRV_OK,
          "set_release accepts index 1");
    CHECK_EQ_U32(qmd.w[26], 0x00000000u, "RELEASE1 ADDR_LOWER");
    CHECK_EQ_U32(qmd.w[27], 0x00000003u, "RELEASE1 ADDR_UPPER");
    CHECK_EQ_U32(qmd.w[28], 0x00000042u, "RELEASE1 PAYLOAD");
    CHECK((qmd.w[4] & (1u << 11)) != 0,
          "RELEASE1 enable bit set in word 4");

    /* NULL safety: no crashes, clean errors. */
    CHECK(vbs_drv_turing_qmd_build(NULL, &qmd) == VBS_DRV_ERR_INVALID_ARG,
          "qmd_build rejects NULL desc");
    CHECK(vbs_drv_turing_qmd_build(
              &(vbs_drv_launch_desc_t){ 0 }, NULL) ==
              VBS_DRV_ERR_INVALID_ARG,
          "qmd_build rejects NULL qmd");
    vbs_drv_turing_qmd_zero(NULL); /* must not crash */
    vbs_drv_turing_qmd_invalidate_caches(NULL); /* must not crash */

    /* qmd_build rejects a description with no code buffer. */
    vbs_drv_launch_desc_t bad = { 0 };
    bad.register_count = 32;
    bad.grid_x = bad.grid_y = bad.grid_z = 1;
    bad.block_x = bad.block_y = bad.block_z = 1;
    bad.code_size = 4096;
    CHECK(vbs_drv_turing_qmd_build(&bad, &qmd) == VBS_DRV_ERR_INVALID_ARG,
          "qmd_build rejects NULL code buffer");
    bad.code = &code;
    bad.code_offset = 1; /* 1 + 4096 > the 4096-byte buffer */
    CHECK(vbs_drv_turing_qmd_build(&bad, &qmd) == VBS_DRV_ERR_INVALID_ARG,
          "qmd_build rejects code range past the buffer end");
    bad.code_offset = 0;
    bad.const_buf_count = 9;
    CHECK(vbs_drv_turing_qmd_build(&bad, &qmd) == VBS_DRV_ERR_INVALID_ARG,
          "qmd_build rejects more than 8 constant buffers");
    bad.const_buf_count = 1; /* const_bufs[0] is NULL */
    CHECK(vbs_drv_turing_qmd_build(&bad, &qmd) == VBS_DRV_ERR_INVALID_ARG,
          "qmd_build rejects a NULL constant buffer");
    bad.const_buf_count = 0;
    bad.use_semaphore = true; /* sem_buf is NULL */
    CHECK(vbs_drv_turing_qmd_build(&bad, &qmd) == VBS_DRV_ERR_INVALID_ARG,
          "qmd_build rejects a semaphore with no buffer");
    bad.use_semaphore = false;
    CHECK(vbs_drv_turing_qmd_build(&bad, &qmd) == VBS_DRV_OK,
          "qmd_build accepts the repaired description");
}

int main(void) {
    test_golden_qmd();
    test_qmd_validation();
    TEST_SUMMARY("test_qmd");
}
