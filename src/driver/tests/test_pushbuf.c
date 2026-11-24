/* test_pushbuf.c — pushbuffer encoding tests.
 *
 * Checks the NV4 (old) command-format header words and the full
 * launch sequence against the values computed from digest §3.3
 * (S6 hw/fifo/dma-pusher.rst) and §3.5 (S1 clc5c0.h).
 *
 * Header math (digest §3.3 bit-string
 * 000CCCCCCCCCCC00SSSMMMMMMMMMMM00):
 *   increasing     = (count << 18) | (subchan << 13) | (method >> 2)
 *   non-increasing = 0x40000000 | (count << 18) | (subchan << 13)
 *                    | (method >> 2)
 */

#include <stdint.h>

#include "test_util.h"
#include "vbs_drv_turing.h"

static void test_headers(void) {
    /* SEND_PCAS_A 0x02b4, subchannel 0, count 1. */
    CHECK_EQ_U32(vbs_drv_turing_pb_hdr_incr(0, 0x02b4, 1), 0x000400ADu,
                 "incr header: SEND_PCAS_A, sub 0, count 1");
    /* LOAD_INLINE_QMD_DATA 0x0320, subchannel 1, count 64,
     * non-increasing. */
    CHECK_EQ_U32(vbs_drv_turing_pb_hdr_nonincr(1, 0x0320, 64), 0x410020C8u,
                 "nonincr header: LOAD_INLINE_QMD_DATA, sub 1, count 64");
    /* Same, increasing. */
    CHECK_EQ_U32(vbs_drv_turing_pb_hdr_incr(0, 0x0320, 64), 0x010000C8u,
                 "incr header: LOAD_INLINE_QMD_DATA, sub 0, count 64");
    /* SET_REPORT_SEMAPHORE_A 0x1b00, subchannel 0, count 4. */
    CHECK_EQ_U32(vbs_drv_turing_pb_hdr_incr(0, 0x1b00, 4), 0x001006C0u,
                 "incr header: SET_REPORT_SEMAPHORE_A, count 4");
    /* Max count (11-bit field): 2047. */
    CHECK_EQ_U32(vbs_drv_turing_pb_hdr_incr(7, 0x0100, 2047), 0x1FFCE040u,
                 "incr header: max count, sub 7, NO_OPERATION");
}

static void test_method_emitters(void) {
    vbs_drv_cmdbuf_t cb;
    CHECK(vbs_drv_cmdbuf_init(&cb, 16) == VBS_DRV_OK, "cmdbuf init");
    CHECK(vbs_drv_turing_pb_emit_method1(&cb, 0, 0x02b4, 0x01000010) ==
              VBS_DRV_OK,
          "emit_method1 succeeds");
    CHECK(cb.count == 2, "emit_method1 writes 2 words");
    CHECK_EQ_U32(cb.words[0], 0x000400ADu, "method1 header word");
    CHECK_EQ_U32(cb.words[1], 0x01000010u, "method1 payload word");

    /* Validation. */
    CHECK(vbs_drv_turing_pb_emit_method1(&cb, 8, 0x02b4, 0) ==
              VBS_DRV_ERR_INVALID_ARG,
          "emit_method1 rejects subchannel 8");
    CHECK(vbs_drv_turing_pb_emit_method1(&cb, 0, 0x02b5, 0) ==
              VBS_DRV_ERR_INVALID_ARG,
          "emit_method1 rejects an unaligned method");
    CHECK(vbs_drv_turing_pb_emit_method1(&cb, 0, 0x4000, 0) ==
              VBS_DRV_ERR_INVALID_ARG,
          "emit_method1 rejects a method past 0x3ffc");
    CHECK(vbs_drv_turing_pb_emit_method1(NULL, 0, 0x02b4, 0) ==
              VBS_DRV_ERR_INVALID_ARG,
          "emit_method1 rejects NULL cmdbuf");
    CHECK(vbs_drv_turing_pb_emit_methods(&cb, 0, 0x0320, NULL, 4) ==
              VBS_DRV_ERR_INVALID_ARG,
          "emit_methods rejects NULL values");
    CHECK(vbs_drv_turing_pb_emit_methods(&cb, 0, 0x3ffc,
                                         (uint32_t[]){ 1, 2 }, 2) ==
              VBS_DRV_ERR_INVALID_ARG,
          "emit_methods rejects a run past the method space");

    /* Report semaphore: one increasing command, 4 payload words. */
    vbs_drv_cmdbuf_reset(&cb);
    CHECK(vbs_drv_turing_pb_emit_report_semaphore(&cb, 0, 0x300000000ULL,
                                                  0xC0FFEE, 0x12345678) ==
              VBS_DRV_OK,
          "emit_report_semaphore succeeds");
    CHECK(cb.count == 5, "report semaphore writes 5 words");
    CHECK_EQ_U32(cb.words[0], 0x001006C0u, "semaphore header word");
    CHECK_EQ_U32(cb.words[1], 0x00000000u, "semaphore addr low");
    CHECK_EQ_U32(cb.words[2], 0x00000003u, "semaphore addr high");
    CHECK_EQ_U32(cb.words[3], 0x00C0FFEEu, "semaphore payload");
    CHECK_EQ_U32(cb.words[4], 0x12345678u, "semaphore control");

    /* Overflow: a 2-word cmdbuf cannot hold a method1 + a nop. */
    vbs_drv_cmdbuf_t small;
    CHECK(vbs_drv_cmdbuf_init(&small, 2) == VBS_DRV_OK, "small cmdbuf init");
    CHECK(vbs_drv_turing_pb_emit_method1(&small, 0, 0x0100, 0) ==
              VBS_DRV_OK,
          "method1 fits in a 2-word cmdbuf");
    CHECK(vbs_drv_turing_pb_emit_nop(&small, 0) == VBS_DRV_ERR_NO_MEMORY,
          "overflow returns NO_MEMORY, not a silent truncate");
    CHECK(small.count == 2, "failed emit leaves the cmdbuf unchanged");
    vbs_drv_cmdbuf_free(&small);
    vbs_drv_cmdbuf_free(&cb);
}

static void test_launch_sequence(void) {
    /* A QMD with a recognizable pattern: word i = 0xA5000000 | i. */
    vbs_drv_turing_qmd_t qmd;
    for (unsigned i = 0; i < VBS_DRV_TURING_QMD_WORDS; i++)
        qmd.w[i] = 0xA5000000u | i;

    /* QMD VA 0x100001000 (4 GiB + 4 KiB): >> 8 = 0x01000010. */
    uint64_t qmd_va = 0x100001000ULL;

    vbs_drv_cmdbuf_t cb;
    CHECK(vbs_drv_cmdbuf_init(&cb, 128) == VBS_DRV_OK,
          "launch cmdbuf init");
    CHECK(vbs_drv_turing_pb_emit_launch_sequence(&cb, 0, qmd_va, &qmd) ==
              VBS_DRV_OK,
          "emit_launch_sequence succeeds");
    /* 2+2 (inline addr A/B) + 1+64 (QMD words) + 2 (invalidate) +
     * 2+2+2 (SEND_PCAS_A/B + launch) = 77 words. */
    CHECK(cb.count == 77, "launch sequence is 77 words");

    CHECK_EQ_U32(cb.words[0], 0x000400C6u,
                 "word 0: SET_INLINE_QMD_ADDRESS_A header");
    CHECK_EQ_U32(cb.words[1], 0x00000000u,
                 "word 1: inline QMD addr high (>>40)");
    CHECK_EQ_U32(cb.words[2], 0x000400C7u,
                 "word 2: SET_INLINE_QMD_ADDRESS_B header");
    CHECK_EQ_U32(cb.words[3], 0x01000010u,
                 "word 3: inline QMD addr low (>>8)");
    CHECK_EQ_U32(cb.words[4], 0x010000C8u,
                 "word 4: LOAD_INLINE_QMD_DATA header, count 64");
    bool qmd_words_ok = true;
    for (unsigned i = 0; i < 64; i++) {
        if (cb.words[5 + i] != (0xA5000000u | i)) {
            qmd_words_ok = false;
            break;
        }
    }
    CHECK(qmd_words_ok, "QMD words 0..63 streamed in order");
    CHECK_EQ_U32(cb.words[69], 0x00040087u,
                 "word 69: INVALIDATE_SHADER_CACHES header");
    CHECK_EQ_U32(cb.words[70], 0x00000000u,
                 "word 70: INVALIDATE_SHADER_CACHES param");
    CHECK_EQ_U32(cb.words[71], 0x000400ADu,
                 "word 71: SEND_PCAS_A header");
    CHECK_EQ_U32(cb.words[72], 0x01000010u,
                 "word 72: SEND_PCAS_A payload (QMD VA >> 8)");
    CHECK_EQ_U32(cb.words[73], 0x000400AEu,
                 "word 73: SEND_PCAS_B header");
    CHECK_EQ_U32(cb.words[74], 0x00000000u,
                 "word 74: SEND_PCAS_B payload (FROM/DELTA 0)");
    CHECK_EQ_U32(cb.words[75], 0x000400AFu,
                 "word 75: SEND_SIGNALING_PCAS_B header");
    CHECK_EQ_U32(cb.words[76], 0x00000003u,
                 "word 76: SEND_SIGNALING_PCAS_B payload (launch = 3)");
    vbs_drv_cmdbuf_free(&cb);

    /* The QMD must live below 2^40 (SEND_PCAS_A carries only the
     * low 32 bits of the >>8-shifted address). */
    CHECK(vbs_drv_cmdbuf_init(&cb, 128) == VBS_DRV_OK,
          "second launch cmdbuf init");
    CHECK(vbs_drv_turing_pb_emit_launch(&cb, 0, 1ULL << 40) ==
              VBS_DRV_ERR_INVALID_ARG,
          "emit_launch rejects a QMD VA at 2^40");
    CHECK(vbs_drv_turing_pb_emit_launch(&cb, 0, 0) ==
              VBS_DRV_ERR_INVALID_ARG,
          "emit_launch rejects a zero QMD VA");
    CHECK(vbs_drv_turing_pb_emit_launch_sequence(&cb, 0, 1ULL << 40, &qmd) ==
              VBS_DRV_ERR_INVALID_ARG,
          "emit_launch_sequence rejects a QMD VA at 2^40");
    vbs_drv_cmdbuf_free(&cb);
}

int main(void) {
    test_headers();
    test_method_emitters();
    test_launch_sequence();
    TEST_SUMMARY("test_pushbuf");
}
