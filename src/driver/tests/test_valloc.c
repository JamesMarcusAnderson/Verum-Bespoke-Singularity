/* test_valloc.c — GPU VA allocator tests (pure logic, no hardware).
 *
 * Region used: base 0x100000000 (4 GiB), size 16 MiB — the same
 * shape as the Turing UMD region, scaled down.
 */

#include <stdint.h>

#include "test_util.h"
#include "vbs_drv.h"

#define VA_BASE 0x100000000ULL
#define VA_SIZE 0x1000000ULL /* 16 MiB */

static void test_basic_alloc_free(void) {
    vbs_drv_va_space_t va;
    CHECK(vbs_drv_va_init(&va, VA_BASE, VA_SIZE) == VBS_DRV_OK,
          "va_init succeeds");
    uint64_t a0 = 0, a1 = 0, a2 = 0;
    CHECK(vbs_drv_va_alloc(&va, 0x1000, 0x1000, &a0) == VBS_DRV_OK,
          "first alloc succeeds");
    CHECK_EQ_U64(a0, VA_BASE, "first alloc is at the region base");
    CHECK(vbs_drv_va_alloc(&va, 0x2000, 0x1000, &a1) == VBS_DRV_OK,
          "second alloc succeeds");
    CHECK_EQ_U64(a1, VA_BASE + 0x1000, "second alloc follows the first");
    CHECK(vbs_drv_va_alloc(&va, 0x1000, 0x1000, &a2) == VBS_DRV_OK,
          "third alloc succeeds");
    CHECK_EQ_U64(a2, VA_BASE + 0x3000, "third alloc follows the second");

    /* Free the middle block and re-allocate: first-fit reuses it. */
    CHECK(vbs_drv_va_free(&va, a1, 0x2000) == VBS_DRV_OK,
          "free of the middle block succeeds");
    uint64_t a3 = 0;
    CHECK(vbs_drv_va_alloc(&va, 0x800, 0x1000, &a3) == VBS_DRV_OK,
          "alloc after free succeeds");
    CHECK_EQ_U64(a3, a1, "first-fit reuses the freed block");
    CHECK(vbs_drv_va_free(&va, a0, 0x1000) == VBS_DRV_OK, "free a0");
    CHECK(vbs_drv_va_free(&va, a2, 0x1000) == VBS_DRV_OK, "free a2");
    CHECK(vbs_drv_va_free(&va, a3, 0x800) == VBS_DRV_OK, "free a3");
    /* Everything freed: the whole region is allocatable again. */
    uint64_t all = 0;
    CHECK(vbs_drv_va_alloc(&va, VA_SIZE, 0x1000, &all) == VBS_DRV_OK,
          "whole region allocatable after all frees (coalescing works)");
    CHECK_EQ_U64(all, VA_BASE, "whole-region alloc starts at the base");
}

static void test_alignment(void) {
    vbs_drv_va_space_t va;
    CHECK(vbs_drv_va_init(&va, VA_BASE, VA_SIZE) == VBS_DRV_OK,
          "va_init succeeds");
    uint64_t a = 0;
    /* Take 4 KiB so the next aligned alloc must skip ahead. */
    CHECK(vbs_drv_va_alloc(&va, 0x1000, 0x1000, &a) == VBS_DRV_OK,
          "unaligned-context alloc succeeds");
    uint64_t b = 0;
    CHECK(vbs_drv_va_alloc(&va, 0x1000, 0x10000, &b) == VBS_DRV_OK,
          "64 KiB-aligned alloc succeeds");
    CHECK((b & 0xffffu) == 0, "aligned alloc is 64 KiB-aligned");
    CHECK_EQ_U64(b, VA_BASE + 0x10000, "aligned alloc skips to the boundary");
    /* The gap left by alignment is still usable. */
    uint64_t c = 0;
    CHECK(vbs_drv_va_alloc(&va, 0x1000, 0x1000, &c) == VBS_DRV_OK,
          "alloc into the alignment gap succeeds");
    CHECK_EQ_U64(c, VA_BASE + 0x1000, "gap alloc uses the skipped bytes");
}

static void test_exhaustion(void) {
    vbs_drv_va_space_t va;
    CHECK(vbs_drv_va_init(&va, VA_BASE, 0x3000) == VBS_DRV_OK,
          "va_init succeeds (12 KiB region)");
    uint64_t a = 0, b = 0, c = 0;
    CHECK(vbs_drv_va_alloc(&va, 0x1000, 0x1000, &a) == VBS_DRV_OK,
          "alloc 1 of 3");
    CHECK(vbs_drv_va_alloc(&va, 0x1000, 0x1000, &b) == VBS_DRV_OK,
          "alloc 2 of 3");
    CHECK(vbs_drv_va_alloc(&va, 0x1000, 0x1000, &c) == VBS_DRV_OK,
          "alloc 3 of 3");
    uint64_t d = 0;
    CHECK(vbs_drv_va_alloc(&va, 0x1000, 0x1000, &d) == VBS_DRV_ERR_NO_MEMORY,
          "alloc past exhaustion fails with NO_MEMORY");
    CHECK(vbs_drv_va_alloc(&va, 0x2000, 0x1000, &d) == VBS_DRV_ERR_NO_MEMORY,
          "oversize alloc fails with NO_MEMORY");
    CHECK(vbs_drv_va_free(&va, b, 0x1000) == VBS_DRV_OK, "free the middle");
    CHECK(vbs_drv_va_alloc(&va, 0x2000, 0x1000, &d) == VBS_DRV_ERR_NO_MEMORY,
          "2-page alloc fails when only 1 page is free (no splitting "
          "across non-adjacent blocks)");
    CHECK(vbs_drv_va_alloc(&va, 0x1000, 0x1000, &d) == VBS_DRV_OK,
          "1-page alloc succeeds in the freed hole");
}

static void test_coalescing(void) {
    vbs_drv_va_space_t va;
    CHECK(vbs_drv_va_init(&va, VA_BASE, VA_SIZE) == VBS_DRV_OK,
          "va_init succeeds");
    uint64_t a = 0, b = 0, c = 0;
    CHECK(vbs_drv_va_alloc(&va, 0x1000, 0x1000, &a) == VBS_DRV_OK, "alloc a");
    CHECK(vbs_drv_va_alloc(&va, 0x1000, 0x1000, &b) == VBS_DRV_OK, "alloc b");
    CHECK(vbs_drv_va_alloc(&va, 0x1000, 0x1000, &c) == VBS_DRV_OK, "alloc c");
    /* Free in an order that forces both merge directions: outer
     * blocks first, then the middle one. */
    CHECK(vbs_drv_va_free(&va, a, 0x1000) == VBS_DRV_OK, "free a");
    CHECK(vbs_drv_va_free(&va, c, 0x1000) == VBS_DRV_OK, "free c");
    CHECK(vbs_drv_va_free(&va, b, 0x1000) == VBS_DRV_OK, "free b");
    CHECK(va.free_count == 1, "three adjacent frees coalesce to one block");
    CHECK_EQ_U64(va.free_blocks[0].addr, VA_BASE,
                 "coalesced block starts at the base");
    CHECK_EQ_U64(va.free_blocks[0].size, VA_SIZE,
                 "coalesced block covers the whole region");
}

static void test_validation(void) {
    vbs_drv_va_space_t va;
    uint64_t a = 0;
    CHECK(vbs_drv_va_init(NULL, VA_BASE, VA_SIZE) == VBS_DRV_ERR_INVALID_ARG,
          "va_init rejects NULL");
    CHECK(vbs_drv_va_init(&va, VA_BASE, 0) == VBS_DRV_ERR_INVALID_ARG,
          "va_init rejects zero size");
    CHECK(vbs_drv_va_init(&va, UINT64_MAX - 0xfff, 0x1000) ==
              VBS_DRV_ERR_INVALID_ARG,
          "va_init rejects a base+size overflow");
    /* Uninitialized (not ready) allocator. */
    vbs_drv_va_space_t dead = { 0 };
    CHECK(vbs_drv_va_alloc(&dead, 0x1000, 0x1000, &a) ==
              VBS_DRV_ERR_INVALID_ARG,
          "alloc on an uninitialized VA space fails");
    CHECK(vbs_drv_va_free(&dead, VA_BASE, 0x1000) == VBS_DRV_ERR_INVALID_ARG,
          "free on an uninitialized VA space fails");

    CHECK(vbs_drv_va_init(&va, VA_BASE, VA_SIZE) == VBS_DRV_OK,
          "va_init succeeds");
    CHECK(vbs_drv_va_alloc(&va, 0, 0x1000, &a) == VBS_DRV_ERR_INVALID_ARG,
          "alloc rejects zero size");
    CHECK(vbs_drv_va_alloc(&va, 0x1000, 0, &a) == VBS_DRV_ERR_INVALID_ARG,
          "alloc rejects zero alignment");
    CHECK(vbs_drv_va_alloc(&va, 0x1000, 3, &a) == VBS_DRV_ERR_INVALID_ARG,
          "alloc rejects non-power-of-two alignment");
    CHECK(vbs_drv_va_alloc(&va, 0x1000, 0x1000, NULL) ==
              VBS_DRV_ERR_INVALID_ARG,
          "alloc rejects NULL out pointer");
    CHECK(vbs_drv_va_alloc(NULL, 0x1000, 0x1000, &a) ==
              VBS_DRV_ERR_INVALID_ARG,
          "alloc rejects NULL VA space");

    /* Free validation: unknown ranges, double frees, overlaps. */
    CHECK(vbs_drv_va_alloc(&va, 0x1000, 0x1000, &a) == VBS_DRV_OK,
          "alloc for free-validation");
    CHECK(vbs_drv_va_free(&va, VA_BASE + 0x1000, 0x1000) ==
              VBS_DRV_ERR_INVALID_ARG,
          "free rejects a range that was never allocated");
    CHECK(vbs_drv_va_free(&va, a, 0x2000) == VBS_DRV_ERR_INVALID_ARG,
          "free rejects a size overlapping a free block");
    CHECK(vbs_drv_va_free(&va, a - 0x1000, 0x1000) ==
              VBS_DRV_ERR_INVALID_ARG,
          "free rejects an address below the region");
    CHECK(vbs_drv_va_free(&va, a, 0) == VBS_DRV_ERR_INVALID_ARG,
          "free rejects zero size");
    CHECK(vbs_drv_va_free(&va, a, 0x1000) == VBS_DRV_OK,
          "free of the real allocation succeeds");
    CHECK(vbs_drv_va_free(&va, a, 0x1000) == VBS_DRV_ERR_INVALID_ARG,
          "double free is rejected");
    /* The allocator is still intact after all the rejected frees. */
    CHECK(va.free_count == 1, "allocator intact after bad frees");
    CHECK_EQ_U64(va.free_blocks[0].size, VA_SIZE,
                 "whole region free after the valid free");
}

int main(void) {
    test_basic_alloc_free();
    test_alignment();
    test_exhaustion();
    test_coalescing();
    test_validation();
    TEST_SUMMARY("test_valloc");
}
