/* test_util.h — tiny shared harness for the driver unit tests.
 *
 * CHECK prints only failures; each test prints a summary line and
 * exits 0 on "ALL PASS", 1 otherwise. No framework, no dependencies.
 */

#ifndef VBS_DRV_TEST_UTIL_H
#define VBS_DRV_TEST_UTIL_H

#include <stdio.h>

static int g_checks;
static int g_fails;

#define CHECK(cond, ...) do { \
    g_checks++; \
    if (!(cond)) { \
        g_fails++; \
        printf("  [FAIL] "); \
        printf(__VA_ARGS__); \
        printf("\n"); \
    } \
} while (0)

#define CHECK_EQ_U32(got, want, what) do { \
    g_checks++; \
    if ((uint32_t)(got) != (uint32_t)(want)) { \
        g_fails++; \
        printf("  [FAIL] %s: got 0x%08x, want 0x%08x\n", \
               (what), (unsigned)(got), (unsigned)(want)); \
    } \
} while (0)

#define CHECK_EQ_U64(got, want, what) do { \
    g_checks++; \
    if ((uint64_t)(got) != (uint64_t)(want)) { \
        g_fails++; \
        printf("  [FAIL] %s: got 0x%016llx, want 0x%016llx\n", \
               (what), (unsigned long long)(got), \
               (unsigned long long)(want)); \
    } \
} while (0)

#define TEST_SUMMARY(name) do { \
    printf("%s: %d checks, %d failures: %s\n", (name), g_checks, g_fails, \
           g_fails == 0 ? "ALL PASS" : "FAILURES PRESENT"); \
    return g_fails == 0 ? 0 : 1; \
} while (0)

#endif /* VBS_DRV_TEST_UTIL_H */
