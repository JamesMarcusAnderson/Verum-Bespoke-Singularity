/* test_backend.c — engine backend wiring tests.
 *
 * Exercises g_vbs_native_backend (the vbs_backend_ops_t table) on a
 * machine with no GPU: init must fail cleanly, every compute entry
 * point must fail cleanly with a clear error, and NULL handling must
 * never crash. On a machine WITH a Turing GPU, init succeeds and the
 * buffer roundtrip is exercised; compilation still fails cleanly
 * (no SASS kernels exist yet — see vbs_backend_native.c).
 */

#include <stdint.h>
#include <string.h>

#include "test_util.h"
#include "vbs_backend.h"

extern const vbs_backend_ops_t g_vbs_native_backend;
extern const char *vbs_native_last_error(void);

static void test_backend_no_gpu(void) {
    const vbs_backend_ops_t *ops = &g_vbs_native_backend;
    CHECK(ops->init != NULL, "backend has init");
    CHECK(ops->get_device_count != NULL, "backend has get_device_count");
    CHECK(ops->get_device != NULL, "backend has get_device");
    CHECK(ops->alloc_buffer != NULL, "backend has alloc_buffer");
    CHECK(ops->compile_primitive_pipeline != NULL,
          "backend has compile_primitive_pipeline");
    CHECK(ops->launch_persistent_kernel == NULL,
          "launch_persistent_kernel is NULL like every other backend");

    bool init_ok = ops->init();
    printf("  (native init: %s)\n", init_ok ? "available" : "unavailable");
    CHECK(vbs_native_last_error() != NULL,
          "last-error string is non-NULL");
    if (!init_ok) {
        CHECK(strstr(vbs_native_last_error(), "no supported GPU") != NULL,
              "last error says no supported GPU was found");
        CHECK(ops->get_device_count() == 0,
              "device count is 0 when init fails");
        CHECK(ops->get_device(0) == NULL,
              "get_device(0) is NULL when init fails");
        /* init is idempotent: a second call gives the same answer. */
        CHECK(ops->init() == false, "second init also fails cleanly");
    } else {
        CHECK(ops->get_device_count() == 1, "device count is 1");
        vbs_device_t *dev = ops->get_device(0);
        CHECK(dev != NULL, "get_device(0) is non-NULL");
        CHECK(ops->get_device(1) == NULL, "get_device(1) is NULL");
        if (dev) {
            CHECK(dev->name[0] != '\0', "device has a name");
            CHECK(dev->max_working_set_bytes > 0,
                  "device reports a working set");
            printf("  (device: %s)\n", dev->name);
            /* Buffer roundtrip on real hardware. */
            vbs_buffer_t *b = ops->alloc_buffer(dev, 4096, false);
            CHECK(b != NULL, "alloc 4 KiB buffer");
            if (b) {
                CHECK(b->host_ptr != NULL, "buffer is host-visible");
                CHECK(b->size == 4096, "buffer size recorded");
                uint8_t src[4096], dst[4096];
                for (size_t i = 0; i < sizeof(src); i++)
                    src[i] = (uint8_t)i;
                CHECK(ops->copy_to_device(b, src, sizeof(src)),
                      "copy to device");
                CHECK(ops->copy_to_host(dst, b, sizeof(dst)),
                      "copy to host");
                CHECK(memcmp(src, dst, sizeof(src)) == 0,
                      "roundtrip preserves bytes");
                CHECK(!ops->copy_to_device(b, src, 8192),
                      "oversize copy is refused");
                ops->free_buffer(b);
            }
            ops->synchronize_device(dev); /* must not crash */
        }
    }
}

static void test_backend_compile_fails_cleanly(void) {
    const vbs_backend_ops_t *ops = &g_vbs_native_backend;
    char err[256];
    memset(err, 0, sizeof(err));
    void *p = ops->compile_primitive_pipeline(
        NULL, VBS_OP_ELEMENTWISE, err, sizeof(err));
    CHECK(p == NULL, "compile_primitive_pipeline returns NULL");
    CHECK(err[0] != '\0', "compile_primitive_pipeline fills err_buf");
    CHECK(strstr(err, "no SASS kernel") != NULL,
          "error message names the missing SASS kernel");
    CHECK(strstr(err, "elementwise") != NULL,
          "error message names the op kind");
    memset(err, 0, sizeof(err));
    p = ops->compile_linear_pipeline(NULL, VBS_MODEL_TYPE_Q8_0, err,
                                     sizeof(err));
    CHECK(p == NULL, "compile_linear_pipeline returns NULL");
    CHECK(strstr(err, "no SASS kernel") != NULL,
          "linear error message names the missing SASS kernel");
    memset(err, 0, sizeof(err));
    p = ops->compile_embed_pipeline(NULL, VBS_MODEL_TYPE_F32, err,
                                    sizeof(err));
    CHECK(p == NULL, "compile_embed_pipeline returns NULL");
    CHECK(strstr(err, "no SASS kernel") != NULL,
          "embed error message names the missing SASS kernel");
    /* NULL err_buf must not crash. */
    CHECK(ops->compile_primitive_pipeline(NULL, VBS_OP_LINEAR, NULL, 0) ==
              NULL,
          "compile with NULL err_buf still returns NULL cleanly");
}

static void test_backend_dispatch_and_nulls(void) {
    const vbs_backend_ops_t *ops = &g_vbs_native_backend;
    CHECK(!ops->dispatch_primitive(NULL, NULL, NULL, 0, 1, 1),
          "dispatch_primitive returns false");
    CHECK(!ops->dispatch_linear_tile(NULL, NULL, NULL, NULL, 0, NULL, NULL,
                                     0, 0, 0, 0, false),
          "dispatch_linear_tile returns false");
    CHECK(ops->alloc_buffer(NULL, 4096, false) == NULL,
          "alloc_buffer with NULL device returns NULL");
    ops->free_buffer(NULL); /* must not crash */
    CHECK(!ops->copy_to_device(NULL, "x", 1),
          "copy_to_device with NULL dst returns false");
    CHECK(!ops->copy_to_host(NULL, NULL, 1),
          "copy_to_host with NULL args returns false");
    ops->synchronize_device(NULL); /* must not crash */
}

int main(void) {
    test_backend_no_gpu();
    test_backend_compile_fails_cleanly();
    test_backend_dispatch_and_nulls();
    TEST_SUMMARY("test_backend");
}
