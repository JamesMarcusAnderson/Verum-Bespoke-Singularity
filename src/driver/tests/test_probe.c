/* test_probe.c — device discovery + framework failure-path tests.
 *
 * On this VM (no GPU, no /dev/dri) probe must find zero devices and
 * open must fail gracefully with NO_DEVICE — never crash, never
 * hang. On a machine WITH a Turing GPU the same test exercises the
 * real open/close path and checks the reported info is sane. The
 * test adapts to whichever machine it runs on.
 *
 * Also covers the framework's NULL-argument handling everywhere a
 * NULL could reach it.
 */

#include <stdint.h>
#include <string.h>

#include "test_drv_helpers.h"
#include "test_util.h"
#include "vbs_drv.h"
#include "vbs_drv_turing.h"

static void test_status_strings(void) {
    static const vbs_drv_status_t all[] = {
        VBS_DRV_OK, VBS_DRV_ERR_NO_DEVICE, VBS_DRV_ERR_NO_MEMORY,
        VBS_DRV_ERR_NOT_SUPPORTED, VBS_DRV_ERR_IO,
        VBS_DRV_ERR_INVALID_ARG, VBS_DRV_ERR_TIMEOUT, VBS_DRV_ERR_BUSY,
        VBS_DRV_ERR_NOT_READY, VBS_DRV_ERR_HW_FAULT,
    };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) {
        const char *s = vbs_drv_status_string(all[i]);
        CHECK(s != NULL && s[0] != '\0',
              "status string for code %d is non-empty", (int)all[i]);
    }
    CHECK(vbs_drv_status_string((vbs_drv_status_t)999) != NULL,
          "status string for an unknown code is non-NULL");
}

static void test_null_safety(void) {
    /* Registration. */
    CHECK(vbs_drv_register_ops(NULL) == VBS_DRV_ERR_INVALID_ARG,
          "register_ops rejects NULL");
    /* Probe. */
    size_t count = 0;
    CHECK(vbs_drv_probe_all(NULL, 1, &count) == VBS_DRV_ERR_INVALID_ARG,
          "probe_all rejects NULL infos");
    vbs_drv_device_info_t infos[1];
    CHECK(vbs_drv_probe_all(infos, 1, NULL) == VBS_DRV_ERR_INVALID_ARG,
          "probe_all rejects NULL count");
    CHECK(vbs_drv_probe_all(infos, 0, &count) == VBS_DRV_ERR_INVALID_ARG,
          "probe_all rejects zero capacity");
    /* Open / close. */
    CHECK(vbs_drv_open(0, NULL) == VBS_DRV_ERR_INVALID_ARG,
          "open rejects NULL out pointer");
    vbs_drv_device_t *dev = NULL;
    CHECK(vbs_drv_open(999, &dev) == VBS_DRV_ERR_NO_DEVICE,
          "open rejects an out-of-range index");
    CHECK(dev == NULL, "failed open leaves the out pointer NULL");
    vbs_drv_close(NULL); /* must not crash */
    /* Buffers. */
    vbs_drv_buffer_t *buf = NULL;
    CHECK(vbs_drv_alloc_buffer(NULL, 4096, 0, &buf) ==
              VBS_DRV_ERR_INVALID_ARG,
          "alloc_buffer rejects NULL device");
    vbs_drv_free_buffer(NULL); /* must not crash */
    CHECK(vbs_drv_map_buffer(NULL) == VBS_DRV_ERR_INVALID_ARG,
          "map_buffer rejects NULL");
    vbs_drv_unmap_buffer(NULL); /* must not crash */
    CHECK(vbs_drv_cpu_begin(NULL, true) == VBS_DRV_ERR_INVALID_ARG,
          "cpu_begin rejects NULL");
    vbs_drv_cpu_end(NULL); /* must not crash */
    CHECK(vbs_drv_copy_to_device(NULL, "x", 1) == VBS_DRV_ERR_INVALID_ARG,
          "copy_to_device rejects NULL dst");
    CHECK(vbs_drv_copy_to_host(NULL, NULL, 1) == VBS_DRV_ERR_INVALID_ARG,
          "copy_to_host rejects NULL args");
    /* Command buffers. */
    vbs_drv_cmdbuf_t cb;
    CHECK(vbs_drv_cmdbuf_init(NULL, 16) == VBS_DRV_ERR_INVALID_ARG,
          "cmdbuf_init rejects NULL");
    CHECK(vbs_drv_cmdbuf_init(&cb, 0) == VBS_DRV_ERR_INVALID_ARG,
          "cmdbuf_init rejects zero capacity");
    CHECK(vbs_drv_cmdbuf_push(NULL, 0) == VBS_DRV_ERR_INVALID_ARG,
          "cmdbuf_push rejects NULL");
    CHECK(vbs_drv_cmdbuf_push_many(NULL, NULL, 0) == VBS_DRV_ERR_INVALID_ARG,
          "cmdbuf_push_many rejects NULL");
    vbs_drv_cmdbuf_free(NULL); /* must not crash */
    vbs_drv_cmdbuf_reset(NULL); /* must not crash */
    /* Submit / fence / launch / synchronize. */
    vbs_drv_fence_t fence = VBS_DRV_FENCE_NONE;
    CHECK(vbs_drv_submit(NULL, &cb, &fence) == VBS_DRV_ERR_INVALID_ARG,
          "submit rejects NULL device");
    CHECK(vbs_drv_fence_wait(NULL, 1, 0) == VBS_DRV_ERR_INVALID_ARG,
          "fence_wait rejects NULL device");
    vbs_drv_fence_free(NULL, 1); /* must not crash */
    CHECK(vbs_drv_submit_and_wait(NULL, &cb, 0) == VBS_DRV_ERR_INVALID_ARG,
          "submit_and_wait rejects NULL device");
    CHECK(vbs_drv_launch(NULL, NULL) == VBS_DRV_ERR_INVALID_ARG,
          "launch rejects NULL device");
    CHECK(vbs_drv_synchronize(NULL) == VBS_DRV_ERR_INVALID_ARG,
          "synchronize rejects NULL device");
    /* Turing last-error is always a valid string. */
    CHECK(vbs_drv_turing_last_error() != NULL,
          "turing last-error string is non-NULL");
}

static void test_probe_and_open(void) {
    CHECK(vbs_drv_turing_register() == VBS_DRV_OK,
          "turing register succeeds");
    CHECK(vbs_drv_turing_register() == VBS_DRV_OK,
          "turing register is idempotent");

    vbs_drv_device_info_t infos[VBS_DRV_MAX_DEVICES];
    size_t count = test_probe_devices(infos, VBS_DRV_MAX_DEVICES);

    if (count == 0) {
        /* No-GPU machine (this VM): open must fail gracefully. */
        test_open_fails_gracefully(0);
        return;
    }

    /* GPU machine: the reported info must be sane, and open/close
     * must work. */
    CHECK(infos[0].name[0] != '\0', "device 0 has a name");
    CHECK(infos[0].pci_device_id != 0, "device 0 has a PCI device id");
    CHECK(infos[0].memory_bytes > 0, "device 0 reports memory");
    printf("  (device 0: %s, %llu MiB)\n", infos[0].name,
           (unsigned long long)(infos[0].memory_bytes >> 20));

    vbs_drv_device_t *dev = test_open_device(0, NULL);
    if (!dev) return;

    /* Buffer alloc/copy roundtrip on real hardware. */
    test_buffer_roundtrip(dev);
    test_close_device(dev);
}

int main(void) {
    test_status_strings();
    test_null_safety();
    test_probe_and_open();
    TEST_SUMMARY("test_probe");
}
