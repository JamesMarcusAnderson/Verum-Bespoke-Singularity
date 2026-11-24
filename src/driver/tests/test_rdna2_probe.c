/* test_rdna2_probe.c — RDNA 2 device discovery + open failure-path tests.
 *
 * On this VM (no GPU, no /dev/dri) probe must find zero devices and
 * open must fail gracefully with NO_DEVICE — never crash, never
 * hang. On a machine WITH a Navi 21 GPU the same test exercises
 * the real open/close path and checks the reported info is sane,
 * plus a buffer alloc/copy roundtrip and synchronize. The test
 * adapts to whichever machine it runs on.
 */

#include <stdint.h>
#include <string.h>

#include "test_drv_helpers.h"
#include "test_util.h"
#include "vbs_drv.h"
#include "vbs_drv_rdna2.h"

static void test_registration(void) {
    CHECK(vbs_drv_rdna2_register() == VBS_DRV_OK,
          "rdna2 register succeeds");
    CHECK(vbs_drv_rdna2_register() == VBS_DRV_OK,
          "rdna2 register is idempotent");
    CHECK(vbs_drv_rdna2_last_error() != NULL,
          "rdna2 last-error string is non-NULL");
}

static void test_probe_and_open(void) {
    vbs_drv_device_info_t infos[VBS_DRV_MAX_DEVICES];
    size_t count = test_probe_devices(infos, VBS_DRV_MAX_DEVICES);

    if (count == 0) {
        /* No-GPU machine (this VM): open must fail gracefully. */
        test_open_fails_gracefully(0);
        /* The failure must not have crashed or hung, and the
         * last-error string stays readable. */
        CHECK(vbs_drv_rdna2_last_error() != NULL,
              "last-error string is readable after a failed open");
        return;
    }

    /* GPU machine: the reported info must be sane, and open/close
     * must work. */
    CHECK(infos[0].name[0] != '\0', "device 0 has a name");
    CHECK(strstr(infos[0].name, "Navi 21") != NULL,
          "device 0 name identifies Navi 21");
    CHECK(infos[0].pci_device_id != 0, "device 0 has a PCI device id");
    CHECK(infos[0].compute_units > 0, "device 0 reports active CUs");
    CHECK(infos[0].memory_bytes > 0, "device 0 reports memory");
    printf("  (device 0: %s, %u CUs, %llu MiB)\n", infos[0].name,
           infos[0].compute_units,
           (unsigned long long)(infos[0].memory_bytes >> 20));

    vbs_drv_device_t *dev = test_open_device(0, "rdna2");
    if (!dev) return;
    CHECK(dev->va_base != 0, "VA space has a nonzero base");

    /* Buffer alloc/copy roundtrip on real hardware. */
    test_buffer_roundtrip(dev);
    /* A device-local (VRAM) buffer without a CPU mapping. */
    vbs_drv_buffer_t *buf = NULL;
    CHECK(vbs_drv_alloc_buffer(dev, 4096, VBS_DRV_BUF_DEVICE_LOCAL,
                               &buf) == VBS_DRV_OK,
          "alloc 4 KiB device-local buffer");
    if (buf) {
        CHECK(buf->gpu_va != 0, "device-local buffer has a GPU VA");
        CHECK(buf->host_ptr == NULL,
              "device-local buffer has no host mapping");
        vbs_drv_free_buffer(buf);
    }
    /* A launch with no code buffer must fail validation, never
     * reach the hardware. */
    vbs_drv_launch_desc_t desc;
    memset(&desc, 0, sizeof(desc));
    CHECK(vbs_drv_launch(dev, &desc) == VBS_DRV_ERR_INVALID_ARG,
          "launch with an empty description is rejected");
    test_close_device(dev);
}

int main(void) {
    test_registration();
    test_probe_and_open();
    TEST_SUMMARY("test_rdna2_probe");
}
