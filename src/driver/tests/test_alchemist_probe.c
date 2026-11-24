/* test_alchemist_probe.c — Alchemist device discovery + failure-path tests.
 *
 * On this VM (no GPU, no /dev/dri) probe must find zero devices and
 * open must fail gracefully with NO_DEVICE — never crash, never
 * hang. On a machine WITH an Alchemist GPU the same test exercises
 * the real open/close path and checks the reported info is sane.
 * The test adapts to whichever machine it runs on.
 *
 * The first probe runs WITHOUT an explicit vbs_drv_alchemist_register()
 * call: the framework auto-registers built-in vendors on first
 * probe, so Alchemist devices must be visible to a caller that only
 * uses vbs_drv_probe_all (this is how the engine backend finds
 * them). Explicit registration is then tested for idempotency.
 */

#include <stdint.h>
#include <string.h>

#include "test_drv_helpers.h"
#include "test_util.h"
#include "vbs_drv.h"
#include "vbs_drv_alchemist.h"

static void test_register_and_error_string(void) {
    CHECK(vbs_drv_alchemist_last_error() != NULL,
          "alchemist last-error string is non-NULL");
    CHECK(vbs_drv_alchemist_register() == VBS_DRV_OK,
          "alchemist register succeeds");
    CHECK(vbs_drv_alchemist_register() == VBS_DRV_OK,
          "alchemist register is idempotent");
}

static void test_probe_and_open(void) {
    /* NOTE: no explicit register before this probe — the framework
     * auto-registers the Alchemist ops on first probe. */
    vbs_drv_device_info_t infos[VBS_DRV_MAX_DEVICES];
    size_t count = test_probe_devices(infos, VBS_DRV_MAX_DEVICES);

    /* Find the first Alchemist device, if any (other vendors' devices
     * may also be present). */
    size_t alchemist_index = (size_t)-1;
    for (size_t i = 0; i < count; i++) {
        if (infos[i].pci_vendor_id == VBS_DRV_ALCHEMIST_PCI_VENDOR) {
            alchemist_index = i;
            break;
        }
    }

    if (alchemist_index == (size_t)-1) {
        /* No Alchemist GPU on this machine (this VM): opening an
         * out-of-range index must fail gracefully. */
        test_open_fails_gracefully(count);
        return;
    }

    /* Alchemist GPU machine: the reported info must be sane, and
     * open/close must work. */
    CHECK(infos[alchemist_index].name[0] != '\0', "Alchemist device has a name");
    CHECK(infos[alchemist_index].pci_device_id != 0,
          "Alchemist device has a PCI device id");
    CHECK(strstr(infos[alchemist_index].name, "Alchemist") != NULL,
          "Alchemist device name mentions Alchemist");
    printf("  (Alchemist device %zu: %s, %llu MiB)\n", alchemist_index,
           infos[alchemist_index].name,
           (unsigned long long)(infos[alchemist_index].memory_bytes >> 20));

    vbs_drv_device_t *dev =
        test_open_device(alchemist_index, "alchemist");
    if (!dev) return;

    /* Buffer alloc/copy roundtrip on real hardware. */
    test_buffer_roundtrip(dev);
    test_close_device(dev);
}

int main(void) {
    /* Probe FIRST: at this point nothing has registered the Alchemist
     * ops yet, so this exercises the framework's auto-registration
     * of built-in vendors. Explicit registration is tested after. */
    test_probe_and_open();
    test_register_and_error_string();
    TEST_SUMMARY("test_alchemist_probe");
}
