/* test_drv_helpers.h — shared setup blocks for the driver unit tests.
 *
 * The three probe tests (Turing / RDNA 2 / Alchemist) and the QMD
 * test share the same probe/open/roundtrip/launch scaffolding; it
 * lives here once instead of being repeated in each test file.
 * Helpers use CHECK from test_util.h, so every test's check count
 * stays the same as before the factoring.
 */

#ifndef VBS_DRV_TEST_DRV_HELPERS_H
#define VBS_DRV_TEST_DRV_HELPERS_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "test_util.h"
#include "vbs_drv.h"

/* Probe all devices and print the count the way the probe tests do.
 * Returns the number of devices found. */
static inline size_t test_probe_devices(vbs_drv_device_info_t *infos,
                                        size_t cap) {
    size_t count = 0;
    CHECK(vbs_drv_probe_all(infos, cap, &count) == VBS_DRV_OK,
          "probe_all succeeds");
    printf("  (probe found %zu device%s)\n", count, count == 1 ? "" : "s");
    return count;
}

/* Open index and expect a graceful NO_DEVICE failure that leaves
 * the out pointer NULL. */
static inline void test_open_fails_gracefully(size_t index) {
    vbs_drv_device_t *dev = NULL;
    CHECK(vbs_drv_open(index, &dev) == VBS_DRV_ERR_NO_DEVICE,
          "open with no device fails with NO_DEVICE");
    CHECK(dev == NULL, "failed open leaves dev NULL");
}

/* Open device index, check it has ops (with the expected name
 * when ops_name is non-NULL) and a ready VA space. Returns the
 * device, or NULL when the open failed. */
static inline vbs_drv_device_t *test_open_device(size_t index,
                                                 const char *ops_name) {
    vbs_drv_device_t *dev = NULL;
    CHECK(vbs_drv_open(index, &dev) == VBS_DRV_OK,
          "open device succeeds");
    if (!dev) return NULL;
    CHECK(dev->ops != NULL, "opened device has ops");
    if (ops_name)
        CHECK(dev->ops->name != NULL &&
                  strcmp(dev->ops->name, ops_name) == 0,
              "opened device uses the expected ops");
    CHECK(dev->va_space.ready, "opened device has a VA space");
    return dev;
}

/* Finish with an open device: synchronize, then close. */
static inline void test_close_device(vbs_drv_device_t *dev) {
    CHECK(vbs_drv_synchronize(dev) == VBS_DRV_OK, "synchronize succeeds");
    vbs_drv_close(dev);
}

/* 4 KiB host-visible alloc + copy roundtrip + oversized-copy
 * rejection, on an already-open device. */
static inline void test_buffer_roundtrip(vbs_drv_device_t *dev) {
    vbs_drv_buffer_t *buf = NULL;
    CHECK(vbs_drv_alloc_buffer(dev, 4096,
                               VBS_DRV_BUF_HOST_VISIBLE |
                                   VBS_DRV_BUF_DEVICE_LOCAL,
                               &buf) == VBS_DRV_OK,
          "alloc 4 KiB host-visible buffer");
    if (buf) {
        CHECK(buf->gpu_va != 0, "buffer has a GPU VA");
        CHECK(buf->host_ptr != NULL, "buffer has a host mapping");
        uint8_t pattern[4096];
        uint8_t back[4096];
        for (size_t i = 0; i < sizeof(pattern); i++)
            pattern[i] = (uint8_t)(i * 2654435761u >> 24);
        CHECK(vbs_drv_copy_to_device(buf, pattern, sizeof(pattern)) ==
                  VBS_DRV_OK,
              "copy 4 KiB to the device");
        CHECK(vbs_drv_copy_to_host(back, buf, sizeof(back)) == VBS_DRV_OK,
              "copy 4 KiB back to the host");
        CHECK(memcmp(pattern, back, sizeof(pattern)) == 0,
              "copy roundtrip preserves bytes");
        CHECK(vbs_drv_copy_to_device(buf, pattern, 8192) ==
                  VBS_DRV_ERR_INVALID_ARG,
              "copy past the buffer end is rejected");
        vbs_drv_free_buffer(buf);
    }
}

/* Fill the standard vector-add launch: code at VA 0x100000000,
 * CB0 at 0x200000000 (256 bytes), semaphore at 0x300000000
 * (payload 0xC0FFEE), grid 1024 x 1 x 1, block 256 x 1 x 1, with a
 * semaphore release. code_size, register_count, and barrier_count
 * vary per test. */
static inline void test_vecadd_launch(vbs_drv_buffer_t *code,
                                      vbs_drv_buffer_t *cb0,
                                      vbs_drv_buffer_t *sem,
                                      vbs_drv_launch_desc_t *desc,
                                      size_t code_size,
                                      uint32_t register_count,
                                      uint32_t barrier_count) {
    memset(code, 0, sizeof(*code));
    code->gpu_va = 0x100000000ULL;
    code->size = code_size;
    memset(cb0, 0, sizeof(*cb0));
    cb0->gpu_va = 0x200000000ULL;
    cb0->size = 256;
    memset(sem, 0, sizeof(*sem));
    sem->gpu_va = 0x300000000ULL;
    sem->size = 4096;
    memset(desc, 0, sizeof(*desc));
    desc->code = code;
    desc->code_offset = 0;
    desc->code_size = code_size;
    desc->register_count = register_count;
    desc->shared_mem_bytes = 0;
    desc->barrier_count = barrier_count;
    desc->grid_x = 1024; desc->grid_y = 1; desc->grid_z = 1;
    desc->block_x = 256; desc->block_y = 1; desc->block_z = 1;
    desc->const_bufs[0] = cb0;
    desc->const_buf_count = 1;
    desc->sem_buf = sem;
    desc->sem_offset = 0;
    desc->sem_payload = 0xC0FFEE;
    desc->use_semaphore = true;
}

#endif /* VBS_DRV_TEST_DRV_HELPERS_H */
