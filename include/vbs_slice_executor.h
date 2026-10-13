#ifndef VBS_SLICE_EXECUTOR_H
#define VBS_SLICE_EXECUTOR_H

#include "vbs_gguf.h"
#include "vbs_backend.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const vbs_gguf_model_t *model;
    const vbs_backend_ops_t *backend;
    vbs_device_t *device;
    size_t window_bytes;

    // Double buffered staging and private windows
    vbs_buffer_t *stage[2];
    vbs_buffer_t *priv[2];
    size_t turn;
} vbs_slice_executor_t;

vbs_slice_executor_t *vbs_slice_executor_create(
    const vbs_gguf_model_t *model,
    const vbs_backend_ops_t *backend,
    vbs_device_t *device,
    size_t window_bytes,
    char *err_buf, size_t err_cap
);

void vbs_slice_executor_free(vbs_slice_executor_t *exec);

bool vbs_slice_executor_run_linear(
    vbs_slice_executor_t *exec,
    const vbs_tensor_t *tensor,
    vbs_buffer_t *input,
    vbs_buffer_t *output,
    char *err_buf, size_t err_cap
);

#ifdef __cplusplus
}
#endif

#endif // VBS_SLICE_EXECUTOR_H
