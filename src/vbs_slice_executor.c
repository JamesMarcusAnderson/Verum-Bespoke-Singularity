#include "vbs_slice_executor.h"
#include <stdio.h>
#include <stdlib.h>

#define MAX_INT(a,b) ((a) > (b) ? (a) : (b))
#define MIN_INT(a,b) ((a) < (b) ? (a) : (b))

vbs_slice_executor_t *vbs_slice_executor_create(
    const vbs_gguf_model_t *model,
    const vbs_backend_ops_t *backend,
    vbs_device_t *device,
    size_t window_bytes,
    char *err_buf, size_t err_cap
) {
    vbs_slice_executor_t *exec = (vbs_slice_executor_t *)calloc(1, sizeof(vbs_slice_executor_t));
    exec->model = model;
    exec->backend = backend;
    exec->device = device;
    exec->window_bytes = MAX_INT(window_bytes, 4096);

    for (int i = 0; i < 2; i++) {
        exec->stage[i] = backend->alloc_buffer(device, exec->window_bytes, true);
        exec->priv[i]  = backend->alloc_buffer(device, exec->window_bytes, false);
        if (!exec->stage[i] || !exec->priv[i]) {
            snprintf(err_buf, err_cap, "Failed allocating bounded execution windows (%zu bytes)", exec->window_bytes);
            vbs_slice_executor_free(exec);
            return NULL;
        }
    }
    return exec;
}

void vbs_slice_executor_free(vbs_slice_executor_t *exec) {
    if (!exec) return;
    for (int i = 0; i < 2; i++) {
        if (exec->stage[i]) exec->backend->free_buffer(exec->stage[i]);
        if (exec->priv[i])  exec->backend->free_buffer(exec->priv[i]);
    }
    free(exec);
}

bool vbs_slice_executor_run_linear(
    vbs_slice_executor_t *exec,
    const vbs_tensor_t *t,
    vbs_buffer_t *x,
    vbs_buffer_t *y,
    char *err_buf, size_t err_cap
) {
    vbs_block_geometry_t g;
    if (!vbs_get_geometry(t->type, &g)) {
        snprintf(err_buf, err_cap, "Unsupported tensor quantization format %u", t->type);
        return false;
    }

    uint64_t max_blocks = exec->window_bytes / g.bytes_per_block;
    if (!max_blocks) {
        snprintf(err_buf, err_cap, "Staging window is smaller than 1 quant block");
        return false;
    }

    uint64_t col_quantum = g.elements_per_block;
    uint64_t max_cols = MAX_INT(col_quantum, (max_blocks / MAX_INT((uint64_t)1, t->rows)) * col_quantum);
    max_cols = MIN_INT(max_cols, (uint64_t)UINT32_MAX - (UINT32_MAX % col_quantum));
    if (max_cols > t->columns) max_cols = t->columns - (t->columns % col_quantum);
    if (!max_cols) max_cols = col_quantum;

    void *pipeline = exec->backend->compile_linear_pipeline(exec->device, t->type, err_buf, err_cap);
    if (!pipeline) return false;

    bool first = true;
    for (uint64_t cs = 0; cs < t->columns; ) {
        uint64_t cc = MIN_INT(max_cols, t->columns - cs);
        cc -= cc % col_quantum;
        if (!cc) cc = col_quantum;

        uint64_t packed_row = (cc / col_quantum) * g.bytes_per_block;
        uint64_t max_rows = MIN_INT(exec->window_bytes / packed_row, (uint64_t)UINT32_MAX);

        for (uint64_t rs = 0; rs < t->rows; ) {
            uint64_t rc = MIN_INT(max_rows, t->rows - rs);
            size_t slot = (exec->turn++) & 1;
            size_t copied = 0;

            if (!vbs_gguf_copy_packed_tile(exec->model, t, rs, rc, cs, cc,
                                           exec->stage[slot]->host_ptr,
                                           exec->window_bytes, &copied,
                                           err_buf, err_cap)) {
                return false;
            }

            if (!exec->backend->dispatch_linear_tile(
                    exec->device, pipeline,
                    exec->priv[slot], exec->stage[slot], copied,
                    x, y,
                    (uint32_t)cc, (uint32_t)rc, (uint32_t)cs, (uint32_t)rs,
                    !first)) {
                return false;
            }
            rs += rc;
        }
        first = false;
        cs += cc;
    }

    exec->backend->synchronize_device(exec->device);
    return true;
}
