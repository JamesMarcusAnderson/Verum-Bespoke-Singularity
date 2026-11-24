/* vbs_dequant.h — dequantization: turning packed weight blocks
 * into floats, and whole tensors into a form the engine can run.
 *
 * Where does the engine stand on a weight type?
 *  - native kernel:   runs directly on GPU in its packed form
 *  - dequantizable:   no native kernel, but convertible to one
 *  - neither:         the engine refuses to run the model (loudly)
 */
#ifndef VBS_DEQUANT_H
#define VBS_DEQUANT_H

#include "vbs_model.h"
#include "vbs_f16.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

bool vbs_dequant_has_native_kernel(vbs_model_type_t t);
bool vbs_dequant_supported(vbs_model_type_t t);

/* Dequantize exactly one block of `type` into `out` (32 or 256
 * floats, depending on the block geometry). Returns the number of
 * values written, or 0 if the type is unsupported. Block layouts
 * follow the model file format's public tensor-encoding
 * specification. */
int vbs_dequant_block(vbs_model_type_t type, const void *block, float *out_values);

/* Requantize one block of 32 floats into a Q8_0 block: f16 scale
 * followed by 32 int8 values (34 bytes total). */
void vbs_requant_q8_0_block(const float *values32, void *out_block);

/* Convert an entire tensor from its stored format into `target`
 * (Q8_0 or F16), optionally multiplying every value by `scale`
 * (used to bake Gemma's sqrt(hidden) embedding scale into the
 * data).
 *
 * On success returns a malloc'd buffer the caller owns, and fills
 * *out_type (== target), *out_packed_row_bytes, *out_total_bytes.
 * The source is read from `src` (packed, row-major, t->rows rows
 * of t->columns elements). Returns NULL and fills `err` on
 * failure. */

/* Arguments for a whole-tensor conversion: the source tensor and
 * data, the target type and optional value scale, and where the
 * results go. */
typedef struct {
    const vbs_tensor_t *t;
    const void *src;
    vbs_model_type_t target;
    float scale;
    vbs_model_type_t *out_type;
    uint64_t *out_packed_row_bytes;
    uint64_t *out_total_bytes;
    char *err;
    size_t err_cap;
} vbs_dequant_request_t;

void *vbs_dequant_tensor(const vbs_dequant_request_t *req);

/* I8G: internal 1-byte/elem target with one f32 scale per row.
 * Buffer = [scales: rows * f32][quants: rows * cols * int8]. */
void *vbs_dequant_tensor_i8g(
    const vbs_tensor_t *t, const void *src, float scale,
    vbs_model_type_t *out_type, uint64_t *out_packed_row_bytes,
    uint64_t *out_total_bytes, char *err, size_t err_cap
);

/* Element-mean of (row amax / 32-block amax): 1.0 = I8G is
 * quality-equivalent to Q8_0 for this tensor; large = outlier
 * rows would inflate quantization error. The engine gates
 * AUTO-selection on this and prints it. */
double vbs_dequant_i8g_quality_ratio(const vbs_tensor_t *t, const void *src);

/* Half-precision helpers (vbs_f16_to_f32 / vbs_f32_to_f16) are
 * defined in vbs_f16.h, included above — the single canonical
 * copy, shared with the hot-kernel modules. */

#ifdef __cplusplus
}
#endif

#endif /* VBS_DEQUANT_H */
