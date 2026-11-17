/* vbs_fuse.h — model-specialized fused FFN kernels (novel mechanism 2).
 *
 * What: at load time, for one specific model, generate a single C11
 * function that fuses the whole feed-forward block — gate projection,
 * up projection, and SiLU*up — into one pass over the shared Q8
 * activations, with the model's exact dimensions (emb, ffn, block counts)
 * baked in as compile-time constants. Compile it once with the system
 * compiler (-march=native when available), cache the .so on disk keyed by
 * the model file's SHA-256 plus the machine fingerprint, and dlopen it on
 * later runs. Before a fused kernel is ever used, it is verified
 * numerically against the reference FFN path on random inputs; a mismatch
 * refuses the load and the engine falls back to the reference path.
 *
 * Why it is faster than the reference path, concretely:
 *  1. The reference quantizes the normed activations to Q8 twice (once
 *     per matrix() call, gate and up). The fused kernel quantizes once.
 *  2. The reference reads the Q8 activation vector once per row of gate
 *     and once per row of up. The fused kernel reads it once per row pair.
 *  3. The reference writes gate[] and up[] to memory, then re-reads them
 *     to apply SiLU. The fused kernel keeps both in registers.
 *
 * How it differs from the existing GPU specializer (vbs_specializer.c):
 * that emitter sequences GPU dispatch calls into a forward pass for the
 * GPU backends. This one generates a CPU fused kernel with compile-time
 * dimensions for the CPU reference path, with its own cache and its own
 * numeric verification. The two do not overlap.
 */
#ifndef VBS_FUSE_H
#define VBS_FUSE_H

#include "vbs_gguf.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Fused FFN ABI: gate_w/up_w are packed weight rows (rows=ffn, cols=emb);
 * xq/qd/qsum are the Q8-quantized activations from quantize_q8();
 * hidden receives ffn outputs. The function is pure (no globals). */
typedef void (*vbs_fused_ffn_fn)(const uint8_t *gate_w, const uint8_t *up_w,
                                 size_t ffn, size_t emb,
                                 const int8_t *xq, const float *qd,
                                 const float *qsum, float *hidden);

/* Return the loaded fused kernel for this exact (type, emb, ffn), or NULL
 * when none is loaded (or the loaded one is for a different model). */
vbs_fused_ffn_fn vbs_fuse_lookup(vbs_ggml_type_t type, size_t emb, size_t ffn);

/* Enable/disable the loaded kernel without unloading it (default: enabled).
 * Used by the --fuse benchmark to interleave reference and fused rounds. */
void vbs_fuse_set_enabled(bool on);

/* Build (or load from cache) the fused kernel for the model at model_path
 * and install it. cache_dir holds fused_<key>.so + fused_<key>.c.
 * Returns false with a message in err when the kernel cannot be built,
 * loaded, or numerically verified; the engine keeps working on the
 * reference path in that case. */
bool vbs_fuse_build_and_load(const char *model_path, const char *cache_dir,
                             char *err, size_t err_cap);

/* Diagnostics about the last build/load attempt. */
bool vbs_fuse_cache_hit(void);
const char *vbs_fuse_loaded_path(void);
double vbs_fuse_build_seconds(void);
double vbs_fuse_verify_max_rel_err(void);

void vbs_fuse_unload(void);

#ifdef __cplusplus
}
#endif

#endif /* VBS_FUSE_H */
