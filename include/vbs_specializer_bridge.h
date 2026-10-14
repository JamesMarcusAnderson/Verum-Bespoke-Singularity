#ifndef VBS_SPECIALIZER_BRIDGE_H
#define VBS_SPECIALIZER_BRIDGE_H

#include "vbs_backend.h"
#include "vbs_fabric.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Bridge setup
void vbs_bridge_bind_runtime(vbs_device_t *dev, const vbs_backend_ops_t *backend, vbs_fabric_t *fabric);
void vbs_bridge_set_activations(vbs_buffer_t *a, vbs_buffer_t *b, vbs_buffer_t *kv_k, vbs_buffer_t *kv_v, vbs_buffer_t *logits);

// Symbols dynamically invoked by the JIT .so module
void *vbs_gpu_get_act_buffer_a(void);
void *vbs_gpu_get_act_buffer_b(void);
void *vbs_bridge_get_block_dptr(size_t index);

void vbs_gpu_dispatch_vram_q8_0(void *w, void *in, void *out, size_t r, size_t c);
void vbs_gpu_dispatch_vram_q4_0(void *w, void *in, void *out, size_t r, size_t c);
void vbs_gpu_dispatch_vram_f32(void *w, void *in, void *out, size_t r, size_t c);

void vbs_gpu_dispatch_embed(void *w, uint32_t token, void *out, size_t dim);
void vbs_gpu_dispatch_norm(void *w, void *in, void *out, size_t dim, float eps);
void vbs_gpu_dispatch_rope(void *in, uint32_t pos, size_t hdim, size_t nh, float theta);
void vbs_gpu_dispatch_attn(void *q, void *k_cache, void *v_cache, void *out, uint32_t pos, size_t nqh, size_t nkh, size_t hd);
void vbs_gpu_dispatch_silu(void *gate, void *up, void *out, size_t n);
void vbs_gpu_dispatch_add(void *a, void *b, void *out, size_t n);

void vbs_gpu_readback_logits(void *d_src, float *h_dst, size_t n);

#ifdef __cplusplus
}
#endif

#endif // VBS_SPECIALIZER_BRIDGE_H
