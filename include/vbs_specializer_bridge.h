#ifndef VBS_SPECIALIZER_BRIDGE_H
#define VBS_SPECIALIZER_BRIDGE_H

#include "vbs_backend.h"
#include "vbs_fabric.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void vbs_bridge_bind_runtime(vbs_device_t *dev, const vbs_backend_ops_t *backend, vbs_fabric_t *fabric);
void vbs_bridge_set_activations(vbs_buffer_t *a, vbs_buffer_t *b, vbs_buffer_t *kv_k, vbs_buffer_t *kv_v, vbs_buffer_t *logits);
void vbs_bridge_set_scratch(vbs_buffer_t **bufs, size_t n);

// Named scratch buffers the generated forward pass routes through.
#define VBS_SCRATCH_Q      0
#define VBS_SCRATCH_K      1
#define VBS_SCRATCH_V      2
#define VBS_SCRATCH_ATTN   3
#define VBS_SCRATCH_GATE   4
#define VBS_SCRATCH_UP     5
#define VBS_SCRATCH_HIDDEN 6
#define VBS_SCRATCH_NORMED 7
#define VBS_SCRATCH_COUNT  8

void *vbs_gpu_get_act_buffer_a(void);
void *vbs_gpu_get_act_buffer_b(void);
void *vbs_gpu_get_kv_k(void);
void *vbs_gpu_get_kv_v(void);
void *vbs_gpu_get_logits(void);
void *vbs_gpu_get_scratch(size_t index);
void *vbs_bridge_get_block_dptr(size_t index);

void vbs_gpu_dispatch_vram_q8_0(void *w, void *in, void *out, size_t r, size_t c);
void vbs_gpu_dispatch_vram_q4_0(void *w, void *in, void *out, size_t r, size_t c);
void vbs_gpu_dispatch_vram_f32(void *w, void *in, void *out, size_t r, size_t c);

// Primitive dispatch argument order is canonical — it matches the PTX and
// MSL kernel signatures slot for slot (B = buffer, S = scalar):
//   embed: B w, S token(u32), B out, S dim(u32)
//   norm:  B in, B w, B out, S dim(u32), S eps(f32)
//   rope:  B x, S pos(u32), S head_dim(u32), S n_heads(u32), S theta(f32)
//   cache: B k, B v, B kc, B vc, S pos(u32), S layer(u32), S max_ctx(u32), S kv_dim(u32)
//   attn:  B q, B kc, B vc, B out, S seq_len(u32), S layer(u32), S max_ctx(u32),
//          S n_q_heads(u32), S n_kv_heads(u32), S head_dim(u32), S scale(f32)
//   silu:  B gate, B up, B out, S n(u32)
//   add:   B a, B b, B out, S n(u32)
void vbs_gpu_dispatch_embed(void *w, uint32_t token, void *out, size_t dim);
void vbs_gpu_dispatch_norm(void *w, void *in, void *out, size_t dim, float eps);
void vbs_gpu_dispatch_rope(void *in, uint32_t pos, size_t hdim, size_t nh, float theta);
void vbs_gpu_dispatch_cache(void *k, void *v, void *kc, void *vc, uint32_t pos, uint32_t layer, uint32_t max_ctx, size_t kv_dim);
void vbs_gpu_dispatch_attn(void *q, void *k_cache, void *v_cache, void *out, uint32_t pos, uint32_t layer, uint32_t max_ctx, size_t nqh, size_t nkh, size_t hd);
void vbs_gpu_dispatch_silu(void *gate, void *up, void *out, size_t n);
void vbs_gpu_dispatch_add(void *a, void *b, void *out, size_t n);

void vbs_gpu_readback_logits(void *d_src, float *h_dst, size_t n);

#ifdef __cplusplus
}
#endif

#endif // VBS_SPECIALIZER_BRIDGE_H
