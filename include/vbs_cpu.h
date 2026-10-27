#ifndef VBS_CPU_H
#define VBS_CPU_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Scalar, non-JIT CPU reference inference for a deliberately restricted
 * LLaMA-style GGUF architecture. See docs/CPU.md for supported tensor types
 * and limitations. Not a general GGUF implementation. */
typedef struct vbs_cpu_model vbs_cpu_model_t;

vbs_cpu_model_t *vbs_cpu_open(const char *path, size_t context_capacity,
                              char *err, size_t err_cap);
void vbs_cpu_close(vbs_cpu_model_t *model);
void vbs_cpu_reset(vbs_cpu_model_t *model);
bool vbs_cpu_step(vbs_cpu_model_t *model, uint32_t token,
                  char *err, size_t err_cap);
bool vbs_cpu_step_ex(vbs_cpu_model_t *model, uint32_t token, bool compute_logits,
                     char *err, size_t err_cap);
const float *vbs_cpu_logits(const vbs_cpu_model_t *model);
size_t vbs_cpu_vocab_size(const vbs_cpu_model_t *model);
uint32_t vbs_cpu_eos_id(const vbs_cpu_model_t *model);
size_t vbs_cpu_context_capacity(const vbs_cpu_model_t *model);

#endif
