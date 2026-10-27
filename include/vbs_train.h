#ifndef VBS_TRAIN_H
#define VBS_TRAIN_H
#include <stddef.h>
#include <stdbool.h>
/* Backprop, honestly scoped.
 * This is gradient descent through the FINAL layer only:
 *   logits = base_logits + bias
 *   p = softmax(logits), loss = -log p[target]
 *   dL/dlogits = p - onehot(target), bias -= lr * dL/dlogits
 * That IS backprop for the bias term, and the saved bias file is
 * usable in --generate (logits += bias before argmax).
 * It is NOT full transformer backprop: Verum has no gradients for
 * attention / MLP / embeddings, and GGUF weights are mostly quantized,
 * so they cannot be SGD-updated in place here. A bias adapter is the
 * trainable part that can be saved, loaded, and verified without
 * pretending to fine-tune every weight.
 */
typedef struct { size_t vocab; float *bias; } vbs_adapter_t;
vbs_adapter_t *vbs_adapter_create(size_t vocab);
void vbs_adapter_free(vbs_adapter_t *a);
bool vbs_adapter_save(const vbs_adapter_t *a, const char *path);
bool vbs_adapter_load(vbs_adapter_t *a, const char *path);
void vbs_adapter_apply(const vbs_adapter_t *a, float *logits);
double vbs_train_bias_step(vbs_adapter_t *a, const float *base_logits, size_t target, float lr, double *prob_out);
/* Full linear-layer backprop, for unit test / future output-layer use:
 * logits[i] = dot(W[i], h) + b[i]; W is vocab x hidden row-major.
 * Returns CE loss, updates W and b in place. */
double vbs_train_linear_step(float *W, float *b, size_t vocab, size_t hidden, const float *h, size_t target, float lr);
#endif
