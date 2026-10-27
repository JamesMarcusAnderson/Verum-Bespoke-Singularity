#ifndef VBS_BLOCK_H
#define VBS_BLOCK_H
#include <stddef.h>
/* Full transformer-block backprop, trainable in C.
 * 1 block, single head, causal attention, ReLU MLP, LayerNorms.
 * Forward caches everything; backward runs the exact chain rule
 * in reverse: head -> final LN -> MLP -> LN2 -> attention -> LN1
 * -> embeddings. All parameters are FP32 and SGD-updated in place.
 * This is a standalone trainable block, not a retrofit of the
 * quantized GGUF inference kernels in vbs_engine.c — those kernels
 * have no gradient storage, so they cannot be backpropped in place.
 */
typedef struct {
    int d, vocab, seq, hidden;
    float *E, *P;
    float *Wq, *Wk, *Wv, *Wo;
    float *W1, *b1, *W2, *b2;
    float *Wout, *bout;
    float *g1, *be1, *g2, *be2, *gf, *bef;
} vbs_block_t;
vbs_block_t *vbs_block_create(int vocab, int d, int seq, int hidden, unsigned seed);
void vbs_block_free(vbs_block_t *b);
/* One SGD step on next-token prediction for the last position.
 * Returns CE loss before the update. */
double vbs_block_train_step(vbs_block_t *b, const int *tokens, int target, float lr);
#endif
/* Inference only: greedy next-token prediction from a trained block. */
int vbs_block_predict(vbs_block_t *b, const int *tokens);
void vbs_block_save(vbs_block_t *b, const char *path);
int vbs_block_load(vbs_block_t *b, const char *path);
double vbs_block_loss(vbs_block_t *b, const int *tokens, int target);
