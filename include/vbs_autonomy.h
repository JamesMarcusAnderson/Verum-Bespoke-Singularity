#ifndef VBS_AUTONOMY_H
#define VBS_AUTONOMY_H

#include "vbs_block.h"

/* Self-directed training, in C, inside the engine itself.
 * The caller gives a pattern and a loss target; the loop picks its
 * own epoch count and learning rate. It stops when the loss drops
 * below target_loss or max_epochs is reached, and halves its own
 * learning rate whenever the loss stalls. Returns the final loss. */
double vbs_autonomous_train(vbs_block_t *b, const int *pattern,
                            int pattern_len, double target_loss,
                            int max_epochs);

#endif /* VBS_AUTONOMY_H */
