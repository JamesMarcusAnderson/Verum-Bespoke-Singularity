#ifndef VBS_AUTONOMY_H
#define VBS_AUTONOMY_H
#include "vbs_block.h"
/* Autonomous learning loop, in C, inside the engine itself.
 * No epochs/lr given by the user: it trains until loss < target
 * or max_epochs, and adjusts its own learning rate when loss stalls.
 * Returns final loss, or -1 on error. */
double vbs_autonomous_train(vbs_block_t *b, const int *pattern, int pattern_len, double target_loss, int max_epochs);
#endif
