/* vbs_autonomy.c — self-directed training loop.
 *
 * Trains a small block on every 3-token window of a repeating
 * pattern until the average loss falls below the target, or the
 * epoch budget runs out. The loop watches for stalls: if the loss
 * fails to improve by more than 1e-4 for 50 epochs in a row, the
 * learning rate is halved. If the rate falls below 0.001 the loop
 * gives up — further training at that rate would not move the loss. */

#include "vbs_autonomy.h"
#include <stdio.h>

double vbs_autonomous_train(vbs_block_t *b, const int *pattern,
                            int pattern_len, double target_loss,
                            int max_epochs) {
    float lr = 0.05f;
    double best = 1e9;
    double loss = 1e9;
    int stall = 0;
    int n_spans = pattern_len - 3;

    for (int ep = 0; ep < max_epochs; ep++) {
        double total = 0.0;
        for (int i = 0; i + 3 < pattern_len; i++) {
            int window[3] = { pattern[i], pattern[i + 1], pattern[i + 2] };
            total += vbs_block_train_step(b, window, pattern[i + 3], lr);
        }
        loss = total / n_spans;

        if (loss < best - 1e-4) {
            best = loss;
            stall = 0;
        } else {
            stall++;
        }
        if (loss < target_loss) break;
        if (stall >= 50) {
            lr *= 0.5f;
            stall = 0;
            printf("autonomy: loss stalled, lr -> %.4f at epoch %d\n", lr, ep + 1);
            if (lr < 0.001f) break;
        }
    }
    printf("autonomy: final loss=%.4f lr=%.4f\n", loss, lr);
    return loss;
}
