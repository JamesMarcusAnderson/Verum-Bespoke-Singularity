/* vbs_scheduler.c — learning-rate schedule.
 *
 * The rate decays by 1% per 100 epochs, with a floor at 10% of the
 * base rate, so late training never stalls completely. The decay
 * counts whole 100-epoch blocks (integer division), so the rate is
 * constant within each block and steps down at block boundaries. */

#include "vbs_scheduler.h"
#include <math.h>

float vbs_scheduler_lr(float base_lr, int epoch) {
    int blocks = epoch / 100;
    float factor = powf(0.99f, (float)blocks);
    if (factor < 0.1f) factor = 0.1f;
    return base_lr * factor;
}
