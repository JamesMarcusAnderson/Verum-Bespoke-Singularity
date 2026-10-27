#include "vbs_scheduler.h"
#include <math.h>
float vbs_scheduler_lr(float base_lr, int epoch){
    /* Decay 1% every 100 epochs, floor at 10% of base. */
    float f = powf(0.99f, (float)(epoch/100));
    if(f < 0.1f) f = 0.1f;
    return base_lr * f;
}
