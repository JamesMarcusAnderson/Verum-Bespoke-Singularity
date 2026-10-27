#include "vbs_autonomy.h"
#include <stdio.h>
double vbs_autonomous_train(vbs_block_t *b, const int *pattern, int pattern_len, double target_loss, int max_epochs){
    float lr = 0.05f;
    double best = 1e9, loss = 1e9;
    int stall = 0;
    for(int ep=0; ep<max_epochs; ep++){
        loss = 0;
        for(int i=0; i+3<pattern_len; i++){
            int win[3]={pattern[i],pattern[i+1],pattern[i+2]};
            loss += vbs_block_train_step(b, win, pattern[i+3], lr);
        }
        loss /= (pattern_len-3);
        if(loss < best-1e-4){ best=loss; stall=0; } else { stall++; }
        if(loss < target_loss) break;
        if(stall>=50){ lr *= 0.5f; stall=0; printf("autonomy: loss stalled, lr -> %.4f at epoch %d\n", lr, ep+1); if(lr<0.001f) break; }
    }
    printf("autonomy: final loss=%.4f lr=%.4f\n", loss, lr);
    return loss;
}
