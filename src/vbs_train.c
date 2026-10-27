#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "vbs_train.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
vbs_adapter_t *vbs_adapter_create(size_t vocab){ vbs_adapter_t *a=calloc(1,sizeof *a); if(!a) return NULL; a->vocab=vocab; a->bias=calloc(vocab,sizeof(float)); if(!a->bias){free(a);return NULL;} return a; }
void vbs_adapter_free(vbs_adapter_t *a){ if(a){free(a->bias);free(a);} }
bool vbs_adapter_save(const vbs_adapter_t *a, const char *path){ FILE *f=fopen(path,"wb"); if(!f) return false; fwrite(&a->vocab,sizeof(size_t),1,f); fwrite(a->bias,sizeof(float),a->vocab,f); fclose(f); return true; }
bool vbs_adapter_load(vbs_adapter_t *a, const char *path){ FILE *f=fopen(path,"rb"); if(!f) return false; size_t v=0; if(fread(&v,sizeof(size_t),1,f)!=1){fclose(f);return false;} if(v!=a->vocab){fclose(f);return false;} int ok=fread(a->bias,sizeof(float),a->vocab,f)==a->vocab; fclose(f); return ok; }
void vbs_adapter_apply(const vbs_adapter_t *a, float *logits){ for(size_t i=0;i<a->vocab;i++) logits[i]+=a->bias[i]; }
static double softmax_ce(const float *logits, size_t vocab, size_t target, float *prob){
    float mx=logits[0]; for(size_t i=1;i<vocab;i++) if(logits[i]>mx) mx=logits[i];
    double sum=0; for(size_t i=0;i<vocab;i++) sum+=exp((double)logits[i]-mx);
    double p=exp((double)logits[target]-mx)/sum; if(prob) *prob=p; return -log(p);
}
double vbs_train_bias_step(vbs_adapter_t *a, const float *base_logits, size_t target, float lr, double *prob_out){
    float *logits=malloc(a->vocab*sizeof(float)); if(!logits) return -1;
    for(size_t i=0;i<a->vocab;i++) logits[i]=base_logits[i]+a->bias[i];
    float mx=logits[0]; for(size_t i=1;i<a->vocab;i++) if(logits[i]>mx) mx=logits[i];
    double sum=0; for(size_t i=0;i<a->vocab;i++) sum+=exp((double)logits[i]-mx);
    double loss=-log(exp((double)logits[target]-mx)/sum);
    for(size_t i=0;i<a->vocab;i++){
        double p=exp((double)logits[i]-mx)/sum;
        double grad=p - (i==target ? 1.0 : 0.0);
        a->bias[i] -= (float)(lr*grad);
    }
    if(prob_out) *prob_out=exp((double)logits[target]-mx)/sum;
    free(logits); return loss;
}
double vbs_train_linear_step(float *W, float *b, size_t vocab, size_t hidden, const float *h, size_t target, float lr){
    float *logits=malloc(vocab*sizeof(float)); if(!logits) return -1;
    for(size_t i=0;i<vocab;i++){ double s=b?b[i]:0; for(size_t j=0;j<hidden;j++) s+=(double)W[i*hidden+j]*h[j]; logits[i]=(float)s; }
    double loss=softmax_ce(logits,vocab,target,NULL);
    float mx=logits[0]; for(size_t i=1;i<vocab;i++) if(logits[i]>mx) mx=logits[i];
    double sum=0; for(size_t i=0;i<vocab;i++) sum+=exp((double)logits[i]-mx);
    for(size_t i=0;i<vocab;i++){
        double p=exp((double)logits[i]-mx)/sum;
        double grad=p - (i==target ? 1.0 : 0.0);
        if(b) b[i]-=(float)(lr*grad);
        for(size_t j=0;j<hidden;j++) W[i*hidden+j]-=(float)(lr*grad*h[j]);
    }
    free(logits); return loss;
}
