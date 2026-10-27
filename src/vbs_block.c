#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "vbs_block.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
static float rnd(unsigned *s){ *s=*s*1664525u+1013904223u; return ((float)(*s>>8)/16777216.0f-0.5f)*0.2f; }
static float *alloc(int n, unsigned *s){ float *p=calloc(n,sizeof(float)); for(int i=0;i<n;i++) p[i]=rnd(s); return p; }
vbs_block_t *vbs_block_create(int vocab,int d,int seq,int hidden,unsigned seed){
    vbs_block_t *b=calloc(1,sizeof *b); b->vocab=vocab; b->d=d; b->seq=seq; b->hidden=hidden;
    b->E=alloc(vocab*d,&seed); b->P=alloc(seq*d,&seed);
    b->Wq=alloc(d*d,&seed); b->Wk=alloc(d*d,&seed); b->Wv=alloc(d*d,&seed); b->Wo=alloc(d*d,&seed);
    b->W1=alloc(d*hidden,&seed); b->b1=calloc(hidden,sizeof(float));
    b->W2=alloc(hidden*d,&seed); b->b2=calloc(d,sizeof(float));
    b->Wout=alloc(d*vocab,&seed); b->bout=calloc(vocab,sizeof(float));
    b->g1=calloc(d,sizeof(float)); b->be1=calloc(d,sizeof(float));
    b->g2=calloc(d,sizeof(float)); b->be2=calloc(d,sizeof(float));
    b->gf=calloc(d,sizeof(float)); b->bef=calloc(d,sizeof(float));
    for(int i=0;i<d;i++){ b->g1[i]=1; b->g2[i]=1; b->gf[i]=1; }
    return b;
}
void vbs_block_free(vbs_block_t *b){ if(!b) return; float *ps[]={b->E,b->P,b->Wq,b->Wk,b->Wv,b->Wo,b->W1,b->b1,b->W2,b->b2,b->Wout,b->bout,b->g1,b->be1,b->g2,b->be2,b->gf,b->bef}; for(int i=0;i<18;i++) free(ps[i]); free(b); }
/* helpers: X is seq x d row-major, W is d x d, Y = X*W */
static void matmul_sd(const float *X,const float *W,float *Y,int seq,int d){ for(int i=0;i<seq;i++) for(int j=0;j<d;j++){ float s=0; for(int k=0;k<d;k++) s+=X[i*d+k]*W[k*d+j]; Y[i*d+j]=s; } }
static void ln_fwd(const float *X,float *Y,float *mean,float *rstd,const float *g,const float *be,int seq,int d){
    for(int i=0;i<seq;i++){ double m=0; for(int j=0;j<d;j++) m+=X[i*d+j]; m/=d; double v=0; for(int j=0;j<d;j++){ double dd=X[i*d+j]-m; v+=dd*dd; } v/=d; double rs=1.0/sqrt(v+1e-5); mean[i]=(float)m; rstd[i]=(float)rs; for(int j=0;j<d;j++) Y[i*d+j]=(float)((X[i*d+j]-m)*rs)*g[j]+be[j]; }
}
/* LN backward: given dY, X, mean, rstd, g -> dX, dg, dbe */
static void ln_bwd(const float *dY,const float *X,const float *mean,const float *rstd,const float *g,float *dX,float *dg,float *dbe,int seq,int d){
    for(int j=0;j<d;j++){ dg[j]=0; dbe[j]=0; }
    for(int i=0;i<seq;i++){
        for(int j=0;j<d;j++){ dbe[j]+=dY[i*d+j]; dg[j]+=dY[i*d+j]*(X[i*d+j]-mean[i])*rstd[i]; }
        double sum1=0,sum2=0; for(int j=0;j<d;j++){ double dy=dY[i*d+j]*g[j]; sum1+=dy; sum2+=dy*(X[i*d+j]-mean[i]); }
        for(int j=0;j<d;j++){ double dy=dY[i*d+j]*g[j]; dX[i*d+j]=(float)(rstd[i]*(dy - sum1/d - (X[i*d+j]-mean[i])*rstd[i]*rstd[i]*sum2/d)); }
    }
}
double vbs_block_train_step(vbs_block_t *b, const int *tokens, int target, float lr){
    int S=b->seq, D=b->d, H=b->hidden, V=b->vocab;
    /* forward caches */
    /* Performance: one allocation for all forward caches, instead of 20+ callocs per step.
       This is the main train-step speedup: allocation was dominating the tiny block. */
    size_t ws_fwd = (size_t)(S*D)*12 + (size_t)S*6 + (size_t)(S*S)*2 + (size_t)(S*H)*2 + (size_t)V + 64;
    float *ws = calloc(ws_fwd, sizeof(float)); float *wp = ws;
    #define TAKE(n) (wp += (n), wp - (n))
    float *x=TAKE(S*D), *ln1=TAKE(S*D), *m1=TAKE(S), *r1=TAKE(S);
    float *Q=TAKE(S*D), *K=TAKE(S*D), *Va=TAKE(S*D);
    float *scores=TAKE(S*S), *probs=TAKE(S*S), *att=TAKE(S*D), *attproj=TAKE(S*D), *res1=TAKE(S*D);
    float *ln2=TAKE(S*D), *m2=TAKE(S), *r2=TAKE(S);
    float *hpre=TAKE(S*H), *hact=TAKE(S*H), *mlp=TAKE(S*D), *res2=TAKE(S*D);
    float *lnf=TAKE(S*D), *mf=TAKE(S), *rf=TAKE(S);
    float *logits=TAKE(V);
    #undef TAKE
    for(int i=0;i<S;i++) for(int j=0;j<D;j++) x[i*D+j]=b->E[tokens[i]*D+j]+b->P[i*D+j];
    ln_fwd(x,ln1,m1,r1,b->g1,b->be1,S,D);
    matmul_sd(ln1,b->Wq,Q,S,D); matmul_sd(ln1,b->Wk,K,S,D); matmul_sd(ln1,b->Wv,Va,S,D);
    double scale=1.0/sqrt((double)D);
    for(int i=0;i<S;i++) for(int j=0;j<S;j++){
        if(j>i){ scores[i*S+j]=-1e9f; probs[i*S+j]=0; continue; }
        double s=0; for(int k=0;k<D;k++) s+=(double)Q[i*D+k]*K[j*D+k]; scores[i*S+j]=(float)(s*scale);
    }
    for(int i=0;i<S;i++){ float mx=-1e9f; for(int j=0;j<=i;j++) if(scores[i*S+j]>mx) mx=scores[i*S+j]; double sum=0; for(int j=0;j<=i;j++){ probs[i*S+j]=(float)exp(scores[i*S+j]-mx); sum+=probs[i*S+j]; } for(int j=0;j<=i;j++) probs[i*S+j]/=(float)sum; }
    for(int i=0;i<S;i++) for(int j=0;j<D;j++){ double s=0; for(int k=0;k<S;k++) s+=(double)probs[i*S+k]*Va[k*D+j]; att[i*D+j]=(float)s; }
    matmul_sd(att,b->Wo,attproj,S,D);
    for(int i=0;i<S*D;i++) res1[i]=x[i]+attproj[i];
    ln_fwd(res1,ln2,m2,r2,b->g2,b->be2,S,D);
    for(int i=0;i<S;i++) for(int j=0;j<H;j++){ double s=b->b1[j]; for(int k=0;k<D;k++) s+=(double)ln2[i*D+k]*b->W1[k*H+j]; hpre[i*H+j]=(float)s; hact[i*H+j]=s>0?(float)s:0; }
    for(int i=0;i<S;i++) for(int j=0;j<D;j++){ double s=b->b2[j]; for(int k=0;k<H;k++) s+=(double)hact[i*H+k]*b->W2[k*D+j]; mlp[i*D+j]=(float)s; }
    for(int i=0;i<S*D;i++) res2[i]=res1[i]+mlp[i];
    ln_fwd(res2,lnf,mf,rf,b->gf,b->bef,S,D);
    int last=S-1;
    for(int v=0;v<V;v++){ double s=b->bout[v]; for(int k=0;k<D;k++) s+=(double)lnf[last*D+k]*b->Wout[k*V+v]; logits[v]=(float)s; }
    float mx=logits[0]; for(int v=1;v<V;v++) if(logits[v]>mx) mx=logits[v];
    double sum=0; for(int v=0;v<V;v++) sum+=exp(logits[v]-mx);
    double loss=-log(exp(logits[target]-mx)/sum);
    /* backward: dlogits for last position only */
    size_t ws_bwd = (size_t)(S*D)*20 + (size_t)(S*S)*3 + (size_t)(S*H)*3 + (size_t)(D*V) + (size_t)(D*D)*4 + (size_t)(D*H)*2 + (size_t)(H*D) + (size_t)V*4 + (size_t)D*10 + (size_t)H*2 + (size_t)(V*D) + 64;
    float *ws2 = calloc(ws_bwd, sizeof(float)); float *wp2 = ws2;
    #define TAKE2(n) (wp2 += (n), wp2 - (n))
    float *dlogits=TAKE2(V);
    for(int v=0;v<V;v++) dlogits[v]=(float)(exp(logits[v]-mx)/sum - (v==target?1:0));
    /* head grads */
    float *dWout=TAKE2(D*V), *dbout=TAKE2(V), *dlnf=TAKE2(S*D);
    for(int v=0;v<V;v++){ dbout[v]=dlogits[v]; for(int k=0;k<D;k++){ dWout[k*V+v]=dlogits[v]*lnf[last*D+k]; dlnf[last*D+k]+=dlogits[v]*b->Wout[k*V+v]; } }
    /* final LN */
    float *dres2=TAKE2(S*D), *dgf=TAKE2(D), *dbef=TAKE2(D);
    ln_bwd(dlnf,res2,mf,rf,b->gf,dres2,dgf,dbef,S,D);
    /* res2 = res1 + mlp */
    float *dres1=TAKE2(S*D), *dmlp=TAKE2(S*D);
    for(int i=0;i<S*D;i++){ dres1[i]=dres2[i]; dmlp[i]=dres2[i]; }
    /* MLP */
    float *dhact=TAKE2(S*H), *dW2=TAKE2(H*D), *db2=TAKE2(D);
    for(int i=0;i<S;i++) for(int j=0;j<D;j++){ db2[j]+=dmlp[i*D+j]; for(int k=0;k<H;k++){ dW2[k*D+j]+=dmlp[i*D+j]*hact[i*H+k]; dhact[i*H+k]+=dmlp[i*D+j]*b->W2[k*D+j]; } }
    float *dhpre=TAKE2(S*H), *dln2=TAKE2(S*D), *dW1=TAKE2(D*H), *db1=TAKE2(H);
    for(int i=0;i<S*H;i++) dhpre[i]=hpre[i]>0?dhact[i]:0;
    for(int i=0;i<S;i++) for(int j=0;j<H;j++){ db1[j]+=dhpre[i*H+j]; for(int k=0;k<D;k++){ dW1[k*H+j]+=dhpre[i*H+j]*ln2[i*D+k]; dln2[i*D+k]+=dhpre[i*H+j]*b->W1[k*H+j]; } }
    /* LN2 */
    float *dres1b=TAKE2(S*D), *dg2=TAKE2(D), *dbe2=TAKE2(D);
    ln_bwd(dln2,res1,m2,r2,b->g2,dres1b,dg2,dbe2,S,D);
    for(int i=0;i<S*D;i++) dres1[i]+=dres1b[i];
    /* res1 = x + attproj, dattproj = dres1, dx gets dres1 too (added later) */
    float *dattproj=TAKE2(S*D), *dx=TAKE2(S*D);
    for(int i=0;i<S*D;i++){ dattproj[i]=dres1[i]; dx[i]=dres1[i]; }
    /* attproj = att * Wo */
    float *datt=TAKE2(S*D), *dWo=TAKE2(D*D);
    for(int i=0;i<S;i++) for(int j=0;j<D;j++){ for(int k=0;k<D;k++){ dWo[k*D+j]+=dattproj[i*D+j]*att[i*D+k]; datt[i*D+k]+=dattproj[i*D+j]*b->Wo[k*D+j]; } }
    /* att = probs * V */
    float *dprobs=TAKE2(S*S), *dVa=TAKE2(S*D);
    for(int i=0;i<S;i++) for(int j=0;j<D;j++) for(int k=0;k<S;k++){ dprobs[i*S+k]+=datt[i*D+j]*Va[k*D+j]; dVa[k*D+j]+=datt[i*D+j]*probs[i*S+k]; }
    /* softmax backward, causal */
    float *dscores=TAKE2(S*S);
    for(int i=0;i<S;i++){ double dot=0; for(int j=0;j<=i;j++) dot+=(double)dprobs[i*S+j]*probs[i*S+j]; for(int j=0;j<=i;j++) dscores[i*S+j]=(float)(probs[i*S+j]*(dprobs[i*S+j]-dot)); }
    /* scores = Q K^T * scale */
    float *dQ=TAKE2(S*D), *dK=TAKE2(S*D);
    for(int i=0;i<S;i++) for(int j=0;j<=i;j++){ double ds=dscores[i*S+j]*scale; for(int k=0;k<D;k++){ dQ[i*D+k]+=ds*K[j*D+k]; dK[j*D+k]+=ds*Q[i*D+k]; } }
    /* QKV grads */
    float *dWq=TAKE2(D*D), *dWk=TAKE2(D*D), *dWv=TAKE2(D*D), *dln1=TAKE2(S*D);
    for(int i=0;i<S;i++){
        for(int j=0;j<D;j++) for(int k=0;k<D;k++){ dWq[k*D+j]+=dQ[i*D+j]*ln1[i*D+k]; dWk[k*D+j]+=dK[i*D+j]*ln1[i*D+k]; dWv[k*D+j]+=dVa[i*D+j]*ln1[i*D+k]; dln1[i*D+k]+=dQ[i*D+j]*b->Wq[k*D+j]+dK[i*D+j]*b->Wk[k*D+j]+dVa[i*D+j]*b->Wv[k*D+j]; }
    }
    /* LN1 */
    float *dx2=TAKE2(S*D), *dg1=TAKE2(D), *dbe1=TAKE2(D);
    ln_bwd(dln1,x,m1,r1,b->g1,dx2,dg1,dbe1,S,D);
    for(int i=0;i<S*D;i++) dx[i]+=dx2[i];
    /* embeddings: dx -> E, P */
    float *dE=TAKE2(V*D), *dP=TAKE2(S*D);
    for(int i=0;i<S;i++){ for(int j=0;j<D;j++){ dE[tokens[i]*D+j]+=dx[i*D+j]; dP[i*D+j]+=dx[i*D+j]; } }
    /* SGD update */
    #define UPD(P,G,N) for(int i=0;i<(N);i++) (P)[i]-=lr*(G)[i]
    UPD(b->Wout,dWout,D*V); UPD(b->bout,dbout,V); UPD(b->gf,dgf,D); UPD(b->bef,dbef,D);
    UPD(b->W2,dW2,H*D); UPD(b->b2,db2,D); UPD(b->W1,dW1,D*H); UPD(b->b1,db1,H);
    UPD(b->g2,dg2,D); UPD(b->be2,dbe2,D); UPD(b->Wo,dWo,D*D);
    UPD(b->Wq,dWq,D*D); UPD(b->Wk,dWk,D*D); UPD(b->Wv,dWv,D*D);
    UPD(b->g1,dg1,D); UPD(b->be1,dbe1,D); UPD(b->E,dE,V*D); UPD(b->P,dP,S*D);
    #undef TAKE2
    free(ws); free(ws2);
    return loss;
}

/* --- Inference: same forward as train_step, no backward, no cache --- */
static void block_forward_logits(vbs_block_t *b, const int *tokens, float *logits_out){
    int S=b->seq, D=b->d, H=b->hidden, V=b->vocab;
    float *x=calloc(S*D,sizeof(float)), *ln1=calloc(S*D,sizeof(float)), *m1=calloc(S,sizeof(float)), *r1=calloc(S,sizeof(float));
    float *Q=calloc(S*D,sizeof(float)), *K=calloc(S*D,sizeof(float)), *Va=calloc(S*D,sizeof(float));
    float *scores=calloc(S*S,sizeof(float)), *probs=calloc(S*S,sizeof(float)), *att=calloc(S*D,sizeof(float)), *attproj=calloc(S*D,sizeof(float)), *res1=calloc(S*D,sizeof(float));
    float *ln2=calloc(S*D,sizeof(float)), *m2=calloc(S,sizeof(float)), *r2=calloc(S,sizeof(float));
    float *hpre=calloc(S*H,sizeof(float)), *hact=calloc(S*H,sizeof(float)), *mlp=calloc(S*D,sizeof(float)), *res2=calloc(S*D,sizeof(float));
    float *lnf=calloc(S*D,sizeof(float)), *mf=calloc(S,sizeof(float)), *rf=calloc(S,sizeof(float));
    for(int i=0;i<S;i++) for(int j=0;j<D;j++) x[i*D+j]=b->E[tokens[i]*D+j]+b->P[i*D+j];
    ln_fwd(x,ln1,m1,r1,b->g1,b->be1,S,D);
    matmul_sd(ln1,b->Wq,Q,S,D); matmul_sd(ln1,b->Wk,K,S,D); matmul_sd(ln1,b->Wv,Va,S,D);
    double scale=1.0/sqrt((double)D);
    for(int i=0;i<S;i++) for(int j=0;j<S;j++){
        if(j>i){ scores[i*S+j]=-1e9f; probs[i*S+j]=0; continue; }
        double s=0; for(int k=0;k<D;k++) s+=(double)Q[i*D+k]*K[j*D+k]; scores[i*S+j]=(float)(s*scale);
    }
    for(int i=0;i<S;i++){ float mx=-1e9f; for(int j=0;j<=i;j++) if(scores[i*S+j]>mx) mx=scores[i*S+j]; double sum=0; for(int j=0;j<=i;j++) sum+=exp(scores[i*S+j]-mx); for(int j=0;j<=i;j++) probs[i*S+j]=(float)(exp(scores[i*S+j]-mx)/sum); }
    for(int i=0;i<S;i++) for(int j=0;j<D;j++){ double s=0; for(int k=0;k<S;k++) s+=(double)probs[i*S+k]*Va[k*D+j]; att[i*D+j]=(float)s; }
    matmul_sd(att,b->Wo,attproj,S,D);
    for(int i=0;i<S*D;i++) res1[i]=x[i]+attproj[i];
    ln_fwd(res1,ln2,m2,r2,b->g2,b->be2,S,D);
    for(int i=0;i<S;i++) for(int j=0;j<H;j++){ double s=b->b1[j]; for(int k=0;k<D;k++) s+=(double)ln2[i*D+k]*b->W1[k*H+j]; hpre[i*H+j]=(float)s; hact[i*H+j]=s>0?(float)s:0; }
    for(int i=0;i<S;i++) for(int j=0;j<D;j++){ double s=b->b2[j]; for(int k=0;k<H;k++) s+=(double)hact[i*H+k]*b->W2[k*D+j]; mlp[i*D+j]=(float)s; }
    for(int i=0;i<S*D;i++) res2[i]=res1[i]+mlp[i];
    ln_fwd(res2,lnf,mf,rf,b->gf,b->bef,S,D);
    int last=S-1;
    for(int v=0;v<V;v++){ double s=b->bout[v]; for(int k=0;k<D;k++) s+=(double)lnf[last*D+k]*b->Wout[k*V+v]; logits_out[v]=(float)s; }
    float *all[]={x,ln1,m1,r1,Q,K,Va,scores,probs,att,attproj,res1,ln2,m2,r2,hpre,hact,mlp,res2,lnf,mf,rf};
    for(unsigned i=0;i<sizeof(all)/sizeof(all[0]);i++) free(all[i]);
}
int vbs_block_predict(vbs_block_t *b, const int *tokens){
    float *logits=calloc(b->vocab,sizeof(float));
    block_forward_logits(b,tokens,logits);
    int best=0; for(int v=1;v<b->vocab;v++) if(logits[v]>logits[best]) best=v;
    free(logits); return best;
}
static void save_arr(FILE *f, float *p, int n){ fwrite(p,sizeof(float),n,f); }
static int load_arr(FILE *f, float *p, int n){ return fread(p,sizeof(float),n,f)==(size_t)n; }
void vbs_block_save(vbs_block_t *b, const char *path){
    FILE *f=fopen(path,"wb"); if(!f) return;
    int hdr[4]={b->vocab,b->d,b->seq,b->hidden}; fwrite(hdr,sizeof(int),4,f);
    save_arr(f,b->E,b->vocab*b->d); save_arr(f,b->P,b->seq*b->d);
    save_arr(f,b->Wq,b->d*b->d); save_arr(f,b->Wk,b->d*b->d); save_arr(f,b->Wv,b->d*b->d); save_arr(f,b->Wo,b->d*b->d);
    save_arr(f,b->W1,b->d*b->hidden); save_arr(f,b->b1,b->hidden); save_arr(f,b->W2,b->hidden*b->d); save_arr(f,b->b2,b->d);
    save_arr(f,b->Wout,b->d*b->vocab); save_arr(f,b->bout,b->vocab);
    save_arr(f,b->g1,b->d); save_arr(f,b->be1,b->d); save_arr(f,b->g2,b->d); save_arr(f,b->be2,b->d); save_arr(f,b->gf,b->d); save_arr(f,b->bef,b->d);
    fclose(f);
}
int vbs_block_load(vbs_block_t *b, const char *path){
    FILE *f=fopen(path,"rb"); if(!f) return 0;
    int hdr[4]; if(fread(hdr,sizeof(int),4,f)!=4){fclose(f);return 0;}
    if(hdr[0]!=b->vocab||hdr[1]!=b->d||hdr[2]!=b->seq||hdr[3]!=b->hidden){fclose(f);return 0;}
    int ok=1;
    ok&=load_arr(f,b->E,b->vocab*b->d); ok&=load_arr(f,b->P,b->seq*b->d);
    ok&=load_arr(f,b->Wq,b->d*b->d); ok&=load_arr(f,b->Wk,b->d*b->d); ok&=load_arr(f,b->Wv,b->d*b->d); ok&=load_arr(f,b->Wo,b->d*b->d);
    ok&=load_arr(f,b->W1,b->d*b->hidden); ok&=load_arr(f,b->b1,b->hidden); ok&=load_arr(f,b->W2,b->hidden*b->d); ok&=load_arr(f,b->b2,b->d);
    ok&=load_arr(f,b->Wout,b->d*b->vocab); ok&=load_arr(f,b->bout,b->vocab);
    ok&=load_arr(f,b->g1,b->d); ok&=load_arr(f,b->be1,b->d); ok&=load_arr(f,b->g2,b->d); ok&=load_arr(f,b->be2,b->d); ok&=load_arr(f,b->gf,b->d); ok&=load_arr(f,b->bef,b->d);
    fclose(f); return ok;
}
double vbs_block_loss(vbs_block_t *b, const int *tokens, int target){
    float *logits=calloc(b->vocab,sizeof(float));
    block_forward_logits(b,tokens,logits);
    float mx=logits[0]; for(int v=1;v<b->vocab;v++) if(logits[v]>mx) mx=logits[v];
    double sum=0; for(int v=0;v<b->vocab;v++) sum+=exp(logits[v]-mx);
    double loss=-log(exp(logits[target]-mx)/sum);
    free(logits); return loss;
}
