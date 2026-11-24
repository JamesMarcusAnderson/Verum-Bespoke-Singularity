/* vbs_block.c — a trainable transformer block.
 *
 * One block, single head, causal attention, ReLU MLP, three
 * LayerNorms. The forward pass caches every intermediate; the
 * backward pass walks the chain rule in reverse (head -> final LN
 * -> MLP -> LN2 -> attention -> LN1 -> embeddings) and SGD-updates
 * all parameters in place.
 *
 * The two forward paths share their stages (embed -> scores and
 * probs -> logits) through forward_scores/forward_rest, but their
 * row softmaxes stay distinct on purpose: the training forward
 * stores softmax numerators then divides by their sum, while the
 * inference forward divides each exponential by the sum directly.
 * Those two orders round differently in the last bit, so unifying
 * them would change training trajectories and predictions. Both
 * softmaxes are preserved exactly as they were.
 *
 * Performance: each train step carves its forward and backward
 * scratch space out of two single allocations (an arena), instead
 * of 20+ separate callocs per step — allocation used to dominate
 * the tiny block's step time.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "vbs_block.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* --- small utilities ------------------------------------------------ */

/* LCG random init in [-0.1, 0.1). The seed is consumed in parameter
 * order (E, P, Wq, Wk, Wv, Wo, W1, W2, Wout), so a given seed always
 * produces the same initial block. */
static float init_random(unsigned *state) {
    *state = *state * 1664525u + 1013904223u;
    return ((float)(*state >> 8) / 16777216.0f - 0.5f) * 0.2f;
}

static float *init_array(int n, unsigned *seed) {
    float *p = (float *)calloc((size_t)n, sizeof(float));
    if (!p) return NULL;
    for (int i = 0; i < n; i++) p[i] = init_random(seed);
    return p;
}

/* A bump arena over one calloc'd float buffer. */
typedef struct {
    float *base;
    float *cursor;
} arena_t;

static float *arena_take(arena_t *a, size_t n) {
    float *p = a->cursor;
    a->cursor += n;
    return p;
}

/* Y = X * W, where X is seq x d row-major and W is d x d. */
static void matmul_square(const float *X, const float *W, float *Y,
                          int seq, int d) {
    for (int i = 0; i < seq; i++) {
        for (int j = 0; j < d; j++) {
            float acc = 0.0f;
            for (int k = 0; k < d; k++) acc += X[i * d + k] * W[k * d + j];
            Y[i * d + j] = acc;
        }
    }
}

/* LayerNorm forward: Y = (X - mean) * rstd * g + be, per row. */
static void layernorm_forward(const float *X, float *Y, float *mean,
                              float *rstd, const float *g, const float *be,
                              int seq, int d) {
    for (int i = 0; i < seq; i++) {
        double m = 0.0;
        for (int j = 0; j < d; j++) m += X[i * d + j];
        m /= d;
        double v = 0.0;
        for (int j = 0; j < d; j++) {
            double diff = X[i * d + j] - m;
            v += diff * diff;
        }
        v /= d;
        double rs = 1.0 / sqrt(v + 1e-5);
        mean[i] = (float)m;
        rstd[i] = (float)rs;
        for (int j = 0; j < d; j++)
            Y[i * d + j] = (float)((X[i * d + j] - m) * rs) * g[j] + be[j];
    }
}

/* LayerNorm backward: given dY, X, mean, rstd, g -> dX, dg, dbe. */
static void layernorm_backward(const float *dY, const float *X,
                               const float *mean, const float *rstd,
                               const float *g, float *dX, float *dg,
                               float *dbe, int seq, int d) {
    for (int j = 0; j < d; j++) {
        dg[j] = 0.0f;
        dbe[j] = 0.0f;
    }
    for (int i = 0; i < seq; i++) {
        for (int j = 0; j < d; j++) {
            dbe[j] += dY[i * d + j];
            dg[j] += dY[i * d + j] * (X[i * d + j] - mean[i]) * rstd[i];
        }
        double sum1 = 0.0, sum2 = 0.0;
        for (int j = 0; j < d; j++) {
            double dy = dY[i * d + j] * g[j];
            sum1 += dy;
            sum2 += dy * (X[i * d + j] - mean[i]);
        }
        for (int j = 0; j < d; j++) {
            double dy = dY[i * d + j] * g[j];
            dX[i * d + j] = (float)(rstd[i] * (dy - sum1 / d -
                (X[i * d + j] - mean[i]) * rstd[i] * rstd[i] * sum2 / d));
        }
    }
}

/* --- lifecycle ------------------------------------------------------ */

vbs_block_t *vbs_block_create(int vocab, int d, int seq, int hidden,
                              unsigned seed) {
    vbs_block_t *b = (vbs_block_t *)calloc(1, sizeof(*b));
    if (!b) return NULL;
    b->vocab = vocab;
    b->d = d;
    b->seq = seq;
    b->hidden = hidden;
    b->E = init_array(vocab * d, &seed);
    b->P = init_array(seq * d, &seed);
    b->Wq = init_array(d * d, &seed);
    b->Wk = init_array(d * d, &seed);
    b->Wv = init_array(d * d, &seed);
    b->Wo = init_array(d * d, &seed);
    b->W1 = init_array(d * hidden, &seed);
    b->b1 = (float *)calloc((size_t)hidden, sizeof(float));
    b->W2 = init_array(hidden * d, &seed);
    b->b2 = (float *)calloc((size_t)d, sizeof(float));
    b->Wout = init_array(d * vocab, &seed);
    b->bout = (float *)calloc((size_t)vocab, sizeof(float));
    b->g1 = (float *)calloc((size_t)d, sizeof(float));
    b->be1 = (float *)calloc((size_t)d, sizeof(float));
    b->g2 = (float *)calloc((size_t)d, sizeof(float));
    b->be2 = (float *)calloc((size_t)d, sizeof(float));
    b->gf = (float *)calloc((size_t)d, sizeof(float));
    b->bef = (float *)calloc((size_t)d, sizeof(float));
    for (int i = 0; i < d; i++) {
        b->g1[i] = 1.0f;
        b->g2[i] = 1.0f;
        b->gf[i] = 1.0f;
    }
    return b;
}

void vbs_block_free(vbs_block_t *b) {
    if (!b) return;
    float *params[] = { b->E, b->P, b->Wq, b->Wk, b->Wv, b->Wo,
                        b->W1, b->b1, b->W2, b->b2, b->Wout, b->bout,
                        b->g1, b->be1, b->g2, b->be2, b->gf, b->bef };
    for (int i = 0; i < 18; i++) free(params[i]);
    free(b);
}

/* --- training ------------------------------------------------------- */

/* SGD update: P[i] -= lr * G[i] for n entries. */
static void sgd_update(float *params, const float *grads, int n, float lr) {
    for (int i = 0; i < n; i++) params[i] -= lr * grads[i];
}

/* The forward pass's scratch buffers. Both the training step
 * (arena) and inference (calloc) fill one of these from their own
 * storage, then run the same stages through it. */
typedef struct {
    float *x, *ln1, *m1, *r1, *Q, *K, *Va, *scores, *probs, *att,
          *attproj, *res1, *ln2, *m2, *r2, *hpre, *hact, *mlp,
          *res2, *lnf, *mf, *rf, *logits;
} fwd_scratch_t;

/* Embed + position, LN1, QKV projections, causal scores.
 * Shared by training and inference. */
static void forward_scores(vbs_block_t *b, const int *tokens, fwd_scratch_t *f) {
    int S = b->seq, D = b->d;
    for (int i = 0; i < S; i++)
        for (int j = 0; j < D; j++)
            f->x[i * D + j] = b->E[tokens[i] * D + j] + b->P[i * D + j];
    layernorm_forward(f->x, f->ln1, f->m1, f->r1, b->g1, b->be1, S, D);
    matmul_square(f->ln1, b->Wq, f->Q, S, D);
    matmul_square(f->ln1, b->Wk, f->K, S, D);
    matmul_square(f->ln1, b->Wv, f->Va, S, D);
    double scale = 1.0 / sqrt((double)D);
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++) {
            if (j > i) {
                f->scores[i * S + j] = -1e9f;
                f->probs[i * S + j] = 0.0f;
                continue;
            }
            double s = 0.0;
            for (int k = 0; k < D; k++) s += (double)f->Q[i * D + k] * f->K[j * D + k];
            f->scores[i * S + j] = (float)(s * scale);
        }
    }
}

/* Row maximum of one causal score row: scores[i*S .. i*S+i].
 * Shared by the training and inference softmaxes, which differ
 * only in how they divide by the sum. */
static float row_max(const float *scores, int S, int i) {
    float mx = -1e9f;
    for (int j = 0; j <= i; j++)
        if (scores[i * S + j] > mx) mx = scores[i * S + j];
    return mx;
}

/* Attention output through the head: probs -> att -> Wo -> res1
 * -> LN2 -> MLP -> res2 -> LNf -> logits (last position only).
 * Shared by training and inference. */
static void forward_rest(vbs_block_t *b, fwd_scratch_t *f) {
    int S = b->seq, D = b->d, H = b->hidden, V = b->vocab;
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < D; j++) {
            double s = 0.0;
            for (int k = 0; k < S; k++) s += (double)f->probs[i * S + k] * f->Va[k * D + j];
            f->att[i * D + j] = (float)s;
        }
    }
    matmul_square(f->att, b->Wo, f->attproj, S, D);
    for (int i = 0; i < S * D; i++) f->res1[i] = f->x[i] + f->attproj[i];

    /* MLP. */
    layernorm_forward(f->res1, f->ln2, f->m2, f->r2, b->g2, b->be2, S, D);
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < H; j++) {
            double s = b->b1[j];
            for (int k = 0; k < D; k++) s += (double)f->ln2[i * D + k] * b->W1[k * H + j];
            f->hpre[i * H + j] = (float)s;
            f->hact[i * H + j] = s > 0 ? (float)s : 0.0f;
        }
    }
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < D; j++) {
            double s = b->b2[j];
            for (int k = 0; k < H; k++) s += (double)f->hact[i * H + k] * b->W2[k * D + j];
            f->mlp[i * D + j] = (float)s;
        }
    }
    for (int i = 0; i < S * D; i++) f->res2[i] = f->res1[i] + f->mlp[i];

    /* Final norm + head (last position only). */
    layernorm_forward(f->res2, f->lnf, f->mf, f->rf, b->gf, b->bef, S, D);
    int last = S - 1;
    for (int v = 0; v < V; v++) {
        double s = b->bout[v];
        for (int k = 0; k < D; k++) s += (double)f->lnf[last * D + k] * b->Wout[k * V + v];
        f->logits[v] = (float)s;
    }
}

double vbs_block_train_step(vbs_block_t *b, const int *tokens, int target,
                            float lr) {
    int S = b->seq, D = b->d, H = b->hidden, V = b->vocab;

    /* Forward caches, carved from one allocation. */
    size_t fwd_size = (size_t)(S * D) * 12 + (size_t)S * 6 +
                      (size_t)(S * S) * 2 + (size_t)(S * H) * 2 + (size_t)V + 64;
    float *fwd_buf = (float *)calloc(fwd_size, sizeof(float));
    arena_t fwd = { fwd_buf, fwd_buf };
    /* Each buffer registers itself in the scratch struct as it is
     * carved, so the shared forward stages see the same buffers
     * the backward pass reads through the locals below. */
    fwd_scratch_t f;
    float *x = (f.x = arena_take(&fwd, (size_t)S * D));
    float *ln1 = (f.ln1 = arena_take(&fwd, (size_t)S * D));
    float *m1 = (f.m1 = arena_take(&fwd, (size_t)S));
    float *r1 = (f.r1 = arena_take(&fwd, (size_t)S));
    float *Q = (f.Q = arena_take(&fwd, (size_t)S * D));
    float *K = (f.K = arena_take(&fwd, (size_t)S * D));
    float *Va = (f.Va = arena_take(&fwd, (size_t)S * D));
    float *scores = (f.scores = arena_take(&fwd, (size_t)S * S));
    float *probs = (f.probs = arena_take(&fwd, (size_t)S * S));
    float *att = (f.att = arena_take(&fwd, (size_t)S * D));
    f.attproj = arena_take(&fwd, (size_t)S * D); /* read only by forward_rest */
    float *res1 = (f.res1 = arena_take(&fwd, (size_t)S * D));
    float *ln2 = (f.ln2 = arena_take(&fwd, (size_t)S * D));
    float *m2 = (f.m2 = arena_take(&fwd, (size_t)S));
    float *r2 = (f.r2 = arena_take(&fwd, (size_t)S));
    float *hpre = (f.hpre = arena_take(&fwd, (size_t)S * H));
    float *hact = (f.hact = arena_take(&fwd, (size_t)S * H));
    f.mlp = arena_take(&fwd, (size_t)S * D); /* read only by forward_rest */
    float *res2 = (f.res2 = arena_take(&fwd, (size_t)S * D));
    float *lnf = (f.lnf = arena_take(&fwd, (size_t)S * D));
    float *mf = (f.mf = arena_take(&fwd, (size_t)S));
    float *rf = (f.rf = arena_take(&fwd, (size_t)S));
    float *logits = (f.logits = arena_take(&fwd, (size_t)V));

    /* Shared forward stages; the row softmax below stays the
     * training variant (numerators stored, then divided). The
     * backward pass needs `scale` and `last`, so they are declared
     * here even though the helpers compute their own copies. */
    double scale = 1.0 / sqrt((double)D);
    int last = S - 1;
    forward_scores(b, tokens, &f);
    /* Row softmax, causal: numerators stored, then divided by the sum. */
    for (int i = 0; i < S; i++) {
        float mx = row_max(scores, S, i);
        double sum = 0.0;
        for (int j = 0; j <= i; j++) {
            probs[i * S + j] = (float)exp(scores[i * S + j] - mx);
            sum += probs[i * S + j];
        }
        for (int j = 0; j <= i; j++) probs[i * S + j] /= (float)sum;
    }
    forward_rest(b, &f);
    float mx = logits[0];
    for (int v = 1; v < V; v++)
        if (logits[v] > mx) mx = logits[v];
    double sum = 0.0;
    for (int v = 0; v < V; v++) sum += exp(logits[v] - mx);
    double loss = -log(exp(logits[target] - mx) / sum);

    /* Backward: dlogits for the last position only. */
    size_t bwd_size = (size_t)(S * D) * 20 + (size_t)(S * S) * 3 +
                      (size_t)(S * H) * 3 + (size_t)(D * V) +
                      (size_t)(D * D) * 4 + (size_t)(D * H) * 2 +
                      (size_t)(H * D) + (size_t)V * 4 + (size_t)D * 10 +
                      (size_t)H * 2 + (size_t)(V * D) + 64;
    float *bwd_buf = (float *)calloc(bwd_size, sizeof(float));
    arena_t bwd = { bwd_buf, bwd_buf };
    float *dlogits = arena_take(&bwd, (size_t)V);
    for (int v = 0; v < V; v++)
        dlogits[v] = (float)(exp(logits[v] - mx) / sum - (v == target ? 1 : 0));

    /* Head gradients. */
    float *dWout = arena_take(&bwd, (size_t)D * V);
    float *dbout = arena_take(&bwd, (size_t)V);
    float *dlnf = arena_take(&bwd, (size_t)S * D);
    for (int v = 0; v < V; v++) {
        dbout[v] = dlogits[v];
        for (int k = 0; k < D; k++) {
            dWout[k * V + v] = dlogits[v] * lnf[last * D + k];
            dlnf[last * D + k] += dlogits[v] * b->Wout[k * V + v];
        }
    }
    /* Final LN. */
    float *dres2 = arena_take(&bwd, (size_t)S * D);
    float *dgf = arena_take(&bwd, (size_t)D);
    float *dbef = arena_take(&bwd, (size_t)D);
    layernorm_backward(dlnf, res2, mf, rf, b->gf, dres2, dgf, dbef, S, D);
    /* res2 = res1 + mlp. */
    float *dres1 = arena_take(&bwd, (size_t)S * D);
    float *dmlp = arena_take(&bwd, (size_t)S * D);
    for (int i = 0; i < S * D; i++) {
        dres1[i] = dres2[i];
        dmlp[i] = dres2[i];
    }
    /* MLP. */
    float *dhact = arena_take(&bwd, (size_t)S * H);
    float *dW2 = arena_take(&bwd, (size_t)H * D);
    float *db2 = arena_take(&bwd, (size_t)D);
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < D; j++) {
            db2[j] += dmlp[i * D + j];
            for (int k = 0; k < H; k++) {
                dW2[k * D + j] += dmlp[i * D + j] * hact[i * H + k];
                dhact[i * H + k] += dmlp[i * D + j] * b->W2[k * D + j];
            }
        }
    }
    float *dhpre = arena_take(&bwd, (size_t)S * H);
    float *dln2 = arena_take(&bwd, (size_t)S * D);
    float *dW1 = arena_take(&bwd, (size_t)D * H);
    float *db1 = arena_take(&bwd, (size_t)H);
    for (int i = 0; i < S * H; i++) dhpre[i] = hpre[i] > 0 ? dhact[i] : 0.0f;
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < H; j++) {
            db1[j] += dhpre[i * H + j];
            for (int k = 0; k < D; k++) {
                dW1[k * H + j] += dhpre[i * H + j] * ln2[i * D + k];
                dln2[i * D + k] += dhpre[i * H + j] * b->W1[k * H + j];
            }
        }
    }
    /* LN2. */
    float *dres1b = arena_take(&bwd, (size_t)S * D);
    float *dg2 = arena_take(&bwd, (size_t)D);
    float *dbe2 = arena_take(&bwd, (size_t)D);
    layernorm_backward(dln2, res1, m2, r2, b->g2, dres1b, dg2, dbe2, S, D);
    for (int i = 0; i < S * D; i++) dres1[i] += dres1b[i];
    /* res1 = x + attproj: dattproj = dres1, dx gets dres1 too. */
    float *dattproj = arena_take(&bwd, (size_t)S * D);
    float *dx = arena_take(&bwd, (size_t)S * D);
    for (int i = 0; i < S * D; i++) {
        dattproj[i] = dres1[i];
        dx[i] = dres1[i];
    }
    /* attproj = att * Wo. */
    float *datt = arena_take(&bwd, (size_t)S * D);
    float *dWo = arena_take(&bwd, (size_t)D * D);
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < D; j++) {
            for (int k = 0; k < D; k++) {
                dWo[k * D + j] += dattproj[i * D + j] * att[i * D + k];
                datt[i * D + k] += dattproj[i * D + j] * b->Wo[k * D + j];
            }
        }
    }
    /* att = probs * V. */
    float *dprobs = arena_take(&bwd, (size_t)S * S);
    float *dVa = arena_take(&bwd, (size_t)S * D);
    for (int i = 0; i < S; i++)
        for (int j = 0; j < D; j++)
            for (int k = 0; k < S; k++) {
                dprobs[i * S + k] += datt[i * D + j] * Va[k * D + j];
                dVa[k * D + j] += datt[i * D + j] * probs[i * S + k];
            }
    /* Softmax backward, causal. */
    float *dscores = arena_take(&bwd, (size_t)S * S);
    for (int i = 0; i < S; i++) {
        double dot = 0.0;
        for (int j = 0; j <= i; j++) dot += (double)dprobs[i * S + j] * probs[i * S + j];
        for (int j = 0; j <= i; j++)
            dscores[i * S + j] = (float)(probs[i * S + j] * (dprobs[i * S + j] - dot));
    }
    /* scores = Q K^T * scale. */
    float *dQ = arena_take(&bwd, (size_t)S * D);
    float *dK = arena_take(&bwd, (size_t)S * D);
    for (int i = 0; i < S; i++) {
        for (int j = 0; j <= i; j++) {
            double ds = dscores[i * S + j] * scale;
            for (int k = 0; k < D; k++) {
                dQ[i * D + k] += ds * K[j * D + k];
                dK[j * D + k] += ds * Q[i * D + k];
            }
        }
    }
    /* QKV gradients. */
    float *dWq = arena_take(&bwd, (size_t)D * D);
    float *dWk = arena_take(&bwd, (size_t)D * D);
    float *dWv = arena_take(&bwd, (size_t)D * D);
    float *dln1 = arena_take(&bwd, (size_t)S * D);
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < D; j++) {
            for (int k = 0; k < D; k++) {
                dWq[k * D + j] += dQ[i * D + j] * ln1[i * D + k];
                dWk[k * D + j] += dK[i * D + j] * ln1[i * D + k];
                dWv[k * D + j] += dVa[i * D + j] * ln1[i * D + k];
                dln1[i * D + k] += dQ[i * D + j] * b->Wq[k * D + j] +
                                   dK[i * D + j] * b->Wk[k * D + j] +
                                   dVa[i * D + j] * b->Wv[k * D + j];
            }
        }
    }
    /* LN1. */
    float *dx2 = arena_take(&bwd, (size_t)S * D);
    float *dg1 = arena_take(&bwd, (size_t)D);
    float *dbe1 = arena_take(&bwd, (size_t)D);
    layernorm_backward(dln1, x, m1, r1, b->g1, dx2, dg1, dbe1, S, D);
    for (int i = 0; i < S * D; i++) dx[i] += dx2[i];
    /* Embeddings: dx -> E, P. */
    float *dE = arena_take(&bwd, (size_t)V * D);
    float *dP = arena_take(&bwd, (size_t)S * D);
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < D; j++) {
            dE[tokens[i] * D + j] += dx[i * D + j];
            dP[i * D + j] += dx[i * D + j];
        }
    }
    /* SGD update. */
    sgd_update(b->Wout, dWout, D * V, lr);
    sgd_update(b->bout, dbout, V, lr);
    sgd_update(b->gf, dgf, D, lr);
    sgd_update(b->bef, dbef, D, lr);
    sgd_update(b->W2, dW2, H * D, lr);
    sgd_update(b->b2, db2, D, lr);
    sgd_update(b->W1, dW1, D * H, lr);
    sgd_update(b->b1, db1, H, lr);
    sgd_update(b->g2, dg2, D, lr);
    sgd_update(b->be2, dbe2, D, lr);
    sgd_update(b->Wo, dWo, D * D, lr);
    sgd_update(b->Wq, dWq, D * D, lr);
    sgd_update(b->Wk, dWk, D * D, lr);
    sgd_update(b->Wv, dWv, D * D, lr);
    sgd_update(b->g1, dg1, D, lr);
    sgd_update(b->be1, dbe1, D, lr);
    sgd_update(b->E, dE, V * D, lr);
    sgd_update(b->P, dP, S * D, lr);
    free(fwd_buf);
    free(bwd_buf);
    return loss;
}

/* --- inference ------------------------------------------------------ */

/* Forward pass for inference: same math as the training forward,
 * without the backward caches. One deliberate difference from the
 * training path: the row softmax divides each exponential by the
 * sum directly instead of storing numerators first. That rounds
 * differently in the last bit, so the two paths are kept separate
 * on purpose (see the file header). */
static void forward_logits(vbs_block_t *b, const int *tokens,
                           float *logits_out) {
    int S = b->seq, D = b->d, H = b->hidden;
    /* Each buffer registers itself in the scratch struct as it is
     * allocated, so the shared forward stages see the same buffers
     * the inference softmax reads through the locals below. */
    fwd_scratch_t f;
    float *x = (f.x = (float *)calloc((size_t)S * D, sizeof(float)));
    float *ln1 = (f.ln1 = (float *)calloc((size_t)S * D, sizeof(float)));
    float *m1 = (f.m1 = (float *)calloc((size_t)S, sizeof(float)));
    float *r1 = (f.r1 = (float *)calloc((size_t)S, sizeof(float)));
    float *Q = (f.Q = (float *)calloc((size_t)S * D, sizeof(float)));
    float *K = (f.K = (float *)calloc((size_t)S * D, sizeof(float)));
    float *Va = (f.Va = (float *)calloc((size_t)S * D, sizeof(float)));
    float *scores = (f.scores = (float *)calloc((size_t)S * S, sizeof(float)));
    float *probs = (f.probs = (float *)calloc((size_t)S * S, sizeof(float)));
    float *att = (f.att = (float *)calloc((size_t)S * D, sizeof(float)));
    float *attproj = (f.attproj = (float *)calloc((size_t)S * D, sizeof(float)));
    float *res1 = (f.res1 = (float *)calloc((size_t)S * D, sizeof(float)));
    float *ln2 = (f.ln2 = (float *)calloc((size_t)S * D, sizeof(float)));
    float *m2 = (f.m2 = (float *)calloc((size_t)S, sizeof(float)));
    float *r2 = (f.r2 = (float *)calloc((size_t)S, sizeof(float)));
    float *hpre = (f.hpre = (float *)calloc((size_t)S * H, sizeof(float)));
    float *hact = (f.hact = (float *)calloc((size_t)S * H, sizeof(float)));
    float *mlp = (f.mlp = (float *)calloc((size_t)S * D, sizeof(float)));
    float *res2 = (f.res2 = (float *)calloc((size_t)S * D, sizeof(float)));
    float *lnf = (f.lnf = (float *)calloc((size_t)S * D, sizeof(float)));
    float *mf = (f.mf = (float *)calloc((size_t)S, sizeof(float)));
    float *rf = (f.rf = (float *)calloc((size_t)S, sizeof(float)));
    f.logits = logits_out;

    /* Shared forward stages; the row softmax below stays the
     * inference variant (each exponential divided by the sum). */
    forward_scores(b, tokens, &f);
    for (int i = 0; i < S; i++) {
        float mx = row_max(scores, S, i);
        double sum = 0.0;
        for (int j = 0; j <= i; j++) sum += exp(scores[i * S + j] - mx);
        for (int j = 0; j <= i; j++)
            probs[i * S + j] = (float)(exp(scores[i * S + j] - mx) / sum);
    }
    forward_rest(b, &f);
    float *all[] = { x, ln1, m1, r1, Q, K, Va, scores, probs, att, attproj,
                     res1, ln2, m2, r2, hpre, hact, mlp, res2, lnf, mf, rf };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); i++) free(all[i]);
}

int vbs_block_predict(vbs_block_t *b, const int *tokens) {
    float *logits = (float *)calloc((size_t)b->vocab, sizeof(float));
    forward_logits(b, tokens, logits);
    int best = 0;
    for (int v = 1; v < b->vocab; v++)
        if (logits[v] > logits[best]) best = v;
    free(logits);
    return best;
}

double vbs_block_loss(vbs_block_t *b, const int *tokens, int target) {
    float *logits = (float *)calloc((size_t)b->vocab, sizeof(float));
    forward_logits(b, tokens, logits);
    float mx = logits[0];
    for (int v = 1; v < b->vocab; v++)
        if (logits[v] > mx) mx = logits[v];
    double sum = 0.0;
    for (int v = 0; v < b->vocab; v++) sum += exp(logits[v] - mx);
    double loss = -log(exp(logits[target] - mx) / sum);
    free(logits);
    return loss;
}

/* --- persistence ---------------------------------------------------- */

/* Parameter arrays in save order, so save and load cannot drift apart. */
typedef struct {
    float **slot;
    int count;
} param_entry_t;

static void fill_param_table(vbs_block_t *b, param_entry_t *table) {
    int d = b->d, v = b->vocab, s = b->seq, h = b->hidden;
    param_entry_t entries[] = {
        { &b->E, v * d }, { &b->P, s * d },
        { &b->Wq, d * d }, { &b->Wk, d * d },
        { &b->Wv, d * d }, { &b->Wo, d * d },
        { &b->W1, d * h }, { &b->b1, h },
        { &b->W2, h * d }, { &b->b2, d },
        { &b->Wout, d * v }, { &b->bout, v },
        { &b->g1, d }, { &b->be1, d },
        { &b->g2, d }, { &b->be2, d },
        { &b->gf, d }, { &b->bef, d },
    };
    memcpy(table, entries, sizeof(entries));
}

void vbs_block_save(vbs_block_t *b, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    int header[4] = { b->vocab, b->d, b->seq, b->hidden };
    fwrite(header, sizeof(int), 4, f);
    param_entry_t table[18];
    fill_param_table(b, table);
    for (int i = 0; i < 18; i++)
        fwrite(*table[i].slot, sizeof(float), (size_t)table[i].count, f);
    fclose(f);
}

int vbs_block_load(vbs_block_t *b, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    int header[4];
    if (fread(header, sizeof(int), 4, f) != 4) {
        fclose(f);
        return 0;
    }
    if (header[0] != b->vocab || header[1] != b->d ||
        header[2] != b->seq || header[3] != b->hidden) {
        fclose(f);
        return 0;
    }
    int ok = 1;
    param_entry_t table[18];
    fill_param_table(b, table);
    for (int i = 0; i < 18; i++) {
        if (fread(*table[i].slot, sizeof(float), (size_t)table[i].count, f)
            != (size_t)table[i].count)
            ok = 0;
    }
    fclose(f);
    return ok;
}
