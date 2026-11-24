/* vbs_train.c — final-layer gradient descent.
 *
 * Two training steps, both cross-entropy on a softmax:
 *   - vbs_train_bias_step: learns a per-token bias added to frozen
 *     base logits (the adapter saved to disk).
 *   - vbs_train_linear_step: learns a full vocab x hidden weight
 *     matrix plus bias from a hidden vector.
 * Both use the same gradient: p - onehot(target), where p is the
 * softmax of the current logits. The softmax is computed with a
 * max-subtraction for numerical stability, and the loss is
 * -log p[target].
 *
 * Note: the pre-rewrite file produced four -Wmaybe-uninitialized
 * warnings (logits[0] read before the compiler could prove vocab
 * was nonzero). The rewrites below guard the empty-vocab case
 * explicitly, so those warnings are gone. */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "vbs_train.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

vbs_adapter_t *vbs_adapter_create(size_t vocab) {
    vbs_adapter_t *a = (vbs_adapter_t *)calloc(1, sizeof(*a));
    if (!a) return NULL;
    a->vocab = vocab;
    a->bias = (float *)calloc(vocab, sizeof(float));
    if (!a->bias) {
        free(a);
        return NULL;
    }
    return a;
}

void vbs_adapter_free(vbs_adapter_t *a) {
    if (!a) return;
    free(a->bias);
    free(a);
}

bool vbs_adapter_save(const vbs_adapter_t *a, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return false;
    fwrite(&a->vocab, sizeof(size_t), 1, f);
    fwrite(a->bias, sizeof(float), a->vocab, f);
    fclose(f);
    return true;
}

bool vbs_adapter_load(vbs_adapter_t *a, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    size_t stored = 0;
    if (fread(&stored, sizeof(size_t), 1, f) != 1) {
        fclose(f);
        return false;
    }
    if (stored != a->vocab) {
        fclose(f);
        return false;
    }
    bool ok = fread(a->bias, sizeof(float), a->vocab, f) == a->vocab;
    fclose(f);
    return ok;
}

void vbs_adapter_apply(const vbs_adapter_t *a, float *logits) {
    for (size_t i = 0; i < a->vocab; i++) logits[i] += a->bias[i];
}

/* Softmax probabilities of `logits` into `probs` (caller-allocated,
 * vocab entries). Returns the max logit used for stabilization, or
 * 0 when vocab is 0. */
static float softmax_probs(const float *logits, size_t vocab, double *probs) {
    if (vocab == 0) return 0.0f;
    float peak = logits[0];
    for (size_t i = 1; i < vocab; i++)
        if (logits[i] > peak) peak = logits[i];
    double total = 0.0;
    for (size_t i = 0; i < vocab; i++)
        total += exp((double)logits[i] - peak);
    for (size_t i = 0; i < vocab; i++)
        probs[i] = exp((double)logits[i] - peak) / total;
    return peak;
}

/* Cross-entropy loss of `logits` for `target`, with the target's
 * probability written to *prob when prob is non-NULL. */
static double softmax_ce(const float *logits, size_t vocab, size_t target,
                         float *prob) {
    if (vocab == 0) return 0.0;
    double *probs = (double *)malloc(vocab * sizeof(double));
    if (!probs) return 0.0;
    softmax_probs(logits, vocab, probs);
    double p = probs[target];
    if (prob) *prob = (float)p;
    free(probs);
    return -log(p);
}

double vbs_train_bias_step(vbs_adapter_t *a, const float *base_logits,
                           size_t target, float lr, double *prob_out) {
    if (a->vocab == 0) return -1.0;
    float *logits = (float *)malloc(a->vocab * sizeof(float));
    double *probs = (double *)malloc(a->vocab * sizeof(double));
    if (!logits || !probs) {
        free(logits);
        free(probs);
        return -1.0;
    }
    for (size_t i = 0; i < a->vocab; i++)
        logits[i] = base_logits[i] + a->bias[i];
    softmax_probs(logits, a->vocab, probs);
    double loss = -log(probs[target]);
    for (size_t i = 0; i < a->vocab; i++) {
        double grad = probs[i] - (i == target ? 1.0 : 0.0);
        a->bias[i] -= (float)(lr * grad);
    }
    if (prob_out) *prob_out = probs[target];
    free(logits);
    free(probs);
    return loss;
}

double vbs_train_linear_step(float *W, float *b, size_t vocab, size_t hidden,
                             const float *h, size_t target, float lr) {
    if (vocab == 0) return -1.0;
    float *logits = (float *)malloc(vocab * sizeof(float));
    double *probs = (double *)malloc(vocab * sizeof(double));
    if (!logits || !probs) {
        free(logits);
        free(probs);
        return -1.0;
    }
    for (size_t i = 0; i < vocab; i++) {
        double acc = b ? b[i] : 0.0;
        for (size_t j = 0; j < hidden; j++)
            acc += (double)W[i * hidden + j] * h[j];
        logits[i] = (float)acc;
    }
    double loss = softmax_ce(logits, vocab, target, NULL);
    softmax_probs(logits, vocab, probs);
    for (size_t i = 0; i < vocab; i++) {
        double grad = probs[i] - (i == target ? 1.0 : 0.0);
        if (b) b[i] -= (float)(lr * grad);
        for (size_t j = 0; j < hidden; j++)
            W[i * hidden + j] -= (float)(lr * grad * h[j]);
    }
    free(logits);
    free(probs);
    return loss;
}
