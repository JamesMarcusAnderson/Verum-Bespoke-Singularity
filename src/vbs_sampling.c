// vbs_sampling.c — sampling strategies for token generation.
//
// Greedy / temperature / top-k / nucleus (top-p) / repetition penalty.
// PRNG: xoshiro256++ (public-domain algorithm by Blackman & Vigna;
// implementation written from the algorithm description). No dependencies.

#include "vbs_sampling.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

void vbs_sampler_init(vbs_sampler_t *s) {
    if (!s) return;
    s->temperature    = 0.0f;
    s->top_k          = 0;
    s->top_p          = 1.0f;
    s->repeat_penalty = 1.0f;
    s->repeat_last_n  = 64;
    s->seed           = 0;
    // PRNG state MUST be zeroed: a stale seed_seeded flag here made
    // sampling read uninitialized state (caught by the reproducibility test)
    s->seed_seeded    = 0;
    memset(s->seed_state, 0, sizeof(s->seed_state));
}

// ---------------------------------------------------------------------------
// xoshiro256++
// ---------------------------------------------------------------------------

static uint64_t rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

uint64_t vbs_rand_next(vbs_sampler_t *s) {
    uint64_t *st = (uint64_t *)s->seed_state;
    if (!s->seed_seeded) {
        uint64_t sm = s->seed ? s->seed : (uint64_t)time(NULL) ^ 0xD1B54A32D192ED03ull;
        for (int i = 0; i < 4; i++) st[i] = splitmix64(&sm);
        s->seed_seeded = 1;
    }
    const uint64_t result = rotl64(st[0] + st[3], 23) + st[0];
    const uint64_t t = st[1] << 17;
    st[2] ^= st[0];
    st[3] ^= st[1];
    st[1] ^= st[2];
    st[0] ^= st[3];
    st[2] ^= t;
    st[3] = rotl64(st[3], 45);
    return result;
}

float vbs_rand_float(vbs_sampler_t *s) {
    // 24 bits of mantissa -> uniform in [0,1)
    return (float)(vbs_rand_next(s) >> 40) * (1.0f / 16777216.0f);
}

// ---------------------------------------------------------------------------
// repetition penalty (CTRL-style: divide positive, multiply negative)
// ---------------------------------------------------------------------------

void vbs_sampler_apply_penalty(const vbs_sampler_t *s,
                               const uint32_t *recent, size_t n_recent,
                               const float *logits, float *out, size_t vocab) {
    if (logits != out) memcpy(out, logits, vocab * sizeof(float));
    if (!recent || n_recent == 0 || s->repeat_penalty == 1.0f) return;

    size_t window = n_recent;
    if (s->repeat_last_n > 0 && window > (size_t)s->repeat_last_n) {
        window = (size_t)s->repeat_last_n;
    }
    const uint32_t *start = recent + (n_recent - window);
    for (size_t i = 0; i < window; i++) {
        uint32_t tok = start[i];
        if (tok >= vocab) continue;
        if (out[tok] > 0.0f) {
            out[tok] /= s->repeat_penalty;
        } else {
            out[tok] *= s->repeat_penalty;
        }
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

typedef struct { float logit; uint32_t idx; } cand_t;

static int cand_cmp(const void *a, const void *b) {
    const cand_t *x = (const cand_t *)a, *y = (const cand_t *)b;
    if (x->logit > y->logit) return -1;
    if (x->logit < y->logit) return 1;
    if (x->idx < y->idx) return -1;
    if (x->idx > y->idx) return 1;
    return 0;
}

uint32_t vbs_sampler_sample(vbs_sampler_t *s,
                            const float *logits, size_t vocab,
                            const uint32_t *recent, size_t n_recent) {
    if (!s || !logits || vocab == 0) return 0;

    float *adj = (float *)malloc(vocab * sizeof(float));
    cand_t *c  = (cand_t *)malloc(vocab * sizeof(cand_t));
    uint32_t result = 0;
    if (!adj || !c) {
        // allocation failure: fall back to plain argmax
        float best = logits[0];
        for (size_t i = 1; i < vocab; i++) if (logits[i] > best) { best = logits[i]; result = (uint32_t)i; }
        free(adj); free(c);
        return result;
    }

    vbs_sampler_apply_penalty(s, recent, n_recent, logits, adj, vocab);

    // greedy fast path (temperature 0 or top_k == 1) — still deterministic
    if (s->temperature <= 0.0f || s->top_k == 1) {
        float best = adj[0];
        for (size_t i = 1; i < vocab; i++) if (adj[i] > best) { best = adj[i]; result = (uint32_t)i; }
        free(adj); free(c);
        return result;
    }

    // temperature scaling
    float inv_t = 1.0f / s->temperature;
    for (size_t i = 0; i < vocab; i++) {
        c[i].logit = adj[i] * inv_t;
        c[i].idx = (uint32_t)i;
    }

    qsort(c, vocab, sizeof(cand_t), cand_cmp);

    // top-k
    size_t n = vocab;
    if (s->top_k > 0 && (size_t)s->top_k < n) n = (size_t)s->top_k;

    // top-p (nucleus): keep the smallest prefix whose mass reaches top_p
    if (s->top_p < 1.0f) {
        float mx = c[0].logit;
        double sum_exp = 0.0;
        for (size_t i = 0; i < n; i++) sum_exp += exp((double)(c[i].logit - mx));
        if (sum_exp > 0.0) {
            double cum = 0.0;
            size_t keep = n;
            for (size_t i = 0; i < n; i++) {
                cum += exp((double)(c[i].logit - mx)) / sum_exp;
                if (cum >= (double)s->top_p) { keep = i + 1; break; }
            }
            n = keep;
        }
    }

    // renormalized softmax sample over the survivors
    float mx = c[0].logit;
    double z = 0.0;
    for (size_t i = 0; i < n; i++) z += exp((double)(c[i].logit - mx));

    double r = (double)vbs_rand_float(s) * z;
    double acc = 0.0;
    size_t pick = n - 1;
    for (size_t i = 0; i < n; i++) {
        acc += exp((double)(c[i].logit - mx));
        if (r < acc) { pick = i; break; }
    }
    result = c[pick].idx;

    free(adj);
    free(c);
    return result;
}
