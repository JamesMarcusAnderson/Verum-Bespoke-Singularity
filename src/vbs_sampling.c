/* vbs_sampling.c — sampling strategies for token generation.
 *
 * Greedy / temperature / top-k / nucleus (top-p) / repetition
 * penalty. The PRNG is xoshiro256++ (a published algorithm,
 * implemented here from the algorithm's description) seeded by
 * splitmix64; a fixed seed reproduces the exact same stream, which
 * the reproducibility test depends on. No dependencies.
 */

#include "vbs_sampling.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

void vbs_sampler_init(vbs_sampler_t *s) {
    if (!s) return;
    s->temperature = 0.0f;
    s->top_k = 0;
    s->top_p = 1.0f;
    s->repeat_penalty = 1.0f;
    s->repeat_last_n = 64;
    s->seed = 0;
    /* PRNG state MUST be zeroed: a stale seed_seeded flag here made
     * sampling read uninitialized state (caught by the
     * reproducibility test). */
    s->seed_seeded = 0;
    memset(s->seed_state, 0, sizeof(s->seed_state));
}

/* ---------------------------------------------------------------- */
/* xoshiro256++                                                     */
/* ---------------------------------------------------------------- */

static uint64_t rotate_left(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

/* splitmix64: one step of the published seeding generator. */
static uint64_t splitmix_step(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

uint64_t vbs_rand_next(vbs_sampler_t *s) {
    uint64_t *st = s->seed_state;
    if (!s->seed_seeded) {
        uint64_t seed = s->seed ? s->seed
                                : (uint64_t)time(NULL) ^ 0xD1B54A32D192ED03ull;
        for (int i = 0; i < 4; i++) st[i] = splitmix_step(&seed);
        s->seed_seeded = 1;
    }
    uint64_t result = rotate_left(st[0] + st[3], 23) + st[0];
    uint64_t t = st[1] << 17;
    st[2] ^= st[0];
    st[3] ^= st[1];
    st[1] ^= st[2];
    st[0] ^= st[3];
    st[2] ^= t;
    st[3] = rotate_left(st[3], 45);
    return result;
}

float vbs_rand_float(vbs_sampler_t *s) {
    /* 24 bits of mantissa -> uniform in [0,1) */
    return (float)(vbs_rand_next(s) >> 40) * (1.0f / 16777216.0f);
}

/* ---------------------------------------------------------------- */
/* repetition penalty (CTRL-style: divide positive, multiply negative) */
/* ---------------------------------------------------------------- */

void vbs_sampler_apply_penalty(const vbs_sampler_t *s,
                               const uint32_t *recent, size_t n_recent,
                               const float *logits, float *out, size_t vocab) {
    if (logits != out) memcpy(out, logits, vocab * sizeof(float));
    if (!recent || n_recent == 0 || s->repeat_penalty == 1.0f) return;

    size_t window = n_recent;
    if (s->repeat_last_n > 0 && window > (size_t)s->repeat_last_n)
        window = (size_t)s->repeat_last_n;
    const uint32_t *start = recent + (n_recent - window);
    for (size_t i = 0; i < window; i++) {
        uint32_t tok = start[i];
        if (tok >= vocab) continue;
        if (out[tok] > 0.0f)
            out[tok] /= s->repeat_penalty;
        else
            out[tok] *= s->repeat_penalty;
    }
}

/* ---------------------------------------------------------------- */
/* sampling pipeline                                                */
/* ---------------------------------------------------------------- */

typedef struct {
    float logit;
    uint32_t idx;
} candidate_t;

/* Sort by logit descending, ties broken by index ascending, so the
 * order is fully deterministic. */
static int candidate_compare(const void *a, const void *b) {
    const candidate_t *x = (const candidate_t *)a;
    const candidate_t *y = (const candidate_t *)b;
    if (x->logit > y->logit) return -1;
    if (x->logit < y->logit) return 1;
    if (x->idx < y->idx) return -1;
    if (x->idx > y->idx) return 1;
    return 0;
}

/* Plain argmax over `logits` (used for the greedy path and as the
 * allocation-failure fallback). */
static uint32_t argmax_token(const float *logits, size_t vocab) {
    uint32_t best = 0;
    for (size_t i = 1; i < vocab; i++)
        if (logits[i] > logits[best]) best = (uint32_t)i;
    return best;
}

uint32_t vbs_sampler_sample(vbs_sampler_t *s,
                            const float *logits, size_t vocab,
                            const uint32_t *recent, size_t n_recent) {
    if (!s || !logits || vocab == 0) return 0;

    float *adjusted = (float *)malloc(vocab * sizeof(float));
    candidate_t *cands = (candidate_t *)malloc(vocab * sizeof(candidate_t));
    if (!adjusted || !cands) {
        /* Allocation failure: fall back to plain argmax. */
        uint32_t fallback = argmax_token(logits, vocab);
        free(adjusted);
        free(cands);
        return fallback;
    }

    vbs_sampler_apply_penalty(s, recent, n_recent, logits, adjusted, vocab);

    /* Greedy fast path (temperature 0 or top_k == 1) — still
     * deterministic. Uses the penalty-adjusted logits. */
    if (s->temperature <= 0.0f || s->top_k == 1) {
        uint32_t best = argmax_token(adjusted, vocab);
        free(adjusted);
        free(cands);
        return best;
    }

    /* Temperature scaling, then sort best-first. */
    float inv_temp = 1.0f / s->temperature;
    for (size_t i = 0; i < vocab; i++) {
        cands[i].logit = adjusted[i] * inv_temp;
        cands[i].idx = (uint32_t)i;
    }
    qsort(cands, vocab, sizeof(candidate_t), candidate_compare);

    /* Top-k: keep only the k best. */
    size_t n = vocab;
    if (s->top_k > 0 && (size_t)s->top_k < n) n = (size_t)s->top_k;

    /* Top-p (nucleus): keep the smallest prefix whose mass reaches top_p. */
    if (s->top_p < 1.0f) {
        float peak = cands[0].logit;
        double total = 0.0;
        for (size_t i = 0; i < n; i++)
            total += exp((double)(cands[i].logit - peak));
        if (total > 0.0) {
            double cumulative = 0.0;
            size_t keep = n;
            for (size_t i = 0; i < n; i++) {
                cumulative += exp((double)(cands[i].logit - peak)) / total;
                if (cumulative >= (double)s->top_p) {
                    keep = i + 1;
                    break;
                }
            }
            n = keep;
        }
    }

    /* Renormalized softmax sample over the survivors. */
    float peak = cands[0].logit;
    double z = 0.0;
    for (size_t i = 0; i < n; i++) z += exp((double)(cands[i].logit - peak));
    double draw = (double)vbs_rand_float(s) * z;
    double acc = 0.0;
    size_t pick = n - 1;
    for (size_t i = 0; i < n; i++) {
        acc += exp((double)(cands[i].logit - peak));
        if (draw < acc) {
            pick = i;
            break;
        }
    }
    uint32_t result = cands[pick].idx;
    free(adjusted);
    free(cands);
    return result;
}
