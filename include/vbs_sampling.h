#ifndef VBS_SAMPLING_H
#define VBS_SAMPLING_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float    temperature;     // 0 = greedy argmax (deterministic)
    int      top_k;           // 0 = disabled; otherwise keep k best
    float    top_p;           // 1.0 = disabled; nucleus sampling
    float    repeat_penalty;  // 1.0 = disabled; classic CTRL-style penalty
    int      repeat_last_n;   // how many recent tokens the penalty sees
    uint64_t seed;            // PRNG seed (0 => derived from clock)
    // internal PRNG state (xoshiro256++); not part of the public config
    int      seed_seeded;
    uint64_t seed_state[4];
} vbs_sampler_t;

// Sensible defaults: greedy, all extras off.
void vbs_sampler_init(vbs_sampler_t *s);

// Apply the repetition penalty to `logits` (vocab entries) for tokens listed
// in `recent` (oldest first, up to s->repeat_last_n entries). Writes the
// adjusted logits to `out` (may alias `logits`).
void vbs_sampler_apply_penalty(const vbs_sampler_t *s,
                               const uint32_t *recent, size_t n_recent,
                               const float *logits, float *out, size_t vocab);

// Full pipeline: penalty -> temperature -> top-k -> top-p -> sample.
// Returns a token id in [0, vocab).
uint32_t vbs_sampler_sample(vbs_sampler_t *s,
                            const float *logits, size_t vocab,
                            const uint32_t *recent, size_t n_recent);

// Deterministic PRNG (xoshiro256++), exposed for tests.
uint64_t vbs_rand_next(vbs_sampler_t *s);
float    vbs_rand_float(vbs_sampler_t *s); // [0,1)

#ifdef __cplusplus
}
#endif

#endif // VBS_SAMPLING_H
