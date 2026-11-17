/* vbs_cpu_smoke.c — minimal CPU inference smoke test (test utility, not engine code).
 * Opens a GGUF model with the CPU reference path, runs N greedy steps from a
 * prompt token list, prints the token ids and tokens/sec. Used to verify a
 * synthetic model loads and produces finite logits before the loop runs.
 * Build: cc -std=c11 -O2 -Iinclude tools/vbs_cpu_smoke.c src/vbs_cpu.c
 *        src/vbs_gguf.c src/vbs_hotpath.c -lm -ldl -o /tmp/vbs_cpu_smoke
 * (vbs_hotpath.c is linked so the smoke test exercises the same dispatch the
 * engine uses; if it is absent, link without it — vbs_cpu.c falls back.)
 */
#include "vbs_cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static uint32_t argmax(const float *logits, size_t n) {
    uint32_t best = 0;
    for (size_t i = 1; i < n; i++) if (logits[i] > logits[best]) best = (uint32_t)i;
    return best;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.gguf prompt_ids... [--tokens N]\n", argv[0]);
        return 64;
    }
    size_t n_tokens = 8;
    uint32_t prompt[64];
    size_t n_prompt = 0;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            n_tokens = (size_t)strtoull(argv[++i], NULL, 10);
        } else if (n_prompt < 64) {
            prompt[n_prompt++] = (uint32_t)strtoul(argv[i], NULL, 10);
        }
    }
    if (n_prompt == 0) { prompt[0] = 1; n_prompt = 1; }

    char err[512] = {0};
    vbs_cpu_model_t *m = vbs_cpu_open(argv[1], 128, err, sizeof(err));
    if (!m) { fprintf(stderr, "open failed: %s\n", err); return 1; }
    printf("model open ok: vocab=%zu ctx=%zu eos=%u\n",
           vbs_cpu_vocab_size(m), vbs_cpu_context_capacity(m), vbs_cpu_eos_id(m));

    for (size_t i = 0; i < n_prompt; i++) {
        if (!vbs_cpu_step_ex(m, prompt[i], i + 1 == n_prompt, err, sizeof(err))) {
            fprintf(stderr, "prompt step %zu failed: %s\n", i, err);
            vbs_cpu_close(m);
            return 1;
        }
    }
    double t0 = now_sec();
    printf("tokens:");
    for (size_t i = 0; i < n_tokens; i++) {
        uint32_t tok = argmax(vbs_cpu_logits(m), vbs_cpu_vocab_size(m));
        printf(" %u", tok);
        fflush(stdout);
        if (tok == vbs_cpu_eos_id(m)) break;
        if (!vbs_cpu_step(m, tok, err, sizeof(err))) {
            fprintf(stderr, "\ngen step %zu failed: %s\n", i, err);
            vbs_cpu_close(m);
            return 1;
        }
    }
    double dt = now_sec() - t0;
    printf("\n%.2f tok/s (%.1f ms/token, %zu tokens)\n",
           (double)n_tokens / dt, dt * 1000.0 / (double)n_tokens, n_tokens);
    vbs_cpu_close(m);
    return 0;
}
