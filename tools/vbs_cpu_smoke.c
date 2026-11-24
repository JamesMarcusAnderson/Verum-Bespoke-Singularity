/* vbs_cpu_smoke.c — minimal CPU inference smoke test (test
 * utility, not engine code). Opens a model file with the CPU
 * reference path, runs N greedy steps from a prompt token list,
 * prints the token ids and tokens/sec. Used to verify a synthetic
 * model loads and produces finite logits before the loop runs.
 * Build: cc -std=c11 -O2 -D_GNU_SOURCE -Iinclude tools/vbs_cpu_smoke.c src/vbs_cpu.c
 *        src/vbs_model.c src/vbs_hotpath.c -lm -ldl -o /tmp/vbs_cpu_smoke
 * (vbs_hotpath.c is linked so the smoke test exercises the same
 * dispatch the engine uses; if it is absent, link without it —
 * vbs_cpu.c falls back.)
 */
#include "vbs_cpu.h"
#include "vbs_port.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_PROMPT 64

typedef struct {
    size_t n_tokens;
    uint32_t prompt[MAX_PROMPT];
    size_t n_prompt;
} smoke_args_t;

static int parse_args(int argc, char **argv, smoke_args_t *out) {
    out->n_tokens = 8;
    out->n_prompt = 0;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            out->n_tokens = (size_t)strtoull(argv[++i], NULL, 10);
        } else if (out->n_prompt < MAX_PROMPT) {
            out->prompt[out->n_prompt++] = (uint32_t)strtoul(argv[i], NULL, 10);
        }
    }
    if (out->n_prompt == 0) {
        out->prompt[0] = 1;
        out->n_prompt = 1;
    }
    return 1;
}

static uint32_t best_token(const float *logits, size_t n) {
    uint32_t best = 0;
    for (size_t i = 1; i < n; i++)
        if (logits[i] > logits[best]) best = (uint32_t)i;
    return best;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.gguf prompt_ids... [--tokens N]\n", argv[0]);
        return 64;
    }
    smoke_args_t args;
    parse_args(argc, argv, &args);

    char err[512] = {0};
    vbs_cpu_model_t *m = vbs_cpu_open(argv[1], 128, err, sizeof(err));
    if (!m) {
        fprintf(stderr, "open failed: %s\n", err);
        return 1;
    }
    printf("model open ok: vocab=%zu ctx=%zu eos=%u\n",
           vbs_cpu_vocab_size(m), vbs_cpu_context_capacity(m), vbs_cpu_eos_id(m));

    for (size_t i = 0; i < args.n_prompt; i++) {
        if (!vbs_cpu_step_ex(m, args.prompt[i], i + 1 == args.n_prompt,
                             err, sizeof(err))) {
            fprintf(stderr, "prompt step %zu failed: %s\n", i, err);
            vbs_cpu_close(m);
            return 1;
        }
    }
    double t0 = vbs_now_sec();
    printf("tokens:");
    for (size_t i = 0; i < args.n_tokens; i++) {
        uint32_t tok = best_token(vbs_cpu_logits(m), vbs_cpu_vocab_size(m));
        printf(" %u", tok);
        fflush(stdout);
        if (tok == vbs_cpu_eos_id(m)) break;
        if (!vbs_cpu_step(m, tok, err, sizeof(err))) {
            fprintf(stderr, "\ngen step %zu failed: %s\n", i, err);
            vbs_cpu_close(m);
            return 1;
        }
    }
    double dt = vbs_now_sec() - t0;
    printf("\n%.2f tok/s (%.1f ms/token, %zu tokens)\n",
           (double)args.n_tokens / dt, dt * 1000.0 / (double)args.n_tokens,
           args.n_tokens);
    vbs_cpu_close(m);
    return 0;
}
