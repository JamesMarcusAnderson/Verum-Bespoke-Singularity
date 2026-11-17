#include "vbs_cpu.h"
#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(FILE *out) {
    fprintf(out, "Usage: vbs-cpu --model MODEL.gguf [--context N] [--max-new N] [--eos ID] [--dump-logits]\n"
                 "Read whitespace-separated prompt token IDs from one line of stdin.\n"
                 "Write one JSON result to stdout. No tokenizer or model is bundled.\n");
}
static bool number(const char *s, unsigned long long *v) {
    char *end = NULL;
    if (!s || !*s || *s == '-') return false;
    errno = 0; *v = strtoull(s, &end, 10);
    return errno == 0 && end && *end == 0;
}

int main(int argc, char **argv) {
    const char *path = NULL;
    unsigned long long ctx = 256, max_new = 32, eos_arg = 0;
    bool override_eos = false, dump_logits = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help")) { usage(stdout); return 0; }
        else if (!strcmp(argv[i], "--model") && i + 1 < argc) path = argv[++i];
        else if (!strcmp(argv[i], "--context") && i + 1 < argc && number(argv[i + 1], &ctx)) i++;
        else if (!strcmp(argv[i], "--max-new") && i + 1 < argc && number(argv[i + 1], &max_new)) i++;
        else if (!strcmp(argv[i], "--eos") && i + 1 < argc && number(argv[i + 1], &eos_arg)) { i++; override_eos = true; }
        else if (!strcmp(argv[i], "--dump-logits")) dump_logits = true;
        else { fprintf(stderr, "Unknown or invalid argument: %s\n", argv[i]); usage(stderr); return 64; }
    }
    if (!path || !ctx || ctx > 65536 || max_new > 4096 || (override_eos && eos_arg > UINT32_MAX)) {
        usage(stderr); return 64;
    }
    char err[512] = {0};
    vbs_cpu_model_t *m = vbs_cpu_open(path, (size_t)ctx, err, sizeof(err));
    if (!m) { fprintf(stderr, "CPU model load failed: %s\n", err[0] ? err : "invalid model"); return 1; }
    uint32_t eos = override_eos ? (uint32_t)eos_arg : vbs_cpu_eos_id(m);
    uint32_t *prompt = (uint32_t *)malloc((size_t)ctx * sizeof(*prompt));
    uint32_t *generated = max_new ? (uint32_t *)malloc((size_t)max_new * sizeof(*generated)) : NULL;
    if (!prompt || (max_new && !generated)) {
        fprintf(stderr, "Out of memory allocating token buffers\n"); free(prompt); free(generated); vbs_cpu_close(m); return 1;
    }
    size_t n_prompt = 0;
    /* Bound input before parsing: at most 11 decimal digits + one space per ID. */
    size_t line_cap = (size_t)ctx * 12 + 2;
    char *line = (char *)malloc(line_cap);
    if (!line || !fgets(line, (int)line_cap, stdin)) {
        fprintf(stderr, "Expected one line of token IDs on stdin\n"); free(line); free(prompt); free(generated); vbs_cpu_close(m); return 64;
    }
    if (!strchr(line, '\n') && !feof(stdin)) {
        fprintf(stderr, "Token input exceeds context limit\n"); free(line); free(prompt); free(generated); vbs_cpu_close(m); return 64;
    }
    char *save = NULL;
    for (char *p = strtok_r(line, " \t\r\n", &save); p; p = strtok_r(NULL, " \t\r\n", &save)) {
        unsigned long long v;
        if (!number(p, &v) || v > UINT32_MAX || v >= vbs_cpu_vocab_size(m) || n_prompt >= (size_t)ctx) {
            fprintf(stderr, "Invalid or too many prompt token IDs\n"); free(line); free(prompt); free(generated); vbs_cpu_close(m); return 64;
        }
        prompt[n_prompt++] = (uint32_t)v;
    }
    free(line);
    if (!n_prompt || (max_new > 0 && n_prompt + (size_t)max_new - 1 > (size_t)ctx)) {
        fprintf(stderr, "Prompt plus generation exceeds CPU context capacity\n"); free(prompt); free(generated); vbs_cpu_close(m); return 64;
    }
    for (size_t i = 0; i < n_prompt; i++) {
        if (!vbs_cpu_step_ex(m, prompt[i], i + 1 == n_prompt, err, sizeof(err))) {
            fprintf(stderr, "CPU inference failed: %s\n", err); free(prompt); free(generated); vbs_cpu_close(m); return 1;
        }
    }
    size_t n_gen = 0;
    bool stopped = false;
    for (size_t i = 0; i < (size_t)max_new; i++) {
        const float *logits = vbs_cpu_logits(m);
        uint32_t best = 0;
        for (size_t k = 1; k < vbs_cpu_vocab_size(m); k++) if (logits[k] > logits[best]) best = (uint32_t)k;
        if (best == eos) { stopped = true; break; }
        generated[n_gen++] = best;
        if (i + 1 < (size_t)max_new && !vbs_cpu_step(m, best, err, sizeof(err))) {
            fprintf(stderr, "CPU inference failed: %s\n", err); free(prompt); free(generated); vbs_cpu_close(m); return 1;
        }
    }
    printf("{\"prompt_tokens\":%zu,\"generated\":[", n_prompt);
    for (size_t i = 0; i < n_gen; i++) printf("%s%" PRIu32, i ? "," : "", generated[i]);
    printf("],\"stopped_on_eos\":%s", stopped ? "true" : "false");
    if (dump_logits) {
        const float *logits = vbs_cpu_logits(m);
        printf(",\"logits\":[");
        for (size_t i = 0; i < vbs_cpu_vocab_size(m); i++) printf("%s%.9g", i ? "," : "", (double)logits[i]);
        printf("]");
    }
    printf("}\n");
    free(prompt); free(generated); vbs_cpu_close(m);
    return 0;
}
