/* vbs_cpu_main.c — the standalone CPU inference binary (vbs-cpu).
 *
 * Reads one line of whitespace-separated prompt token IDs from
 * stdin, runs greedy CPU inference, and writes one JSON result to
 * stdout. No tokenizer or model is bundled: the caller supplies
 * token IDs and the model path. Exit codes: 0 success, 1 runtime
 * failure, 64 usage error.
 */

#include "vbs_cpu.h"
#include "vbs_port.h"
#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(FILE *out) {
    fprintf(out,
            "Usage: vbs-cpu --model MODEL.gguf [--context N] [--max-new N] [--eos ID] [--dump-logits]\n"
            "Read whitespace-separated prompt token IDs from one line of stdin.\n"
            "Write one JSON result to stdout. No tokenizer or model is bundled.\n");
}

/* Parse a non-negative decimal integer, fully consumed, no overflow. */
static bool parse_number(const char *s, unsigned long long *v) {
    char *end = NULL;
    if (!s || !*s || *s == '-') return false;
    errno = 0;
    *v = strtoull(s, &end, 10);
    return errno == 0 && end && *end == 0;
}

typedef struct {
    const char *model_path;
    unsigned long long context;
    unsigned long long max_new;
    unsigned long long eos_arg;
    bool override_eos;
    bool dump_logits;
} cpu_options_t;

/* Parse argv. Returns 0 on success, 64 on any unknown or invalid
 * argument (usage already printed). A flag whose value is missing
 * or fails to parse is treated as unknown, exactly like a flag that
 * was never recognized. */
static int parse_options(int argc, char **argv, cpu_options_t *o) {
    o->model_path = NULL;
    o->context = 256;
    o->max_new = 32;
    o->eos_arg = 0;
    o->override_eos = false;
    o->dump_logits = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            print_usage(stdout);
            return 1; /* help shown; caller exits 0 */
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            o->model_path = argv[++i];
        } else if (strcmp(argv[i], "--context") == 0 && i + 1 < argc &&
                   parse_number(argv[i + 1], &o->context)) {
            i++;
        } else if (strcmp(argv[i], "--max-new") == 0 && i + 1 < argc &&
                   parse_number(argv[i + 1], &o->max_new)) {
            i++;
        } else if (strcmp(argv[i], "--eos") == 0 && i + 1 < argc &&
                   parse_number(argv[i + 1], &o->eos_arg)) {
            i++;
            o->override_eos = true;
        } else if (strcmp(argv[i], "--dump-logits") == 0) {
            o->dump_logits = true;
        } else {
            fprintf(stderr, "Unknown or invalid argument: %s\n", argv[i]);
            print_usage(stderr);
            return 64;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    cpu_options_t o;
    int prc = parse_options(argc, argv, &o);
    if (prc == 1) return 0;
    if (prc != 0) return prc;
    if (!o.model_path || !o.context || o.context > 65536 || o.max_new > 4096 ||
        (o.override_eos && o.eos_arg > UINT32_MAX)) {
        print_usage(stderr);
        return 64;
    }

    char err[512] = { 0 };
    vbs_cpu_model_t *m = vbs_cpu_open(o.model_path, (size_t)o.context,
                                      err, sizeof(err));
    if (!m) {
        fprintf(stderr, "CPU model load failed: %s\n",
                err[0] ? err : "invalid model");
        return 1;
    }
    uint32_t eos = o.override_eos ? (uint32_t)o.eos_arg : vbs_cpu_eos_id(m);
    uint32_t *prompt = (uint32_t *)malloc((size_t)o.context * sizeof(*prompt));
    uint32_t *generated = o.max_new
        ? (uint32_t *)malloc((size_t)o.max_new * sizeof(*generated))
        : NULL;
    char *line = NULL;
    int rc = 0;
    if (!prompt || (o.max_new && !generated)) {
        fprintf(stderr, "Out of memory allocating token buffers\n");
        rc = 1;
        goto done;
    }

    /* Read one line of token IDs, bounded before parsing: at most
     * 11 decimal digits + one space per ID. */
    size_t n_prompt = 0;
    size_t line_cap = (size_t)o.context * 12 + 2;
    line = (char *)malloc(line_cap);
    if (!line || !fgets(line, (int)line_cap, stdin)) {
        fprintf(stderr, "Expected one line of token IDs on stdin\n");
        rc = 64;
        goto done;
    }
    if (!strchr(line, '\n') && !feof(stdin)) {
        fprintf(stderr, "Token input exceeds context limit\n");
        rc = 64;
        goto done;
    }
    char *save = NULL;
    for (char *p = vbs_strtok_r(line, " \t\r\n", &save); p;
         p = vbs_strtok_r(NULL, " \t\r\n", &save)) {
        unsigned long long v;
        if (!parse_number(p, &v) || v > UINT32_MAX ||
            v >= vbs_cpu_vocab_size(m) || n_prompt >= (size_t)o.context) {
            fprintf(stderr, "Invalid or too many prompt token IDs\n");
            rc = 64;
            goto done;
        }
        prompt[n_prompt++] = (uint32_t)v;
    }
    free(line);
    line = NULL;
    if (!n_prompt ||
        (o.max_new > 0 && n_prompt + (size_t)o.max_new - 1 > (size_t)o.context)) {
        fprintf(stderr, "Prompt plus generation exceeds CPU context capacity\n");
        rc = 64;
        goto done;
    }

    for (size_t i = 0; i < n_prompt; i++) {
        if (!vbs_cpu_step_ex(m, prompt[i], i + 1 == n_prompt, err, sizeof(err))) {
            fprintf(stderr, "CPU inference failed: %s\n", err);
            rc = 1;
            goto done;
        }
    }
    size_t n_gen = 0;
    bool stopped = false;
    for (size_t i = 0; i < (size_t)o.max_new; i++) {
        const float *logits = vbs_cpu_logits(m);
        uint32_t best = 0;
        for (size_t k = 1; k < vbs_cpu_vocab_size(m); k++)
            if (logits[k] > logits[best]) best = (uint32_t)k;
        if (best == eos) {
            stopped = true;
            break;
        }
        generated[n_gen++] = best;
        if (i + 1 < (size_t)o.max_new &&
            !vbs_cpu_step(m, best, err, sizeof(err))) {
            fprintf(stderr, "CPU inference failed: %s\n", err);
            rc = 1;
            goto done;
        }
    }
    printf("{\"prompt_tokens\":%zu,\"generated\":[", n_prompt);
    for (size_t i = 0; i < n_gen; i++)
        printf("%s%" PRIu32, i ? "," : "", generated[i]);
    printf("],\"stopped_on_eos\":%s", stopped ? "true" : "false");
    if (o.dump_logits) {
        const float *logits = vbs_cpu_logits(m);
        printf(",\"logits\":[");
        for (size_t i = 0; i < vbs_cpu_vocab_size(m); i++)
            printf("%s%.9g", i ? "," : "", (double)logits[i]);
        printf("]");
    }
    printf("}\n");
done:
    free(line);
    free(prompt);
    free(generated);
    vbs_cpu_close(m);
    return rc;
}
