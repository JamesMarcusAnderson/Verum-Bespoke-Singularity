#include "vbs_engine.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void usage(void) {
    fprintf(stderr,
        "Verum Bespoke Singularity (VBS) — Pure C GPU LLM Engine\n"
        "\n"
        "Usage:\n"
        "  ./verum-slice --inspect model.gguf\n"
        "  ./verum-slice --compile [--window-mib N] model.gguf\n"
        "  ./verum-slice --benchmark [--tokens N] [--window-mib N] model.gguf\n"
        "  ./verum-slice --generate --tokenizer tokenizer.json --prompt \"Hello\" model.gguf\n"
    );
}

int main(int argc, const char *argv[]) {
    if (argc < 3) {
        usage();
        return 64;
    }

    const char *mode = NULL;
    const char *model_path = NULL;
    const char *tokenizer_path = NULL;
    const char *prompt_text = "The universe is";
    size_t window_mib = 256;
    size_t bench_tokens = 128;
    size_t max_gen_tokens = 64;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--inspect") == 0 ||
            strcmp(argv[i], "--compile") == 0 ||
            strcmp(argv[i], "--benchmark") == 0 ||
            strcmp(argv[i], "--generate") == 0) {
            mode = argv[i];
        } else if (strcmp(argv[i], "--window-mib") == 0 && i + 1 < argc) {
            window_mib = (size_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            bench_tokens = (size_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            tokenizer_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt_text = argv[++i];
        } else if (argv[i][0] != '-') {
            model_path = argv[i];
        }
    }

    if (!mode || !model_path) {
        usage();
        return 64;
    }

    char err_buf[512] = {0};
    vbs_engine_t *engine = vbs_engine_create(model_path, window_mib, err_buf, sizeof(err_buf));
    if (!engine) {
        fprintf(stderr, "Initialization failed: %s\n", err_buf);
        return 1;
    }

    if (strcmp(mode, "--inspect") == 0) {
        vbs_engine_print_inspection(engine);
        vbs_engine_destroy(engine);
        return 0;
    }

    if (strcmp(mode, "--compile") == 0) {
        if (!vbs_engine_compile(engine, err_buf, sizeof(err_buf))) {
            fprintf(stderr, "Compilation refused: %s\n", err_buf);
            vbs_engine_destroy(engine);
            return 2;
        }
        vbs_engine_print_inspection(engine);
        vbs_engine_destroy(engine);
        return 0;
    }

    if (strcmp(mode, "--benchmark") == 0) {
        if (!vbs_engine_compile(engine, err_buf, sizeof(err_buf))) {
            fprintf(stderr, "Compilation refused: %s\n", err_buf);
            vbs_engine_destroy(engine);
            return 2;
        }

        uint32_t prompt_tokens[1] = { 1 };
        size_t vocab_size = engine->model_profile->vocab_size ? engine->model_profile->vocab_size : 32000;
        float *dummy_logits = (float *)calloc(vocab_size, sizeof(float));

        vbs_engine_forward(engine, prompt_tokens, 1, dummy_logits);

        double t_start = get_time_sec();
        for (size_t i = 0; i < bench_tokens; i++) {
            vbs_engine_forward(engine, prompt_tokens, 1, dummy_logits);
        }
        double t_end = get_time_sec();

        double elapsed = t_end - t_start;
        double tok_per_sec = (double)bench_tokens / (elapsed > 0.0 ? elapsed : 1e-6);
        double model_gb = (double)engine->model_profile->total_model_bytes / 1e9;

        printf("\n=== Performance Telemetry ===\n");
        printf("Decode Throughput:   %.2f tokens/sec\n", tok_per_sec);
        printf("Effective Bandwidth: %.2f GB/s\n", tok_per_sec * model_gb);
        printf("=============================\n\n");

        free(dummy_logits);
        vbs_engine_destroy(engine);
        return 0;
    }

    if (strcmp(mode, "--generate") == 0) {
        if (!tokenizer_path) {
            fprintf(stderr, "Error: --generate requires --tokenizer /path/to/tokenizer.json\n");
            vbs_engine_destroy(engine);
            return 1;
        }
        tokenizer_t *tok = tokenizer_load(tokenizer_path);
        if (!tok) {
            fprintf(stderr, "Error: Failed to load tokenizer from %s\n", tokenizer_path);
            vbs_engine_destroy(engine);
            return 1;
        }

        if (!vbs_engine_compile(engine, err_buf, sizeof(err_buf))) {
            fprintf(stderr, "Compilation refused: %s\n", err_buf);
            tokenizer_free(tok);
            vbs_engine_destroy(engine);
            return 2;
        }

        uint32_t prompt_tokens[2048];
        size_t n_prompt = tokenizer_encode(tok, prompt_text, prompt_tokens, 2048);

        size_t vocab_size = engine->model_profile->vocab_size ? engine->model_profile->vocab_size : 32000;
        float *logits = (float *)calloc(vocab_size, sizeof(float));

        // Prefill prompt tokens
        for (size_t i = 0; i < n_prompt; i++) {
            vbs_engine_forward(engine, &prompt_tokens[i], 1, logits);
        }

        printf("%s", prompt_text);
        fflush(stdout);

        // Simple argmax sample
        uint32_t cur_token = 0;
        float max_logit = -1e9f;
        for (size_t i = 0; i < vocab_size; i++) {
            if (logits[i] > max_logit) { max_logit = logits[i]; cur_token = (uint32_t)i; }
        }

        for (size_t i = 0; i < max_gen_tokens; i++) {
            if ((int)cur_token == tokenizer_eos_id(tok)) break;

            char *piece = tokenizer_decode(tok, &cur_token, 1);
            if (piece) {
                printf("%s", piece);
                fflush(stdout);
                free(piece);
            }

            vbs_engine_forward(engine, &cur_token, 1, logits);

            max_logit = -1e9f;
            for (size_t j = 0; j < vocab_size; j++) {
                if (logits[j] > max_logit) { max_logit = logits[j]; cur_token = (uint32_t)j; }
            }
        }
        printf("\n");

        free(logits);
        tokenizer_free(tok);
        vbs_engine_destroy(engine);
        return 0;
    }

    usage();
    vbs_engine_destroy(engine);
    return 64;
}
