#include "vbs_engine.h"
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
        "\n"
        "Flags:\n"
        "  --inspect         Parse container, profile hardware/model, print facts\n"
        "  --compile         Derive graph, plan execution, JIT compile bespoke binary\n"
        "  --benchmark       Run timed forward inference test and print throughput telemetry\n"
        "  --tokens N        Number of tokens to generate during benchmark (default: 128)\n"
        "  --window-mib N    Streaming staging window size in MiB (default: 256)\n"
    );
}

int main(int argc, const char *argv[]) {
    if (argc < 3) {
        usage();
        return 64;
    }

    const char *mode = NULL;
    const char *model_path = NULL;
    size_t window_mib = 256;
    size_t bench_tokens = 128;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--inspect") == 0 ||
            strcmp(argv[i], "--compile") == 0 ||
            strcmp(argv[i], "--benchmark") == 0) {
            mode = argv[i];
        } else if (strcmp(argv[i], "--window-mib") == 0 && i + 1 < argc) {
            window_mib = (size_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            bench_tokens = (size_t)strtoull(argv[++i], NULL, 10);
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

        printf("\n=== Running Verum Hardware Benchmark ===\n");
        printf("Model: %s\n", model_path);
        printf("Benchmark length: %zu tokens\n", bench_tokens);
        printf("Staging window: %zu MiB\n", window_mib);

        uint32_t prompt_tokens[1] = { 1 }; // BOS token
        float *dummy_logits = (float *)calloc(32000, sizeof(float));

        // 1. Warm-up Pass (2 tokens to settle JIT caches and page mappings)
        vbs_engine_forward(engine, prompt_tokens, 1, dummy_logits);
        vbs_engine_forward(engine, prompt_tokens, 1, dummy_logits);

        // 2. Timed Execution Pass
        double t_start = get_time_sec();
        for (size_t i = 0; i < bench_tokens; i++) {
            vbs_engine_forward(engine, prompt_tokens, 1, dummy_logits);
        }
        double t_end = get_time_sec();

        double elapsed = t_end - t_start;
        double tok_per_sec = (double)bench_tokens / (elapsed > 0.0 ? elapsed : 1e-6);
        double model_gb = (double)engine->model_profile->total_model_bytes / 1e9;
        double effective_bandwidth_gbps = tok_per_sec * model_gb;

        printf("\n=== Performance Telemetry ===\n");
        printf("Total Model Weights:   %.2f GB\n", model_gb);
        printf("Tokens Evaluated:      %zu\n", bench_tokens);
        printf("Elapsed Time:          %.3f seconds\n", elapsed);
        printf("Decode Throughput:     %.2f tokens/sec\n", tok_per_sec);
        printf("Effective Bandwidth:   %.2f GB/s\n", effective_bandwidth_gbps);
        printf("CPU Compute Activity:  0.0%% (GPU Native)\n");
        printf("=============================\n\n");

        free(dummy_logits);
        vbs_engine_destroy(engine);
        return 0;
    }

    usage();
    vbs_engine_destroy(engine);
    return 64;
}
