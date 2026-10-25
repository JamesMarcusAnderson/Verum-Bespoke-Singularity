#include "vbs_engine.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <unistd.h>

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
        "  ./verum-slice --backends\n"
        "  ./verum-slice --inspect model.gguf\n"
        "  ./verum-slice --compile [--window-mib N] model.gguf\n"
        "  ./verum-slice --benchmark [--tokens N] [--window-mib N] model.gguf\n"
        "  ./verum-slice --generate --tokenizer tokenizer.json --prompt \"Hello\" model.gguf\n"
        "  ./verum-slice --evolve --tokenizer tokenizer.json --prompt \"Hello\" model.gguf\n"
        "  ./verum-slice --autonomy --tokenizer tokenizer.json --prompt \"Hello\" [--tokens N] model.gguf\n"
    );
}

/* Additive diagnostic, added without changing any existing mode:
 * reports which GPU backends were compiled in and, by probing each
 * backend's own init(), whether a usable device is present right now.
 * It needs no model file and no GPU, so it is safe to run on CI and on
 * a machine with no GPU at all. A backend with no device is reported
 * as "available: no", not as an error, so this mode always exits 0
 * when the report itself was produced. */
static void print_one_backend(const char *label, const vbs_backend_ops_t *ops) {
    bool available = false;
    size_t count = 0;
    if (ops && ops->init) {
        available = ops->init();
        if (available && ops->get_device_count) {
            count = ops->get_device_count();
        }
    }
    printf("  %s: available: %s, devices: %zu\n", label, available ? "yes" : "no", count);
    if (available && ops && ops->get_device) {
        for (size_t i = 0; i < count; i++) {
            vbs_device_t *dev = ops->get_device(i);
            if (dev) {
                printf("    [%zu] %s\n", i, dev->name);
            }
        }
    }
}

static int print_backends(void) {
    printf("=== Backend capability report ===\n");
#if defined(VBS_ENABLE_CUDA) && VBS_ENABLE_CUDA
    printf("  CUDA: compiled: yes\n");
#else
    printf("  CUDA: compiled: no\n");
#endif
#if defined(VBS_ENABLE_HIP) && VBS_ENABLE_HIP
    printf("  HIP: compiled: yes\n");
#else
    printf("  HIP: compiled: no\n");
#endif
#if defined(VBS_ENABLE_METAL) && VBS_ENABLE_METAL
    printf("  Metal: compiled: yes\n");
#else
    printf("  Metal: compiled: no\n");
#endif
    print_one_backend("CUDA", &g_vbs_cuda_backend);
    print_one_backend("HIP", &g_vbs_hip_backend);
#if defined(VBS_ENABLE_METAL) && VBS_ENABLE_METAL
    print_one_backend("Metal", &g_vbs_metal_backend);
#endif
    printf("===================================\n");
    return 0;
}

static double evaluate_fitness(const char *text, size_t len) {
    if (!text || len == 0) return -1000.0;
    double score = (double)len;
    int open_braces = 0, close_braces = 0;
    for (size_t i = 0; i < len; i++) {
        if (text[i] == '{') open_braces++;
        if (text[i] == '}') close_braces++;
        if (isalnum((unsigned char)text[i])) score += 1.0;
    }
    if (open_braces > 0 && open_braces == close_braces) score += 50.0;
    return score;
}

int main(int argc, const char *argv[]) {
    /* Model-free diagnostic: handled before the model requirement below,
     * so "./verum-slice --backends" works with a single argument. */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--backends") == 0) {
            return print_backends();
        }
    }

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
            strcmp(argv[i], "--generate") == 0 ||
            strcmp(argv[i], "--autonomy") == 0 ||
            strcmp(argv[i], "--evolve") == 0) {
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

    if (strcmp(mode, "--autonomy") == 0) {
        /* PGAE Phase 1: standalone, --generate / --benchmark / --evolve
         * above are untouched. Engine owns the generation loop; main
         * only waits, then decodes. --tokens sets max generation
         * tokens when given, otherwise 64. */
        if (!tokenizer_path) {
            fprintf(stderr, "Error: --autonomy requires --tokenizer /path/to/tokenizer.json\n");
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
        if (n_prompt == 0) { prompt_tokens[0] = 1; n_prompt = 1; }
        size_t max_gen = (bench_tokens != 128) ? bench_tokens : max_gen_tokens;
        /* Note: bench_tokens defaults to 128, so a user who did not
         * pass --tokens gets 64 here, matching --generate. */
        if (argc > 0) { /* silence unused-variable warnings in some builds */ }
        uint32_t eos_id = (uint32_t)tokenizer_eos_id(tok);
        if (tokenizer_eos_id(tok) < 0) eos_id = VBS_PGAE_EOS_NONE;
        printf("PGAE PHASE 1 ENGAGED: engine owns the generation loop, main thread waiting.\n");
        printf("Note: this is NOT a persistent device kernel yet. Per-token launches still go through the JIT forward path; see vbs_engine.h for Phase 2.\n");
        fflush(stdout);
        uint32_t *gen_tokens = NULL;
        size_t n_gen = 0;
        bool stopped_on_eos = false;
        if (!vbs_engine_forward_autonomy(engine, prompt_tokens, n_prompt, max_gen, eos_id, &gen_tokens, &n_gen, &stopped_on_eos)) {
            fprintf(stderr, "Error: --autonomy generation failed\n");
            tokenizer_free(tok);
            vbs_engine_destroy(engine);
            return 2;
        }
        printf("%s", prompt_text);
        fflush(stdout);
        if (gen_tokens) {
            char *piece = tokenizer_decode(tok, gen_tokens, n_gen);
            if (piece) { printf("%s", piece); free(piece); }
            free(gen_tokens);
        }
        printf("\n");
        printf("PGAE control: state=EXIT, position=%u, generated=%zu, stopped_on_eos=%s\n",
               engine->autonomy_control.position, n_gen, stopped_on_eos ? "yes" : "no");
        tokenizer_free(tok);
        vbs_engine_destroy(engine);
        return 0;
    }

    if (strcmp(mode, "--evolve") == 0) {
        /* Standalone path: --generate / --benchmark above are untouched. */
        if (!tokenizer_path) {
            fprintf(stderr, "Error: --evolve requires --tokenizer /path/to/tokenizer.json\n");
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
        /* Output projection: the plan node named "output" that is not
         * "output_norm". This is more precise than assuming the last
         * plan slot is the output projection. */
        size_t out_idx = engine->plan->n_plans;
        for (size_t i = 0; i < engine->plan->n_plans; i++) {
            const vbs_node_t *nd = engine->plan->plans[i].node;
            if (nd && nd->weight && strstr(nd->weight->name, "output") && !strstr(nd->weight->name, "output_norm")) {
                out_idx = i;
                break;
            }
        }
        if (out_idx >= engine->plan->n_plans) {
            fprintf(stderr, "Error: --evolve could not find an output projection plan node\n");
            tokenizer_free(tok);
            vbs_engine_destroy(engine);
            return 1;
        }
        const vbs_node_t *out_node = engine->plan->plans[out_idx].node;
        bool out_is_f32 = (out_node->weight->type == VBS_GGML_TYPE_F32);
        vbs_memory_block_t *out_block = NULL;
        if (engine->fabric && out_node->tensor_index < engine->fabric->n_blocks) {
            out_block = engine->fabric->blocks[out_node->tensor_index];
        }
        if (!out_is_f32) {
            fprintf(stderr, "Note: --evolve mutation skipped: output tensor %s is type %d, not FP32\n",
                    out_node->weight->name, (int)out_node->weight->type);
        }

        uint32_t prompt_tokens[2048];
        size_t n_prompt = tokenizer_encode(tok, prompt_text, prompt_tokens, 2048);
        if (n_prompt == 0) { prompt_tokens[0] = 1; n_prompt = 1; }

        size_t vocab_size = engine->model_profile->vocab_size ? engine->model_profile->vocab_size : 32000;
        float *logits = (float *)calloc(vocab_size, sizeof(float));
        float *best_snapshot = NULL;
        size_t best_snapshot_bytes = 0;
        double global_best_fitness = -1e9;
        srand((unsigned)(time(NULL) ^ (unsigned)getpid()));

        printf("=== INITIATING EVOLUTIONARY SINGULARITY ===\n");
        int rc = 0;
        for (int epoch = 0; epoch < 20; epoch++) {
            vbs_engine_reset_energy();
            for (size_t i = 0; i < n_prompt; i++) {
                vbs_engine_forward(engine, &prompt_tokens[i], 1, logits);
            }
            uint32_t best_token = 0;
            float max_logit = -1e9f;
            for (size_t i = 0; i < vocab_size; i++) {
                if (logits[i] > max_logit) { max_logit = logits[i]; best_token = (uint32_t)i; }
            }
            char *decoded = tokenizer_decode(tok, &best_token, 1);
            double fitness = evaluate_fitness(decoded, decoded ? strlen(decoded) : 0);
            printf("Epoch %02d | Token: %s | Fitness: %.2f | ", epoch, decoded ? decoded : "", fitness);

            if (fitness > global_best_fitness) {
                printf("[KEEP MUTATION]\n");
                global_best_fitness = fitness;
                /* Snapshot the current (kept) weights, for a real revert. */
                if (out_is_f32 && out_block && out_block->data) {
                    if (!best_snapshot || best_snapshot_bytes != out_block->size) {
                        free(best_snapshot);
                        best_snapshot = (float *)malloc(out_block->size);
                        best_snapshot_bytes = best_snapshot ? out_block->size : 0;
                    }
                    if (best_snapshot) memcpy(best_snapshot, out_block->data, out_block->size);
                }
            } else {
                printf("[REVERT MUTATION]\n");
                if (best_snapshot && out_block && out_block->data) {
                    memcpy(out_block->data, best_snapshot, best_snapshot_bytes);
                    if (out_block->current_tier == VBS_TIER_VRAM && out_block->gpu_buffer && engine->backend && engine->backend->copy_to_device) {
                        engine->backend->copy_to_device(out_block->gpu_buffer, out_block->data, best_snapshot_bytes);
                    }
                }
            }

            if (out_is_f32) {
                vbs_engine_mutate_layer(engine, out_idx, 0.05f);
            }

            if (epoch % 5 == 0 && epoch > 0) {
                printf("--- HYDRA PROTOCOL TRIGGERED: Recompiling JIT ===\n");
                err_buf[0] = '\0';
                vbs_engine_recompile(engine, err_buf, sizeof(err_buf));
                if (err_buf[0] != '\0') {
                    fprintf(stderr, "Recompile failed: %s\n", err_buf);
                    rc = 2;
                    free(decoded);
                    break;
                }
                printf("--- HYDRA RECOMPILE DONE (prunes only layers with energy < 0.0001; reset sets all to 1.0) ---\n");
            }
            free(decoded);
        }

        printf("=== EVOLUTION COMPLETE ===\n");
        free(best_snapshot);
        free(logits);
        tokenizer_free(tok);
        vbs_engine_destroy(engine);
        return rc;
    }

    usage();
    vbs_engine_destroy(engine);
    return 64;
}
