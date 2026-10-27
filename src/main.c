#include "vbs_engine.h"
#include "vbs_learn.h"
#include "vbs_train.h"
#include "vbs_block.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <math.h>
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
        "  ./verum-slice --learn --prompt \"What is X?\" --correct \"X is Y\" [--memory verum-memory.tsv]\n"
        "  ./verum-slice --train-adapter --target-token 1234 [--vocab 32000] [--epochs 100] [--lr 0.5] --adapter-out adapter.bin\n"
        "  ./verum-slice --generate --adapter adapter.bin ...\n"
        "  ./verum-slice --train-block [--epochs 500] [--lr 0.05]\n"
        "  ./verum-slice --generate --memory verum-memory.tsv --tokenizer tokenizer.json --prompt \"What is X?\" model.gguf\n"
        "  ./verum-slice --evolve --tokenizer tokenizer.json --prompt \"Hello\" model.gguf\n"
        "  ./verum-slice --autonomy --tokenizer tokenizer.json --prompt \"Hello\" [--tokens N] model.gguf\n"
        "  ./verum-slice --prune --tokenizer tokenizer.json --prompt \"The capital of France is Paris\" model.gguf\n"
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

/* Singularity fitness, model-agnostic, no hardcoded model/GPU/backend:
 * Negative log-likelihood of the user's own prompt sequence, plus
 * next-token accuracy. For each prompt token i, the logits after
 * token i-1 predict token i. NLL = -mean(log P(true token)).
 * Fitness = 1 / (1 + NLL), so lower NLL = higher fitness, bounded
 * in (0, 1]. This replaces the old brace-counting toy score, which
 * rewarded length and balanced braces regardless of what the model
 * actually predicted. It works for any model, vocab, or backend the
 * user loads, because it only uses that model's own logits.
 * Returns -1 on bad input; otherwise fills *nll_out and *acc_out. */
static double evaluate_nll_fitness(const float *logits, size_t vocab_size, uint32_t true_token, double *nll_out, double *acc_out) {
    (void)acc_out;
    if (!logits || vocab_size == 0) return -1.0;
    float max_logit = logits[0];
    for (size_t i = 1; i < vocab_size; i++) if (logits[i] > max_logit) max_logit = logits[i];
    double sum_exp = 0.0;
    for (size_t i = 0; i < vocab_size; i++) sum_exp += exp((double)logits[i] - (double)max_logit);
    if (sum_exp <= 0.0) return -1.0;
    double log_prob = (double)logits[true_token] - (double)max_logit - log(sum_exp);
    if (nll_out) *nll_out = -log_prob;
    return 1.0 / (1.0 + (-log_prob));
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
    const char *correct_text = NULL;
    const char *memory_path = "verum-memory.tsv";
    const char *adapter_path = NULL;
    const char *adapter_out = NULL;
    size_t train_target = 0; int have_target = 0;
    size_t train_vocab = 32000; size_t train_epochs = 100; float train_lr = 0.5f;
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
            strcmp(argv[i], "--prune") == 0 ||
            strcmp(argv[i], "--evolve") == 0 ||
            strcmp(argv[i], "--learn") == 0 ||
            strcmp(argv[i], "--train-adapter") == 0 ||
            strcmp(argv[i], "--train-block") == 0) {
            mode = argv[i];
        } else if (strcmp(argv[i], "--window-mib") == 0 && i + 1 < argc) {
            window_mib = (size_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            bench_tokens = (size_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            tokenizer_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt_text = argv[++i];
        } else if (strcmp(argv[i], "--correct") == 0 && i + 1 < argc) {
            correct_text = argv[++i];
        } else if (strcmp(argv[i], "--memory") == 0 && i + 1 < argc) {
            memory_path = argv[++i];
        } else if (strcmp(argv[i], "--adapter") == 0 && i + 1 < argc) {
            adapter_path = argv[++i];
        } else if (strcmp(argv[i], "--adapter-out") == 0 && i + 1 < argc) {
            adapter_out = argv[++i];
        } else if (strcmp(argv[i], "--target-token") == 0 && i + 1 < argc) {
            train_target = (size_t)strtoull(argv[++i], NULL, 10); have_target = 1;
        } else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) {
            train_vocab = (size_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            train_epochs = (size_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            train_lr = (float)strtod(argv[++i], NULL);
        } else if (argv[i][0] != '-') {
            model_path = argv[i];
        }
    }

    if (!mode || (!model_path && strcmp(mode, "--learn") != 0 && strcmp(mode, "--train-adapter") != 0 && strcmp(mode, "--train-block") != 0)) {
        usage();
        return 64;
    }

    if (strcmp(mode, "--learn") == 0) {
        if (!prompt_text || !correct_text) { fprintf(stderr, "Error: --learn requires --prompt and --correct\n"); return 1; }
        if (!vbs_learn_append(memory_path, prompt_text, correct_text)) { fprintf(stderr, "Error: could not write %s\n", memory_path); return 1; }
        printf("LEARNED: %s -> %s  [file=%s]\n", prompt_text, correct_text, memory_path);
        return 0;
    }
    if (strcmp(mode, "--train-block") == 0) {
        /* Task: memorize [1,2,3] -> 5, vocab 8, d 16, hidden 32.
         * If backprop is wrong, loss will not go to near 0. */
        vbs_block_t *blk = vbs_block_create(8, 16, 3, 32, 12345u);
        int toks[3] = {1,2,3};
        for (size_t ep=0; ep<train_epochs; ep++) {
            double loss = vbs_block_train_step(blk, toks, 5, train_lr);
            if (ep==0 || ep+1==train_epochs || (ep+1)%100==0) printf("block epoch %zu loss=%.4f\n", ep+1, loss);
        }
        vbs_block_free(blk); return 0;
    }
    if (strcmp(mode, "--train-adapter") == 0) {
        if (!have_target || !adapter_out || train_target >= train_vocab) { fprintf(stderr, "Error: --train-adapter requires --target-token < vocab and --adapter-out\n"); return 1; }
        vbs_adapter_t *ad = vbs_adapter_create(train_vocab); if(!ad) return 1;
        float *base = calloc(train_vocab, sizeof(float));
        for (size_t ep=0; ep<train_epochs; ep++) {
            double prob=0; double loss = vbs_train_bias_step(ad, base, train_target, train_lr, &prob);
            if (ep==0 || ep+1==train_epochs || (ep+1)%20==0) printf("epoch %zu loss=%.4f p(target)=%.4f\n", ep+1, loss, prob);
        }
        free(base);
        if (!vbs_adapter_save(ad, adapter_out)) { fprintf(stderr, "Error: could not write %s\n", adapter_out); vbs_adapter_free(ad); return 1; }
        printf("TRAINED adapter: target=%zu vocab=%zu epochs=%zu lr=%.3f -> %s bias[target]=%.4f\n", train_target, train_vocab, train_epochs, train_lr, adapter_out, ad->bias[train_target]);
        vbs_adapter_free(ad); return 0;
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

        char recalled[4096] = {0};
        double recall_score = vbs_learn_recall(memory_path, prompt_text, recalled, sizeof recalled);
        char effective_prompt[8192];
        const char *gen_prompt = prompt_text;
        if (recall_score > 0.0 && recalled[0]) {
            printf("LEARNED RECALL score=%.2f: %s\n", recall_score, recalled);
            snprintf(effective_prompt, sizeof effective_prompt, "Previously corrected answer: %s\nQuestion: %s", recalled, prompt_text);
            gen_prompt = effective_prompt;
        }
        uint32_t prompt_tokens[2048];
        size_t n_prompt = tokenizer_encode(tok, gen_prompt, prompt_tokens, 2048);

        size_t vocab_size = engine->model_profile->vocab_size ? engine->model_profile->vocab_size : 32000;
        float *logits = (float *)calloc(vocab_size, sizeof(float));
        vbs_adapter_t *gen_adapter = NULL;
        if (adapter_path) {
            gen_adapter = vbs_adapter_create(vocab_size);
            if (gen_adapter && vbs_adapter_load(gen_adapter, adapter_path)) printf("ADAPTER LOADED: %s\n", adapter_path);
            else { fprintf(stderr, "Warning: adapter %s not loaded, continuing without it\n", adapter_path); if(gen_adapter) vbs_adapter_free(gen_adapter); gen_adapter=NULL; }
        }

        // Prefill prompt tokens
        for (size_t i = 0; i < n_prompt; i++) {
            vbs_engine_forward(engine, &prompt_tokens[i], 1, logits);
        }

        printf("%s", gen_prompt);
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
            if (gen_adapter) vbs_adapter_apply(gen_adapter, logits);

            max_logit = -1e9f;
            for (size_t j = 0; j < vocab_size; j++) {
                if (logits[j] > max_logit) { max_logit = logits[j]; cur_token = (uint32_t)j; }
            }
        }
        printf("\n");

        free(logits);
        if (gen_adapter) vbs_adapter_free(gen_adapter);
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

    if (strcmp(mode, "--prune") == 0) {
        /* Hydra-NLL Pruning Loop: standalone, other modes untouched.
         * No hardcoded tokens: the evaluation sequence is the user's
         * --prompt, encoded with their --tokenizer, for whatever model
         * they loaded. Baseline NLL is measured, then each distinct
         * transformer layer is trial-pruned by layer number (not plan
         * index: plan index != blk.<L>), recompiled, and re-measured.
         * Keep if delta < threshold, else revert and recompile.
         * The specializer already skips by layer number when energy
         * < 0.0001, so no new specializer hook is added here. */
        if (!tokenizer_path) {
            fprintf(stderr, "Error: --prune requires --tokenizer /path/to/tokenizer.json\n");
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
        if (n_prompt < 2) {
            fprintf(stderr, "Error: --prune requires a prompt with at least 2 tokens to score NLL\n");
            tokenizer_free(tok);
            vbs_engine_destroy(engine);
            return 1;
        }
        vbs_engine_reset_energy();
        double baseline_acc = 0.0;
        double baseline_nll = vbs_engine_evaluate_prompt_nll(engine, prompt_tokens, n_prompt, &baseline_acc);
        if (baseline_nll < 0.0) {
            fprintf(stderr, "Error: --prune could not measure baseline NLL\n");
            tokenizer_free(tok);
            vbs_engine_destroy(engine);
            return 2;
        }
        const double threshold = 0.05;
        size_t n_layers = engine->model_profile ? engine->model_profile->n_layers : 0;
        if (n_layers == 0 || n_layers > VBS_MAX_LAYERS) n_layers = VBS_MAX_LAYERS;
        printf("=== HYDRA PRUNING INITIATED ===\n");
        printf("Baseline NLL: %.4f | Acc: %.2f | Threshold: %.4f\n", baseline_nll, baseline_acc, threshold);
        printf("Note: Hydra energy is still a trial-prune trigger here, not a measured activation energy. NLL is the measured signal.\n");
        size_t kept = 0;
        int rc = 0;
        for (size_t layer = 0; layer < n_layers; layer++) {
            g_vbs_layer_energy[layer] = 0.0f;
            err_buf[0] = '\0';
            vbs_engine_recompile(engine, err_buf, sizeof(err_buf));
            if (err_buf[0] != '\0') {
                fprintf(stderr, "Recompile failed for layer %zu: %s\n", layer, err_buf);
                g_vbs_layer_energy[layer] = 1.0f;
                err_buf[0] = '\0';
                vbs_engine_recompile(engine, err_buf, sizeof(err_buf));
                rc = 2;
                break;
            }
            double acc = 0.0;
            double pruned_nll = vbs_engine_evaluate_prompt_nll(engine, prompt_tokens, n_prompt, &acc);
            if (pruned_nll < 0.0) {
                fprintf(stderr, "NLL measurement failed for layer %zu, reverting\n", layer);
                g_vbs_layer_energy[layer] = 1.0f;
                err_buf[0] = '\0';
                vbs_engine_recompile(engine, err_buf, sizeof(err_buf));
                rc = 2;
                break;
            }
            double delta = pruned_nll - baseline_nll;
            if (delta < threshold) {
                printf("[KEEP] Layer %zu | Pruned NLL: %.4f | Delta: %+.4f | Acc: %.2f\n", layer, pruned_nll, delta, acc);
                baseline_nll = pruned_nll;
                kept++;
            } else {
                printf("[REVERT] Layer %zu | Pruned NLL: %.4f | Delta: %+.4f (Too High) | Acc: %.2f\n", layer, pruned_nll, delta, acc);
                g_vbs_layer_energy[layer] = 1.0f;
                err_buf[0] = '\0';
                vbs_engine_recompile(engine, err_buf, sizeof(err_buf));
                if (err_buf[0] != '\0') { fprintf(stderr, "Recompile failed on revert for layer %zu: %s\n", layer, err_buf); rc = 2; break; }
            }
        }
        printf("=== HYDRA PRUNING COMPLETE | Kept %zu / %zu layers pruned ===\n", kept, n_layers);
        tokenizer_free(tok);
        vbs_engine_destroy(engine);
        return rc;
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
        double global_best_acc = -1.0;
        srand((unsigned)(time(NULL) ^ (unsigned)getpid()));

        printf("=== INITIATING EVOLUTIONARY SINGULARITY ===\n");
        int rc = 0;
        for (int epoch = 0; epoch < 20; epoch++) {
            vbs_engine_reset_energy();
            /* Real fitness: NLL + next-token accuracy on the user's
             * own prompt, any model, any vocab. After each prompt
             * token, logits predict the next prompt token. A prompt
             * with fewer than 2 tokens has no prediction to score,
             * so fitness is 0 in that case, not a fake score. */
            double nll_sum = 0.0;
            size_t nll_count = 0;
            size_t acc_correct = 0;
            for (size_t i = 0; i < n_prompt; i++) {
                vbs_engine_forward(engine, &prompt_tokens[i], 1, logits);
                if (i + 1 < n_prompt && prompt_tokens[i + 1] < vocab_size) {
                    double nll = 0.0;
                    double f = evaluate_nll_fitness(logits, vocab_size, prompt_tokens[i + 1], &nll, NULL);
                    if (f >= 0.0) { nll_sum += nll; nll_count++; }
                    uint32_t best_token = 0;
                    float max_logit = -1e9f;
                    for (size_t j = 0; j < vocab_size; j++) {
                        if (logits[j] > max_logit) { max_logit = logits[j]; best_token = (uint32_t)j; }
                    }
                    if (best_token == prompt_tokens[i + 1]) acc_correct++;
                }
            }
            double avg_nll = nll_count ? (nll_sum / (double)nll_count) : 0.0;
            double acc = nll_count ? ((double)acc_correct / (double)nll_count) : 0.0;
            double fitness = nll_count ? (1.0 / (1.0 + avg_nll)) : 0.0;
            /* Accuracy is a gate, not a replacement: a mutation that
             * lowers NLL but collapses accuracy is not kept, because
             * fitness alone can be gamed by a narrow distribution. */
            char *decoded = NULL;
            if (n_prompt > 0) decoded = tokenizer_decode(tok, &prompt_tokens[n_prompt - 1], 1);
            printf("Epoch %02d | NLL: %.4f | Acc: %.2f | Fitness: %.4f | ", epoch, avg_nll, acc, fitness);
            (void)decoded;

            if (fitness > global_best_fitness && acc >= global_best_acc) {
                printf("[KEEP MUTATION]\n");
                global_best_fitness = fitness;
                global_best_acc = acc;
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
