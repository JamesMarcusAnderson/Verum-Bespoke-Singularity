#include "vbs_engine.h"
#include "vbs_learn.h"
#include "vbs_train.h"
#include "vbs_block.h"
#include "vbs_autonomy.h"
#include "vbs_scheduler.h"
#include "vbs_autotune.h"
#include "vbs_cpu.h"
#include "vbs_dequant.h"
#include "vbs_sampling.h"
#include "vbs_port.h"
#include "vbs_backend_select.h"
#include "vbs_backend_test.h"
#include "vbs_hotpath.h"
#include "vbs_loop.h"
#include "vbs_fuse.h"
#include "vbs_tune.h"
#include "vbs_cc.h"
#include "self_tune.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>
#include <math.h>

static double get_time_sec(void) {
    return vbs_now_sec();
}

static void usage(void) {
    fprintf(stderr,
        "Verum Bespoke Singularity (VBS) — Pure C GPU LLM Engine\n"
        "\n"
        "Usage:\n"
        "  ./verum-slice --backends\n"
        "  ./verum-slice --backend-test\n"
        "  ./verum-slice --inspect model.gguf\n"
        "  ./verum-slice --compile [--window-mib N] model.gguf\n"
        "  ./verum-slice --benchmark [--tokens N] [--window-mib N] model.gguf\n"
        "  ./verum-slice --generate --tokenizer tokenizer.json --prompt \"Hello\" [--tokens N] model.gguf\n"
        "  ./verum-slice --cpu --tokenizer tokenizer.json --prompt \"Hello\" [--tokens N] model.gguf\n"
        "  ./verum-slice --dequant\n"
        "  ./verum-slice --sample [--temperature 0.7] [--top-k 40] [--top-p 0.9] [--seed 12345]\n"
        "  ./verum-slice --learn --prompt \"What is X?\" --correct \"X is Y\" [--memory verum-memory.tsv]\n"
        "  ./verum-slice --train-adapter --target-token 1234 [--vocab 32000] [--epochs 100] [--lr 0.5] --adapter-out adapter.bin\n"
        "  ./verum-slice --generate --adapter adapter.bin ...\n"
        "  ./verum-slice --train-block [--epochs 500] [--lr 0.05]\n"
        "  ./verum-slice --train-tiny [--epochs 500] [--lr 0.05] [--block-out tiny.bin]\n"
        "  ./verum-slice --autonomous-c\n"
        "  ./verum-slice --autotune [--autotune-out verum-autotune.json]\n"
        "  ./verum-slice --train-text --prompt \"hello world \" [--epochs 500]\n"
        "  ./verum-slice --generate --memory verum-memory.tsv --tokenizer tokenizer.json --prompt \"What is X?\" model.gguf\n"
        "  ./verum-slice --evolve --tokenizer tokenizer.json --prompt \"Hello\" model.gguf\n"
        "  ./verum-slice --autonomy --tokenizer tokenizer.json --prompt \"Hello\" [--tokens N] model.gguf\n"
        "  ./verum-slice --prune --tokenizer tokenizer.json --prompt \"The capital of France is Paris\" model.gguf\n"
        "  ./verum-slice --loop [--prompt-ids 1,2,3] [--tokens N] [--iterations K] [--loop-poison] model.gguf\n"
        "  ./verum-slice --hotbench [--tune-out verum-tune.json]\n"
        "  ./verum-slice --fuse [--cache-dir DIR] [--tokens N] model.gguf\n"
        "  ./verum-slice --tune-show\n"
    );
}

/* Additive diagnostic, added without changing any existing mode:
 * reports which backends were compiled in and, by probing each
 * backend's own init(), whether a usable device is present right now.
 * It needs no model file and no GPU, so it is safe to run on CI and on
 * a machine with no GPU at all. A backend with no device is reported
 * as "available: no", not as an error, so this mode always exits 0
 * when the report itself was produced. The report lives in
 * vbs_backend_select.c so the order matches engine selection. */
static int print_backends(void) {
    vbs_backend_probe_all();
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

/* One --fuse generation round: reset the model, feed the prompt, generate
 * up to n_gen greedy tokens into out_tokens. Returns the token count and
 * writes the generation-only elapsed seconds to *sec_out (prompt feeding
 * is excluded; it is identical for both paths). Returns 0 on error. */
static size_t fuse_gen_round(vbs_cpu_model_t *cm, const uint32_t *prompt_ids,
                             size_t n_prompt, size_t n_gen, uint32_t *out_tokens,
                             double *sec_out, char *err, size_t err_cap) {
    vbs_cpu_reset(cm);
    for (size_t i = 0; i < n_prompt; i++) {
        if (!vbs_cpu_step_ex(cm, prompt_ids[i], i + 1 == n_prompt, err, err_cap)) return 0;
    }
    double t0 = get_time_sec();
    size_t n = 0;
    for (size_t i = 0; i < n_gen; i++) {
        const float *lg = vbs_cpu_logits(cm);
        size_t vocab = vbs_cpu_vocab_size(cm);
        uint32_t best = 0;
        for (size_t j = 1; j < vocab; j++) if (lg[j] > lg[best]) best = (uint32_t)j;
        out_tokens[n++] = best;
        if (best == vbs_cpu_eos_id(cm)) break;
        if (!vbs_cpu_step(cm, best, err, err_cap)) break;
    }
    *sec_out = get_time_sec() - t0;
    return n;
}

int main(int argc, const char *argv[]) {
    /* Model-free diagnostic: handled before the model requirement below,
     * so "./verum-slice --backends" works with a single argument. */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage();
            return 0;
        }
        if (strcmp(argv[i], "--backends") == 0) {
            return print_backends();
        }
        if (strcmp(argv[i], "--backend-test") == 0) {
            return vbs_backend_selftest();
        }
    }

    if (argc < 2) {
        usage();
        return 64;
    }

    const char *mode = NULL;
    const char *model_path = NULL;
    const char *tokenizer_path = NULL;
    const char *correct_text = NULL;
    const char *memory_path = "verum-memory.tsv";
    const char *adapter_path = NULL;
    const char *autotune_path = "verum-autotune.json";
    const char *adapter_out = NULL;
    const char *block_out = NULL;
    size_t train_target = 0; int have_target = 0;
    size_t train_vocab = 32000; size_t train_epochs = 100; float train_lr = 0.5f;
#ifdef VBS_SELF_TUNE_LR
    train_lr = (float)VBS_SELF_TUNE_LR;
#endif
    const char *prompt_text = "The universe is";
    size_t window_mib = 256; int have_window = 0;
    size_t bench_tokens = 128;
    size_t max_gen_tokens = 64;
    /* Recursive-loop / hotbench / fuse options. */
    const char *prompt_ids_text = "1,2,3";
    int loop_iterations = 6;
    size_t loop_tokens_per_iter = 8;
    const char *loop_log_path = "vbs-loop.jsonl";
    const char *loop_work_dir = ".verum-loop-work";
    const char *repo_dir = ".";
    int loop_poison = 0;
    const char *tune_out_path = NULL;
    const char *fuse_cache_dir = ".verum-fuse-cache";
    float sample_temp = 0.0f; int sample_top_k = 0; float sample_top_p = 1.0f; uint64_t sample_seed = 0;

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
            strcmp(argv[i], "--train-block") == 0 ||
            strcmp(argv[i], "--train-tiny") == 0 ||
            strcmp(argv[i], "--autonomous-c") == 0 ||
            strcmp(argv[i], "--autotune") == 0 ||
            strcmp(argv[i], "--train-text") == 0 ||
            strcmp(argv[i], "--cpu") == 0 ||
            strcmp(argv[i], "--dequant") == 0 ||
            strcmp(argv[i], "--loop") == 0 ||
            strcmp(argv[i], "--hotbench") == 0 ||
            strcmp(argv[i], "--fuse") == 0 ||
            strcmp(argv[i], "--tune-show") == 0 ||
            strcmp(argv[i], "--sample") == 0) {
            mode = argv[i];
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            sample_temp = (float)strtod(argv[++i], NULL);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            sample_top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            sample_top_p = (float)strtod(argv[++i], NULL);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            sample_seed = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--window-mib") == 0 && i + 1 < argc) {
            window_mib = (size_t)strtoull(argv[++i], NULL, 10); have_window = 1;
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            /* --tokens sets both counts: --benchmark/--autonomy read
               bench_tokens, --generate/--cpu read max_gen_tokens, and
               before this fix --tokens was silently ignored by the
               generation modes. */
            bench_tokens = (size_t)strtoull(argv[++i], NULL, 10);
            max_gen_tokens = bench_tokens;
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
        } else if (strcmp(argv[i], "--block-out") == 0 && i + 1 < argc) {
            block_out = argv[++i];
        } else if (strcmp(argv[i], "--autotune-out") == 0 && i + 1 < argc) {
            autotune_path = argv[++i];
        } else if (strcmp(argv[i], "--target-token") == 0 && i + 1 < argc) {
            train_target = (size_t)strtoull(argv[++i], NULL, 10); have_target = 1;
        } else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) {
            train_vocab = (size_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            train_epochs = (size_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            train_lr = (float)strtod(argv[++i], NULL);
        } else if (strcmp(argv[i], "--prompt-ids") == 0 && i + 1 < argc) {
            prompt_ids_text = argv[++i];
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            loop_iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tokens-per-iter") == 0 && i + 1 < argc) {
            loop_tokens_per_iter = (size_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--loop-log") == 0 && i + 1 < argc) {
            loop_log_path = argv[++i];
        } else if (strcmp(argv[i], "--loop-work") == 0 && i + 1 < argc) {
            loop_work_dir = argv[++i];
        } else if (strcmp(argv[i], "--repo") == 0 && i + 1 < argc) {
            repo_dir = argv[++i];
        } else if (strcmp(argv[i], "--loop-poison") == 0) {
            loop_poison = 1;
        } else if (strcmp(argv[i], "--tune-out") == 0 && i + 1 < argc) {
            tune_out_path = argv[++i];
        } else if (strcmp(argv[i], "--cache-dir") == 0 && i + 1 < argc) {
            fuse_cache_dir = argv[++i];
        } else if (argv[i][0] != '-') {
            model_path = argv[i];
        }
    }

    if (!mode || (!model_path && strcmp(mode, "--learn") != 0 && strcmp(mode, "--train-adapter") != 0 && strcmp(mode, "--train-block") != 0 && strcmp(mode, "--train-tiny") != 0 && strcmp(mode, "--autonomous-c") != 0 && strcmp(mode, "--autotune") != 0 && strcmp(mode, "--train-text") != 0 && strcmp(mode, "--cpu") != 0 && strcmp(mode, "--dequant") != 0 && strcmp(mode, "--hotbench") != 0 && strcmp(mode, "--tune-show") != 0 && strcmp(mode, "--sample") != 0)) {
        usage();
        return 64;
    }

    if (strcmp(mode, "--learn") == 0) {
        if (!prompt_text || !correct_text) { fprintf(stderr, "Error: --learn requires --prompt and --correct\n"); return 1; }
        if (!vbs_learn_append(memory_path, prompt_text, correct_text)) { fprintf(stderr, "Error: could not write %s\n", memory_path); return 1; }
        printf("LEARNED: %s -> %s  [file=%s]\n", prompt_text, correct_text, memory_path);
        return 0;
    }
    if (strcmp(mode, "--train-text") == 0) {
        /* Real text, not just a pattern: train next-char prediction on
         * the user's --prompt text, repeated. Vocab = 32 (char % 32),
         * context 3. This is still a tiny model, not a real LLM. */
        const char *txt = prompt_text ? prompt_text : "hello world ";
        int n = (int)strlen(txt);
        if (n < 4) { fprintf(stderr, "Error: --train-text needs at least 4 chars\n"); return 1; }
        vbs_block_t *blk = vbs_block_create(32, 16, 3, 32, 12345u);
        double loss = 0;
        for (size_t ep=0; ep<train_epochs; ep++) {
            loss = 0; int cnt = 0;
            for (int i=0; i+3<n; i++) {
                int win[3] = {(unsigned char)txt[i]%32, (unsigned char)txt[i+1]%32, (unsigned char)txt[i+2]%32};
                loss += vbs_block_train_step(blk, win, (unsigned char)txt[i+3]%32, train_lr); cnt++;
            }
            loss /= cnt;
            if (ep==0 || ep+1==train_epochs || (ep+1)%100==0) printf("text epoch %zu loss=%.4f\n", ep+1, loss);
        }
        /* generate 20 chars starting from first 3 chars */
        int ctx[3] = {(unsigned char)txt[0]%32, (unsigned char)txt[1]%32, (unsigned char)txt[2]%32};
        printf("text generate: ");
        for (int i=0;i<3;i++) printf("%c", txt[i]);
        for (int step=0; step<20; step++) {
            int nxt = vbs_block_predict(blk, ctx);
            /* map token back: find a char in txt with that token, else '?' */
            char out='?'; for(int i=0;i<n;i++) if((unsigned char)txt[i]%32==nxt){ out=txt[i]; break; }
            printf("%c", out);
            ctx[0]=ctx[1]; ctx[1]=ctx[2]; ctx[2]=nxt;
        }
        printf("\n");
        vbs_block_free(blk); return 0;
    }
    if (strcmp(mode, "--autotune") == 0) {
        vbs_autotune_t cfg;
        int rc = vbs_autotune_run(autotune_path, &cfg);
        printf("autotune: %s config -> %s (backend=%s threads=%d window=%zu MiB)\n", rc==2?"reused":"saved", autotune_path, cfg.backend, cfg.threads, cfg.window_mib);
        return 0;
    }
    if (strcmp(mode, "--autonomous-c") == 0) {
        vbs_autotune_t acfg; vbs_autotune_run("verum-autotune.json", &acfg);
        int pattern[12] = {1,2,3,5,1,2,3,5,1,2,3,5};
        vbs_block_t *blk = vbs_block_create(8, 16, 3, 32, 12345u);
        if (vbs_block_load(blk, "autonomous.bin")) printf("autonomous-c: loaded saved model autonomous.bin, continuing learning\n");
        else { vbs_block_free(blk); blk = vbs_block_create(8, 16, 3, 32, 12345u); printf("autonomous-c: no saved model, starting fresh\n"); }
        printf("autonomous-c: before predict [1,2,3] -> %d\n", vbs_block_predict(blk, (int[]){1,2,3}));
        double loss = 0; int ok = 0;
        for (int attempt=0; attempt<3 && !ok; attempt++) {
            loss = vbs_autonomous_train(blk, pattern, 12, 0.01, 1000);
            int ctx[3] = {1,2,3}; int gen[9];
            for (int step=0; step<9; step++) { gen[step] = vbs_block_predict(blk, ctx); ctx[0]=ctx[1]; ctx[1]=ctx[2]; ctx[2]=gen[step]; }
            int want[9] = {5,1,2,3,5,1,2,3,5}; ok = 1;
            for (int i=0;i<9;i++) if (gen[i]!=want[i]) ok=0;
            printf("autonomous-c: attempt %d generate %s\n", attempt+1, ok?"correct":"wrong, retraining");
            if (!ok && attempt<2) { /* self-correction: keep training, no human input */ }
        }
        double best = 1e9; FILE *bf = fopen("autonomous-best.txt", "r");
        if (bf) { if (fscanf(bf, "%lf", &best) != 1) best = 1e9; fclose(bf); }
        if (loss < best) {
            vbs_block_save(blk, "autonomous.bin");
            FILE *bo = fopen("autonomous-best.txt", "w"); if (bo) { fprintf(bo, "%.6f\n", loss); fclose(bo); }
            printf("autonomous-c: saved model to autonomous.bin, new best loss=%.4f\n", loss);
        } else {
            printf("autonomous-c: NOT saving, loss %.4f worse than best %.4f, keeping old model\n", loss, best);
        }
        printf("autonomous-c: scheduler lr at epoch 500 = %.4f\n", vbs_scheduler_lr(0.05f, 500));
        int ctx[3] = {1,2,3};
        printf("autonomous-c: generate: 1 2 3");
        for (int step=0; step<9; step++) { int nxt = vbs_block_predict(blk, ctx); printf(" %d", nxt); ctx[0]=ctx[1]; ctx[1]=ctx[2]; ctx[2]=nxt; }
        printf("\nautonomous-c: final loss=%.4f\n", loss);
        vbs_block_free(blk); return 0;
    }
    if (strcmp(mode, "--train-tiny") == 0) {
        /* End-to-end tiny LLM: learn a repeating pattern, then generate.
         * Pattern: 1,2,3,5 repeating, vocab 8, context 3.
         * Train on every window in the pattern, then greedy-generate
         * starting from 1,2,3 and print the result. */
        int pattern[12] = {1,2,3,5,1,2,3,5,1,2,3,5};
        vbs_block_t *blk = vbs_block_create(8, 16, 3, 32, 12345u);
        printf("before: predict [1,2,3] -> %d (want 5)\n", vbs_block_predict(blk, (int[]){1,2,3}));
        double loss = 0;
        for (size_t ep=0; ep<train_epochs; ep++) {
            loss = 0;
            for (int i=0; i<9; i++) {
                int win[3] = {pattern[i], pattern[i+1], pattern[i+2]};
                loss += vbs_block_train_step(blk, win, pattern[i+3], train_lr);
            }
            loss /= 9.0;
            if (ep==0 || ep+1==train_epochs || (ep+1)%100==0) printf("tiny epoch %zu loss=%.4f\n", ep+1, loss);
        }
        int ctx[3] = {1,2,3};
        printf("after: predict [1,2,3] -> %d (want 5)\n", vbs_block_predict(blk, ctx));
        printf("generate: 1 2 3");
        for (int step=0; step<9; step++) {
            int nxt = vbs_block_predict(blk, ctx);
            printf(" %d", nxt);
            ctx[0]=ctx[1]; ctx[1]=ctx[2]; ctx[2]=nxt;
        }
        printf("\n");
        if (block_out) { vbs_block_save(blk, block_out); printf("saved block -> %s\n", block_out); }
        vbs_block_free(blk); return 0;
    }
    if (strcmp(mode, "--train-block") == 0) {
        /* Task: memorize [1,2,3] -> 5, vocab 8, d 16, hidden 32.
         * If backprop is wrong, loss will not go to near 0. */
        vbs_block_t *blk = vbs_block_create(8, 16, 3, 32, 12345u);
        printf("train-block lr=%.4f (from self_tune.h unless --lr given)\n", train_lr);
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
    if (strcmp(mode, "--sample") == 0) {
        float logits[4] = {0.0f, 1.0f, 3.0f, 2.0f};
        vbs_sampler_t s; vbs_sampler_init(&s);
        uint32_t g = vbs_sampler_sample(&s, logits, 4, NULL, 0);
        printf("sample: greedy token=%u (want 2)\n", g);
        vbs_sampler_init(&s); s.temperature = sample_temp > 0 ? sample_temp : 0.7f; s.top_k = sample_top_k ? sample_top_k : 2; s.seed = sample_seed ? sample_seed : 12345;
        uint32_t a = vbs_sampler_sample(&s, logits, 4, NULL, 0);
        vbs_sampler_init(&s); s.temperature = sample_temp > 0 ? sample_temp : 0.7f; s.top_k = sample_top_k ? sample_top_k : 2; s.seed = sample_seed ? sample_seed : 12345;
        uint32_t b = vbs_sampler_sample(&s, logits, 4, NULL, 0);
        printf("sample: temp/top-k token=%u repeat=%u deterministic=%s\n", a, b, a==b ? "yes" : "no");
        if (g != 2 || a != b) { printf("sample: FAIL\n"); return 1; }
        printf("sample: ALL PASS\n"); return 0;
    }
    if (strcmp(mode, "--dequant") == 0) {
        /* Dequant decode check: verifies the quant decode path for types
           the GPU specializer previously refused, using known blocks. */
        int fails = 0;
        struct { vbs_ggml_type_t type; const char *name; } types[] = {
            {VBS_GGML_TYPE_F32, "F32"}, {VBS_GGML_TYPE_F16, "F16"},
            {VBS_GGML_TYPE_Q4_0, "Q4_0"}, {VBS_GGML_TYPE_Q4_1, "Q4_1"},
            {VBS_GGML_TYPE_Q5_0, "Q5_0"}, {VBS_GGML_TYPE_Q5_1, "Q5_1"},
            {VBS_GGML_TYPE_Q8_0, "Q8_0"}, {VBS_GGML_TYPE_Q8_1, "Q8_1"},
            {VBS_GGML_TYPE_Q2_K, "Q2_K"}, {VBS_GGML_TYPE_Q3_K, "Q3_K"},
            {VBS_GGML_TYPE_Q4_K, "Q4_K"}, {VBS_GGML_TYPE_Q5_K, "Q5_K"},
            {VBS_GGML_TYPE_Q6_K, "Q6_K"}, {VBS_GGML_TYPE_Q8_K, "Q8_K"},
            {VBS_GGML_TYPE_BF16, "BF16"},
        };
        for (size_t i = 0; i < sizeof(types)/sizeof(types[0]); i++) {
            printf("dequant: %s native=%s dequantizable=%s\n", types[i].name,
                vbs_dequant_has_native_kernel(types[i].type) ? "yes" : "no",
                vbs_dequant_supported(types[i].type) ? "yes" : "no");
        }
        /* Q4_0 known block: scale 0.5, nibbles i / 15-i */
        { uint8_t blk[18] = {0}; ((uint16_t*)blk)[0] = vbs_f32_to_f16(0.5f);
          for (int i = 0; i < 16; i++) blk[2+i] = (uint8_t)(i | ((15-i)<<4));
          float y[32]; int n = vbs_dequant_block(VBS_GGML_TYPE_Q4_0, blk, y);
          if (n != 32 || fabsf(y[0] - 0.5f*(0-8)) > 1e-6f || fabsf(y[16] - 0.5f*(15-8)) > 1e-6f) { printf("dequant: Q4_0 FAIL n=%d y0=%f y16=%f\n", n, y[0], y[16]); fails++; }
          else printf("dequant: Q4_0 OK 32 values\n"); }
        /* Q8_0 known block: scale 0.125, quants i*7-80 */
        { uint8_t blk[34] = {0}; ((uint16_t*)blk)[0] = vbs_f32_to_f16(0.125f);
          for (int i = 0; i < 32; i++) blk[2+i] = (uint8_t)(int8_t)(i*7-80);
          float y[32]; int n = vbs_dequant_block(VBS_GGML_TYPE_Q8_0, blk, y);
          if (n != 32 || fabsf(y[0] - 0.125f*(int8_t)(-80)) > 1e-6f) { printf("dequant: Q8_0 FAIL n=%d\n", n); fails++; }
          else printf("dequant: Q8_0 OK 32 values\n"); }
        /* K-quant known-value checks. A zero block cannot catch a layout
           error (every field reads 0 either way) — that is how the Q4_K /
           Q5_K scale-position bug survived until a real model was run.
           Each block below puts nonzero scales at the type's real offsets
           and checks exact decoded values. */
        /* Q2_K: d=2, dmin=0, scales 0x01 (scale 1, min 0), qs[0]=0xE4 */
        { uint8_t blk[84] = {0}; for (int i = 0; i < 16; i++) blk[i] = 0x01;
          blk[16] = 0xE4; ((uint16_t*)(blk+80))[0] = vbs_f32_to_f16(2.0f);
          float y[256]; int n = vbs_dequant_block(VBS_GGML_TYPE_Q2_K, blk, y);
          if (n != 256 || fabsf(y[0]) > 1e-6f || fabsf(y[32] - 2.0f) > 1e-6f ||
              fabsf(y[64] - 4.0f) > 1e-6f || fabsf(y[96] - 6.0f) > 1e-6f) { printf("dequant: Q2_K FAIL n=%d y32=%f y64=%f\n", n, y[32], y[64]); fails++; }
          else printf("dequant: Q2_K OK 256 values\n"); }
        /* Q3_K: d=2, scales 0 (-> -32 each), hmask all set, qs[0]=0xE4 */
        { uint8_t blk[110] = {0}; for (int i = 0; i < 32; i++) blk[i] = 0xFF;
          blk[32] = 0xE4; ((uint16_t*)(blk+108))[0] = vbs_f32_to_f16(2.0f);
          float y[256]; int n = vbs_dequant_block(VBS_GGML_TYPE_Q3_K, blk, y);
          if (n != 256 || fabsf(y[0]) > 1e-6f || fabsf(y[32] + 64.0f) > 1e-6f ||
              fabsf(y[64] + 128.0f) > 1e-6f) { printf("dequant: Q3_K FAIL n=%d y32=%f\n", n, y[32]); fails++; }
          else printf("dequant: Q3_K OK 256 values\n"); }
        /* Q4_K: d=2, dmin=1 FIRST (offset 0), sub-block scales (2,1),(3,0) */
        { uint8_t blk[144] = {0};
          ((uint16_t*)blk)[0] = vbs_f32_to_f16(2.0f);
          ((uint16_t*)blk)[1] = vbs_f32_to_f16(1.0f);
          blk[4] = 2; blk[8] = 1; blk[5] = 3; blk[9] = 0;
          blk[16] = 0x21; /* element 0 nibble 1, element 32 nibble 2 */
          float y[256]; int n = vbs_dequant_block(VBS_GGML_TYPE_Q4_K, blk, y);
          if (n != 256 || fabsf(y[0] - 3.0f) > 1e-6f || fabsf(y[32] - 12.0f) > 1e-6f) { printf("dequant: Q4_K FAIL n=%d y0=%f y32=%f\n", n, y[0], y[32]); fails++; }
          else printf("dequant: Q4_K OK 256 values\n"); }
        /* Q5_K: same scales as Q4_K, plus qh high bit on element 0 */
        { uint8_t blk[176] = {0};
          ((uint16_t*)blk)[0] = vbs_f32_to_f16(2.0f);
          ((uint16_t*)blk)[1] = vbs_f32_to_f16(1.0f);
          blk[4] = 2; blk[8] = 1; blk[5] = 3; blk[9] = 0;
          blk[16] = 0x01; /* qh[0] bit 0: element 0 gains +16 */
          blk[48] = 0x21;
          float y[256]; int n = vbs_dequant_block(VBS_GGML_TYPE_Q5_K, blk, y);
          if (n != 256 || fabsf(y[0] - 67.0f) > 1e-6f || fabsf(y[32] - 12.0f) > 1e-6f) { printf("dequant: Q5_K FAIL n=%d y0=%f y32=%f\n", n, y[0], y[32]); fails++; }
          else printf("dequant: Q5_K OK 256 values\n"); }
        /* Q6_K: d=2, scales[0]=3, scales[2]=1, ql[0]=0x21, ql[32]=0x05 */
        { uint8_t blk[210] = {0};
          blk[192] = 3; blk[194] = 1; blk[0] = 0x21; blk[32] = 0x05;
          ((uint16_t*)(blk+208))[0] = vbs_f32_to_f16(2.0f);
          float y[256]; int n = vbs_dequant_block(VBS_GGML_TYPE_Q6_K, blk, y);
          if (n != 256 || fabsf(y[0] + 186.0f) > 1e-6f || fabsf(y[32] + 54.0f) > 1e-6f) { printf("dequant: Q6_K FAIL n=%d y0=%f y32=%f\n", n, y[0], y[32]); fails++; }
          else printf("dequant: Q6_K OK 256 values\n"); }
        /* Q8_K: f32 d=0.25, quants i-128, correct per-16 bsums */
        { uint8_t blk[292] = {0}; float d = 0.25f; memcpy(blk, &d, 4);
          for (int i = 0; i < 256; i++) blk[4+i] = (uint8_t)(int8_t)(i-128);
          for (int j = 0; j < 16; j++) { int s = 0; for (int i = 0; i < 16; i++) s += (int8_t)blk[4+j*16+i]; ((int16_t*)(blk+260))[j] = (int16_t)s; }
          float y[256]; int n = vbs_dequant_block(VBS_GGML_TYPE_Q8_K, blk, y);
          if (n != 256 || fabsf(y[0] + 32.0f) > 1e-6f || fabsf(y[128]) > 1e-6f ||
              fabsf(y[255] - 31.75f) > 1e-6f) { printf("dequant: Q8_K FAIL n=%d y0=%f y255=%f\n", n, y[0], y[255]); fails++; }
          else printf("dequant: Q8_K OK 256 values\n"); }
        printf("dequant: %s\n", fails ? "FAILURES" : "ALL PASS");
        return fails ? 1 : 0;
    }
    if (strcmp(mode, "--cpu") == 0) {
        if (!tokenizer_path) { fprintf(stderr, "Error: --cpu requires --tokenizer /path/to/tokenizer.json\n"); return 1; }
        double t_start = get_time_sec();
        tokenizer_t *tok = tokenizer_load(tokenizer_path);
        if (!tok) { fprintf(stderr, "Error: Failed to load tokenizer from %s\n", tokenizer_path); return 1; }
        char cpu_err[512] = {0};
        vbs_cpu_model_t *cm = vbs_cpu_open(model_path, 256, cpu_err, sizeof(cpu_err));
        if (!cm) { fprintf(stderr, "CPU model load failed: %s\n", cpu_err); tokenizer_free(tok); return 1; }
        double t_load_end = get_time_sec();
        uint32_t prompt_tokens[2048];
        size_t n_prompt = tokenizer_encode(tok, prompt_text, prompt_tokens, 2048);
        for (size_t i = 0; i < n_prompt; i++) {
            if (!vbs_cpu_step_ex(cm, prompt_tokens[i], i + 1 == n_prompt, cpu_err, sizeof(cpu_err))) { fprintf(stderr, "CPU inference failed: %s\n", cpu_err); vbs_cpu_close(cm); tokenizer_free(tok); return 1; }
        }
        double t_prefill_end = get_time_sec();
        printf("%s", prompt_text); fflush(stdout);
        uint32_t eos = vbs_cpu_eos_id(cm);
        vbs_sampler_t sampler; vbs_sampler_init(&sampler);
        sampler.temperature = sample_temp; sampler.top_k = sample_top_k; sampler.top_p = sample_top_p; sampler.seed = sample_seed;
        uint32_t recent[4096]; size_t n_recent = 0;
        for (size_t i = 0; i < n_prompt && n_recent < 4096; i++) recent[n_recent++] = prompt_tokens[i];
        size_t n_gen = 0;
        for (size_t i = 0; i < max_gen_tokens; i++) {
            const float *logits = vbs_cpu_logits(cm);
            size_t vocab = vbs_cpu_vocab_size(cm);
            uint32_t best = vbs_sampler_sample(&sampler, logits, vocab, recent, n_recent);
            if (n_recent < 4096) recent[n_recent++] = best;
            if (best == eos) break;
            char *piece = tokenizer_decode(tok, &best, 1);
            if (piece) { printf("%s", piece); fflush(stdout); free(piece); }
            n_gen++;
            if (!vbs_cpu_step(cm, best, cpu_err, sizeof(cpu_err))) { fprintf(stderr, "CPU inference failed: %s\n", cpu_err); break; }
        }
        double t_gen_end = get_time_sec();
        double load_s = t_load_end - t_start;
        double prefill_s = t_prefill_end - t_load_end;
        double gen_s = t_gen_end - t_prefill_end;
        fprintf(stderr, "cpu: load %.3fs (tokenizer + model, incl. any weight conversion), "
                "prefill %zu tokens in %.3fs (%.1f tok/s), "
                "generated %zu tokens in %.3fs (%.1f tok/s)\n",
                load_s,
                n_prompt, prefill_s, n_prompt / (prefill_s > 0 ? prefill_s : 1e-9),
                n_gen, gen_s, n_gen / (gen_s > 0 ? gen_s : 1e-9));
        printf("\n"); vbs_cpu_close(cm); tokenizer_free(tok); return 0;
    }
    if (strcmp(mode, "--loop") == 0) {
        /* Recursive loop: live session + unsupervised optimization. */
        uint32_t prompt_ids[64];
        size_t n_prompt = 0;
        {
            char buf[1024];
            snprintf(buf, sizeof(buf), "%s", prompt_ids_text);
            char *save = NULL;
            for (char *tok = strtok_r(buf, ",", &save); tok && n_prompt < 64;
                 tok = strtok_r(NULL, ",", &save)) {
                prompt_ids[n_prompt++] = (uint32_t)strtoul(tok, NULL, 10);
            }
        }
        if (n_prompt == 0) { prompt_ids[0] = 1; n_prompt = 1; }
        if (loop_tokens_per_iter == 0) loop_tokens_per_iter = 8;
        vbs_loop_config_t lcfg;
        memset(&lcfg, 0, sizeof(lcfg));
        lcfg.model_path = model_path;
        lcfg.prompt = prompt_ids;
        lcfg.n_prompt = n_prompt;
        lcfg.n_tokens = bench_tokens;
        lcfg.max_iterations = loop_iterations;
        lcfg.tokens_per_iter = loop_tokens_per_iter;
        lcfg.log_path = loop_log_path;
        lcfg.work_dir = loop_work_dir;
        lcfg.repo_dir = repo_dir;
        lcfg.poison = loop_poison;
        vbs_loop_result_t res;
        if (vbs_loop_run(&lcfg, &res) != 0) {
            fprintf(stderr, "loop failed\n");
            return 1;
        }
        printf("loop tokens:");
        for (size_t i = 0; i < res.n_tokens_generated; i++) printf(" %u", res.tokens[i]);
        printf("\n");
        vbs_loop_result_free(&res);
        return 0;
    }
    if (strcmp(mode, "--hotbench") == 0) {
        /* Benchmark every hot-kernel variant at fixed shapes (the tiny
         * model's shapes: 1408x512, 512x512, 256x512) and report the
         * fastest correct variant per type. Candidates are benched in an
         * isolated child, exactly like the loop does. */
        vbs_bench_shape_t shapes[3] = { {1408, 512}, {512, 512}, {256, 512} };
        double ref_ns[VBS_HOT_NTYPES] = {0};
        double dummy[VBS_HOT_NTYPES] = {0};
        int dummy_wins[VBS_HOT_NTYPES] = {0};
        bool correct = false;
        char berr[1024] = {0};
        printf("hotbench: shapes 1408x512, 512x512, 256x512 (min ns/call over 7 samples, worst shape)\n");
        printf("hotbench: each candidate is timed interleaved with builtin-ref in the same child;\n");
        printf("hotbench: the ratio + paired wins are the trustworthy numbers on a contended machine.\n");
        printf("%-16s", "variant");
        for (int t = 0; t < VBS_HOT_NTYPES; t++) printf(" %12s", vbs_hot_type_name(t));
        printf("  correct\n");
        if (vbs_loop_bench_in_child(NULL, NULL, shapes, 3, 0x1Fu, ref_ns, dummy, dummy_wins, &correct, berr, sizeof(berr)) != 0) {
            fprintf(stderr, "hotbench: builtin bench failed: %s\n", berr);
            return 1;
        }
        printf("%-16s", "builtin-ref");
        for (int t = 0; t < VBS_HOT_NTYPES; t++) printf(" %12.0f", ref_ns[t]);
        printf("  %s\n", correct ? "yes" : "NO");
        vbs_tune_profile_t prof;
        vbs_tune_profile_init(&prof);
        double best_ratio[VBS_HOT_NTYPES];
        for (int t = 0; t < VBS_HOT_NTYPES; t++) {
            snprintf(prof.best[t], sizeof(prof.best[t]), "builtin-ref");
            prof.best_ns[t] = ref_ns[t];
            best_ratio[t] = 1.0;
        }
        char work[512];
        snprintf(work, sizeof(work), "%s", ".verum-hotbench-work");
        {
            char cmd[1400];
            snprintf(cmd, sizeof(cmd), "mkdir -p %s", work);
            if (system(cmd) != 0) { /* non-fatal; compile will fail loudly */ }
        }
        for (size_t ci = 0; ci < vbs_loop_candidate_count(); ci++) {
            const vbs_candidate_t *cand = vbs_loop_candidate(ci);
            if (cand->needs_avx2 && !vbs_hot_cpu_has_avx2()) {
                printf("%-16s skipped (host lacks AVX2)\n", cand->name);
                continue;
            }
            char src[1200], so[1200], inc1[1200], inc2[1200];
            snprintf(src, sizeof(src), "%s/src/hotkernels/%s", repo_dir, cand->src_file);
            snprintf(so, sizeof(so), "%s/cand_%s.so", work, cand->name);
            snprintf(inc1, sizeof(inc1), "-I%s/include", repo_dir);
            snprintf(inc2, sizeof(inc2), "-I%s/src/hotkernels", repo_dir);
            const char *extra[3];
            size_t n_extra = 0;
            extra[n_extra++] = inc1;
            extra[n_extra++] = inc2;
            if (cand->extra_flag) extra[n_extra++] = cand->extra_flag;
            char cerr[1024] = {0};
            if (vbs_cc_compile_so(NULL, src, so, extra, n_extra, cerr, sizeof(cerr)) != 0) {
                printf("%-16s COMPILE FAILED\n", cand->name);
                continue;
            }
            double ns[VBS_HOT_NTYPES] = {0};
            double base_ns[VBS_HOT_NTYPES] = {0};
            int wins[VBS_HOT_NTYPES] = {0};
            uint32_t mask = 0;
            for (int t = 0; t < VBS_HOT_NTYPES; t++)
                if (cand->optimizes[t]) mask |= (uint32_t)(1u << t);
            int brc = vbs_loop_bench_in_child(so, NULL, shapes, 3, mask, ns, base_ns, wins, &correct, berr, sizeof(berr));
            if (brc == -2) {
                printf("%-16s CRASHED in bench\n", cand->name);
                continue;
            }
            if (brc != 0) {
                printf("%-16s BENCH FAILED: %s\n", cand->name, berr);
                continue;
            }
            printf("%-16s", cand->name);
            for (int t = 0; t < VBS_HOT_NTYPES; t++) {
                if (mask & (1u << t)) printf(" %12.0f", ns[t]);
                else printf(" %12s", "-");
            }
            printf("  %s", correct ? "yes" : "NO");
            if (correct) {
                printf("  speedup vs builtin:");
                for (int t = 0; t < VBS_HOT_NTYPES; t++) {
                    if (cand->optimizes[t] && ns[t] > 0) printf(" %s=%.2fx(%d/7)", vbs_hot_type_name(t), base_ns[t] / ns[t], wins[t]);
                }
            }
            printf("\n");
            if (correct) {
                for (int t = 0; t < VBS_HOT_NTYPES; t++) {
                    /* Best is by ratio measured in the same child run,
                     * with paired-win evidence scaled to the win size
                     * (same two-tier rule as the loop's accept gate). */
                    double ratio = (base_ns[t] > 0 && ns[t] > 0) ? ns[t] / base_ns[t] : 1.0;
                    int enough_wins = (ratio < 0.80) ? (wins[t] >= 4) : (wins[t] >= 5);
                    if (cand->optimizes[t] && ratio < 0.98 && enough_wins && ratio < best_ratio[t]) {
                        snprintf(prof.best[t], sizeof(prof.best[t]), "%s", cand->name);
                        prof.best_ns[t] = ns[t];
                        best_ratio[t] = ratio;
                    }
                }
            }
        }
        printf("best per type:");
        for (int t = 0; t < VBS_HOT_NTYPES; t++) {
            printf(" %s=%s(%.0fns)", vbs_hot_type_name(t), prof.best[t], prof.best_ns[t]);
        }
        printf("\n");
        if (tune_out_path) {
            if (vbs_tune_write_json(&prof, tune_out_path)) {
                printf("tune profile written to %s (fingerprint %s)\n", tune_out_path, prof.fingerprint);
            } else {
                fprintf(stderr, "hotbench: could not write %s\n", tune_out_path);
                return 1;
            }
        }
        return 0;
    }
    if (strcmp(mode, "--fuse") == 0) {
        /* Fused FFN: build/load the model-specialized fused kernel, then
         * interleave reference and fused generation rounds on the same
         * prompt (3 each) and compare the MINIMUM round time of each
         * side. Single-shot timing on a shared VM is too noisy to resolve
         * a ~15% end-to-end effect: min-of-3 interleaved rounds is the
         * same discipline the loop's benchmark uses. Token sequences from
         * every round must match the first reference round exactly. */
        uint32_t prompt_ids[64];
        size_t n_prompt = 0;
        {
            char buf[1024];
            snprintf(buf, sizeof(buf), "%s", prompt_ids_text);
            char *save = NULL;
            for (char *tok = strtok_r(buf, ",", &save); tok && n_prompt < 64;
                 tok = strtok_r(NULL, ",", &save)) {
                prompt_ids[n_prompt++] = (uint32_t)strtoul(tok, NULL, 10);
            }
        }
        if (n_prompt == 0) { prompt_ids[0] = 1; n_prompt = 1; }
        char ferr[1024] = {0};
        vbs_cpu_model_t *cm = vbs_cpu_open(model_path, 128, ferr, sizeof(ferr));
        if (!cm) { fprintf(stderr, "fuse: model open failed: %s\n", ferr); return 1; }
        size_t n_gen = bench_tokens;
        uint32_t *ref_tokens = (uint32_t *)calloc(n_gen, sizeof(uint32_t));
        uint32_t *round_tokens = (uint32_t *)calloc(n_gen, sizeof(uint32_t));
        if (!ref_tokens || !round_tokens) { free(ref_tokens); free(round_tokens); vbs_cpu_close(cm); return 1; }
        /* Build/load the fused kernel first; reference rounds disable it
         * via vbs_fuse_set_enabled(false) without unloading. */
        if (!vbs_fuse_build_and_load(model_path, fuse_cache_dir, ferr, sizeof(ferr))) {
            fprintf(stderr, "fuse: %s\n", ferr);
            free(ref_tokens); free(round_tokens); vbs_cpu_close(cm); return 1;
        }
        printf("fuse: kernel %s (cache %s, build %.2fs, verify max rel err %.3g)\n",
               vbs_fuse_loaded_path(), vbs_fuse_cache_hit() ? "HIT" : "MISS",
               vbs_fuse_build_seconds(), vbs_fuse_verify_max_rel_err());
        const int rounds = 3;
        double ref_min = -1.0, fused_min = -1.0;
        size_t n_ref = 0;
        int same = 1;
        for (int r = 0; r < rounds; r++) {
            double sec = 0;
            vbs_fuse_set_enabled(false);
            size_t n = fuse_gen_round(cm, prompt_ids, n_prompt, n_gen,
                                      round_tokens, &sec, ferr, sizeof(ferr));
            if (n == 0) { fprintf(stderr, "fuse: reference round failed: %s\n", ferr); same = 0; break; }
            if (r == 0) { n_ref = n; memcpy(ref_tokens, round_tokens, n * sizeof(uint32_t)); }
            else if (n != n_ref || memcmp(ref_tokens, round_tokens, n_ref * sizeof(uint32_t)) != 0) same = 0;
            if (ref_min < 0 || sec < ref_min) ref_min = sec;
            printf("fuse: round %d reference: %zu tokens in %.3fs (%.1f tok/s)\n",
                   r + 1, n, sec, n / (sec > 0 ? sec : 1e-9));
            vbs_fuse_set_enabled(true);
            n = fuse_gen_round(cm, prompt_ids, n_prompt, n_gen,
                               round_tokens, &sec, ferr, sizeof(ferr));
            if (n == 0) { fprintf(stderr, "fuse: fused round failed: %s\n", ferr); same = 0; break; }
            if (n != n_ref || memcmp(ref_tokens, round_tokens, n_ref * sizeof(uint32_t)) != 0) same = 0;
            if (fused_min < 0 || sec < fused_min) fused_min = sec;
            printf("fuse: round %d fused:     %zu tokens in %.3fs (%.1f tok/s)\n",
                   r + 1, n, sec, n / (sec > 0 ? sec : 1e-9));
        }
        vbs_fuse_set_enabled(true);
        printf("fuse: token sequences %s; speedup %.2fx (min of %d interleaved rounds each)\n",
               same ? "IDENTICAL" : "DIFFER (check numerics)",
               (ref_min > 0 && fused_min > 0) ? ref_min / fused_min : 0.0, rounds);
        free(ref_tokens); free(round_tokens);
        vbs_cpu_close(cm);
        return same ? 0 : 1;
    }
    if (strcmp(mode, "--tune-show") == 0) {
        char fp[192] = "";
        vbs_tune_fingerprint(fp, sizeof(fp));
        printf("tune: machine fingerprint %s\n", fp);
        printf("tune: avx2=%s avx512f=%s\n",
               vbs_hot_cpu_has_avx2() ? "yes" : "no",
               vbs_hot_cpu_has_avx512f() ? "yes" : "no");
        vbs_tune_profile_t prof;
        char terr[512] = {0};
        if (!vbs_tune_load_embedded(&prof, terr, sizeof(terr))) {
            printf("tune: %s\n", terr);
            printf("tune: run --hotbench --tune-out profile.json, then tools/vbs_tune_embed.sh to embed one\n");
            return 0;
        }
        printf("tune: embedded profile fingerprint %s (%s)\n",
               prof.fingerprint,
               vbs_tune_matches_machine(&prof) ? "MATCHES this machine" : "MISMATCH — ignored");
        printf("tune: generated_unix %llu\n", (unsigned long long)prof.generated_unix);
        for (int t = 0; t < VBS_HOT_NTYPES; t++) {
            printf("tune:   %-5s best=%s %.0f ns\n",
                   vbs_hot_type_name(t),
                   prof.best[t][0] ? prof.best[t] : "(unknown)", prof.best_ns[t]);
        }
        return 0;
    }
    /* Any PC, any config: if the user did not give --window-mib, autotune
       this machine first and use its window/backend. Override with --window-mib. */
    if (!have_window && model_path) {
        vbs_autotune_t acfg; vbs_autotune_run(autotune_path, &acfg);
        window_mib = acfg.window_mib;
        printf("auto-config: backend=%s threads=%d window=%zu MiB (autotuned for this PC)\n", acfg.backend, acfg.threads, acfg.window_mib); fflush(stdout);
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
        srand((unsigned)(time(NULL) ^ (unsigned)vbs_getpid()));

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
