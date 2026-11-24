/* vbs_loop.c — the closed recursive optimization loop.
 *
 * Architecture: one live process holds the model session (KV cache
 * and position survive everything below). Between tokens it runs
 * optimization iterations. Each iteration reads the structured
 * profiling counters to find the current hotspot, compiles the
 * next candidate for that hotspot with the system compiler, and
 * benchmark-verifies it in a forked child. Only a candidate that
 * is correct AND measurably faster is hot-swapped into the live
 * process, and the swap is watched live: a regression rolls back
 * to the previous module.
 *
 * Hot-swap mechanism: dlopen + atomic function-table swap between
 * tokens. Chosen over supervisor re-exec because (a) the CPU
 * inference path is single-threaded, so a token boundary is a
 * guaranteed quiescent point -- no call can be in flight; (b) the
 * kernel ABI is pure (no state in the module), so there is nothing
 * to migrate; (c) a swap costs microseconds and cannot drop the
 * session, where a re-exec would need a checkpoint of the KV cache
 * and a replay window. Re-exec stays the right answer for a
 * multi-threaded engine; for this one it would add failure modes
 * for no benefit. See REPORT.md for the full justification.
 */
#include "vbs_loop.h"
#include "vbs_cc.h"
#include "vbs_cpu.h"
#include "vbs_model.h"
#include "vbs_tune.h"
#include "vbs_port.h"
#include "vbs_preamble.h"
#if !defined(_WIN32)
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#define ACCEPT_MARGIN 0.98   /* candidate must beat best by >= 2% */
#define REGRESS_MARGIN 1.05  /* live regression beyond +5% rolls back */
#define CORRECT_TOL 1e-4     /* relative tolerance vs reference */

static uint64_t now_ms(void) {
    return (uint64_t)(vbs_wall_sec() * 1000.0);
}

/* --- Candidate registry --- */

static const vbs_candidate_t g_candidates[] = {
    { "unroll4", "cand_unroll.c", NULL, false,
      { true, true, true, false, false },
      "row-unrolled x4, shared activation loads" },
    { "avx2", "cand_avx2.c", "-mavx2", true,
      { true, true, true, false, false },
      "AVX2 int8 dots, shared widened activations" },
    { "slow-wasteful", "cand_slow.c", "-O0", false,
      { true, false, false, false, false },
      "TEST: correct but wasteful — must be rejected by the benchmark gate" },
    { "wrong-result", "cand_wrong.c", NULL, false,
      { true, false, false, false, false },
      "TEST: off-by-one dot — must be rejected by the correctness gate" },
    { "broken-syntax", "cand_nocompile.c", NULL, false,
      { true, false, false, false, false },
      "TEST: syntax error — must be rejected by the compile gate" },
};

size_t vbs_loop_candidate_count(void) {
    return sizeof(g_candidates) / sizeof(g_candidates[0]);
}

const vbs_candidate_t *vbs_loop_candidate(size_t idx) {
    if (idx >= vbs_loop_candidate_count()) return NULL;
    return &g_candidates[idx];
}

/* --- Benchmark core --- */

static uint32_t rng_state = 0x9e3779b9u;
/* Test-data randoms use vbs_xorshift32_next(&rng_state) from vbs_port.h. */

/* One benchmark workspace: random inputs for one (type, shape). */
typedef struct {
    uint8_t *w;
    int8_t *xq;
    float *x;
    float *qd, *qsum;
    float *y_ref, *y_cand;
    size_t rows, cols, nblocks, wbytes;
} bench_ws_t;

static void bench_ws_free(bench_ws_t *w) {
    free(w->w);
    free(w->xq);
    free(w->x);
    free(w->qd);
    free(w->qsum);
    free(w->y_ref);
    free(w->y_cand);
}

/* Run one matvec of the given type through a kernel table. */
static void run_matvec(const vbs_hot_kernels_t *t, int type, const bench_ws_t *w, float *out) {
    switch (type) {
        case 0: t->matvec_q8_0(w->w, w->rows, w->cols, w->xq, w->qd, out); break;
        case 1: t->matvec_q4_0(w->w, w->rows, w->cols, w->xq, w->qd, out); break;
        case 2: t->matvec_q4_1(w->w, w->rows, w->cols, w->xq, w->qd, w->qsum, out); break;
        case 3: t->matvec_f32((const float *)w->w, w->rows, w->cols, w->x, out); break;
        default: t->matvec_f16((const uint16_t *)w->w, w->rows, w->cols, w->x, out); break;
    }
}

/* Fill a workspace with random test data for one (type, shape).
 * Returns false on allocation failure. */
static bool bench_ws_fill(bench_ws_t *w, int type, size_t rows, size_t cols) {
    static const size_t stride[VBS_HOT_NTYPES] = { 34, 18, 20, 4, 2 };
    memset(w, 0, sizeof(*w));
    w->rows = rows;
    w->cols = cols;
    w->nblocks = cols / 32;
    w->wbytes = (type <= 2) ? rows * w->nblocks * stride[type] : rows * cols * stride[type];
    w->w = (uint8_t *)malloc(w->wbytes ? w->wbytes : 1);
    w->xq = (int8_t *)malloc(cols);
    w->x = (float *)malloc(cols * sizeof(float));
    w->qd = (float *)malloc((w->nblocks ? w->nblocks : 1) * sizeof(float));
    w->qsum = (float *)malloc((w->nblocks ? w->nblocks : 1) * sizeof(float));
    w->y_ref = (float *)malloc(rows * sizeof(float));
    w->y_cand = (float *)malloc(rows * sizeof(float));
    if (!w->w || !w->xq || !w->x || !w->qd || !w->qsum || !w->y_ref || !w->y_cand) {
        bench_ws_free(w);
        return false;
    }
    for (size_t i = 0; i < w->wbytes; i++) w->w[i] = (uint8_t)(vbs_xorshift32_next(&rng_state) & 0xff);
    for (size_t i = 0; i < cols; i++) {
        w->xq[i] = (int8_t)((int)(vbs_xorshift32_next(&rng_state) % 255) - 127);
        w->x[i] = ((float)(vbs_xorshift32_next(&rng_state) % 2000) - 1000.0f) / 1000.0f;
    }
    for (size_t b = 0; b < w->nblocks; b++) {
        w->qd[b] = 0.01f;
        w->qsum[b] = 0.5f;
    }
    /* Sane f16 scales in the packed headers (1.0 / 0.5 / 0.25). */
    if (type <= 2) {
        for (size_t r = 0; r < rows; r++) {
            for (size_t b = 0; b < w->nblocks; b++) {
                uint8_t *blk = w->w + r * w->nblocks * stride[type] + b * stride[type];
                blk[0] = 0x00; blk[1] = 0x3c;
                if (type == 2) { blk[2] = 0x00; blk[3] = 0x34; }
            }
        }
    }
    /* F32/F16 weights must be finite values, not random bit
     * patterns (which include NaN/Inf and would fail the
     * correctness check). */
    if (type == 3) {
        float *wf = (float *)w->w;
        for (size_t i = 0; i < rows * cols; i++)
            wf[i] = ((float)(vbs_xorshift32_next(&rng_state) % 2000) - 1000.0f) / 1000.0f;
    }
    if (type == 4) {
        static const uint16_t sane[4] = { 0x3c00, 0x3800, 0xbc00, 0x0000 };
        uint16_t *wh = (uint16_t *)w->w;
        for (size_t i = 0; i < rows * cols; i++) wh[i] = sane[vbs_xorshift32_next(&rng_state) % 4];
    }
    return true;
}

/* Calibrate a repetition count so one sample window is ~20ms. */
static size_t calibrate_reps(const vbs_hot_kernels_t *t, int type, bench_ws_t *w) {
    size_t reps = 1;
    double t0 = vbs_now_sec();
    for (;;) {
        for (size_t k = 0; k < reps; k++) run_matvec(t, type, w, w->y_cand);
        double dt = vbs_now_sec() - t0;
        if (dt >= 0.02 || reps >= 100000) break;
        reps *= 4;
        t0 = vbs_now_sec();
    }
    return reps;
}

/* Benchmark one kernel type. Correctness is always checked
 * (candidate vs builtin reference). Timing runs only when
 * (time_mask & (1<<type)): the loop times only the hotspot type,
 * and hotbench times only the types a candidate optimizes --
 * timing the rest wastes most of the bench on code the candidate
 * does not touch (and the slowest types dominate).
 *
 * Timing method (designed for a contended VM, where absolute
 * times jump):
 *  - reps are calibrated PER FUNCTION so both functions' samples
 *    are ~20ms windows (a shared reps count would make the slow
 *    function's samples much longer and more likely to be
 *    descheduled, inflating it).
 *  - candidate and base samples alternate in the same process, so
 *    each pair sees nearly the same contention.
 *  - the reported time per function is the MINIMUM of 7 samples
 *    (the least-contended sample), not the median: on this VM the
 *    minimum is stable across runs while the median jumps 2-3x.
 *  - the decision uses BOTH the minimum ratio and the paired win
 *    count (samples where the candidate beat the base): a real
 *    improvement wins most pairs; noise does not.
 */
static int bench_one_type(const vbs_hot_kernels_t *cand,
                          const vbs_hot_kernels_t *base,
                          const vbs_hot_kernels_t *ref,
                          int type, const vbs_bench_shape_t *shapes, size_t n_shapes,
                          uint32_t time_mask,
                          double *cand_ns_out, double *base_ns_out,
                          int *wins_out, bool *correct_out) {
    double worst_cand = 0.0, worst_base = 0.0;
    int worst_shape_wins = -1;
    bool correct = true;
    bool timed = (time_mask & (1u << type)) != 0;

    for (size_t s = 0; s < n_shapes; s++) {
        bench_ws_t ws;
        if (!bench_ws_fill(&ws, type, shapes[s].rows, shapes[s].cols)) return -1;

        /* Correctness: candidate vs the builtin reference on the same inputs. */
        run_matvec(ref, type, &ws, ws.y_ref);
        run_matvec(cand, type, &ws, ws.y_cand);
        for (size_t i = 0; i < ws.rows; i++) {
            if (!isfinite(ws.y_cand[i])) {
                correct = false;
                break;
            }
            double denom = fabs((double)ws.y_ref[i]);
            if (denom < 1e-6) denom = 1e-6;
            if (fabs((double)ws.y_cand[i] - (double)ws.y_ref[i]) / denom > CORRECT_TOL) {
                correct = false;
                break;
            }
        }

        if (!timed) {
            bench_ws_free(&ws);
            continue;
        }

        /* Per-function reps calibration, then 7 alternating
         * samples (base first, then candidate) per shape. */
        size_t reps_cand = calibrate_reps(cand, type, &ws);
        size_t reps_base = calibrate_reps(base, type, &ws);
        double cand_min = 1e30, base_min = 1e30;
        int shape_wins = 0;
        for (int k = 0; k < 7; k++) {
            double t0 = vbs_now_sec();
            for (size_t j = 0; j < reps_base; j++) run_matvec(base, type, &ws, ws.y_cand);
            double base_sample = (vbs_now_sec() - t0) / (double)reps_base;
            t0 = vbs_now_sec();
            for (size_t j = 0; j < reps_cand; j++) run_matvec(cand, type, &ws, ws.y_cand);
            double cand_sample = (vbs_now_sec() - t0) / (double)reps_cand;
            if (cand_sample < base_sample) shape_wins++;
            if (cand_sample < cand_min) cand_min = cand_sample;
            if (base_sample < base_min) base_min = base_sample;
        }
        /* Minimum of 7, not the median: on a contended VM the
         * fastest sample is the least-contended one, and the
         * minimum is stable across runs where the median jumps
         * 2-3x (verified: ref q8_0 min was 176-177us in three
         * separate rounds while its median moved 197-252us). */
        /* Keep the slowest shape's time: the loop optimizes the
         * worst case. The reported win count is the wins AT THAT
         * SHAPE, so the wins test and the ratio test judge the
         * same shape (a candidate can win on small shapes while
         * losing where it matters). */
        if (cand_min * 1e9 > worst_cand) {
            worst_cand = cand_min * 1e9;
            worst_shape_wins = shape_wins;
        }
        if (base_min * 1e9 > worst_base) worst_base = base_min * 1e9;
        /* The checksum keeps the timing loop honest (outputs are consumed). */
        volatile float sink = 0.0f;
        for (size_t i = 0; i < ws.rows; i++) sink += ws.y_cand[i];
        (void)sink;

        bench_ws_free(&ws);
    }
    *cand_ns_out = worst_cand;
    *base_ns_out = worst_base;
    *wins_out = worst_shape_wins;
    *correct_out = correct;
    return 0;
}

static const vbs_hot_kernels_t *load_table(const char *so_path, void **handle_out,
                                           char *err, size_t err_cap) {
    if (!so_path) {
        *handle_out = NULL;
        return vbs_hot_builtin();
    }
    void *h = vbs_dlopen(so_path, 1);
    if (!h) {
        snprintf(err, err_cap, "bench: dlopen failed: %s", vbs_dlerror());
        return NULL;
    }
    uint32_t (*abi_fn)(void) = (uint32_t (*)(void))vbs_dlsym(h, "vbs_hot_abi_version");
    const vbs_hot_kernels_t *(*table_fn)(void) =
        (const vbs_hot_kernels_t *(*)(void))vbs_dlsym(h, "vbs_hot_kernels");
    if (!abi_fn || !table_fn || abi_fn() != VBS_HOT_ABI_VERSION) {
        snprintf(err, err_cap, "bench: bad ABI exports");
        vbs_dlclose(h);
        return NULL;
    }
    const vbs_hot_kernels_t *t = table_fn();
    if (!t || !t->matvec_q8_0 || !t->matvec_q4_0 || !t->matvec_q4_1 ||
        !t->matvec_f32 || !t->matvec_f16) {
        snprintf(err, err_cap, "bench: incomplete kernel table");
        vbs_dlclose(h);
        return NULL;
    }
    *handle_out = h;
    return t;
}

int vbs_loop_bench_module(const vbs_bench_args_t *a) {
    const vbs_hot_kernels_t *ref = vbs_hot_builtin();
    void *h_cand = NULL, *h_base = NULL;
    const vbs_hot_kernels_t *cand = load_table(a->so_path, &h_cand, a->err, a->err_cap);
    if (!cand) return -1;
    const vbs_hot_kernels_t *base = load_table(a->base_so_path, &h_base, a->err, a->err_cap);
    if (!base) {
        if (h_cand) vbs_dlclose(h_cand);
        return -1;
    }
    bool all_correct = true;
    for (int t = 0; t < VBS_HOT_NTYPES; t++) {
        bool c = true;
        double cns = 0.0, bns = 0.0;
        int w = -1;
        if (bench_one_type(cand, base, ref, t, a->shapes, a->n_shapes, a->time_mask, &cns, &bns, &w, &c) != 0) {
            snprintf(a->err, a->err_cap, "bench: out of memory");
            if (h_cand) vbs_dlclose(h_cand);
            if (h_base) vbs_dlclose(h_base);
            return -1;
        }
        a->median_ns[t] = cns;
        a->base_ns[t] = bns;
        a->wins[t] = w;
        if (!c) all_correct = false;
    }
    *a->correct = all_correct;
    if (h_cand) vbs_dlclose(h_cand);
    if (h_base) vbs_dlclose(h_base);
    return 0;
}

/* Child bench: fork, run the benchmark, write results to a file,
 * exit. The parent treats a signal death as a candidate crash. */
int vbs_loop_bench_in_child(const vbs_bench_args_t *a) {
#if defined(_WIN32)
    /* The forked-child benchmark is POSIX-only: Windows has no
     * fork(). The in-process benchmark (vbs_loop_bench_module)
     * still works. */
    snprintf(a->err, a->err_cap, "bench: forked child benchmark is POSIX-only");
    return -1;
#else
    /* Result file in the current directory, not /tmp: /tmp is a
     * small tmpfs on this machine and can be full (other agents'
     * models). The file is unlinked after the parent reads it. */
    char result_path[512];
    snprintf(result_path, sizeof(result_path), ".vbs_loop_bench_%d.txt", (int)getpid());
    pid_t pid = fork();
    if (pid < 0) {
        snprintf(a->err, a->err_cap, "bench: fork failed");
        return -1;
    }
    if (pid == 0) {
        double ns[VBS_HOT_NTYPES] = {0};
        double bns[VBS_HOT_NTYPES] = {0};
        int w[VBS_HOT_NTYPES] = {0};
        bool c = false;
        char berr[1024] = {0};
        /* The child runs the same benchmark with local result
         * buffers; the parent reads them back from the file. */
        vbs_bench_args_t child = *a;
        child.median_ns = ns;
        child.base_ns = bns;
        child.wins = w;
        child.correct = &c;
        child.err = berr;
        child.err_cap = sizeof(berr);
        int rc = vbs_loop_bench_module(&child);
        FILE *f = fopen(result_path, "w");
        if (f) {
            fprintf(f, "rc=%d correct=%d\n", rc, c ? 1 : 0);
            for (int t = 0; t < VBS_HOT_NTYPES; t++) fprintf(f, "ns_%d=%.1f\n", t, ns[t]);
            for (int t = 0; t < VBS_HOT_NTYPES; t++) fprintf(f, "base_%d=%.1f\n", t, bns[t]);
            for (int t = 0; t < VBS_HOT_NTYPES; t++) fprintf(f, "wins_%d=%d\n", t, w[t]);
            fprintf(f, "err=%s\n", berr);
            fclose(f);
        }
        _exit(0);
    }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) { }
    if (WIFSIGNALED(status)) {
        snprintf(a->err, a->err_cap, "bench: candidate crashed (signal %d)", WTERMSIG(status));
        unlink(result_path);
        return -2;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        snprintf(a->err, a->err_cap, "bench: child exited abnormally");
        unlink(result_path);
        return -1;
    }
    FILE *f = fopen(result_path, "r");
    if (!f) {
        snprintf(a->err, a->err_cap, "bench: no result file from child");
        return -1;
    }
    int rc = -1;
    int corr = 0;
    char line[1200];
    char child_err[1024] = {0};
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "rc=%d correct=%d", &rc, &corr) == 2) continue;
        for (int t = 0; t < VBS_HOT_NTYPES; t++) {
            char pat[32];
            snprintf(pat, sizeof(pat), "ns_%d=", t);
            if (strncmp(line, pat, strlen(pat)) == 0) {
                a->median_ns[t] = strtod(line + strlen(pat), NULL);
            }
            snprintf(pat, sizeof(pat), "base_%d=", t);
            if (strncmp(line, pat, strlen(pat)) == 0) {
                a->base_ns[t] = strtod(line + strlen(pat), NULL);
            }
            snprintf(pat, sizeof(pat), "wins_%d=", t);
            if (strncmp(line, pat, strlen(pat)) == 0) {
                a->wins[t] = atoi(line + strlen(pat));
            }
        }
        if (strncmp(line, "err=", 4) == 0) {
            snprintf(child_err, sizeof(child_err), "%.1023s", line + 4);
            size_t n = strlen(child_err);
            while (n && (child_err[n - 1] == '\n')) child_err[--n] = '\0';
        }
    }
    fclose(f);
    unlink(result_path);
    if (rc != 0) {
        snprintf(a->err, a->err_cap, "bench: %s", child_err[0] ? child_err : "child bench failed");
        return -1;
    }
    *a->correct = corr != 0;
    return 0;
#endif /* _WIN32 */
}

/* --- Structured JSONL log --- */

typedef struct {
    FILE *f;
    int quiet;
} loop_log_t;

static void log_event(loop_log_t *lg, const char *event, const char *fields) {
    if (lg->f) {
        fprintf(lg->f, "{\"ts_ms\":%llu,\"event\":\"%s\"%s}\n",
                (unsigned long long)now_ms(), event, fields ? fields : "");
        fflush(lg->f);
    }
}

static double counters_total_ns(void) {
    vbs_hot_counter_t c[VBS_HOT_NTYPES];
    vbs_hot_counters(c);
    uint64_t total = 0;
    for (int i = 0; i < VBS_HOT_NTYPES; i++) total += c[i].ns;
    return (double)total;
}

static int hotspot_type(void) {
    vbs_hot_counter_t c[VBS_HOT_NTYPES];
    vbs_hot_counters(c);
    int best = 0;
    for (int i = 1; i < VBS_HOT_NTYPES; i++) {
        if (c[i].ns > c[best].ns) best = i;
    }
    return best;
}

/* --- Live session helpers --- */

static uint32_t argmax_token(const float *logits, size_t n) {
    uint32_t best = 0;
    for (size_t i = 1; i < n; i++) if (logits[i] > logits[best]) best = (uint32_t)i;
    return best;
}

/* Generate up to n tokens greedily, appending to res->tokens.
 * Returns the number actually generated (stops early on EOS or
 * error). */
static size_t gen_tokens(vbs_cpu_model_t *m, size_t n, vbs_loop_result_t *res,
                         char *err, size_t err_cap) {
    size_t done = 0;
    uint32_t eos = vbs_cpu_eos_id(m);
    for (size_t i = 0; i < n; i++) {
        uint32_t tok = argmax_token(vbs_cpu_logits(m), vbs_cpu_vocab_size(m));
        res->tokens[res->n_tokens_generated++] = tok;
        done++;
        if (tok == eos) break;
        if (!vbs_cpu_step(m, tok, err, err_cap)) return done;
    }
    return done;
}

/* Like gen_tokens, but records per-token matvec time (counter
 * delta around each forward pass) and returns the MINIMUM and
 * mean. The live regression monitor compares minima, not means:
 * on a contended VM a window's mean swings 2x between identical
 * modules (verified), while the minimum -- the least-contended
 * forward pass in the window -- is stable. A module that is
 * genuinely slower has a higher minimum; contention can only
 * inflate a measurement, never deflate it below the module's
 * true cost. */
static size_t gen_tokens_measured(vbs_cpu_model_t *m, size_t n, vbs_loop_result_t *res,
                                  double *min_ns, double *mean_ns,
                                  char *err, size_t err_cap) {
    double total = 0.0, mn = 1e30;
    size_t measured = 0, done = 0;
    uint32_t eos = vbs_cpu_eos_id(m);
    for (size_t i = 0; i < n; i++) {
        double c0 = counters_total_ns();
        uint32_t tok = argmax_token(vbs_cpu_logits(m), vbs_cpu_vocab_size(m));
        res->tokens[res->n_tokens_generated++] = tok;
        done++;
        if (tok == eos) break;
        if (!vbs_cpu_step(m, tok, err, err_cap)) break;
        double c1 = counters_total_ns();
        double dt = c1 - c0;
        total += dt;
        if (dt < mn) mn = dt;
        measured++;
    }
    *min_ns = measured ? mn : 0.0;
    *mean_ns = measured ? total / (double)measured : 0.0;
    return done;
}

/* One measured live window: reset the counters, generate up to
 * `cap` tokens (bounded by the remaining budget), and report the
 * count plus min/mean matvec time per token. */
static size_t live_window(vbs_cpu_model_t *m, size_t cap, size_t *tokens_left,
                          vbs_loop_result_t *res, double *min_ns, double *mean_ns,
                          char *err, size_t err_cap) {
    vbs_hot_counters_reset();
    size_t w = *tokens_left < cap ? *tokens_left : cap;
    size_t gen = gen_tokens_measured(m, w, res, min_ns, mean_ns, err, err_cap);
    *tokens_left -= gen;
    return gen;
}

/* Derive benchmark shapes from the model: (ffn,emb), (emb,emb),
 * (vocab,emb). */
static size_t model_shapes(const char *model_path, vbs_bench_shape_t *out, size_t cap) {
    char err[512] = {0};
    vbs_model_t *g = vbs_model_open(model_path, err, sizeof(err));
    if (!g) return 0;
    uint32_t emb = 0, ffn = 0, vocab = 0;
    size_t n = 0;
    if (vbs_model_get_uint32_metadata(g, "llama.embedding_length", &emb) &&
        vbs_model_get_uint32_metadata(g, "llama.feed_forward_length", &ffn) &&
        vbs_model_get_uint32_metadata(g, "llama.vocab_size", &vocab)) {
        if (n < cap) out[n++] = (vbs_bench_shape_t){ ffn, emb };
        if (n < cap) out[n++] = (vbs_bench_shape_t){ emb, emb };
        if (n < cap) out[n++] = (vbs_bench_shape_t){ vocab, emb };
    }
    vbs_model_close(g);
    return n;
}

void vbs_loop_result_free(vbs_loop_result_t *res) {
    if (!res) return;
    free(res->tokens);
    res->tokens = NULL;
}

/* Escape a string for embedding in a JSON string value: quotes
 * and backslashes are backslash-escaped, newlines become spaces. */
static void escape_json(const char *src, char *dst, size_t cap) {
    size_t j = 0;
    for (size_t i = 0; src[i] && j + 2 < cap; i++) {
        if (src[i] == '"' || src[i] == '\\') dst[j++] = '\\';
        if (src[i] == '\n' || src[i] == '\r') {
            dst[j++] = ' ';
            continue;
        }
        dst[j++] = src[i];
    }
    dst[j] = '\0';
}

/* Record a candidate rejection: console line (unless quiet) plus
 * the candidate_decision log event. `console_suffix` is the text
 * inside the REJECTED parentheses; `extra_fields` is appended to
 * the log event's fields (may be ""). */
static void reject_candidate(loop_log_t *lg, const vbs_loop_config_t *cfg,
                             vbs_loop_result_t *res, int iter,
                             const vbs_candidate_t *cand, const char *reason,
                             const char *console_suffix, const char *extra_fields) {
    res->rejected++;
    if (!cfg->quiet) {
        printf("[loop] iter %d: %s REJECTED (%s)\n", iter, cand->name, console_suffix);
    }
    char fields[2400];
    snprintf(fields, sizeof(fields),
             ",\"iter\":%d,\"candidate\":\"%s\",\"decision\":\"rejected\",\"reason\":\"%s\"%s",
             iter, cand->name, reason, extra_fields ? extra_fields : "");
    log_event(lg, "candidate_decision", fields);
}

/* --- The loop --- */

int vbs_loop_run(const vbs_loop_config_t *cfg, vbs_loop_result_t *res) {
    memset(res, 0, sizeof(*res));
    snprintf(res->best_variant, sizeof(res->best_variant), "builtin-ref");
    res->tokens = (uint32_t *)calloc(cfg->n_tokens + 1, sizeof(uint32_t));
    if (!res->tokens) return -1;

    const char *log_path = cfg->log_path ? cfg->log_path : "vbs-loop.jsonl";
    const char *work_dir = cfg->work_dir ? cfg->work_dir : ".verum-loop-work";
    const char *repo_dir = cfg->repo_dir ? cfg->repo_dir : ".";
    /* Create the work dir (no shell: work_dir comes from the CLI). */
    vbs_mkdir_p(work_dir);

    loop_log_t lg = { NULL, cfg->quiet };
    lg.f = fopen(log_path, "a");
    if (!lg.f) {
        fprintf(stderr, "loop: cannot open log %s\n", log_path);
        return -1;
    }

    char err[1024] = {0};
    vbs_cpu_model_t *m = vbs_cpu_open(cfg->model_path, 128, err, sizeof(err));
    if (!m) {
        fprintf(stderr, "loop: model open failed: %s\n", err);
        fclose(lg.f);
        return -1;
    }

    /* Prompt: feed all prompt tokens, computing logits only on the last. */
    for (size_t i = 0; i < cfg->n_prompt; i++) {
        if (!vbs_cpu_step_ex(m, cfg->prompt[i], i + 1 == cfg->n_prompt, err, sizeof(err))) {
            fprintf(stderr, "loop: prompt step failed: %s\n", err);
            vbs_cpu_close(m);
            fclose(lg.f);
            return -1;
        }
    }

    vbs_bench_shape_t shapes[4];
    size_t n_shapes = model_shapes(cfg->model_path, shapes, 4);
    if (n_shapes == 0) {
        shapes[0] = (vbs_bench_shape_t){ 1408, 512 };
        shapes[1] = (vbs_bench_shape_t){ 512, 512 };
        n_shapes = 2;
    }

    if (!cfg->quiet) {
        printf("[loop] session started: model=%s prompt_tokens=%zu target_tokens=%zu\n",
               cfg->model_path, cfg->n_prompt, cfg->n_tokens);
        printf("[loop] hotspot shapes:");
        for (size_t i = 0; i < n_shapes; i++) printf(" %zux%zu", shapes[i].rows, shapes[i].cols);
        printf("\n");
    }
    {
        char fields[1024];
        snprintf(fields, sizeof(fields),
                 ",\"model\":\"%s\",\"prompt_tokens\":%zu,\"target_tokens\":%zu,"
                 "\"max_iterations\":%d",
                 cfg->model_path, cfg->n_prompt, cfg->n_tokens, cfg->max_iterations);
        log_event(&lg, "loop_start", fields);
    }

    /* Baseline: measure the builtin table the same way candidates
     * are measured (isolated child bench), and run a live token
     * window. */
    double best_ns[VBS_HOT_NTYPES] = {0};
    bool base_correct = false;
    double dummy_base[VBS_HOT_NTYPES] = {0};
    int dummy_wins[VBS_HOT_NTYPES] = {0};
    vbs_bench_args_t bargs = { .so_path = NULL, .base_so_path = NULL,
                               .shapes = shapes, .n_shapes = n_shapes, .time_mask = 0x1Fu,
                               .median_ns = best_ns, .base_ns = dummy_base, .wins = dummy_wins,
                               .correct = &base_correct, .err = err, .err_cap = sizeof(err) };
    if (vbs_loop_bench_in_child(&bargs) != 0) {
        fprintf(stderr, "loop: baseline bench failed: %s\n", err);
        vbs_cpu_close(m);
        fclose(lg.f);
        return -1;
    }
    if (!cfg->quiet) {
        printf("[loop] baseline (builtin-ref):");
        for (int t = 0; t < VBS_HOT_NTYPES; t++) printf(" %s=%.0fns", vbs_hot_type_name(t), best_ns[t]);
        printf("\n");
    }
    {
        char fields[1024];
        snprintf(fields, sizeof(fields),
                 ",\"variant\":\"builtin-ref\",\"q8_0_ns\":%.1f,\"q4_0_ns\":%.1f,"
                 "\"q4_1_ns\":%.1f,\"f32_ns\":%.1f,\"f16_ns\":%.1f",
                 best_ns[0], best_ns[1], best_ns[2], best_ns[3], best_ns[4]);
        log_event(&lg, "baseline", fields);
    }

    /* Embedded tune profile (if any): order candidates
     * best-known-first. */
    int order[16];
    size_t n_cand = vbs_loop_candidate_count();
    for (size_t i = 0; i < n_cand; i++) order[i] = (int)i;
    vbs_tune_profile_t prof;
    char terr[256] = {0};
    if (vbs_tune_load_embedded(&prof, terr, sizeof(terr)) && vbs_tune_matches_machine(&prof)) {
        if (!cfg->quiet) printf("[loop] embedded tune profile matches this machine; ordering candidates by it\n");
        log_event(&lg, "tune_profile", ",\"source\":\"embedded\",\"match\":true");
        /* Simple stable preference: move a profile-best variant to the front. */
        for (int t = 0; t < VBS_HOT_NTYPES; t++) {
            if (!prof.best[t][0]) continue;
            for (size_t i = 1; i < n_cand; i++) {
                if (strcmp(g_candidates[order[i]].name, prof.best[t]) == 0) {
                    int tmp = order[i];
                    memmove(order + 1, order, i * sizeof(int));
                    order[0] = tmp;
                    break;
                }
            }
        }
    }

    double t_start = vbs_now_sec();
    size_t tokens_left = cfg->n_tokens;
    bool tried[16] = { false };
    int hotspot = 0;
    double live_baseline_ns_per_token = 0.0;
    bool poison_done = false;

    /* Live baseline window: a few tokens on the builtin table, so
     * the live regression monitor has a pre-swap number to compare
     * against. */
    {
        double min_ns = 0.0, mean_ns = 0.0;
        size_t gen = live_window(m, cfg->tokens_per_iter, &tokens_left, res,
                                 &min_ns, &mean_ns, err, sizeof(err));
        if (gen > 0) live_baseline_ns_per_token = min_ns;
        hotspot = hotspot_type();
        snprintf(res->hotspot_type, sizeof(res->hotspot_type), "%s", vbs_hot_type_name(hotspot));
        if (!cfg->quiet) {
            printf("[loop] live baseline: min %.0f ns/token matvec (mean %.0f), hotspot=%s\n",
                   min_ns, mean_ns, res->hotspot_type);
        }
        char fields[512];
        snprintf(fields, sizeof(fields),
                 ",\"hotspot\":\"%s\",\"live_min_ns_per_token\":%.1f,\"live_mean_ns_per_token\":%.1f",
                 res->hotspot_type, min_ns, mean_ns);
        log_event(&lg, "hotspot", fields);
    }

    for (int iter = 1; iter <= cfg->max_iterations; iter++) {
        /* Pick the next untested candidate for the hotspot type. */
        const vbs_candidate_t *cand = NULL;
        int cand_idx = -1;
        for (size_t k = 0; k < n_cand; k++) {
            int idx = order[k];
            if (tried[idx]) continue;
            if (!g_candidates[idx].optimizes[hotspot]) continue;
            if (g_candidates[idx].needs_avx2 && !vbs_hot_cpu_has_avx2()) {
                tried[idx] = true;
                if (!cfg->quiet) printf("[loop] iter %d: skip %s (host lacks AVX2)\n", iter, g_candidates[idx].name);
                char fields[256];
                snprintf(fields, sizeof(fields), ",\"iter\":%d,\"candidate\":\"%s\",\"reason\":\"no-avx2\"",
                         iter, g_candidates[idx].name);
                log_event(&lg, "skipped", fields);
                continue;
            }
            cand = &g_candidates[idx];
            cand_idx = idx;
            break;
        }
        if (!cand) {
            if (!cfg->quiet) printf("[loop] iter %d: no candidates left for hotspot %s — loop terminated\n", iter, res->hotspot_type);
            char fields[256];
            snprintf(fields, sizeof(fields), ",\"iter\":%d,\"hotspot\":\"%s\"", iter, res->hotspot_type);
            log_event(&lg, "exhausted", fields);
            break;
        }
        tried[cand_idx] = true;
        res->iterations_run++;

        if (!cfg->quiet) printf("[loop] iter %d: candidate %s (%s)\n", iter, cand->name, cand->note);
        {
            char fields[512];
            snprintf(fields, sizeof(fields), ",\"iter\":%d,\"candidate\":\"%s\",\"hotspot\":\"%s\"",
                     iter, cand->name, res->hotspot_type);
            log_event(&lg, "candidate_start", fields);
        }

        /* Compile. */
        char src_path[1200], so_path[1200];
        snprintf(src_path, sizeof(src_path), "%s/src/hotkernels/%s", repo_dir, cand->src_file);
        snprintf(so_path, sizeof(so_path), "%s/cand_%s.so", work_dir, cand->name);
        char inc1[1200], inc2[1200];
        snprintf(inc1, sizeof(inc1), "-I%s/include", repo_dir);
        snprintf(inc2, sizeof(inc2), "-I%s/src/hotkernels", repo_dir);
        const char *extra[3];
        size_t n_extra = 0;
        extra[n_extra++] = inc1;
        extra[n_extra++] = inc2;
        if (cand->extra_flag) extra[n_extra++] = cand->extra_flag;
        char cerr[2048] = {0};
        double ct0 = vbs_now_sec();
        int crc = vbs_cc_compile_so(NULL, src_path, so_path, extra, n_extra, cerr, sizeof(cerr));
        double compile_sec = vbs_now_sec() - ct0;
        if (crc != 0) {
            char cerr_esc[2048];
            escape_json(cerr, cerr_esc, sizeof(cerr_esc));
            char extra_fields[2400];
            snprintf(extra_fields, sizeof(extra_fields),
                     ",\"rc\":%d,\"compile_sec\":%.2f,\"stderr\":\"%s\"",
                     crc, compile_sec, cerr_esc);
            char console[64];
            snprintf(console, sizeof(console), "compile failed, rc=%d", crc);
            reject_candidate(&lg, cfg, res, iter, cand, "compile_failed", console, extra_fields);
            continue;
        }

        /* Benchmark-verify in an isolated child, against the
         * CURRENT live module as the base -- both timed interleaved
         * in the same child, so the accept decision compares
         * numbers measured under identical conditions. Only the
         * hotspot type is timed. */
        double cand_ns[VBS_HOT_NTYPES] = {0};
        double cur_ns[VBS_HOT_NTYPES] = {0};
        int wins[VBS_HOT_NTYPES] = {0};
        bool correct = false;
        const char *cur_path = vbs_hot_current_path();
        vbs_bench_args_t bargs = { .so_path = so_path,
                                   .base_so_path = cur_path[0] ? cur_path : NULL,
                                   .shapes = shapes, .n_shapes = n_shapes,
                                   .time_mask = (uint32_t)(1u << hotspot),
                                   .median_ns = cand_ns, .base_ns = cur_ns, .wins = wins,
                                   .correct = &correct, .err = err, .err_cap = sizeof(err) };
        int brc = vbs_loop_bench_in_child(&bargs);
        if (brc == -2) {
            reject_candidate(&lg, cfg, res, iter, cand, "crash", "crashed in child bench", "");
            continue;
        }
        if (brc != 0) {
            char console[1200];
            snprintf(console, sizeof(console), "bench failed: %s", err);
            reject_candidate(&lg, cfg, res, iter, cand, "bench_failed", console, "");
            continue;
        }
        if (!correct) {
            reject_candidate(&lg, cfg, res, iter, cand, "incorrect", "wrong results", "");
            continue;
        }
        /* Accept only when the candidate beats the CURRENT module
         * by the margin, measured in the same child run -- with
         * paired-win evidence scaled to the size of the win. A big
         * ratio win (>1.25x) needs only a majority of paired wins;
         * a marginal win needs strong paired evidence, because a
         * 2% median edge on a contended VM is as likely to be
         * noise as signal. */
        double before = cur_ns[hotspot];
        double after = cand_ns[hotspot];
        int w = wins[hotspot];
        int n_pairs = 7; /* wins are counted at the worst shape only */
        int strong_win = (after < before * 0.80);
        int enough_wins = strong_win ? (w >= 4) : (w >= 5);
        if (!(after < before * ACCEPT_MARGIN) || !enough_wins) {
            char console[256];
            snprintf(console, sizeof(console),
                     "%.0f ns vs current %.0f ns, wins %d/%d — no reliable improvement",
                     after, before, w, n_pairs);
            char extra_fields[256];
            snprintf(extra_fields, sizeof(extra_fields),
                     ",\"candidate_ns\":%.1f,\"current_ns\":%.1f,\"wins\":%d,\"pairs\":%d",
                     after, before, w, n_pairs);
            reject_candidate(&lg, cfg, res, iter, cand, "no_improvement", console, extra_fields);
            continue;
        }

        /* Accept: hot-swap between tokens. */
        char lerr[512] = {0};
        const char *prev_name = vbs_hot_current_name();
        if (!vbs_hot_load(so_path, lerr, sizeof(lerr))) {
            char console[600];
            snprintf(console, sizeof(console), "load failed: %s", lerr);
            reject_candidate(&lg, cfg, res, iter, cand, "load_failed", console, "");
            continue;
        }
        res->accepted++;
        res->swaps++;
        double prev_best = best_ns[hotspot];
        best_ns[hotspot] = after;
        snprintf(res->best_variant, sizeof(res->best_variant), "%s", cand->name);
        res->best_ns = after;
        /* Rollback record: the best values belonging to the module
         * now in the rollback slot (the one we just replaced). */
        double rb_best_ns = prev_best;
        char rb_best_variant[64];
        snprintf(rb_best_variant, sizeof(rb_best_variant), "%s", prev_name);
        if (!cfg->quiet) printf("[loop] iter %d: %s ACCEPTED (%.0f ns vs %.0f ns, %.1fx, wins %d/%d) — hot-swapped in\n",
               iter, cand->name, after, before, before / after, w, n_pairs);
        {
            char fields[768];
            snprintf(fields, sizeof(fields),
                     ",\"iter\":%d,\"candidate\":\"%s\",\"decision\":\"accepted\","
                     "\"candidate_ns\":%.1f,\"current_ns\":%.1f,\"speedup\":%.3f,"
                     "\"wins\":%d,\"pairs\":%d,"
                     "\"swap_from\":\"%s\",\"swap_to\":\"%s\"",
                     iter, cand->name, after, before, before / after,
                     w, n_pairs,
                     prev_name, cand->name);
            log_event(&lg, "candidate_decision", fields);
        }

        /* Fault injection (poison mode): after the first real swap,
         * force in the deliberately slow module as if it had been
         * accepted, so the live monitor must catch the regression
         * and roll back. */
        if (cfg->poison && !poison_done && res->swaps == 1) {
            poison_done = true;
            char slow_src[1200], slow_so[1200];
            snprintf(slow_src, sizeof(slow_src), "%s/src/hotkernels/cand_slow.c", repo_dir);
            snprintf(slow_so, sizeof(slow_so), "%s/cand_poison.so", work_dir);
            const char *pextra[3] = { inc1, inc2, "-O0" };
            char perr[2048] = {0};
            if (vbs_cc_compile_so(NULL, slow_src, slow_so, pextra, 3, perr, sizeof(perr)) == 0 &&
                vbs_hot_load(slow_so, lerr, sizeof(lerr))) {
                /* The rollback slot now holds the accepted
                 * candidate, so a rollback restores the
                 * candidate's best time, not the pre-acceptance
                 * baseline. */
                rb_best_ns = after;
                snprintf(rb_best_variant, sizeof(rb_best_variant), "%s", cand->name);
                if (!cfg->quiet) printf("[loop] POISON INJECTED: slow module force-loaded, live monitor must roll it back\n");
                log_event(&lg, "poison_inject", ",\"module\":\"slow-wasteful\"");
            } else {
                if (!cfg->quiet) printf("[loop] poison injection failed (compile/load): %s\n", lerr[0] ? lerr : perr);
            }
        }

        /* Live regression monitor: generate the next window on the
         * new module and compare the MINIMUM per-token matvec time
         * against the pre-swap minimum (see
         * gen_tokens_measured). */
        if (tokens_left > 0) {
            double min_ns = 0.0, mean_ns = 0.0;
            size_t gen = live_window(m, cfg->tokens_per_iter, &tokens_left, res,
                                     &min_ns, &mean_ns, err, sizeof(err));
            if (gen > 0) {
                double live_ns = min_ns;
                if (!cfg->quiet) printf("[loop] live monitor: min %.0f ns/token (mean %.0f, baseline min %.0f)\n",
                       min_ns, mean_ns, live_baseline_ns_per_token);
                if (live_ns > live_baseline_ns_per_token * REGRESS_MARGIN) {
                    char rerr[512] = {0};
                    if (vbs_hot_rollback(rerr, sizeof(rerr))) {
                        res->rollbacks++;
                        best_ns[hotspot] = rb_best_ns;
                        snprintf(res->best_variant, sizeof(res->best_variant), "%s", rb_best_variant);
                        res->best_ns = rb_best_ns;
                        if (!cfg->quiet) printf("[loop] LIVE REGRESSION — rolled back to %s\n", rb_best_variant);
                        char fields[768];
                        snprintf(fields, sizeof(fields),
                                 ",\"iter\":%d,\"candidate\":\"%s\",\"live_min_ns_per_token\":%.1f,"
                                 "\"live_mean_ns_per_token\":%.1f,"
                                 "\"baseline_min_ns_per_token\":%.1f,\"rolled_back_to\":\"%s\"",
                                 iter, cand->name, min_ns, mean_ns, live_baseline_ns_per_token, rb_best_variant);
                        log_event(&lg, "rollback", fields);
                        /* Refresh the live baseline on the restored
                         * module so the next iteration's monitor
                         * compares against the module actually
                         * running. */
                        if (tokens_left > 0) {
                            double rmin = 0.0, rmean = 0.0;
                            size_t gen2 = live_window(m, cfg->tokens_per_iter, &tokens_left, res,
                                                      &rmin, &rmean, err, sizeof(err));
                            if (gen2 > 0) live_baseline_ns_per_token = rmin;
                        }
                    } else {
                        if (!cfg->quiet) printf("[loop] LIVE REGRESSION but rollback failed: %s\n", rerr);
                        char fields[512];
                        snprintf(fields, sizeof(fields),
                                 ",\"iter\":%d,\"candidate\":\"%s\",\"error\":\"rollback_failed\"",
                                 iter, cand->name);
                        log_event(&lg, "rollback_failed", fields);
                    }
                } else {
                    live_baseline_ns_per_token = live_ns;
                    char fields[512];
                    snprintf(fields, sizeof(fields),
                             ",\"iter\":%d,\"candidate\":\"%s\",\"live_min_ns_per_token\":%.1f,"
                             "\"live_mean_ns_per_token\":%.1f",
                             iter, cand->name, min_ns, mean_ns);
                    log_event(&lg, "live_ok", fields);
                }
            }
        }

        /* Keep the session going between iterations. */
        if (tokens_left > 0) {
            size_t w = tokens_left < cfg->tokens_per_iter ? tokens_left : cfg->tokens_per_iter;
            size_t gen = gen_tokens(m, w, res, err, sizeof(err));
            tokens_left -= gen;
        }
    }

    /* Finish the remaining tokens on whatever module is current. */
    while (tokens_left > 0) {
        size_t gen = gen_tokens(m, tokens_left, res, err, sizeof(err));
        if (gen == 0) break;
        tokens_left -= gen;
    }

    double total_sec = vbs_now_sec() - t_start;
    res->tokens_per_sec = total_sec > 0 ? (double)res->n_tokens_generated / total_sec : 0.0;
    if (res->best_ns == 0.0) res->best_ns = best_ns[hotspot];

    {
        char fields[1024];
        snprintf(fields, sizeof(fields),
                 ",\"iterations\":%d,\"accepted\":%d,\"rejected\":%d,\"swaps\":%d,"
                 "\"rollbacks\":%d,\"best_variant\":\"%s\",\"best_ns\":%.1f,"
                 "\"hotspot\":\"%s\",\"tokens_generated\":%zu,\"tokens_per_sec\":%.1f,"
                 "\"final_kernel\":\"%s\"",
                 res->iterations_run, res->accepted, res->rejected, res->swaps,
                 res->rollbacks, res->best_variant, res->best_ns,
                 res->hotspot_type, res->n_tokens_generated, res->tokens_per_sec,
                 vbs_hot_current_name());
        log_event(&lg, "loop_end", fields);
    }
    if (!cfg->quiet) {
        printf("[loop] done: %d iterations, %d accepted, %d rejected, %d swaps, %d rollbacks\n",
               res->iterations_run, res->accepted, res->rejected, res->swaps, res->rollbacks);
        printf("[loop] best variant: %s (%.0f ns at hotspot %s); final kernel: %s\n",
               res->best_variant, res->best_ns, res->hotspot_type, vbs_hot_current_name());
        printf("[loop] session: %zu tokens, %.1f tok/s (session state never dropped)\n",
               res->n_tokens_generated, res->tokens_per_sec);
    }

    vbs_cpu_close(m);
    fclose(lg.f);
    return 0;
}
