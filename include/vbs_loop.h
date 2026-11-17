/* vbs_loop.h — the closed recursive optimization loop.
 *
 * Logs -> hotspot analysis -> generate candidate -> compile -> benchmark
 * (in an isolated child) -> hot-swap between tokens -> live measurement ->
 * rollback on regression -> repeat. Unsupervised: the daemon runs it, and
 * every decision is written to a structured JSONL log.
 *
 * Safety invariants:
 *  - A candidate is NEVER loaded into the live process before it has
 *    passed compile + correctness + benchmark in a forked child. A
 *    candidate that crashes kills only the child.
 *  - A swap happens only between tokens, when no kernel call is in flight,
 *    and only swaps the function-pointer table — model state (KV cache,
 *    position, weights) is never touched, so the session cannot drop.
 *  - After a swap, the next token window is measured live; a regression
 *    beyond the threshold rolls back to the previous module immediately.
 *  - The loop terminates: when the candidate list is exhausted, or after
 *    max_iterations, it reports the best variant found (or honestly that
 *    nothing beat the baseline).
 */
#ifndef VBS_LOOP_H
#define VBS_LOOP_H

#include "vbs_hotpath.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --- Candidate registry --- */

typedef struct {
    const char *name;          /* variant name, e.g. "avx2" */
    const char *src_file;      /* under <repo>/src/hotkernels/ */
    const char *extra_flag;    /* extra compiler flag, e.g. "-mavx2" or "-O0" */
    bool needs_avx2;           /* skip on hosts without AVX2 */
    bool optimizes[VBS_HOT_NTYPES]; /* which kernel types this candidate improves */
    const char *note;          /* what it does (or what gate it tests) */
} vbs_candidate_t;

size_t vbs_loop_candidate_count(void);
const vbs_candidate_t *vbs_loop_candidate(size_t idx);

/* --- Benchmarking --- */

typedef struct { size_t rows, cols; } vbs_bench_shape_t;

/* Benchmark one module (so_path == NULL means the builtin reference table)
 * against a base module (base_so_path == NULL means builtin) at the given
 * shapes. Fills minimum ns per matvec call per type for both, the paired
 * win count per type (of 7 samples at the candidate's worst shape, where
 * the candidate beat the base; -1 when the type was not timed), and sets *correct false when any type's output
 * disagrees with the reference beyond tolerance. Correctness is always
 * checked against the builtin reference; timing interleaves candidate and
 * base in the same process (alternating samples, per-function reps), so
 * the ratio between them is meaningful even on a contended machine where
 * absolute times jump around. time_mask selects which types are timed
 * (bit t); untimed types are still correctness-checked. Returns 0 on
 * success, -1 on load/bench failure (message in err). Runs in-process;
 * the loop calls it in a forked child for isolation (see
 * vbs_loop_bench_in_child). */
int vbs_loop_bench_module(const char *so_path, const char *base_so_path,
                          const vbs_bench_shape_t *shapes, size_t n_shapes,
                          uint32_t time_mask,
                          double median_ns[VBS_HOT_NTYPES],
                          double base_ns[VBS_HOT_NTYPES],
                          int wins[VBS_HOT_NTYPES],
                          bool *correct, char *err, size_t err_cap);

/* Same benchmark, run in a forked child so a crashing candidate cannot
 * take down the caller. Returns 0 on success; -2 when the child died on a
 * signal (candidate crash); -1 on other failures. */
int vbs_loop_bench_in_child(const char *so_path, const char *base_so_path,
                            const vbs_bench_shape_t *shapes, size_t n_shapes,
                            uint32_t time_mask,
                            double median_ns[VBS_HOT_NTYPES],
                            double base_ns[VBS_HOT_NTYPES],
                            int wins[VBS_HOT_NTYPES],
                            bool *correct, char *err, size_t err_cap);

/* --- The loop --- */

typedef struct {
    const char *model_path;
    const uint32_t *prompt;
    size_t n_prompt;
    size_t n_tokens;        /* tokens to generate in the live session */
    int max_iterations;     /* max optimization iterations */
    size_t tokens_per_iter; /* tokens generated between iterations */
    const char *log_path;   /* structured JSONL log (NULL = vbs-loop.jsonl) */
    const char *work_dir;   /* temp dir for candidate .so files */
    const char *repo_dir;   /* repo root holding src/hotkernels/ */
    int poison;             /* fault injection: force a bad swap, expect rollback */
    int quiet;              /* suppress human-readable progress lines */
} vbs_loop_config_t;

typedef struct {
    int iterations_run;
    int accepted;
    int rejected;
    int swaps;
    int rollbacks;
    double best_ns;                 /* best min ns at the hotspot shape */
    char best_variant[64];
    char hotspot_type[16];
    uint32_t *tokens;               /* generated token ids (caller frees) */
    size_t n_tokens_generated;
    double tokens_per_sec;
} vbs_loop_result_t;

/* Run the live session + recursive loop. Returns 0 on a completed run
 * (even when no candidate was accepted — that is an honest outcome, not
 * an error), -1 on setup failure (bad model, no candidates, ...). */
int vbs_loop_run(const vbs_loop_config_t *cfg, vbs_loop_result_t *res);

void vbs_loop_result_free(vbs_loop_result_t *res);

#ifdef __cplusplus
}
#endif

#endif /* VBS_LOOP_H */
