#ifndef VBS_AUTOTUNE_H
#define VBS_AUTOTUNE_H

#include <stddef.h>
#include <stdint.h>

/* Hardware-aware autotune, learned across runs.
 *   1. Detect: cores and RAM from the OS, backend from the single
 *      backend-selection point (first available GPU, else CPU).
 *   2. Choose: threads = cores; window_mib = clamp(RAM/8, 64, 1024)
 *      MiB, so a small machine does not OOM and a big machine does
 *      not under-use RAM.
 *   3. Benchmark: time the tiny transformer block for 100 epochs and
 *      record epochs/sec as this machine's performance score.
 *   4. Learn: save config + fingerprint to the config file (default
 *      verum-autotune.json). Next run, if the fingerprint matches,
 *      reuse the learned config instead of re-tuning; if the
 *      hardware changed, re-tune and update the file.
 * This is performance autotuning for the host, not model training.
 * The config file is one flat key=value line (despite the .json
 * name); see vbs_autotune_load for the exact field order. */
typedef struct {
    char backend[32];
    int threads;
    size_t window_mib;
    int cores;
    uint64_t ram_bytes;
    double score_epochs_per_sec;
    char fingerprint[128];
} vbs_autotune_t;

int vbs_autotune_detect(vbs_autotune_t *cfg);
double vbs_autotune_benchmark(void);
int vbs_autotune_run(const char *config_path, vbs_autotune_t *out);
int vbs_autotune_load(const char *path, vbs_autotune_t *out);

#endif /* VBS_AUTOTUNE_H */
