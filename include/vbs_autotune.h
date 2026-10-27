#ifndef VBS_AUTOTUNE_H
#define VBS_AUTOTUNE_H
#include <stddef.h>
#include <stdint.h>
/* Hardware-aware autotune, self-learning across runs.
 * 1. Detect: OS, CPU cores, RAM, GPU backends (CUDA/HIP/Metal via backend init).
 * 2. Choose: backend = first available GPU, else CPU; threads = cores;
 *    window_mib = clamp(RAM/8, 64, 1024) MiB, so a small machine does not
 *    OOM and a big machine does not under-use RAM.
 * 3. Benchmark: time the tiny transformer block for 100 epochs, record
 *    epochs/sec as the performance score for this machine.
 * 4. Learn: save config + fingerprint to verum-autotune.json. Next run,
 *    if fingerprint matches, reuse the learned config instead of re-tuning.
 *    If hardware changed, re-tune and update the file.
 * This is performance autotuning for the host, not model weight training.
 */
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
#endif
