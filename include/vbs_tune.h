/* vbs_tune.h — persistent per-machine autotune profiles that travel with
 * the binary.
 *
 * Idea: the best kernel variant for a hot path depends on the machine
 * (CPU model, SIMD level, cache sizes). Measuring it costs a benchmark
 * sweep every run. Instead, --hotbench measures once and writes a small
 * JSON profile; tools/vbs_tune_embed.sh stores that profile inside the
 * binary itself as an ELF section (.vbs_tune) with objcopy. The binary
 * then carries its tune wherever it is copied — no sidecar file, no
 * config directory, no network. At startup the engine reads its own
 * section, checks the machine fingerprint, and applies the profile only
 * when it matches this machine; a profile from another machine is
 * ignored, never trusted.
 *
 * The profile is advisory: it orders the loop's candidate list (best-known
 * variant first) and seeds the loop's best-time baseline. Every candidate
 * is still benchmark-verified before it is accepted, so a stale or wrong
 * profile can only cost a few extra benchmark runs, never a wrong kernel.
 */
#ifndef VBS_TUNE_H
#define VBS_TUNE_H

#include "vbs_hotpath.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Stable fingerprint of this machine's performance-relevant traits:
 * CPU model, SIMD levels, core count, L3 size, machine-id. Two machines
 * with the same fingerprint are expected to rank kernels the same way. */
bool vbs_tune_fingerprint(char *out, size_t cap);

typedef struct {
    char fingerprint[192];
    /* Best-known kernel variant name per hot type, "" if unknown. */
    char best[VBS_HOT_NTYPES][64];
    /* Best-known median time (ns) per matvec call at the benchmark shape. */
    double best_ns[VBS_HOT_NTYPES];
    uint64_t generated_unix;
} vbs_tune_profile_t;

void vbs_tune_profile_init(vbs_tune_profile_t *p);

/* JSON serialization (the format objcopy embeds). */
bool vbs_tune_write_json(const vbs_tune_profile_t *p, const char *path);
bool vbs_tune_read_json(vbs_tune_profile_t *p, const char *path);
bool vbs_tune_parse_json(vbs_tune_profile_t *p, const char *json, size_t len);

/* Read the .vbs_tune section embedded in the running binary (/proc/self/exe).
 * Returns false with a message in err when there is no embedded profile. */
bool vbs_tune_load_embedded(vbs_tune_profile_t *p, char *err, size_t err_cap);

/* True when the profile's fingerprint matches this machine. */
bool vbs_tune_matches_machine(const vbs_tune_profile_t *p);

#ifdef __cplusplus
}
#endif

#endif /* VBS_TUNE_H */
