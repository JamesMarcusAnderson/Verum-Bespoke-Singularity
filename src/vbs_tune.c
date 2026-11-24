/* vbs_tune.c — per-machine autotune profiles embedded in the binary. */
#include "vbs_tune.h"
#include "vbs_sha256.h"
#include "vbs_port.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__linux__)
#include <elf.h>
#endif

void vbs_tune_profile_init(vbs_tune_profile_t *p) {
    memset(p, 0, sizeof(*p));
    vbs_tune_fingerprint(p->fingerprint, sizeof(p->fingerprint));
    p->generated_unix = (uint64_t)time(NULL);
}

/* Append a field to a trait string; a field that does not fit
 * whole is dropped, never truncated. */
static void append_field(char *buf, size_t cap, size_t *len, const char *s) {
    size_t n = strlen(s);
    if (*len + n + 1 >= cap) return;
    memcpy(buf + *len, s, n);
    *len += n;
    buf[*len] = '\0';
}

/* Trim trailing newlines and spaces from a length. */
static size_t trim_right_len(const char *s, size_t n) {
    while (n && (s[n - 1] == '\n' || s[n - 1] == ' ')) n--;
    return n;
}

/* Read the first line of a small sysfs/proc file, trailing-trimmed. */
static bool read_trimmed_line(const char *path, char *out, size_t cap) {
    FILE *f = fopen(path, "r");
    if (!f) return false;
    bool ok = false;
    if (fgets(out, (int)cap, f)) {
        size_t n = trim_right_len(out, strlen(out));
        out[n] = '\0';
        ok = true;
    }
    fclose(f);
    return ok;
}

/* Read the first "model name" line and the flags line from /proc/cpuinfo. */
static void cpuinfo_fields(char *model, size_t model_cap, char *flags, size_t flags_cap) {
    model[0] = '\0';
    flags[0] = '\0';
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return;
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        const char *v;
        if (strncmp(line, "model name", 10) == 0 && !model[0]) {
            v = strchr(line, ':');
            if (v) {
                v++;
                while (*v == ' ' || *v == '\t') v++;
                size_t n = trim_right_len(v, strlen(v));
                if (n >= model_cap) n = model_cap - 1;
                memcpy(model, v, n);
                model[n] = '\0';
            }
        } else if (strncmp(line, "flags", 5) == 0 && !flags[0]) {
            v = strchr(line, ':');
            if (v) {
                v++;
                size_t n = strlen(v);
                if (n >= flags_cap) n = flags_cap - 1;
                memcpy(flags, v, n);
                flags[n] = '\0';
            }
        }
        if (model[0] && flags[0]) break;
    }
    fclose(f);
}

/* Whole-word flag search: the name must be bounded by spaces (or
 * the string ends) on both sides. */
static bool has_flag(const char *flags, const char *name) {
    size_t n = strlen(name);
    const char *p = flags;
    while ((p = strstr(p, name)) != NULL) {
        bool left_ok = (p == flags) || p[-1] == ' ';
        bool right_ok = (p[n] == ' ' || p[n] == '\n' || p[n] == '\0');
        if (left_ok && right_ok) return true;
        p += n;
    }
    return false;
}

bool vbs_tune_fingerprint(char *out, size_t cap) {
    char model[256], flags[8192];
    cpuinfo_fields(model, sizeof(model), flags, sizeof(flags));
    if (!model[0]) snprintf(model, sizeof(model), "unknown-cpu");

    /* Normalize: keep only the tokens that affect kernel ranking. */
    char traits[1024];
    size_t len = 0;
    traits[0] = '\0';
    append_field(traits, sizeof(traits), &len, model);
    append_field(traits, sizeof(traits), &len, "|");
    static const char *k_rank_flags[] = { "avx512f", "avx2", "sse4_2", "fma" };
    for (size_t i = 0; i < sizeof(k_rank_flags) / sizeof(k_rank_flags[0]); i++) {
        if (has_flag(flags, k_rank_flags[i])) {
            char tagged[32];
            snprintf(tagged, sizeof(tagged), "%s,", k_rank_flags[i]);
            append_field(traits, sizeof(traits), &len, tagged);
        }
    }
    char tmp[64];
    long cores = (long)vbs_cpu_count();
    snprintf(tmp, sizeof(tmp), "|cores=%ld", cores > 0 ? cores : 1);
    append_field(traits, sizeof(traits), &len, tmp);
    char sz[32] = "";
    if (read_trimmed_line("/sys/devices/system/cpu/cpu0/cache/index3/size", sz, sizeof(sz))) {
        snprintf(tmp, sizeof(tmp), "|l3=%s", sz);
        append_field(traits, sizeof(traits), &len, tmp);
    }
    char mid[64] = "";
    if (read_trimmed_line("/etc/machine-id", mid, sizeof(mid))) {
        snprintf(tmp, sizeof(tmp), "|m=%.12s", mid);
        append_field(traits, sizeof(traits), &len, tmp);
    }

    /* Fingerprint = sha256 of the trait string, first 16 hex chars. */
    vbs_sha256_ctx_t ctx;
    vbs_sha256_init(&ctx);
    vbs_sha256_update(&ctx, traits, strlen(traits));
    uint8_t digest[32];
    vbs_sha256_final(&ctx, digest);
    char hex[17];
    for (int i = 0; i < 8; i++) sprintf(hex + i * 2, "%02x", digest[i]);
    hex[16] = '\0';
    snprintf(out, cap, "%s", hex);
    return true;
}

/* --- JSON (machine-written format; parser is intentionally strict) --- */

bool vbs_tune_write_json(const vbs_tune_profile_t *p, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) return false;
    fprintf(f, "{\n  \"format\": \"vbs-tune-profile-v1\",\n");
    fprintf(f, "  \"fingerprint\": \"%s\",\n", p->fingerprint);
    fprintf(f, "  \"generated_unix\": %llu,\n", (unsigned long long)p->generated_unix);
    fprintf(f, "  \"best\": {\n");
    for (int i = 0; i < VBS_HOT_NTYPES; i++) {
        fprintf(f, "    \"%s\": {\"variant\": \"%s\", \"ns\": %.1f}%s\n",
                vbs_hot_type_name(i), p->best[i], p->best_ns[i],
                i + 1 < VBS_HOT_NTYPES ? "," : "");
    }
    fprintf(f, "  }\n}\n");
    fclose(f);
    return true;
}

static bool json_string_field(const char *json, const char *key, char *out, size_t cap) {
    char pat[96];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) return false;
    p = strchr(p + strlen(pat), '"');
    if (!p) return false;
    p++;
    const char *end = strchr(p, '"');
    if (!end) return false;
    size_t n = (size_t)(end - p);
    if (n >= cap) n = cap - 1;
    memcpy(out, p, n);
    out[n] = '\0';
    return true;
}

static bool json_double_field(const char *json, const char *key, double *out) {
    char pat[96];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p) return false;
    p = strchr(p + strlen(pat), ':');
    if (!p) return false;
    *out = strtod(p + 1, NULL);
    return true;
}

bool vbs_tune_parse_json(vbs_tune_profile_t *p, const char *json, size_t len) {
    (void)len;
    memset(p, 0, sizeof(*p));
    char fmt[64];
    if (!json_string_field(json, "format", fmt, sizeof(fmt)) ||
        strcmp(fmt, "vbs-tune-profile-v1") != 0) {
        return false;
    }
    if (!json_string_field(json, "fingerprint", p->fingerprint, sizeof(p->fingerprint))) {
        return false;
    }
    double gen = 0;
    if (json_double_field(json, "generated_unix", &gen)) p->generated_unix = (uint64_t)gen;
    /* Per-type entries: find each type's object, then its variant/ns. */
    for (int i = 0; i < VBS_HOT_NTYPES; i++) {
        char pat[96];
        snprintf(pat, sizeof(pat), "\"%s\"", vbs_hot_type_name(i));
        const char *sec = strstr(json, pat);
        if (!sec) continue;
        const char *obj_end = strchr(sec, '}');
        if (!obj_end) continue;
        char tmp[256];
        size_t n = (size_t)(obj_end - sec);
        if (n >= sizeof(tmp)) n = sizeof(tmp) - 1;
        memcpy(tmp, sec, n);
        tmp[n] = '\0';
        json_string_field(tmp, "variant", p->best[i], sizeof(p->best[i]));
        double ns = 0;
        if (json_double_field(tmp, "ns", &ns)) p->best_ns[i] = ns;
    }
    return true;
}

bool vbs_tune_read_json(vbs_tune_profile_t *p, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    char buf[4096];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = '\0';
    return vbs_tune_parse_json(p, buf, n);
}

/* --- Embedded section read-back (Linux ELF64) --- */

bool vbs_tune_load_embedded(vbs_tune_profile_t *p, char *err, size_t err_cap) {
#if !defined(__linux__)
    snprintf(err, err_cap, "embedded tune profiles are Linux-only in this build");
    return false;
#else
    FILE *f = fopen("/proc/self/exe", "rb");
    if (!f) {
        snprintf(err, err_cap, "cannot open /proc/self/exe");
        return false;
    }
    Elf64_Shdr *sh = NULL;
    char *strtab = NULL;
    char *json = NULL;
    size_t json_len = 0;
    bool ok = false;

    Elf64_Ehdr eh;
    if (fread(&eh, 1, sizeof(eh), f) != sizeof(eh) ||
        memcmp(eh.e_ident, ELFMAG, SELFMAG) != 0 ||
        eh.e_ident[EI_CLASS] != ELFCLASS64 ||
        eh.e_shoff == 0 || eh.e_shentsize != sizeof(Elf64_Shdr) ||
        eh.e_shnum == 0 || eh.e_shstrndx >= eh.e_shnum) {
        snprintf(err, err_cap, "not a readable ELF64 binary");
        goto done;
    }
    sh = (Elf64_Shdr *)calloc(eh.e_shnum, sizeof(Elf64_Shdr));
    if (!sh) {
        snprintf(err, err_cap, "out of memory");
        goto done;
    }
    fseek(f, (long)eh.e_shoff, SEEK_SET);
    if (fread(sh, sizeof(Elf64_Shdr), eh.e_shnum, f) != eh.e_shnum) {
        snprintf(err, err_cap, "cannot read section headers");
        goto done;
    }
    /* Section-name string table. */
    {
        Elf64_Shdr *strsh = &sh[eh.e_shstrndx];
        strtab = (char *)malloc(strsh->sh_size ? (size_t)strsh->sh_size : 1);
        if (!strtab) {
            snprintf(err, err_cap, "out of memory");
            goto done;
        }
        fseek(f, (long)strsh->sh_offset, SEEK_SET);
        if (fread(strtab, 1, (size_t)strsh->sh_size, f) != (size_t)strsh->sh_size) {
            snprintf(err, err_cap, "cannot read section names");
            goto done;
        }
        for (size_t i = 0; i < eh.e_shnum; i++) {
            if (sh[i].sh_name >= strsh->sh_size) continue;
            if (strcmp(strtab + sh[i].sh_name, ".vbs_tune") != 0) continue;
            json_len = (size_t)sh[i].sh_size;
            json = (char *)malloc(json_len + 1);
            if (!json) break;
            fseek(f, (long)sh[i].sh_offset, SEEK_SET);
            if (fread(json, 1, json_len, f) != json_len) {
                free(json);
                json = NULL;
                break;
            }
            json[json_len] = '\0';
            ok = true;
            break;
        }
    }
    if (!ok || !json) {
        snprintf(err, err_cap, "no embedded .vbs_tune section in this binary");
        goto done;
    }
    if (!vbs_tune_parse_json(p, json, json_len)) {
        snprintf(err, err_cap, "embedded .vbs_tune section is not a valid profile");
        ok = false;
        goto done;
    }
    ok = true;
done:
    free(json);
    free(strtab);
    free(sh);
    fclose(f);
    return ok;
#endif
}

bool vbs_tune_matches_machine(const vbs_tune_profile_t *p) {
    char fp[192];
    if (!vbs_tune_fingerprint(fp, sizeof(fp))) return false;
    return strcmp(fp, p->fingerprint) == 0;
}
