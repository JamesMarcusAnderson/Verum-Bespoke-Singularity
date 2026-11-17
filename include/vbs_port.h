/* =============================================================================
 * vbs_port.h — the ONE place for platform differences in the Verum engine.
 *
 * Rule: no .c file includes <unistd.h>, <dlfcn.h>, <sys/mman.h>, <fcntl.h>,
 * <strings.h> or <sys/stat.h> directly for a portability reason — it uses
 * the wrappers below instead. The POSIX branches are the exact calls the
 * engine used before, so Linux/macOS behavior is unchanged; the _WIN32
 * branches are the native Windows equivalents (no POSIX emulation layer).
 *
 * Requirements: C11. Windows: MinGW-w64 gcc, or MSVC 2019+ (needs /std:c11
 * and the VS2015+ conformant %zu/snprintf). No other dependencies.
 * POSIX builds must define _GNU_SOURCE (or _POSIX_C_SOURCE >= 200809L) —
 * the Makefile and CMakeLists.txt both do — because clock_gettime,
 * ftruncate, posix_memalign, mmap and friends are POSIX, not C11, and
 * glibc hides their declarations without it.
 * =========================================================================== */
#ifndef VBS_PORT_H
#define VBS_PORT_H

#include <ctype.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <direct.h>    /* _mkdir */
#include <malloc.h>    /* _aligned_malloc / _aligned_free */
#include <sys/stat.h>  /* _stat64 */
#define VBS_DYLIB_EXT ".dll"
#elif defined(__APPLE__)
#define VBS_DYLIB_EXT ".dylib"
#else
#define VBS_DYLIB_EXT ".so"
#endif

#if !defined(_WIN32)
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

/* ---------------------------------------------------------------------------
 * Monotonic clock, seconds. Same call on every OS; the Windows branch uses
 * QueryPerformanceCounter, the POSIX branch clock_gettime(CLOCK_MONOTONIC).
 * ------------------------------------------------------------------------- */
static inline double vbs_now_sec(void) {
#if defined(_WIN32)
    static LARGE_INTEGER freq;
    static int have_freq = 0;
    LARGE_INTEGER now;
    if (!have_freq) {
        QueryPerformanceFrequency(&freq);
        have_freq = 1;
    }
    QueryPerformanceCounter(&now);
    return freq.QuadPart ? (double)now.QuadPart / (double)freq.QuadPart : 0.0;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
#endif
}

/* ---------------------------------------------------------------------------
 * Wall clock, seconds since the Unix epoch. POSIX uses
 * clock_gettime(CLOCK_REALTIME); Windows uses GetSystemTimeAsFileTime.
 * For log timestamps only — never for measuring durations (use
 * vbs_now_sec for that).
 * ------------------------------------------------------------------------- */
static inline double vbs_wall_sec(void) {
#if defined(_WIN32)
    FILETIME ft;
    ULARGE_INTEGER u;
    GetSystemTimeAsFileTime(&ft);
    u.LowPart = ft.dwLowDateTime;
    u.HighPart = ft.dwHighDateTime;
    /* 100-ns intervals since 1601-01-01; the Unix epoch is 11644473600 s
     * after that. */
    return (double)u.QuadPart / 1e7 - 11644473600.0;
#else
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
#endif
}

/* ---------------------------------------------------------------------------
 * Dynamic libraries: dlopen/dlsym/dlclose on POSIX, LoadLibraryA /
 * GetProcAddress / FreeLibrary on Windows. `now` selects RTLD_NOW|RTLD_LOCAL
 * (specializer) vs RTLD_LAZY (GPU backends); Windows loads eagerly either way.
 * ------------------------------------------------------------------------- */
static inline void *vbs_dlopen(const char *path, int now) {
#if defined(_WIN32)
    (void)now;
    return (void *)LoadLibraryA(path);
#else
    return dlopen(path, now ? (RTLD_NOW | RTLD_LOCAL) : RTLD_LAZY);
#endif
}

static inline void *vbs_dlsym(void *handle, const char *name) {
#if defined(_WIN32)
    return (void *)(uintptr_t)GetProcAddress((HMODULE)handle, name);
#else
    return dlsym(handle, name);
#endif
}

static inline void vbs_dlclose(void *handle) {
    if (!handle) return;
#if defined(_WIN32)
    FreeLibrary((HMODULE)handle);
#else
    dlclose(handle);
#endif
}

/* Last dynamic-load error as a string. POSIX returns dlerror();
 * Windows formats GetLastError(). Never returns NULL. */
static inline const char *vbs_dlerror(void) {
#if defined(_WIN32)
    static char buf[512];
    DWORD e = GetLastError();
    if (e == 0) return "unknown error";
    if (FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                       NULL, e, 0, buf, (DWORD)sizeof(buf), NULL) == 0)
        snprintf(buf, sizeof(buf), "error %lu", (unsigned long)e);
    return buf;
#else
    const char *e = dlerror();
    return e ? e : "unknown error";
#endif
}

/* ---------------------------------------------------------------------------
 * Host facts: logical CPU count and total physical RAM. Windows uses
 * GetSystemInfo / GlobalMemoryStatusEx; POSIX uses sysconf. Both fall
 * back to sane defaults (1 core, 1 GiB) instead of failing.
 * ------------------------------------------------------------------------- */
static inline unsigned int vbs_cpu_count(void) {
#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors > 0 ? (unsigned int)si.dwNumberOfProcessors : 1;
#else
    long c = sysconf(_SC_NPROCESSORS_ONLN);
    return c > 0 ? (unsigned int)c : 1;
#endif
}

static inline uint64_t vbs_ram_bytes(void) {
#if defined(_WIN32)
    MEMORYSTATUSEX ms;
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms) && ms.ullTotalPhys > 0)
        return (uint64_t)ms.ullTotalPhys;
    return 1ULL << 30;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0)
        return (uint64_t)pages * (uint64_t)page_size;
    return 1ULL << 30;
#endif
}

/* ---------------------------------------------------------------------------
 * Environment variables: setenv/unsetenv on POSIX, SetEnvironmentVariableA
 * on Windows. Returns 0 on success, non-zero on failure.
 * ------------------------------------------------------------------------- */
static inline int vbs_setenv(const char *name, const char *value) {
#if defined(_WIN32)
    return SetEnvironmentVariableA(name, value) ? 0 : -1;
#else
    return setenv(name, value, 1);
#endif
}

static inline int vbs_unsetenv(const char *name) {
#if defined(_WIN32)
    return SetEnvironmentVariableA(name, NULL) ? 0 : -1;
#else
    return unsetenv(name);
#endif
}

/* ---------------------------------------------------------------------------
 * Small POSIX string helpers, reimplemented in pure C11 so no _GNU_SOURCE /
 * <strings.h> dependency remains. Behavior matches the POSIX originals in
 * the "C" locale (the engine never calls setlocale, so this is exact).
 * ------------------------------------------------------------------------- */
static inline char *vbs_strdup(const char *s) {
    size_t n;
    char *d;
    if (!s) return NULL;
    n = strlen(s) + 1;
    d = (char *)malloc(n);
    if (d) memcpy(d, s, n);
    return d;
}

static inline int vbs_strncasecmp(const char *a, const char *b, size_t n) {
    size_t i;
    for (i = 0; i < n; i++) {
        unsigned char ca = (unsigned char)a[i];
        unsigned char cb = (unsigned char)b[i];
        int d = tolower(ca) - tolower(cb);
        if (d != 0) return d;
        if (ca == '\0') return 0;
    }
    return 0;
}

static inline char *vbs_strtok_r(char *s, const char *delim, char **save) {
    char *p = s ? s : *save;
    char *tok;
    if (!p) return NULL;
    p += strspn(p, delim);
    if (*p == '\0') {
        *save = p;
        return NULL;
    }
    tok = p;
    p += strcspn(p, delim);
    if (*p != '\0') *p++ = '\0';
    *save = p;
    return tok;
}

/* ---------------------------------------------------------------------------
 * Filesystem: mkdir / stat. Windows takes path separators '/' as well as
 * '\\', so no separator translation is needed anywhere in the engine.
 * ------------------------------------------------------------------------- */
static inline int vbs_mkdir(const char *path) {
#if defined(_WIN32)
    return _mkdir(path);
#else
    return mkdir(path, 0755);
#endif
}

/* 1 if the path exists (file or directory), 0 otherwise. */
static inline int vbs_stat_exists(const char *path) {
#if defined(_WIN32)
    struct _stat64 st;
    return _stat64(path, &st) == 0;
#else
    struct stat st;
    return stat(path, &st) == 0;
#endif
}

/* 0 and *out = size on success, -1 if the path cannot be statted. */
static inline int vbs_stat_size(const char *path, size_t *out) {
#if defined(_WIN32)
    struct _stat64 st;
    if (_stat64(path, &st) != 0) return -1;
    *out = (size_t)st.st_size;
    return 0;
#else
    struct stat st;
    if (stat(path, &st) != 0) return -1;
    *out = (size_t)st.st_size;
    return 0;
#endif
}

/* ---------------------------------------------------------------------------
 * Process id (seed mixing only — never used for signalling or control).
 * ------------------------------------------------------------------------- */
static inline unsigned long vbs_getpid(void) {
#if defined(_WIN32)
    return (unsigned long)GetCurrentProcessId();
#else
    return (unsigned long)getpid();
#endif
}

/* ---------------------------------------------------------------------------
 * Endianness. GGUF is a little-endian format and the parsers read multi-byte
 * integers with memcpy, so a big-endian host is refused with a clear error
 * instead of silently misreading every model. All supported hosts (x86-64
 * and ARM Windows/Linux/macOS) are little-endian.
 * ------------------------------------------------------------------------- */
static inline int vbs_is_little_endian(void) {
    const uint32_t t = 1;
    return *(const unsigned char *)&t == 1;
}

/* ---------------------------------------------------------------------------
 * File mappings. One struct owns the platform handles on both sides so the
 * callers (GGUF reader, NVMe swap tier) hold no fd/HANDLE themselves.
 * POSIX: open + mmap. Windows: CreateFileA + CreateFileMappingA +
 * MapViewOfFile. The mapped view is `ptr`; it stays valid until
 * vbs_mmap_close. Open-failure codes let callers keep their exact
 * pre-existing error messages.
 * ------------------------------------------------------------------------- */
#define VBS_MMAP_OK 0
#define VBS_MMAP_ERR_OPEN 1 /* file could not be opened */
#define VBS_MMAP_ERR_SIZE 2 /* file size unusable (empty or unreadable) */
#define VBS_MMAP_ERR_MAP 3  /* mapping itself failed */

typedef struct {
    void *ptr;     /* mapped view, NULL if unmapped */
    uint64_t size; /* mapped length in bytes */
#if defined(_WIN32)
    void *win_file;    /* HANDLE from CreateFileA */
    void *win_mapping; /* HANDLE from CreateFileMappingA */
#else
    int fd; /* -1 if closed */
#endif
} vbs_mmap_t;

static inline void vbs_mmap_init(vbs_mmap_t *m) {
    m->ptr = NULL;
    m->size = 0;
#if defined(_WIN32)
    m->win_file = NULL;
    m->win_mapping = NULL;
#else
    m->fd = -1;
#endif
}

/* Map a whole file read-only. */
static inline int vbs_mmap_open_ro(const char *path, vbs_mmap_t *m) {
    vbs_mmap_init(m);
#if defined(_WIN32)
    {
        HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        HANDLE map;
        LARGE_INTEGER sz;
        void *view;
        if (h == INVALID_HANDLE_VALUE) return VBS_MMAP_ERR_OPEN;
        if (!GetFileSizeEx(h, &sz) || sz.QuadPart <= 0) {
            CloseHandle(h);
            return VBS_MMAP_ERR_SIZE;
        }
        map = CreateFileMappingA(h, NULL, PAGE_READONLY, 0, 0, NULL);
        if (!map) {
            CloseHandle(h);
            return VBS_MMAP_ERR_MAP;
        }
        view = MapViewOfFile(map, FILE_MAP_READ, 0, 0, 0);
        if (!view) {
            CloseHandle(map);
            CloseHandle(h);
            return VBS_MMAP_ERR_MAP;
        }
        m->win_file = (void *)h;
        m->win_mapping = (void *)map;
        m->ptr = view;
        m->size = (uint64_t)sz.QuadPart;
        return VBS_MMAP_OK;
    }
#else
    {
        int fd = open(path, O_RDONLY);
        struct stat st;
        void *p;
        if (fd < 0) return VBS_MMAP_ERR_OPEN;
        if (fstat(fd, &st) != 0 || st.st_size <= 0) {
            close(fd);
            return VBS_MMAP_ERR_SIZE;
        }
        p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (p == MAP_FAILED) {
            close(fd);
            return VBS_MMAP_ERR_MAP;
        }
        m->fd = fd;
        m->ptr = p;
        m->size = (uint64_t)st.st_size;
        return VBS_MMAP_OK;
    }
#endif
}

/* Create (or replace) a file of `size` bytes and map it read-write/shared. */
static inline int vbs_mmap_create_rw(const char *path, uint64_t size, vbs_mmap_t *m) {
    vbs_mmap_init(m);
    if (size == 0) return VBS_MMAP_ERR_SIZE;
#if defined(_WIN32)
    {
        HANDLE h = CreateFileA(path, GENERIC_READ | GENERIC_WRITE, 0, NULL,
                               CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
        HANDLE map;
        LARGE_INTEGER li;
        void *view;
        if (h == INVALID_HANDLE_VALUE) return VBS_MMAP_ERR_OPEN;
        li.QuadPart = (LONGLONG)size;
        if (!SetFilePointerEx(h, li, NULL, FILE_BEGIN) || !SetEndOfFile(h)) {
            CloseHandle(h);
            return VBS_MMAP_ERR_SIZE;
        }
        map = CreateFileMappingA(h, NULL, PAGE_READWRITE,
                                 (DWORD)(size >> 32),
                                 (DWORD)(size & 0xFFFFFFFFu), NULL);
        if (!map) {
            CloseHandle(h);
            return VBS_MMAP_ERR_MAP;
        }
        view = MapViewOfFile(map, FILE_MAP_WRITE, 0, 0, 0);
        if (!view) {
            CloseHandle(map);
            CloseHandle(h);
            return VBS_MMAP_ERR_MAP;
        }
        m->win_file = (void *)h;
        m->win_mapping = (void *)map;
        m->ptr = view;
        m->size = size;
        return VBS_MMAP_OK;
    }
#else
    {
        int fd = open(path, O_RDWR | O_CREAT, 0600);
        void *p;
        if (fd < 0) return VBS_MMAP_ERR_OPEN;
        if (ftruncate(fd, (off_t)size) != 0) {
            close(fd);
            return VBS_MMAP_ERR_SIZE;
        }
        p = mmap(NULL, (size_t)size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) {
            close(fd);
            return VBS_MMAP_ERR_MAP;
        }
        m->fd = fd;
        m->ptr = p;
        m->size = size;
        return VBS_MMAP_OK;
    }
#endif
}

static inline void vbs_mmap_close(vbs_mmap_t *m) {
    if (!m) return;
#if defined(_WIN32)
    if (m->ptr) UnmapViewOfFile(m->ptr);
    if (m->win_mapping) CloseHandle((HANDLE)m->win_mapping);
    if (m->win_file) CloseHandle((HANDLE)m->win_file);
#else
    /* The fd is closed only with a live view: a calloc-zeroed struct has
     * fd == 0 (stdin), which must never be closed. Every successful open
     * sets ptr, so this loses no real close. */
    if (m->ptr) {
        munmap(m->ptr, (size_t)m->size);
        if (m->fd >= 0) close(m->fd);
    }
#endif
    vbs_mmap_init(m);
}

/* ---------------------------------------------------------------------------
 * Aligned allocation. POSIX: posix_memalign (freed with free).
 * Windows: _aligned_malloc (must be freed with _aligned_free — never free).
 * ------------------------------------------------------------------------- */
static inline int vbs_aligned_alloc(void **out, size_t alignment, size_t size) {
#if defined(_WIN32)
    *out = _aligned_malloc(size, alignment);
    return *out ? 0 : -1;
#else
    return posix_memalign(out, alignment, size);
#endif
}

static inline void vbs_aligned_free(void *p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}

/* ---------------------------------------------------------------------------
 * Temp directory. POSIX: $TMPDIR, else /tmp. Windows: GetTempPathA
 * ($TEMP/$TMP, else the Windows directory). Trailing separators stripped.
 * ------------------------------------------------------------------------- */
static inline const char *vbs_temp_dir(char *buf, size_t cap) {
#if defined(_WIN32)
    DWORD n = GetTempPathA((DWORD)cap, buf);
    size_t len;
    if (n == 0 || n >= cap) {
        snprintf(buf, cap, ".");
        return buf;
    }
    len = strlen(buf);
    while (len > 1 && (buf[len - 1] == '\\' || buf[len - 1] == '/'))
        buf[--len] = '\0';
    return buf;
#else
    const char *t = getenv("TMPDIR");
    if (!t || !*t) t = "/tmp";
    snprintf(buf, cap, "%s", t);
    return buf;
#endif
}

/* Join dir + suffix into buf, truncating the dir (never the suffix) so the
 * result always ends with the full suffix. Empty string if cap is too small
 * to hold even the suffix. */
static inline void vbs_join_suffix(char *buf, size_t cap, const char *dir, const char *suffix) {
    size_t sl;
    if (cap == 0) return;
    sl = strlen(suffix);
    if (cap <= sl + 1) {
        buf[0] = '\0';
        return;
    }
    snprintf(buf, cap, "%.*s%s", (int)(cap - sl - 1), dir, suffix);
}

/* Default on-disk locations (overridable with VBS_CACHE_DIR / VBS_SWAP_PATH).
 * Forward slashes work in every Windows file API, so one spelling serves. */
static inline void vbs_default_cache_dir(char *buf, size_t cap) {
    char tmp[512];
    vbs_temp_dir(tmp, sizeof tmp);
    vbs_join_suffix(buf, cap, tmp, "/verum_cache");
}

static inline void vbs_default_swap_path(char *buf, size_t cap) {
    char tmp[512];
    vbs_temp_dir(tmp, sizeof tmp);
    vbs_join_suffix(buf, cap, tmp, "/verum_swap.bin");
}

#if defined(_WIN32)
/* Full path of the running exe. The specializer links its DLL against it so
 * the bridge symbols the generated code calls resolve at load time. */
static inline int vbs_exe_path(char *buf, size_t cap) {
    DWORD n = GetModuleFileNameA(NULL, buf, (DWORD)cap);
    return (n > 0 && n < cap) ? 0 : -1;
}
#endif

#endif /* VBS_PORT_H */
