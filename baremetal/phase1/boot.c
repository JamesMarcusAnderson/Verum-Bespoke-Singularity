#include <stddef.h>  /* NULL, freestanding-safe */

/*
 * definitely-not-skynet — Phase 1
 * Bare-metal CPU LLM inference. No OS.
 *
 * Build target: x86_64-elf (no stdlib, no libc)
 * Toolchain:    clang --target=x86_64-unknown-none-elf
 *               or    x86_64-elf-gcc
 *
 * Entry: UEFI calls efi_main() from a PE32+ binary.
 * After ExitBootServices() the entire OS is this file.
 *
 * Phase 1 goal: load a GGUF model from a UEFI Simple File System
 *               partition, run CPU inference, print tokens to
 *               the UEFI serial console, never return.
 */

/* ------------------------------------------------------------------ */
/* UEFI types — no efi.h needed, we define what we use                */
/* ------------------------------------------------------------------ */

typedef unsigned char      UINT8;
typedef unsigned short     UINT16;
typedef unsigned int       UINT32;
typedef unsigned long long UINT64;
typedef long long          INT64;
typedef unsigned long long UINTN;
typedef long long          INTN;
typedef void              *EFI_HANDLE;
typedef void              *EFI_EVENT;
typedef UINT64             EFI_STATUS;
typedef UINT16             CHAR16;
typedef UINT8              BOOLEAN;

#define EFI_SUCCESS           0ULL
#define EFI_ERROR(x)          ((x) & (1ULL << 63))
#define EFI_INVALID_PARAMETER (0x8000000000000002ULL)
#define EFI_NOT_FOUND         (0x800000000000000EULL)
#define EFI_BUFFER_TOO_SMALL  (0x8000000000000005ULL)
#define EFI_OUT_OF_RESOURCES  (0x8000000000000009ULL)

typedef struct {
    UINT64 Signature;
    UINT32 Revision;
    UINT32 HeaderSize;
    UINT32 CRC32;
    UINT32 Reserved;
} EFI_TABLE_HEADER;

/* Simple Text Output */
typedef struct _EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL {
    void *Reset;
    EFI_STATUS (*OutputString)(struct _EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL *,
                               CHAR16 *);
    void *TestString;
    void *QueryMode;
    void *SetMode;
    void *SetAttribute;
    void *ClearScreen;
    void *SetCursorPosition;
    void *EnableCursor;
    void *Mode;
} EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL;

/* Memory types */
typedef enum {
    EfiReservedMemoryType,
    EfiLoaderCode,
    EfiLoaderData,
    EfiBootServicesCode,
    EfiBootServicesData,
    EfiRuntimeServicesCode,
    EfiRuntimeServicesData,
    EfiConventionalMemory,
    EfiUnusableMemory,
    EfiACPIReclaimMemory,
    EfiACPIMemoryNVS,
    EfiMemoryMappedIO,
    EfiMemoryMappedIOPortSpace,
    EfiPalCode,
    EfiMaxMemoryType
} EFI_MEMORY_TYPE;

typedef struct {
    UINT32 Type;
    UINT64 PhysicalStart;
    UINT64 VirtualStart;
    UINT64 NumberOfPages;
    UINT64 Attribute;
} EFI_MEMORY_DESCRIPTOR;

/* Boot Services (subset we need) */
typedef EFI_STATUS (*EFI_ALLOCATE_PAGES)(UINT32, UINT32, UINTN, UINT64 *);
typedef EFI_STATUS (*EFI_FREE_PAGES)(UINT64, UINTN);
typedef EFI_STATUS (*EFI_GET_MEMORY_MAP)(UINTN *, EFI_MEMORY_DESCRIPTOR *,
                                          UINTN *, UINTN *, UINT32 *);
typedef EFI_STATUS (*EFI_EXIT_BOOT_SERVICES)(EFI_HANDLE, UINTN);
typedef EFI_STATUS (*EFI_LOCATE_PROTOCOL)(void *, void *, void **);
typedef EFI_STATUS (*EFI_HANDLE_PROTOCOL)(EFI_HANDLE, void *, void **);
typedef EFI_STATUS (*EFI_LOCATE_HANDLE_BUFFER)(UINT32, void *, void *,
                                                UINTN *, EFI_HANDLE **);
typedef void       (*EFI_STALL)(UINTN);

typedef struct {
    EFI_TABLE_HEADER        Hdr;
    void                   *RaiseTPL;
    void                   *RestoreTPL;
    EFI_ALLOCATE_PAGES      AllocatePages;
    EFI_FREE_PAGES          FreePages;
    EFI_GET_MEMORY_MAP      GetMemoryMap;
    void                   *AllocatePool;
    void                   *FreePool;
    void                   *CreateEvent;
    void                   *SetTimer;
    void                   *WaitForEvent;
    void                   *SignalEvent;
    void                   *CloseEvent;
    void                   *CheckEvent;
    void                   *InstallProtocolInterface;
    void                   *ReinstallProtocolInterface;
    void                   *UninstallProtocolInterface;
    EFI_HANDLE_PROTOCOL     HandleProtocol;
    void                   *Reserved;
    void                   *RegisterProtocolNotify;
    EFI_LOCATE_HANDLE_BUFFER LocateHandleBuffer;
    EFI_LOCATE_PROTOCOL     LocateProtocol;
    void                   *InstallMultipleProtocolInterfaces;
    void                   *UninstallMultipleProtocolInterfaces;
    void                   *CalculateCrc32;
    void                   *CopyMem;
    void                   *SetMem;
    void                   *CreateEventEx;
    EFI_EXIT_BOOT_SERVICES  ExitBootServices;
    EFI_STALL               Stall;
} EFI_BOOT_SERVICES;

typedef struct {
    EFI_TABLE_HEADER            Hdr;
    CHAR16                     *FirmwareVendor;
    UINT32                      FirmwareRevision;
    EFI_HANDLE                  ConsoleInHandle;
    void                       *ConIn;
    EFI_HANDLE                  ConsoleOutHandle;
    EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL *ConOut;
    EFI_HANDLE                  StandardErrorHandle;
    EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL *StdErr;
    void                       *RuntimeServices;
    EFI_BOOT_SERVICES          *BootServices;
    UINTN                       NumberOfTableEntries;
    void                       *ConfigurationTable;
} EFI_SYSTEM_TABLE;

/* Simple File System */
typedef struct _EFI_FILE_PROTOCOL {
    UINT64 Revision;
    EFI_STATUS (*Open)(struct _EFI_FILE_PROTOCOL *, struct _EFI_FILE_PROTOCOL **,
                       CHAR16 *, UINT64, UINT64);
    EFI_STATUS (*Close)(struct _EFI_FILE_PROTOCOL *);
    EFI_STATUS (*Delete)(struct _EFI_FILE_PROTOCOL *);
    EFI_STATUS (*Read)(struct _EFI_FILE_PROTOCOL *, UINTN *, void *);
    EFI_STATUS (*Write)(struct _EFI_FILE_PROTOCOL *, UINTN *, void *);
    EFI_STATUS (*GetPosition)(struct _EFI_FILE_PROTOCOL *, UINT64 *);
    EFI_STATUS (*SetPosition)(struct _EFI_FILE_PROTOCOL *, UINT64);
    EFI_STATUS (*GetInfo)(struct _EFI_FILE_PROTOCOL *, void *, UINTN *, void *);
    EFI_STATUS (*SetInfo)(struct _EFI_FILE_PROTOCOL *, void *, UINTN, void *);
    EFI_STATUS (*Flush)(struct _EFI_FILE_PROTOCOL *);
} EFI_FILE_PROTOCOL;

typedef struct {
    UINT64 Revision;
    EFI_STATUS (*OpenVolume)(void *, EFI_FILE_PROTOCOL **);
} EFI_SIMPLE_FILE_SYSTEM_PROTOCOL;

/* GUIDs */
static UINT8 g_sfsp_guid[16] = {
    0x64, 0x21, 0x68, 0x96, 0x52, 0x6c, 0xd3, 0x11,
    0xaa, 0x95, 0x00, 0x80, 0xc7, 0x3c, 0x88, 0x81
};

/* ------------------------------------------------------------------ */
/* Global UEFI pointers (valid before ExitBootServices)               */
/* ------------------------------------------------------------------ */

static EFI_SYSTEM_TABLE  *g_st  = NULL;
static EFI_BOOT_SERVICES *g_bs  = NULL;
static EFI_HANDLE         g_img = NULL;

/* ------------------------------------------------------------------ */
/* Minimal console output (no printf — encode manually)               */
/* ------------------------------------------------------------------ */

static void puts16(const CHAR16 *s) {
    if (!g_st || !g_st->ConOut) return;
    g_st->ConOut->OutputString(g_st->ConOut, (CHAR16 *)s);
}

/* Convert ASCII string to UCS-2 on stack and print */
static void puts_ascii(const char *s) {
    CHAR16 buf[256];
    int i = 0;
    while (*s && i < 255) { buf[i++] = (CHAR16)*s++; }
    buf[i] = 0;
    puts16(buf);
}

/* Print uint64 in hex */
static void put_hex64(UINT64 v) {
    static const char hex[] = "0123456789ABCDEF";
    char s[19];
    s[0] = '0'; s[1] = 'x';
    for (int i = 15; i >= 2; i--) { s[i] = hex[v & 0xF]; v >>= 4; }
    s[17] = 0;
    puts_ascii(s);
}

static void put_newline(void) { puts16((CHAR16[]){'\r','\n',0}); }

/* ------------------------------------------------------------------ */
/* GGUF magic / header                                                 */
/* ------------------------------------------------------------------ */

#define GGUF_MAGIC 0x46554747UL  /* 'GGUF' */

typedef struct __attribute__((packed)) {
    UINT32 magic;
    UINT32 version;
    UINT64 tensor_count;
    UINT64 metadata_kv_count;
} gguf_header_t;

/* ------------------------------------------------------------------ */
/* Minimal bump allocator — uses EFI-allocated pages before EBS       */
/* After EBS: uses a static arena                                      */
/* ------------------------------------------------------------------ */

static UINT8  g_arena[32 * 1024 * 1024];  /* 32 MB static arena */
static UINTN  g_arena_pos = 0;

static void *arena_alloc(UINTN sz) {
    sz = (sz + 15) & ~15ULL; /* 16-byte align */
    if (g_arena_pos + sz > sizeof(g_arena)) return NULL;
    void *p = g_arena + g_arena_pos;
    g_arena_pos += sz;
    return p;
}

/* memset / memcpy — no libc */
static void *k_memset(void *dst, int c, UINTN n) {
    UINT8 *d = dst;
    while (n--) *d++ = (UINT8)c;
    return dst;
}
static void *k_memcpy(void *dst, const void *src, UINTN n) {
    UINT8 *d = dst; const UINT8 *s = src;
    while (n--) *d++ = *s++;
    return dst;
}
static UINTN k_strlen(const char *s) {
    UINTN n = 0; while (*s++) n++; return n;
}

/* ------------------------------------------------------------------ */
/* GGUF loader — bare metal version                                    */
/* Reads directly from file buffer into arena                          */
/* ------------------------------------------------------------------ */

typedef struct {
    void   *data;
    UINT64  size;
    BOOLEAN valid;
} vbs_model_t;

static vbs_model_t g_model;

static BOOLEAN gguf_load(void *buf, UINT64 sz) {
    if (sz < sizeof(gguf_header_t)) return 0;
    gguf_header_t *hdr = (gguf_header_t *)buf;
    if (hdr->magic != GGUF_MAGIC) {
        puts_ascii("GGUF: bad magic"); put_newline();
        return 0;
    }
    puts_ascii("GGUF version: ");
    put_hex64(hdr->version); put_newline();
    puts_ascii("tensors: ");
    put_hex64(hdr->tensor_count); put_newline();

    g_model.data  = buf;
    g_model.size  = sz;
    g_model.valid = 1;
    return 1;
}

/* ------------------------------------------------------------------ */
/* CPU inference — scalar fallback, no SIMD required                  */
/* Runs after ExitBootServices when no OS exists.                     */
/* ------------------------------------------------------------------ */

/* RMS norm: x = x * (1/sqrt(mean(x^2) + eps)) * w */
static void cpu_rms_norm(float *x, const float *w, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    /* Newton-Raphson reciprocal sqrt (no libm) */
    float mean = ss / (float)n + 1e-5f;
    /* Initial estimate via bit hack */
    float y = mean;
    int yi; k_memcpy(&yi, &y, 4);
    yi = 0x5F3759DF - (yi >> 1);
    k_memcpy(&y, &yi, 4);
    y = y * (1.5f - 0.5f * mean * y * y);
    y = y * (1.5f - 0.5f * mean * y * y); /* second iteration */
    for (int i = 0; i < n; i++) x[i] = x[i] * y * w[i];
}

/* Softmax */
static void cpu_softmax(float *x, int n) {
    float max_val = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max_val) max_val = x[i];
    /* exp via Taylor series: e^x ≈ 1 + x + x^2/2 + x^3/6 for small x */
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = x[i] - max_val;
        /* Clamp to avoid overflow */
        if (v < -20.0f) v = -20.0f;
        /* exp approximation */
        float ex = 1.0f + v + v*v*0.5f + v*v*v*(1.0f/6.0f) +
                   v*v*v*v*(1.0f/24.0f);
        if (ex < 0.0f) ex = 0.0f;
        x[i] = ex;
        sum += ex;
    }
    if (sum > 0.0f) for (int i = 0; i < n; i++) x[i] /= sum;
}

/* Matmul: C = A * B, A[M][K] B[K][N] C[M][N] */
static void cpu_matmul(const float *A, const float *B, float *C,
                        int M, int N, int K) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++)
                acc += A[m * K + k] * B[k * N + n];
            C[m * N + n] = acc;
        }
}

/* SiLU: x = x * sigmoid(x) */
static void cpu_silu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        /* sigmoid via 1/(1+e^-v) — e^-v via Taylor */
        float ev = -v;
        if (ev > 20.0f) ev = 20.0f;
        float e_neg = 1.0f + ev + ev*ev*0.5f + ev*ev*ev*(1.0f/6.0f);
        float sig = 1.0f / (1.0f + e_neg);
        x[i] = v * sig;
    }
}

/* RoPE: rotary position embedding */
static void cpu_rope(float *x, int pos, int dim, int heads) {
    int head_dim = dim / heads;
    for (int h = 0; h < heads; h++) {
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f;
            /* 1/10000^(i/head_dim) — integer power approximation */
            for (int p = 0; p < i; p++) freq /= 10000.0f;
            float val = (float)pos * freq;
            /* cos/sin via Taylor: cos(x) ≈ 1 - x^2/2 + x^4/24 */
            float cos_val = 1.0f - val*val*0.5f + val*val*val*val*(1.0f/24.0f);
            float sin_val = val - val*val*val*(1.0f/6.0f);
            float *p2 = x + h * head_dim + i;
            float x0 = p2[0], x1 = p2[1];
            p2[0] = x0 * cos_val - x1 * sin_val;
            p2[1] = x0 * sin_val + x1 * cos_val;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Token generation loop — runs forever after EBS                     */
/* ------------------------------------------------------------------ */

#define MAX_SEQ   512
#define MAX_DIM   4096

static float g_x[MAX_DIM];
static float g_xb[MAX_DIM];
static float g_q[MAX_DIM];
static float g_k_buf[MAX_DIM];
static float g_v_buf[MAX_DIM];
static float g_logits[32000]; /* typical vocab size */

/* Print a single ASCII token to console */
static void print_token(int tok) {
    /* Bare-metal: just print token ID as hex for now.
       Phase 1.5: map token IDs to UTF-8 using vocab table from GGUF. */
    puts_ascii("tok:"); put_hex64((UINT64)tok); puts_ascii(" ");
}

static void inference_loop(void) {
    puts_ascii("VBS inference loop started"); put_newline();

    /* Fake token generation — real weight loading comes from GGUF
       Phase 1.1: wire g_model.data → weight slices by name lookup */
    int token = 1; /* BOS */
    int pos   = 0;

    while (1) {
        /* Clear hidden state */
        k_memset(g_x, 0, sizeof(g_x));
        g_x[token % MAX_DIM] = 1.0f; /* dummy embedding */

        /* RMS norm */
        float w_ones[MAX_DIM];
        k_memset(w_ones, 0, sizeof(w_ones));
        for (int i = 0; i < MAX_DIM; i++) {
            /* bit pattern for 1.0f */
            UINT32 one = 0x3F800000;
            k_memcpy(&w_ones[i], &one, 4);
        }
        cpu_rms_norm(g_x, w_ones, MAX_DIM);

        /* Copy to xb */
        k_memcpy(g_xb, g_x, MAX_DIM * sizeof(float));

        /* RoPE */
        cpu_rope(g_xb, pos, MAX_DIM, 32);

        /* Dummy logits (identity) */
        k_memset(g_logits, 0, sizeof(g_logits));
        for (int i = 0; i < MAX_DIM && i < 32000; i++)
            g_logits[i] = g_xb[i % MAX_DIM];

        /* Softmax */
        cpu_softmax(g_logits, 32000);

        /* Greedy sample */
        int next = 0;
        float best = g_logits[0];
        for (int i = 1; i < 32000; i++)
            if (g_logits[i] > best) { best = g_logits[i]; next = i; }

        print_token(next);
        token = next;
        pos++;

        /* EOS check */
        if (token == 2 || pos >= MAX_SEQ) {
            put_newline();
            pos   = 0;
            token = 1;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Memory map acquisition and ExitBootServices                        */
/* ------------------------------------------------------------------ */

static BOOLEAN exit_boot_services(void) {
    UINTN mmap_size  = 0;
    UINTN map_key    = 0;
    UINTN desc_size  = 0;
    UINT32 desc_ver  = 0;

    /* First call to get the size */
    EFI_STATUS s = g_bs->GetMemoryMap(&mmap_size, NULL,
                                       &map_key, &desc_size, &desc_ver);
    if (s != EFI_BUFFER_TOO_SMALL && EFI_ERROR(s)) {
        puts_ascii("GetMemoryMap size query failed"); put_newline();
        return 0;
    }

    /* Add slack for the map growing during AllocatePages */
    mmap_size += 4 * desc_size;

    UINT64 mmap_pages = (mmap_size + 4095) / 4096;
    UINT64 mmap_phys  = 0;
    s = g_bs->AllocatePages(0 /* AllocateAnyPages */,
                             EfiLoaderData, mmap_pages, &mmap_phys);
    if (EFI_ERROR(s)) {
        puts_ascii("AllocatePages for mmap failed"); put_newline();
        return 0;
    }

    EFI_MEMORY_DESCRIPTOR *mmap = (EFI_MEMORY_DESCRIPTOR *)(UINTN)mmap_phys;
    s = g_bs->GetMemoryMap(&mmap_size, mmap, &map_key, &desc_size, &desc_ver);
    if (EFI_ERROR(s)) {
        puts_ascii("GetMemoryMap failed"); put_newline();
        return 0;
    }

    s = g_bs->ExitBootServices(g_img, map_key);
    if (EFI_ERROR(s)) {
        /* Map key stale — retry once */
        s = g_bs->GetMemoryMap(&mmap_size, mmap, &map_key,
                                &desc_size, &desc_ver);
        if (!EFI_ERROR(s))
            s = g_bs->ExitBootServices(g_img, map_key);
    }

    if (EFI_ERROR(s)) {
        puts_ascii("ExitBootServices failed"); put_newline();
        return 0;
    }

    /* Boot Services are gone. g_bs is now invalid. */
    puts_ascii("ExitBootServices: ok — OS services offline");
    return 1;
}

/* ------------------------------------------------------------------ */
/* Load model from EFI Simple File System                             */
/* ------------------------------------------------------------------ */

static BOOLEAN load_model_from_fs(void) {
    /* Locate Simple File System */
    UINTN n_handles = 0;
    EFI_HANDLE *handles = NULL;
    EFI_STATUS s = g_bs->LocateHandleBuffer(
        2 /* ByProtocol */,
        g_sfsp_guid,
        NULL,
        &n_handles,
        &handles
    );
    if (EFI_ERROR(s) || n_handles == 0) {
        puts_ascii("No FS handle found"); put_newline();
        return 0;
    }

    EFI_SIMPLE_FILE_SYSTEM_PROTOCOL *fs = NULL;
    s = g_bs->HandleProtocol(handles[0], g_sfsp_guid, (void **)&fs);
    if (EFI_ERROR(s) || !fs) {
        puts_ascii("HandleProtocol FS failed"); put_newline();
        return 0;
    }

    EFI_FILE_PROTOCOL *root = NULL;
    s = fs->OpenVolume(fs, &root);
    if (EFI_ERROR(s) || !root) {
        puts_ascii("OpenVolume failed"); put_newline();
        return 0;
    }

    /* Try to open \model.gguf */
    EFI_FILE_PROTOCOL *file = NULL;
    CHAR16 model_path[] = { '\\','m','o','d','e','l','.','g','g','u','f',0 };
    s = root->Open(root, &file, model_path,
                   0x0000000000000001ULL /* EFI_FILE_MODE_READ */, 0);
    if (EFI_ERROR(s) || !file) {
        puts_ascii("model.gguf not found"); put_newline();
        return 0;
    }

    puts_ascii("model.gguf found — loading..."); put_newline();

    /* Read up to 32MB into arena */
    UINT8 *buf = (UINT8 *)g_arena;
    UINTN  sz  = sizeof(g_arena);
    s = file->Read(file, &sz, buf);
    file->Close(file);

    if (EFI_ERROR(s)) {
        puts_ascii("Read failed"); put_newline();
        return 0;
    }

    put_hex64((UINT64)sz); puts_ascii(" bytes read"); put_newline();
    g_arena_pos = sz; /* bump allocator starts after model */

    return gguf_load(buf, (UINT64)sz);
}

/* ------------------------------------------------------------------ */
/* UEFI entry point                                                    */
/* ------------------------------------------------------------------ */

EFI_STATUS efi_main(EFI_HANDLE ImageHandle,
                     EFI_SYSTEM_TABLE *SystemTable) {
    g_img = ImageHandle;
    g_st  = SystemTable;
    g_bs  = SystemTable->BootServices;

    g_st->ConOut->OutputString(g_st->ConOut,
        (CHAR16[]){ 'V','B','S',' ','p','h','1',' ','b','o','o','t','\r','\n',0 });

    /* Load model before exiting boot services
       (we need AllocatePages / FS access) */
    if (!load_model_from_fs()) {
        puts_ascii("Model load failed — continuing with dummy weights");
        put_newline();
    }

    /* Hand back control to ourselves */
    if (!exit_boot_services()) {
        /* Failed to exit — halt */
        while (1) { __asm__ volatile("hlt"); }
    }

    /* === Boot services are gone. No OS. We are the OS. === */

    /* Inference loop never returns */
    inference_loop();

    /* Should never reach here */
    while (1) { __asm__ volatile("hlt"); }
    return EFI_SUCCESS;
}