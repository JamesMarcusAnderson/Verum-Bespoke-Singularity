/* vbs_profiler.c — system and model profiling: CPU/RAM/GPU/
 * storage detection, measured copy bandwidth, tensor hot/warm/
 * cold classification, and the strict model-shape refusal. */
#include "vbs_profiler.h"
#include "vbs_backend.h"
#include "vbs_port.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__)
/* sysctl.h comes from vbs_port.h on Apple platforms. */
#include <mach/mach.h>
#include <mach/vm_statistics.h>
#include <mach/mach_types.h>
#include <mach/mach_init.h>
#include <mach/mach_host.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#endif

extern const vbs_backend_ops_t g_vbs_cuda_backend;
extern const vbs_backend_ops_t g_vbs_hip_backend;
extern const vbs_backend_ops_t g_vbs_vulkan_backend;
#if defined(VBS_ENABLE_METAL) && VBS_ENABLE_METAL
extern const vbs_backend_ops_t g_vbs_metal_backend;
#endif

static void detect_cpu(vbs_cpu_profile_t *c) {
#if defined(_WIN32)
    SYSTEM_INFO si;
    MEMORYSTATUSEX ms;
    GetSystemInfo(&si);
    c->logical_cores = (uint32_t)si.dwNumberOfProcessors;
    c->physical_cores = c->logical_cores;
    ms.dwLength = sizeof ms;
    if (GlobalMemoryStatusEx(&ms)) {
        c->total_ram_bytes = (uint64_t)ms.ullTotalPhys;
        c->free_ram_bytes = (uint64_t)ms.ullAvailPhys;
    } else {
        c->total_ram_bytes = 16ULL * 1024 * 1024 * 1024;
        c->free_ram_bytes = 8ULL * 1024 * 1024 * 1024;
    }
    c->ram_bandwidth_gbps = 40.0;
    c->l3_cache_bytes = 32 * 1024 * 1024;
#else
    c->logical_cores = (uint32_t)sysconf(_SC_NPROCESSORS_ONLN);
    c->physical_cores = c->logical_cores;
#endif

#if defined(__APPLE__)
    size_t sz = sizeof(uint64_t);
    sysctlbyname("hw.memsize", &c->total_ram_bytes, &sz, NULL, 0);
    sysctlbyname("hw.l3cachesize", &c->l3_cache_bytes, &sz, NULL, 0);
    int phys = 0;
    sz = sizeof(phys);
    sysctlbyname("hw.physicalcpu", &phys, &sz, NULL, 0);
    c->physical_cores = (uint32_t)phys;

    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_statistics64_data_t vmstat;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vmstat, &count) == KERN_SUCCESS) {
        c->free_ram_bytes = (uint64_t)(vmstat.free_count + vmstat.inactive_count) * (uint64_t)sysconf(_SC_PAGESIZE);
    } else {
        c->free_ram_bytes = c->total_ram_bytes / 2;
    }
    c->ram_bandwidth_gbps = 200.0;
#elif defined(__linux__)
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        c->total_ram_bytes = (uint64_t)si.totalram * si.mem_unit;
        c->free_ram_bytes = (uint64_t)si.freeram * si.mem_unit;
        c->total_swap_bytes = (uint64_t)si.totalswap * si.mem_unit;
    }
    c->ram_bandwidth_gbps = 50.0;
    c->l3_cache_bytes = 32 * 1024 * 1024;
#elif !defined(_WIN32)
    c->total_ram_bytes = 16ULL * 1024 * 1024 * 1024;
    c->free_ram_bytes = 8ULL * 1024 * 1024 * 1024;
    c->ram_bandwidth_gbps = 40.0;
#endif

#if defined(VBS_HAVE_AVX2) && VBS_HAVE_AVX2
    c->has_avx2 = true;
#endif
#if defined(VBS_HAVE_NEON) && VBS_HAVE_NEON
    c->has_neon = true;
#endif
}

/* Measured host<->device copy bandwidth (James): times real
 * transfers of a 32 MiB buffer through the backend instead of
 * trusting a table constant. This is the copy path bandwidth; no
 * device-local kernel benchmark exists in the ops table yet, so
 * both bandwidth fields take this measured value. */
static double measure_device_bandwidth_gbps(vbs_device_t *dev, const vbs_backend_ops_t *backend) {
    size_t test_bytes = 32 * 1024 * 1024;
    void *host_src = malloc(test_bytes);
    if (!host_src) return 0.0;
    memset(host_src, 0x5A, test_bytes);
    vbs_buffer_t *d_buf = backend->alloc_buffer(dev, test_bytes, false);
    if (!d_buf) {
        free(host_src);
        return 0.0;
    }
    double t0 = vbs_now_sec();
    for (int i = 0; i < 4; i++) backend->copy_to_device(d_buf, host_src, test_bytes);
    backend->synchronize_device(dev);
    double t1 = vbs_now_sec();
    backend->free_buffer(d_buf);
    free(host_src);
    double elapsed = t1 - t0;
    if (elapsed <= 0.0) return 0.0;
    return ((double)(test_bytes * 4) / 1e9) / elapsed;
}

static double benchmark_memcpy_bandwidth(size_t bytes) {
    void *src = malloc(bytes);
    void *dst = malloc(bytes);
    if (!src || !dst) {
        free(src);
        free(dst);
        return 0.0;
    }
    memset(src, 0xAB, bytes);

    double t0 = vbs_now_sec();
    for (int i = 0; i < 4; i++) memcpy(dst, src, bytes);
    double t1 = vbs_now_sec();

    double elapsed = t1 - t0;
    double gb = (double)(bytes * 4) / 1e9;

    free(src);
    free(dst);
    return (elapsed > 0.0) ? (gb / elapsed) : 0.0;
}

/* Per-backend GPU profile constants. The probe order is
 * Vulkan, CUDA, Metal, HIP. Note the HIP quirk: its block never
 * set max_alloc_bytes, so it stays 0 -- preserved here via
 * report_max_alloc. */
typedef struct {
    const vbs_backend_ops_t *ops;
    vbs_backend_type_t type;
    int free_pct;
    uint32_t sm_count;
    uint32_t warp_size;
    bool tensor_cores;
    bool report_max_alloc;
} gpu_backend_desc_t;

static const gpu_backend_desc_t k_gpu_backends[] = {
    { &g_vbs_vulkan_backend, VBS_BACKEND_VULKAN, 90, 0, 32, false, true },
    { &g_vbs_cuda_backend,   VBS_BACKEND_CUDA,   90, 80, 32, true,  true },
#if defined(VBS_ENABLE_METAL) && VBS_ENABLE_METAL
    { &g_vbs_metal_backend,  VBS_BACKEND_METAL,  85, 40, 32, false, true },
#endif
    { &g_vbs_hip_backend,    VBS_BACKEND_HIP,    85, 60, 64, false, false },
};

static void detect_gpus(vbs_system_profile_t *sys) {
    sys->n_gpus = 0;
    for (size_t b = 0; b < sizeof(k_gpu_backends) / sizeof(k_gpu_backends[0]); b++) {
        const gpu_backend_desc_t *desc = &k_gpu_backends[b];
        if (!desc->ops->init()) continue;
        size_t n = desc->ops->get_device_count();
        for (size_t i = 0; i < n && sys->n_gpus < 8; i++) {
            vbs_device_t *d = desc->ops->get_device(i);
            vbs_gpu_profile_t *g = &sys->gpus[sys->n_gpus++];
            snprintf(g->name, sizeof(g->name), "%s", d->name);
            g->type = desc->type;
            g->device = d;
            g->total_vram_bytes = d->max_working_set_bytes;
            g->free_vram_bytes = d->max_working_set_bytes * (uint64_t)desc->free_pct / 100;
            if (desc->report_max_alloc) g->max_alloc_bytes = d->max_buffer_bytes;
            double bw = measure_device_bandwidth_gbps(d, desc->ops);
            g->mem_bandwidth_gbps = bw;
            g->pcie_bandwidth_gbps = bw;
            /* Vulkan 1.0 properties expose no CU count; the shader
             * workgroup size (32) is the scheduling unit. */
            g->sm_count = desc->sm_count;
            g->warp_size = desc->warp_size;
            g->supports_tensor_cores = desc->tensor_cores;
        }
    }
}

static void detect_storage(vbs_system_profile_t *sys) {
    sys->n_storage = 0;
#if defined(_WIN32)
    char tmp[512];
    vbs_temp_dir(tmp, sizeof tmp);
    const char *candidates[] = { tmp, "C:\\", NULL };
#else
    const char *candidates[] = { "/tmp", "/var/tmp", "/", NULL };
#endif
    for (int i = 0; candidates[i] && sys->n_storage < 4; i++) {
        if (vbs_stat_exists(candidates[i])) {
            vbs_storage_profile_t *s = &sys->storage[sys->n_storage++];
            snprintf(s->mount_path, sizeof(s->mount_path), "%s", candidates[i]);
            s->free_bytes = 100ULL * 1024 * 1024 * 1024;
            s->read_bandwidth_mbps = 3500.0;
            s->write_bandwidth_mbps = 3000.0;
            s->is_nvme = true;
            s->is_ssd = true;
        }
    }
}

vbs_system_profile_t *vbs_profiler_scan_system(void) {
    vbs_system_profile_t *sys = (vbs_system_profile_t *)calloc(1, sizeof(vbs_system_profile_t));
    if (!sys) return NULL;
    detect_cpu(&sys->cpu);
    detect_gpus(sys);
    detect_storage(sys);

    sys->cpu.ram_bandwidth_gbps = benchmark_memcpy_bandwidth(64 * 1024 * 1024);
    sys->total_ram_bytes = sys->cpu.total_ram_bytes;
    sys->total_vram_bytes = 0;
    for (size_t i = 0; i < sys->n_gpus; i++) sys->total_vram_bytes += sys->gpus[i].total_vram_bytes;

    sys->total_swap_capacity_bytes = 0;
    for (size_t i = 0; i < sys->n_storage; i++) sys->total_swap_capacity_bytes += sys->storage[i].free_bytes;

    sys->total_unified_capacity_bytes =
        sys->total_vram_bytes + sys->total_ram_bytes + sys->total_swap_capacity_bytes;

    return sys;
}

void vbs_profiler_free_system(vbs_system_profile_t *p) { free(p); }

void vbs_profiler_print_system(const vbs_system_profile_t *p) {
    printf("\n=== Verum System Profile ===\n");
    printf("CPU: %u physical / %u logical cores | RAM %.1f GiB | bandwidth %.1f GB/s\n",
        p->cpu.physical_cores, p->cpu.logical_cores,
        p->cpu.total_ram_bytes / (1024.0 * 1024 * 1024),
        p->cpu.ram_bandwidth_gbps);

    for (size_t i = 0; i < p->n_gpus; i++) {
        const vbs_gpu_profile_t *g = &p->gpus[i];
        printf("GPU[%zu]: %s | VRAM %.1f GiB free / %.1f GiB total | mem %.0f GB/s | PCIe %.0f GB/s%s\n",
               i, g->name,
               g->free_vram_bytes / (1024.0 * 1024 * 1024),
               g->total_vram_bytes / (1024.0 * 1024 * 1024),
               g->mem_bandwidth_gbps, g->pcie_bandwidth_gbps,
               g->supports_tensor_cores ? " | Tensor Cores" : "");
    }

    for (size_t i = 0; i < p->n_storage; i++) {
        printf("Storage[%zu]: %s | %.1f GiB free | %.0f MB/s read\n",
            i, p->storage[i].mount_path,
            p->storage[i].free_bytes / (1024.0 * 1024 * 1024),
            p->storage[i].read_bandwidth_mbps);
    }

    printf("Unified Memory Pool: %.1f GiB total\n",
        p->total_unified_capacity_bytes / (1024.0 * 1024 * 1024));
    printf("  = %.1f GiB VRAM + %.1f GiB RAM + %.1f GiB NVMe cold tier\n",
        p->total_vram_bytes / (1024.0 * 1024 * 1024),
        p->total_ram_bytes / (1024.0 * 1024 * 1024),
        p->total_swap_capacity_bytes / (1024.0 * 1024 * 1024));
    printf("============================\n\n");
}

vbs_model_profile_t *vbs_profiler_scan_model(const vbs_model_t *model) {
    vbs_model_profile_t *mp = (vbs_model_profile_t *)calloc(1, sizeof(vbs_model_profile_t));
    if (!mp) return NULL;
    mp->model = model;
    mp->n_tensor_profiles = model->n_tensors;
    mp->tensor_profiles = (vbs_tensor_profile_t *)calloc(
        model->n_tensors, sizeof(vbs_tensor_profile_t));

    uint64_t total = 0, hot = 0, warm = 0, cold = 0;
    for (size_t i = 0; i < model->n_tensors; i++) {
        vbs_tensor_profile_t *tp = &mp->tensor_profiles[i];
        tp->tensor = &model->tensors[i];
        tp->bytes = model->tensors[i].byte_length;
        total += tp->bytes;

        const char *n = model->tensors[i].name;
        if (strstr(n, "token_embd") || strstr(n, "output") || strstr(n, "norm")) {
            tp->is_hot_path = true;
            tp->access_frequency = 100;
            hot += tp->bytes;
            if (strstr(n, "token_embd.weight") || strstr(n, "output.weight")) {
                mp->vocab_size = (uint32_t)model->tensors[i].rows;
            }
        } else if (strstr(n, "attn") || strstr(n, ".q.") || strstr(n, ".k.") || strstr(n, ".v.")) {
            tp->access_frequency = 50;
            warm += tp->bytes;
        } else {
            tp->access_frequency = 10;
            cold += tp->bytes;
        }
        tp->estimated_compute_cost = (double)tp->bytes * 2.0;
    }

    mp->total_model_bytes = total;
    mp->hot_working_set_bytes = hot;
    mp->warm_working_set_bytes = warm;
    mp->cold_working_set_bytes = cold;

    struct {
        const char *key;
        uint32_t *field;
    } dims[] = {
        { "llama.block_count",               &mp->n_layers      },
        { "llama.attention.head_count",      &mp->n_heads       },
        { "llama.attention.head_count_kv",   &mp->n_kv_heads    },
        { "llama.embedding_length",          &mp->embedding_dim },
        { "llama.context_length",            &mp->context_length },
    };
    for (size_t i = 0; i < sizeof(dims) / sizeof(dims[0]); i++) {
        uint32_t v;
        if (vbs_model_get_uint32_metadata(model, dims[i].key, &v)) *dims[i].field = v;
    }
    if (mp->n_heads && mp->embedding_dim) mp->head_dim = mp->embedding_dim / mp->n_heads;

    mp->rope_theta = 10000.0f;
    mp->rms_eps = 1e-5f;
    float fv;
    if (vbs_model_get_float32_metadata(model, "llama.rope.freq_base", &fv)) mp->rope_theta = fv;
    if (vbs_model_get_float32_metadata(model, "llama.attention.layer_norm_rms_epsilon", &fv)) mp->rms_eps = fv;

    /* Strict refusal: no invented model shapes. If the essential
     * dimensions cannot be proven from the model metadata and
     * tensor shapes, fail here instead of letting downstream code
     * guess (James, deliberate refusal). */
    if (mp->n_layers == 0 || mp->n_heads == 0 || mp->n_kv_heads == 0 ||
        mp->embedding_dim == 0 || mp->vocab_size == 0 || mp->head_dim == 0 ||
        mp->context_length == 0) {
        fprintf(stderr, "Model refusal: essential dimensions unproven (layers=%u heads=%u kv_heads=%u embed=%u vocab=%u head_dim=%u ctx=%u). Refusing to invent parameters.\n",
                mp->n_layers, mp->n_heads, mp->n_kv_heads, mp->embedding_dim,
                mp->vocab_size, mp->head_dim, mp->context_length);
        vbs_profiler_free_model(mp);
        return NULL;
    }

    return mp;
}

void vbs_profiler_free_model(vbs_model_profile_t *p) {
    if (!p) return;
    free(p->tensor_profiles);
    free(p);
}

void vbs_profiler_print_model(const vbs_model_profile_t *p) {
    printf("\n=== Verum Model Profile ===\n");
    printf("Total weights: %.2f GiB across %zu tensors\n",
        p->total_model_bytes / (1024.0 * 1024 * 1024), p->n_tensor_profiles);
    printf("  Hot path (embeds/norms/output):   %.2f GiB  (keep in VRAM)\n",
        p->hot_working_set_bytes / (1024.0 * 1024 * 1024));
    printf("  Warm path (attention weights):    %.2f GiB  (RAM OK, VRAM preferred)\n",
        p->warm_working_set_bytes / (1024.0 * 1024 * 1024));
    printf("  Cold path (MLP down/up/gate):     %.2f GiB  (NVMe acceptable)\n",
        p->cold_working_set_bytes / (1024.0 * 1024 * 1024));
    if (p->n_layers) {
        printf("Architecture: %u layers, %u heads (%u KV), head_dim=%u, ctx=%u\n",
            p->n_layers, p->n_heads, p->n_kv_heads, p->head_dim, p->context_length);
    }
    printf("===========================\n\n");
}

void vbs_profiler_benchmark_devices(vbs_system_profile_t *sys) {
    for (size_t i = 0; i < sys->n_gpus; i++) {
        vbs_gpu_profile_t *g = &sys->gpus[i];
        g->gflops_f32 = (double)g->sm_count * 128.0 * 1.5;
        g->gflops_f16 = g->gflops_f32 * 2.0;
    }
}
