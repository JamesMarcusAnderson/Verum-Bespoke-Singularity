#include "vbs_profiler.h"
#include "vbs_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
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
#if defined(VBS_ENABLE_METAL) && VBS_ENABLE_METAL
extern const vbs_backend_ops_t g_vbs_metal_backend;
#endif

static void detect_cpu(vbs_cpu_profile_t *c) {
    c->logical_cores = (uint32_t)sysconf(_SC_NPROCESSORS_ONLN);
    c->physical_cores = c->logical_cores;

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
#else
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

static double benchmark_memcpy_bandwidth(size_t bytes) {
    void *src = malloc(bytes);
    void *dst = malloc(bytes);
    if (!src || !dst) { free(src); free(dst); return 0.0; }
    memset(src, 0xAB, bytes);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < 4; i++) memcpy(dst, src, bytes);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    double gb = (double)(bytes * 4) / 1e9;

    free(src); free(dst);
    return (elapsed > 0.0) ? (gb / elapsed) : 0.0;
}

static void detect_gpus(vbs_system_profile_t *sys) {
    sys->n_gpus = 0;

    if (g_vbs_cuda_backend.init()) {
        size_t n = g_vbs_cuda_backend.get_device_count();
        for (size_t i = 0; i < n && sys->n_gpus < 8; i++) {
            vbs_device_t *d = g_vbs_cuda_backend.get_device(i);
            vbs_gpu_profile_t *g = &sys->gpus[sys->n_gpus++];
            snprintf(g->name, sizeof(g->name), "%s", d->name);
            g->type = VBS_BACKEND_CUDA;
            g->device = d;
            g->total_vram_bytes = d->max_working_set_bytes;
            g->free_vram_bytes = d->max_working_set_bytes * 90 / 100;
            g->max_alloc_bytes = d->max_buffer_bytes;
            g->mem_bandwidth_gbps = 500.0;
            g->pcie_bandwidth_gbps = 32.0;
            g->sm_count = 80;
            g->warp_size = 32;
            g->supports_tensor_cores = true;
        }
    }

#if defined(VBS_ENABLE_METAL) && VBS_ENABLE_METAL
    if (g_vbs_metal_backend.init()) {
        size_t n = g_vbs_metal_backend.get_device_count();
        for (size_t i = 0; i < n && sys->n_gpus < 8; i++) {
            vbs_device_t *d = g_vbs_metal_backend.get_device(i);
            vbs_gpu_profile_t *g = &sys->gpus[sys->n_gpus++];
            snprintf(g->name, sizeof(g->name), "%s", d->name);
            g->type = VBS_BACKEND_METAL;
            g->device = d;
            g->total_vram_bytes = d->max_working_set_bytes;
            g->free_vram_bytes = d->max_working_set_bytes * 85 / 100;
            g->max_alloc_bytes = d->max_buffer_bytes;
            g->mem_bandwidth_gbps = 400.0;
            g->pcie_bandwidth_gbps = 400.0;
            g->sm_count = 40;
            g->warp_size = 32;
            g->supports_tensor_cores = false;
        }
    }
#endif

    if (g_vbs_hip_backend.init()) {
        size_t n = g_vbs_hip_backend.get_device_count();
        for (size_t i = 0; i < n && sys->n_gpus < 8; i++) {
            vbs_device_t *d = g_vbs_hip_backend.get_device(i);
            vbs_gpu_profile_t *g = &sys->gpus[sys->n_gpus++];
            snprintf(g->name, sizeof(g->name), "%s", d->name);
            g->type = VBS_BACKEND_HIP;
            g->device = d;
            g->total_vram_bytes = d->max_working_set_bytes;
            g->free_vram_bytes = d->max_working_set_bytes * 85 / 100;
            g->mem_bandwidth_gbps = 800.0;
            g->pcie_bandwidth_gbps = 32.0;
            g->sm_count = 60;
            g->warp_size = 64;
        }
    }
}

static void detect_storage(vbs_system_profile_t *sys) {
    sys->n_storage = 0;
    const char *candidates[] = { "/tmp", "/var/tmp", "/", NULL };
    for (int i = 0; candidates[i] && sys->n_storage < 4; i++) {
        struct stat st;
        if (stat(candidates[i], &st) == 0) {
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

vbs_model_profile_t *vbs_profiler_scan_model(const vbs_gguf_model_t *model) {
    vbs_model_profile_t *mp = (vbs_model_profile_t *)calloc(1, sizeof(vbs_model_profile_t));
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

    uint32_t v;
    if (vbs_gguf_get_uint32_metadata(model, "llama.block_count", &v)) mp->n_layers = v;
    if (vbs_gguf_get_uint32_metadata(model, "llama.attention.head_count", &v)) mp->n_heads = v;
    if (vbs_gguf_get_uint32_metadata(model, "llama.attention.head_count_kv", &v)) mp->n_kv_heads = v;
    if (vbs_gguf_get_uint32_metadata(model, "llama.embedding_length", &v)) mp->embedding_dim = v;
    if (vbs_gguf_get_uint32_metadata(model, "llama.context_length", &v)) mp->context_length = v;
    if (mp->n_heads && mp->embedding_dim) mp->head_dim = mp->embedding_dim / mp->n_heads;
    if (mp->vocab_size == 0) mp->vocab_size = 32000;

    mp->rope_theta = 10000.0f;
    mp->rms_eps = 1e-5f;
    float fv;
    if (vbs_gguf_get_float32_metadata(model, "llama.rope.freq_base", &fv)) mp->rope_theta = fv;
    if (vbs_gguf_get_float32_metadata(model, "llama.attention.layer_norm_rms_epsilon", &fv)) mp->rms_eps = fv;

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
