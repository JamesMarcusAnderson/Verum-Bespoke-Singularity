#ifndef VBS_PROFILER_H
#define VBS_PROFILER_H

#include "vbs_backend.h"
#include "vbs_gguf.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char name[128];
    vbs_backend_type_t type;
    vbs_device_t *device;

    uint64_t total_vram_bytes;
    uint64_t free_vram_bytes;
    uint64_t max_alloc_bytes;

    // Measured capabilities
    double  gflops_f32;
    double  gflops_f16;
    double  mem_bandwidth_gbps;      // VRAM bandwidth
    double  pcie_bandwidth_gbps;     // Host <-> device bandwidth
    double  peer_bandwidth_gbps[8];  // GPU <-> GPU bandwidth (if peer access)

    uint32_t compute_cap_major;
    uint32_t compute_cap_minor;
    uint32_t sm_count;
    uint32_t warp_size;
    bool     supports_tensor_cores;
    bool     supports_peer_access[8];
} vbs_gpu_profile_t;

typedef struct {
    uint64_t total_ram_bytes;
    uint64_t free_ram_bytes;
    uint64_t total_swap_bytes;
    uint32_t numa_node_count;
    uint32_t logical_cores;
    uint32_t physical_cores;
    uint64_t l3_cache_bytes;
    double   ram_bandwidth_gbps;
    bool     has_avx2;
    bool     has_avx512;
    bool     has_neon;
} vbs_cpu_profile_t;

typedef struct {
    char   mount_path[256];
    uint64_t free_bytes;
    double read_bandwidth_mbps;
    double write_bandwidth_mbps;
    bool   is_nvme;
    bool   is_ssd;
} vbs_storage_profile_t;

typedef struct {
    vbs_cpu_profile_t cpu;

    size_t n_gpus;
    vbs_gpu_profile_t gpus[8];

    size_t n_storage;
    vbs_storage_profile_t storage[4];

    // Aggregated unified memory capacity
    uint64_t total_unified_capacity_bytes;
    uint64_t total_vram_bytes;
    uint64_t total_ram_bytes;
    uint64_t total_swap_capacity_bytes;
} vbs_system_profile_t;

typedef struct {
    // Per-tensor access pattern
    const vbs_tensor_t *tensor;
    uint64_t access_frequency;      // Hot = embed/output; Warm = attention; Cold = MLP
    uint64_t bytes;
    double   estimated_compute_cost;
    bool     is_hot_path;
    bool     is_shared_across_layers;
} vbs_tensor_profile_t;

typedef struct {
    const vbs_gguf_model_t *model;
    size_t n_tensor_profiles;
    vbs_tensor_profile_t *tensor_profiles;

    uint64_t total_model_bytes;
    uint64_t hot_working_set_bytes;    // Must stay in VRAM
    uint64_t warm_working_set_bytes;   // Can live in RAM
    uint64_t cold_working_set_bytes;   // NVMe/mmap acceptable

    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t context_length;
    uint32_t vocab_size;
    uint32_t embedding_dim;
} vbs_model_profile_t;

vbs_system_profile_t *vbs_profiler_scan_system(void);
void                  vbs_profiler_free_system(vbs_system_profile_t *p);
void                  vbs_profiler_print_system(const vbs_system_profile_t *p);

vbs_model_profile_t  *vbs_profiler_scan_model(const vbs_gguf_model_t *model);
void                  vbs_profiler_free_model(vbs_model_profile_t *p);
void                  vbs_profiler_print_model(const vbs_model_profile_t *p);

// Measure actual throughput by running micro-benchmarks on each device
void vbs_profiler_benchmark_devices(vbs_system_profile_t *sys);

#ifdef __cplusplus
}
#endif
#endif
