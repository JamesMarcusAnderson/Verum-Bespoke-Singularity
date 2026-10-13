#ifndef VBS_FABRIC_H
#define VBS_FABRIC_H

#include "vbs_backend.h"
#include "vbs_profiler.h"
#include "vbs_gguf.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VBS_TIER_VRAM = 0,         // Fastest: in-GPU VRAM
    VBS_TIER_PEER_VRAM = 1,    // Another GPU's VRAM via peer access
    VBS_TIER_HOST_PINNED = 2,  // DMA-capable system RAM
    VBS_TIER_HOST_PAGED = 3,   // Normal system RAM
    VBS_TIER_NVME_MMAP = 4,    // mmap-backed cold storage
    VBS_TIER_FILE_SOURCE = 5   // Original GGUF mmap region
} vbs_memory_tier_t;

typedef struct vbs_memory_block {
    void *data;                       // host-visible pointer (if applicable)
    vbs_buffer_t *gpu_buffer;         // GPU buffer handle (if resident on GPU)
    size_t size;
    vbs_memory_tier_t current_tier;
    vbs_device_t *resident_device;    // NULL for host-only
    const vbs_tensor_t *tensor;       // Source tensor (if applicable)
    uint64_t last_access_tick;
    uint32_t pin_count;               // >0 means do not evict
} vbs_memory_block_t;

typedef struct vbs_fabric {
    const vbs_system_profile_t *sys;
    const vbs_backend_ops_t *backend;

    // Per-device allocation tracking
    uint64_t vram_in_use[8];
    uint64_t vram_budget[8];           // Soft cap per GPU

    // Pinned host memory pool for DMA transfers
    void   *pinned_pool;
    size_t  pinned_pool_size;
    size_t  pinned_pool_offset;

    // NVMe cold-tier swap file
    int     nvme_swap_fd;
    void   *nvme_swap_map;
    size_t  nvme_swap_size;
    size_t  nvme_swap_offset;

    // Block registry
    vbs_memory_block_t **blocks;
    size_t n_blocks;
    size_t block_cap;

    uint64_t global_tick;
} vbs_fabric_t;

vbs_fabric_t *vbs_fabric_create(
    const vbs_system_profile_t *sys,
    const vbs_backend_ops_t *backend,
    size_t pinned_pool_mib,
    size_t nvme_swap_gib,
    const char *nvme_swap_path
);

void vbs_fabric_free(vbs_fabric_t *f);

// Register a tensor into the fabric. Chooses initial tier based on hints.
vbs_memory_block_t *vbs_fabric_register_tensor(
    vbs_fabric_t *f,
    const vbs_tensor_t *tensor,
    const void *source_data,
    vbs_memory_tier_t preferred_tier,
    vbs_device_t *preferred_device
);

// Guarantee a block is resident on a specific device's VRAM. Streams in if needed.
// Evicts cold blocks if VRAM is full. Returns the GPU buffer handle.
vbs_buffer_t *vbs_fabric_fault_in_to_device(
    vbs_fabric_t *f,
    vbs_memory_block_t *block,
    vbs_device_t *device
);

// Pin a block so it cannot be evicted during execution of an op
void vbs_fabric_pin(vbs_fabric_t *f, vbs_memory_block_t *block);
void vbs_fabric_unpin(vbs_fabric_t *f, vbs_memory_block_t *block);

// Prefetch: schedule a block move in the background
void vbs_fabric_prefetch(
    vbs_fabric_t *f,
    vbs_memory_block_t *block,
    vbs_device_t *target_device
);

// Evict cold blocks from a device until at least `bytes_needed` is free
bool vbs_fabric_make_room(vbs_fabric_t *f, vbs_device_t *device, size_t bytes_needed);

// Report statistics
void vbs_fabric_print_stats(const vbs_fabric_t *f);

#ifdef __cplusplus
}
#endif
#endif
