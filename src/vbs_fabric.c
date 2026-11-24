/* vbs_fabric.c — the memory fabric.
 *
 * Tracks where every tensor's weights live (VRAM, host RAM, NVMe,
 * or the model file) and moves them between tiers on demand.
 * VRAM is a soft-budgeted cache: faulting a block in evicts the
 * least-recently-used unpinned block on that device until there is
 * room. Blocks start at FILE_SOURCE (the model mmap) and are
 * uploaded only when a VRAM preference fits the budget at
 * registration time, or when faulted in later.
 */

#include "vbs_fabric.h"
#include "vbs_port.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

vbs_fabric_t *vbs_fabric_create(
    const vbs_system_profile_t *sys,
    const vbs_backend_ops_t *backend,
    size_t pinned_pool_mib,
    size_t nvme_swap_gib,
    const char *nvme_swap_path) {
    vbs_fabric_t *f = (vbs_fabric_t *)calloc(1, sizeof(vbs_fabric_t));
    if (!f) return NULL;
    f->sys = sys;
    f->backend = backend;
    vbs_mmap_init(&f->nvme_swap);

    for (size_t i = 0; i < sys->n_gpus && i < 8; i++)
        f->vram_budget[i] = sys->gpus[i].free_vram_bytes * 90 / 100;

    f->pinned_pool_size = pinned_pool_mib * 1024ULL * 1024ULL;
    if (vbs_aligned_alloc(&f->pinned_pool, 4096, f->pinned_pool_size) != 0) {
        f->pinned_pool = NULL;
        f->pinned_pool_size = 0;
    }

    if (nvme_swap_gib > 0 && nvme_swap_path) {
        f->nvme_swap_size = nvme_swap_gib * 1024ULL * 1024ULL * 1024ULL;
        if (vbs_mmap_create_rw(nvme_swap_path, f->nvme_swap_size,
                               &f->nvme_swap) != VBS_MMAP_OK) {
            f->nvme_swap_size = 0;
        }
    }

    f->block_cap = 1024;
    f->blocks = (vbs_memory_block_t **)calloc(
        f->block_cap, sizeof(vbs_memory_block_t *));
    if (!f->blocks) {
        vbs_fabric_free(f);
        return NULL;
    }
    return f;
}

void vbs_fabric_free(vbs_fabric_t *f) {
    if (!f) return;
    for (size_t i = 0; i < f->n_blocks; i++) {
        vbs_memory_block_t *b = f->blocks[i];
        if (!b) continue;
        if (b->gpu_buffer && f->backend) f->backend->free_buffer(b->gpu_buffer);
        free(b);
    }
    free(f->blocks);
    if (f->pinned_pool) vbs_aligned_free(f->pinned_pool);
    vbs_mmap_close(&f->nvme_swap);
    free(f);
}

/* Index of a device in the system profile, or -1 when unknown. */
static int find_device_index(const vbs_fabric_t *f, const vbs_device_t *d) {
    for (size_t i = 0; i < f->sys->n_gpus; i++) {
        if (f->sys->gpus[i].device == d) return (int)i;
    }
    return -1;
}

vbs_memory_block_t *vbs_fabric_register_tensor(
    vbs_fabric_t *f,
    const vbs_tensor_t *tensor,
    const void *source_data,
    vbs_memory_tier_t preferred_tier,
    vbs_device_t *preferred_device) {
    if (f->n_blocks >= f->block_cap) {
        size_t new_cap = f->block_cap * 2;
        vbs_memory_block_t **grown = (vbs_memory_block_t **)realloc(
            f->blocks, new_cap * sizeof(vbs_memory_block_t *));
        if (!grown) return NULL;
        f->blocks = grown;
        f->block_cap = new_cap;
    }
    vbs_memory_block_t *b =
        (vbs_memory_block_t *)calloc(1, sizeof(vbs_memory_block_t));
    if (!b) return NULL;
    b->size = tensor->byte_length;
    b->tensor = tensor;
    b->data = (void *)source_data;
    b->current_tier = VBS_TIER_FILE_SOURCE;
    b->resident_device = NULL;

    if (preferred_tier == VBS_TIER_VRAM && preferred_device) {
        int di = find_device_index(f, preferred_device);
        if (di >= 0 && f->vram_in_use[di] + b->size <= f->vram_budget[di]) {
            b->gpu_buffer =
                f->backend->alloc_buffer(preferred_device, b->size, false);
            if (b->gpu_buffer) {
                if (source_data && f->backend->copy_to_device)
                    f->backend->copy_to_device(b->gpu_buffer, source_data,
                                               b->size);
                f->vram_in_use[di] += b->size;
                b->current_tier = VBS_TIER_VRAM;
                b->resident_device = preferred_device;
            }
        }
    }

    f->blocks[f->n_blocks++] = b;
    return b;
}

bool vbs_fabric_make_room(vbs_fabric_t *f, vbs_device_t *device,
                          size_t bytes_needed) {
    int di = find_device_index(f, device);
    if (di < 0) return false;

    uint64_t available = f->vram_budget[di] - f->vram_in_use[di];
    if (available >= bytes_needed) return true;

    while (f->vram_in_use[di] + bytes_needed > f->vram_budget[di]) {
        /* Evict the least-recently-used unpinned block on this
         * device. Ties keep the earliest-registered block. */
        vbs_memory_block_t *victim = NULL;
        uint64_t oldest_tick = UINT64_MAX;
        for (size_t i = 0; i < f->n_blocks; i++) {
            vbs_memory_block_t *b = f->blocks[i];
            if (b->resident_device == device && b->pin_count == 0 &&
                b->last_access_tick < oldest_tick) {
                victim = b;
                oldest_tick = b->last_access_tick;
            }
        }
        if (!victim) return false;
        if (victim->gpu_buffer) {
            f->backend->free_buffer(victim->gpu_buffer);
            victim->gpu_buffer = NULL;
            f->vram_in_use[di] -= victim->size;
            victim->resident_device = NULL;
            victim->current_tier = VBS_TIER_FILE_SOURCE;
        }
    }
    return true;
}

vbs_buffer_t *vbs_fabric_fault_in_to_device(
    vbs_fabric_t *f,
    vbs_memory_block_t *block,
    vbs_device_t *device) {
    f->global_tick++;
    block->last_access_tick = f->global_tick;

    if (block->resident_device == device && block->gpu_buffer)
        return block->gpu_buffer;
    if (!vbs_fabric_make_room(f, device, block->size)) return NULL;

    block->gpu_buffer = f->backend->alloc_buffer(device, block->size, false);
    if (!block->gpu_buffer) return NULL;
    if (block->data && f->backend->copy_to_device)
        f->backend->copy_to_device(block->gpu_buffer, block->data, block->size);

    int di = find_device_index(f, device);
    if (di >= 0) f->vram_in_use[di] += block->size;
    block->resident_device = device;
    block->current_tier = VBS_TIER_VRAM;
    return block->gpu_buffer;
}

void vbs_fabric_pin(vbs_fabric_t *f, vbs_memory_block_t *b) {
    (void)f;
    b->pin_count++;
}

void vbs_fabric_unpin(vbs_fabric_t *f, vbs_memory_block_t *b) {
    (void)f;
    if (b->pin_count) b->pin_count--;
}

void vbs_fabric_prefetch(vbs_fabric_t *f, vbs_memory_block_t *b,
                         vbs_device_t *d) {
    vbs_fabric_fault_in_to_device(f, b, d);
}

void vbs_fabric_print_stats(const vbs_fabric_t *f) {
    printf("\n=== Fabric Memory Stats ===\n");
    for (size_t i = 0; i < f->sys->n_gpus; i++) {
        printf("GPU[%zu]: VRAM %.2f / %.2f GiB in use\n", i,
               f->vram_in_use[i] / (1024.0 * 1024 * 1024),
               f->vram_budget[i] / (1024.0 * 1024 * 1024));
    }
    printf("Pinned host pool: %.1f MiB\n", f->pinned_pool_size / (1024.0 * 1024));
    printf("NVMe swap tier:   %.1f GiB\n",
           f->nvme_swap_size / (1024.0 * 1024 * 1024));
    printf("Registered blocks: %zu\n", f->n_blocks);
    printf("===========================\n\n");
}
