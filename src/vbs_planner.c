#include "vbs_planner.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static vbs_device_t *select_primary_device(const vbs_system_profile_t *sys) {
    if (sys->n_gpus == 0) return NULL;
    size_t best = 0;
    double best_score = 0;
    for (size_t i = 0; i < sys->n_gpus; i++) {
        double score = sys->gpus[i].mem_bandwidth_gbps *
                       (double)(sys->gpus[i].total_vram_bytes / (1024 * 1024));
        if (score > best_score) { best = i; best_score = score; }
    }
    return sys->gpus[best].device;
}

static vbs_execution_strategy_t choose_strategy(
    const vbs_tensor_t *t,
    uint64_t available_vram,
    uint64_t available_ram,
    uint64_t available_nvme
) {
    uint64_t size = t->byte_length;
    if (size <= available_vram) return VBS_STRATEGY_IN_VRAM;
    if (size <= available_vram + available_ram) return VBS_STRATEGY_HOST_STREAMED;
    if (size <= available_vram + available_ram + available_nvme) return VBS_STRATEGY_NVME_STREAMED;
    return VBS_STRATEGY_RECOMPUTE;
}

static void choose_tile_dims(
    const vbs_tensor_t *t,
    uint64_t vram_budget_for_tile,
    uint32_t *out_rows,
    uint32_t *out_cols
) {
    uint64_t packed_row_bytes = t->packed_row_bytes;
    uint64_t max_rows = vram_budget_for_tile / packed_row_bytes;
    if (max_rows == 0) max_rows = 1;
    if (max_rows > t->rows) max_rows = t->rows;
    *out_rows = (uint32_t)max_rows;
    *out_cols = (uint32_t)t->columns;
}

vbs_execution_plan_t *vbs_planner_build(
    const vbs_graph_t *graph,
    const vbs_system_profile_t *sys,
    const vbs_model_profile_t *mp,
    vbs_fabric_t *fabric
) {
    vbs_execution_plan_t *plan = (vbs_execution_plan_t *)calloc(1, sizeof(vbs_execution_plan_t));
    plan->graph = graph;
    plan->sys = sys;
    plan->mp = mp;

    plan->primary_device = select_primary_device(sys);
    plan->vram_budget_per_device = sys->n_gpus ? sys->gpus[0].free_vram_bytes * 90 / 100 : 0;
    plan->host_cache_budget = sys->cpu.total_ram_bytes * 60 / 100;
    plan->nvme_overflow_budget = sys->total_swap_capacity_bytes * 50 / 100;

    uint64_t total_capacity =
        plan->vram_budget_per_device +
        plan->host_cache_budget +
        plan->nvme_overflow_budget;

    plan->n_plans = graph->n_nodes;
    plan->plans = (vbs_op_plan_t *)calloc(plan->n_plans, sizeof(vbs_op_plan_t));

    uint64_t vram_remaining = plan->vram_budget_per_device;
    uint64_t ram_remaining = plan->host_cache_budget;
    uint64_t nvme_remaining = plan->nvme_overflow_budget;

    for (size_t i = 0; i < graph->n_nodes; i++) {
        const vbs_node_t *n = &graph->nodes[i];
        vbs_op_plan_t *p = &plan->plans[i];
        p->node = n;
        p->target_device = plan->primary_device;

        if (!n->weight) {
            p->strategy = VBS_STRATEGY_IN_VRAM;
            p->resident_tier = VBS_TIER_VRAM;
            continue;
        }

        // Model profiles tell us which tensors are hot vs cold
        bool is_hot = false;
        for (size_t j = 0; j < mp->n_tensor_profiles; j++) {
            if (mp->tensor_profiles[j].tensor == n->weight) {
                is_hot = mp->tensor_profiles[j].is_hot_path;
                break;
            }
        }

        p->strategy = choose_strategy(n->weight, vram_remaining, ram_remaining, nvme_remaining);
        switch (p->strategy) {
            case VBS_STRATEGY_IN_VRAM:
                p->resident_tier = VBS_TIER_VRAM;
                vram_remaining -= n->weight->byte_length;
                break;
            case VBS_STRATEGY_HOST_STREAMED:
                p->resident_tier = VBS_TIER_HOST_PINNED;
                ram_remaining -= n->weight->byte_length;
                break;
            case VBS_STRATEGY_NVME_STREAMED:
                p->resident_tier = VBS_TIER_NVME_MMAP;
                nvme_remaining -= n->weight->byte_length;
                break;
            case VBS_STRATEGY_PEER_SHARED:
                p->resident_tier = VBS_TIER_PEER_VRAM;
                break;
            case VBS_STRATEGY_RECOMPUTE:
                p->resident_tier = VBS_TIER_FILE_SOURCE;
                break;
        }

        // Allocate a tile-sized VRAM budget for streaming kernels
        uint64_t tile_budget = plan->vram_budget_per_device / (uint64_t)graph->n_nodes;
        // No floor: tile size derives from the VRAM budget and the
        // tensor row geometry alone (choose_tile_dims fits whole
        // quant-aligned rows into whatever budget exists).
        choose_tile_dims(n->weight, tile_budget, &p->tile_rows, &p->tile_cols);

        p->prefetch_depth = is_hot ? 0 : 2;
        p->use_tensor_cores = sys->gpus[0].supports_tensor_cores &&
                              (n->weight->type == VBS_GGML_TYPE_F16 ||
                               n->weight->type == VBS_GGML_TYPE_BF16);

        p->estimated_vram_footprint = (uint64_t)p->tile_rows * p->tile_cols;
    }

    // Register all weights into the fabric
    if (fabric) {
        for (size_t i = 0; i < plan->n_plans; i++) {
            vbs_op_plan_t *p = &plan->plans[i];
            if (p->node->weight) {
                vbs_fabric_register_tensor(fabric,
                    p->node->weight, NULL, p->resident_tier, p->target_device);
            }
        }
    }

    // Compute theoretical token throughput
    uint64_t bytes_per_token = mp->hot_working_set_bytes + mp->warm_working_set_bytes;
    double effective_bw = plan->primary_device ?
        sys->gpus[0].mem_bandwidth_gbps * 1e9 : sys->cpu.ram_bandwidth_gbps * 1e9;
    plan->estimated_tokens_per_second = effective_bw / (double)bytes_per_token;
    plan->estimated_memory_pressure =
        (double)mp->total_model_bytes / (double)total_capacity;

    snprintf(plan->strategy_summary, sizeof(plan->strategy_summary),
        "VRAM-resident: %zu ops | Host-streamed: %zu ops | NVMe-streamed: %zu ops",
        0UL, 0UL, 0UL);

    size_t in_vram = 0, in_host = 0, in_nvme = 0;
    for (size_t i = 0; i < plan->n_plans; i++) {
        switch (plan->plans[i].strategy) {
            case VBS_STRATEGY_IN_VRAM: in_vram++; break;
            case VBS_STRATEGY_HOST_STREAMED: in_host++; break;
            case VBS_STRATEGY_NVME_STREAMED: in_nvme++; break;
            default: break;
        }
    }
    snprintf(plan->strategy_summary, sizeof(plan->strategy_summary),
        "VRAM: %zu ops | Host: %zu ops | NVMe: %zu ops",
        in_vram, in_host, in_nvme);

    return plan;
}

void vbs_planner_free(vbs_execution_plan_t *plan) {
    if (!plan) return;
    free(plan->plans);
    free(plan);
}

void vbs_planner_print(const vbs_execution_plan_t *plan) {
    printf("\n=== Verum Autonomous Execution Plan ===\n");
    printf("Primary device:    %s\n", plan->primary_device ? plan->primary_device->name : "CPU");
    printf("VRAM budget:       %.2f GiB per device\n", plan->vram_budget_per_device / (1024.0 * 1024 * 1024));
    printf("Host cache budget: %.2f GiB\n", plan->host_cache_budget / (1024.0 * 1024 * 1024));
    printf("NVMe overflow:     %.2f GiB\n", plan->nvme_overflow_budget / (1024.0 * 1024 * 1024));
    printf("Strategy mix:      %s\n", plan->strategy_summary);
    printf("Memory pressure:   %.2fx model fits in available tiers\n",
        1.0 / (plan->estimated_memory_pressure > 0 ? plan->estimated_memory_pressure : 1.0));
    printf("Theoretical throughput: %.1f tokens/sec\n", plan->estimated_tokens_per_second);
    printf("=======================================\n\n");
}
