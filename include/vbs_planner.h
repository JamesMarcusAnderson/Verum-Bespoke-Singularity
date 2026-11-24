#ifndef VBS_PLANNER_H
#define VBS_PLANNER_H

#include "vbs_preamble.h"
#include "vbs_graph.h"
#include "vbs_profiler.h"
#include "vbs_fabric.h"

VBS_EXTERN_C_BEGIN

/* Where a tensor's weights live during execution. */
typedef enum {
    VBS_STRATEGY_IN_VRAM,       /* fits entirely in VRAM, no streaming */
    VBS_STRATEGY_PEER_SHARED,   /* split across multiple GPUs' VRAM via peer access */
    VBS_STRATEGY_HOST_STREAMED, /* weights in system RAM, DMA tiles to GPU */
    VBS_STRATEGY_NVME_STREAMED, /* weights on NVMe, double-buffer through RAM to GPU */
    VBS_STRATEGY_RECOMPUTE      /* keep nothing resident, recompute dequant each pass */
} vbs_execution_strategy_t;

/* The execution decision for one graph node. */
typedef struct {
    const vbs_node_t *node;
    vbs_execution_strategy_t strategy;
    vbs_device_t *target_device;
    vbs_memory_tier_t resident_tier;

    uint32_t tile_rows;
    uint32_t tile_cols;
    uint32_t prefetch_depth;

    bool use_tensor_cores;
    bool fuse_with_next; /* kernel fusion opportunity */
    uint32_t fused_group_id;

    double estimated_latency_ms;
    double estimated_bandwidth_gbps;
    uint64_t estimated_vram_footprint;
} vbs_op_plan_t;

/* The whole-model plan: one op plan per node plus global budgets
 * and throughput estimates. */
typedef struct {
    const vbs_graph_t *graph;
    const vbs_system_profile_t *sys;
    const vbs_model_profile_t *mp;

    size_t n_plans;
    vbs_op_plan_t *plans;

    vbs_device_t *primary_device;
    uint64_t vram_budget_per_device;
    uint64_t host_cache_budget;
    uint64_t nvme_overflow_budget;

    double estimated_tokens_per_second;
    double estimated_memory_pressure;
    char strategy_summary[256];
} vbs_execution_plan_t;

vbs_execution_plan_t *vbs_planner_build(
    const vbs_graph_t *graph,
    const vbs_system_profile_t *sys,
    const vbs_model_profile_t *mp,
    vbs_fabric_t *fabric);

void vbs_planner_free(vbs_execution_plan_t *plan);
void vbs_planner_print(const vbs_execution_plan_t *plan);

VBS_EXTERN_C_END

#endif /* VBS_PLANNER_H */
