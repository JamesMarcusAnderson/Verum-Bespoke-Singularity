#ifndef VBS_GRAPH_H
#define VBS_GRAPH_H

#include "vbs_model.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Operation kinds the graph compiler can derive from a model file.
 * The engine's execution path only handles a subset today (embedding,
 * linear, normalize); the rest are recognized so the graph can name
 * them, and the specializer refuses what it cannot run. */
typedef enum {
    VBS_OP_EMBEDDING,
    VBS_OP_LINEAR,
    VBS_OP_NORMALIZE,
    VBS_OP_ATTENTION,
    VBS_OP_ROTARY,
    VBS_OP_ACTIVATION,
    VBS_OP_ELEMENTWISE,
    VBS_OP_STATE_SPACE,
    VBS_OP_CONVOLUTION,
    VBS_OP_CACHE,
    VBS_OP_PROJECTION
} vbs_op_kind_t;

/* One executable operation, built from one model tensor. */
typedef struct {
    char id[128];
    vbs_op_kind_t kind;
    const vbs_tensor_t *weight;
    uint32_t tensor_index; /* index of weight in the model's tensor array (fabric registration order) */
    char inputs[4][128];
    size_t n_inputs;
    char outputs[2][128];
    size_t n_outputs;
    uint64_t rows;
    uint64_t columns;
} vbs_node_t;

/* The compiled graph: nodes plus provenance. */
typedef struct {
    vbs_node_t *nodes;
    size_t n_nodes;
    size_t node_cap;
    char fingerprint[65];    /* sha256 hex over (kind:id:rows,columns;) per node */
    char declared_arch[64];  /* "general.architecture" metadata, or "unspecified" */
    char semantic_source[64];/* how the roles were derived */
} vbs_graph_t;

/* Derive an execution graph from a loaded model. Returns NULL (with
 * err_buf filled) when the model has unresolved matrix semantics or
 * yields no executable operations. */
vbs_graph_t *vbs_graph_compile(const vbs_model_t *model,
                               char *err_buf, size_t err_cap);
void vbs_graph_free(vbs_graph_t *graph);

#ifdef __cplusplus
}
#endif

#endif /* VBS_GRAPH_H */
