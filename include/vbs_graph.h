#ifndef VBS_GRAPH_H
#define VBS_GRAPH_H

#include "vbs_gguf.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

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

typedef struct {
    char id[128];
    vbs_op_kind_t kind;
    const vbs_tensor_t *weight;
    uint32_t tensor_index; // index of weight in the model's tensor array (fabric registration order)
    char inputs[4][128];
    size_t n_inputs;
    char outputs[2][128];
    size_t n_outputs;
    uint64_t rows;
    uint64_t columns;
} vbs_node_t;

typedef struct {
    vbs_node_t *nodes;
    size_t n_nodes;
    size_t node_cap;
    char fingerprint[65];
    char declared_arch[64];
    char semantic_source[64];
} vbs_graph_t;

vbs_graph_t *vbs_graph_compile(const vbs_gguf_model_t *model, char *err_buf, size_t err_cap);
void vbs_graph_free(vbs_graph_t *graph);

#ifdef __cplusplus
}
#endif

#endif // VBS_GRAPH_H
