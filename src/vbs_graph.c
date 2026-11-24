/* vbs_graph.c — derive an execution graph from a model file.
 *
 * Each tensor gets a role from its name (lowercased, substring
 * matched), and each role maps to an operation kind. Tensors whose
 * role cannot be determined are skipped — unless they are matrices,
 * in which case the whole model is refused: a matrix the engine
 * cannot name is a matrix it cannot execute, and silently dropping
 * it would change the model's math. The graph's fingerprint is a
 * SHA-256 over every node's kind, id, and geometry, so two models
 * that compile to the same graph have the same fingerprint.
 */

#include "vbs_graph.h"
#include "vbs_sha256.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

typedef enum {
    ROLE_UNKNOWN,
    ROLE_EMBED,
    ROLE_NORM,
    ROLE_QUERY,
    ROLE_KEY,
    ROLE_VALUE,
    ROLE_ATTN_OUT,
    ROLE_GATE,
    ROLE_UP,
    ROLE_DOWN,
    ROLE_CONV,
    ROLE_STATE,
    ROLE_PROJECTION
} tensor_role_t;

/* One substring rule: if the lowercased name contains `needle`, the
 * tensor has `role`. Rules are tried in table order and the first
 * match wins, so the order below is part of the behavior (e.g.
 * "attn_output" must reach the "output" rule before the
 * "attn_output" rule — the former wins and the tensor becomes a
 * projection, matching the original classifier). */
typedef struct {
    const char *needle;
    tensor_role_t role;
} role_rule_t;

static const role_rule_t k_role_rules[] = {
    { "token", ROLE_EMBED },
    { "tok", ROLE_EMBED },
    { "embed", ROLE_EMBED },
    { "embd", ROLE_EMBED },
    { "norm", ROLE_NORM },
    { "ln", ROLE_NORM },
    { "rms", ROLE_NORM },
    { "query", ROLE_QUERY },
    { ".q.", ROLE_QUERY },
    { "attn_q", ROLE_QUERY },
    { "key", ROLE_KEY },
    { ".k.", ROLE_KEY },
    { "attn_k", ROLE_KEY },
    { "value", ROLE_VALUE },
    { ".v.", ROLE_VALUE },
    { "attn_v", ROLE_VALUE },
    { "gate", ROLE_GATE },
    { "up", ROLE_UP },
    { "expand", ROLE_UP },
    { "down", ROLE_DOWN },
    { "contract", ROLE_DOWN },
    { "conv", ROLE_CONV },
    { "ssm", ROLE_STATE },
    { "state", ROLE_STATE },
    { "output", ROLE_PROJECTION },
    { "proj", ROLE_PROJECTION },
    { "attention", ROLE_ATTN_OUT },
    { "attn_output", ROLE_ATTN_OUT },
};

static tensor_role_t classify_tensor(const char *name) {
    char lowered[256];
    size_t len = strlen(name);
    if (len >= sizeof(lowered)) len = sizeof(lowered) - 1;
    for (size_t i = 0; i < len; i++)
        lowered[i] = (char)tolower((unsigned char)name[i]);
    lowered[len] = '\0';
    for (size_t i = 0; i < sizeof(k_role_rules) / sizeof(k_role_rules[0]); i++) {
        if (strstr(lowered, k_role_rules[i].needle))
            return k_role_rules[i].role;
    }
    return ROLE_UNKNOWN;
}

/* The op kind a role executes as. Most roles are plain linear maps;
 * only embedding, normalization, convolution, and state-space get
 * their own kinds. */
static vbs_op_kind_t kind_for_role(tensor_role_t role) {
    switch (role) {
    case ROLE_EMBED: return VBS_OP_EMBEDDING;
    case ROLE_NORM: return VBS_OP_NORMALIZE;
    case ROLE_CONV: return VBS_OP_CONVOLUTION;
    case ROLE_STATE: return VBS_OP_STATE_SPACE;
    default: return VBS_OP_LINEAR;
    }
}

static void fingerprint_graph(vbs_graph_t *g) {
    vbs_sha256_ctx_t ctx;
    vbs_sha256_init(&ctx);
    for (size_t i = 0; i < g->n_nodes; i++) {
        char record[256];
        int written = snprintf(record, sizeof(record), "%u:%s:%llu,%llu;",
                               g->nodes[i].kind, g->nodes[i].id,
                               (unsigned long long)g->nodes[i].rows,
                               (unsigned long long)g->nodes[i].columns);
        vbs_sha256_update(&ctx, record, (size_t)written);
    }
    uint8_t digest[32];
    vbs_sha256_final(&ctx, digest);
    for (int i = 0; i < 32; i++)
        sprintf(g->fingerprint + (i * 2), "%02x", digest[i]);
    g->fingerprint[64] = '\0';
}

vbs_graph_t *vbs_graph_compile(const vbs_model_t *model,
                               char *err_buf, size_t err_cap) {
    vbs_graph_t *g = (vbs_graph_t *)calloc(1, sizeof(vbs_graph_t));
    if (!g) return NULL;
    g->node_cap = model->n_tensors * 2 + 16;
    g->nodes = (vbs_node_t *)calloc(g->node_cap, sizeof(vbs_node_t));
    if (!g->nodes) {
        free(g);
        return NULL;
    }

    size_t unknown_matrices = 0;
    for (size_t i = 0; i < model->n_tensors; i++) {
        const vbs_tensor_t *t = &model->tensors[i];
        tensor_role_t role = classify_tensor(t->name);
        if (role == ROLE_UNKNOWN) {
            if (t->n_dims >= 2 && t->rows > 0 && t->columns > 0)
                unknown_matrices++;
            continue;
        }
        vbs_node_t *n = &g->nodes[g->n_nodes++];
        snprintf(n->id, sizeof(n->id), "%s", t->name);
        n->weight = t;
        n->tensor_index = (uint32_t)i;
        n->rows = t->rows;
        n->columns = t->columns;
        n->kind = kind_for_role(role);
    }

    if (unknown_matrices > 0) {
        if (err_buf)
            snprintf(err_buf, err_cap,
                     "Graph compiler refused model file: %zu unresolved matrix semantics",
                     unknown_matrices);
        vbs_graph_free(g);
        return NULL;
    }
    if (g->n_nodes == 0) {
        if (err_buf)
            snprintf(err_buf, err_cap,
                     "No executable operations could be derived from the model file");
        vbs_graph_free(g);
        return NULL;
    }

    const char *arch = vbs_model_get_string_metadata(model, "general.architecture");
    snprintf(g->declared_arch, sizeof(g->declared_arch), "%s",
             arch ? arch : "unspecified");
    snprintf(g->semantic_source, sizeof(g->semantic_source),
             "inferred_metadata_and_geometry");
    fingerprint_graph(g);
    return g;
}

void vbs_graph_free(vbs_graph_t *g) {
    if (!g) return;
    free(g->nodes);
    free(g);
}
