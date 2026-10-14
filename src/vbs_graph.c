#include "vbs_graph.h"
#include "vbs_sha256.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

typedef enum {
    ROLE_UNKNOWN, ROLE_EMBED, ROLE_NORM, ROLE_QUERY, ROLE_KEY, ROLE_VALUE,
    ROLE_ATTN_OUT, ROLE_GATE, ROLE_UP, ROLE_DOWN, ROLE_CONV, ROLE_STATE, ROLE_PROJECTION
} tensor_role_t;

static tensor_role_t role_for_name(const char *name) {
    char buf[256];
    size_t len = strlen(name);
    if (len >= sizeof(buf)) len = sizeof(buf) - 1;
    for (size_t i = 0; i < len; i++) buf[i] = (char)tolower((unsigned char)name[i]);
    buf[len] = '\0';

    if (strstr(buf, "token") || strstr(buf, "tok") || strstr(buf, "embed") || strstr(buf, "embd")) return ROLE_EMBED;
    if (strstr(buf, "norm") || strstr(buf, "ln") || strstr(buf, "rms")) return ROLE_NORM;
    if (strstr(buf, "query") || strstr(buf, ".q.") || strstr(buf, "attn_q")) return ROLE_QUERY;
    if (strstr(buf, "key")   || strstr(buf, ".k.") || strstr(buf, "attn_k")) return ROLE_KEY;
    if (strstr(buf, "value") || strstr(buf, ".v.") || strstr(buf, "attn_v")) return ROLE_VALUE;
    if (strstr(buf, "gate")) return ROLE_GATE;
    if (strstr(buf, "up") || strstr(buf, "expand")) return ROLE_UP;
    if (strstr(buf, "down") || strstr(buf, "contract")) return ROLE_DOWN;
    if (strstr(buf, "conv")) return ROLE_CONV;
    if (strstr(buf, "ssm") || strstr(buf, "state")) return ROLE_STATE;
    if (strstr(buf, "output") || strstr(buf, "proj")) return ROLE_PROJECTION;
    if (strstr(buf, "attention") || strstr(buf, "attn_output")) return ROLE_ATTN_OUT;
    return ROLE_UNKNOWN;
}

static void compute_fingerprint(vbs_graph_t *g) {
    vbs_sha256_ctx_t ctx;
    vbs_sha256_init(&ctx);
    for (size_t i = 0; i < g->n_nodes; i++) {
        char buf[256];
        int l = snprintf(buf, sizeof(buf), "%u:%s:%llu,%llu;",
                         g->nodes[i].kind, g->nodes[i].id,
                         (unsigned long long)g->nodes[i].rows,
                         (unsigned long long)g->nodes[i].columns);
        vbs_sha256_update(&ctx, buf, (size_t)l);
    }
    uint8_t digest[32];
    vbs_sha256_final(&ctx, digest);
    for (int i = 0; i < 32; i++) {
        sprintf(g->fingerprint + (i * 2), "%02x", digest[i]);
    }
    g->fingerprint[64] = '\0';
}

vbs_graph_t *vbs_graph_compile(const vbs_gguf_model_t *model, char *err_buf, size_t err_cap) {
    vbs_graph_t *g = (vbs_graph_t *)calloc(1, sizeof(vbs_graph_t));
    g->node_cap = model->n_tensors * 2 + 16;
    g->nodes = (vbs_node_t *)calloc(g->node_cap, sizeof(vbs_node_t));

    size_t unk_matrices = 0;

    for (size_t i = 0; i < model->n_tensors; i++) {
        const vbs_tensor_t *t = &model->tensors[i];
        tensor_role_t r = role_for_name(t->name);

        if (r == ROLE_UNKNOWN) {
            if (t->n_dims >= 2 && t->rows > 0 && t->columns > 0) {
                unk_matrices++;
            }
            continue;
        }

        vbs_node_t *n = &g->nodes[g->n_nodes++];
        snprintf(n->id, sizeof(n->id), "%s", t->name);
        n->weight = t;
        n->rows = t->rows;
        n->columns = t->columns;

        switch (r) {
            case ROLE_EMBED:
                n->kind = VBS_OP_EMBEDDING;
                break;
            case ROLE_NORM:
                n->kind = VBS_OP_NORMALIZE;
                break;
            case ROLE_CONV:
                n->kind = VBS_OP_CONVOLUTION;
                break;
            case ROLE_STATE:
                n->kind = VBS_OP_STATE_SPACE;
                break;
            default:
                n->kind = VBS_OP_LINEAR;
                break;
        }
    }

    if (unk_matrices > 0) {
        snprintf(err_buf, err_cap, "Graph compiler refused GGUF: %zu unresolved matrix semantics", unk_matrices);
        vbs_graph_free(g);
        return NULL;
    }

    if (g->n_nodes == 0) {
        snprintf(err_buf, err_cap, "No executable operations could be derived from GGUF");
        vbs_graph_free(g);
        return NULL;
    }

    const char *arch = vbs_gguf_get_string_metadata(model, "general.architecture");
    snprintf(g->declared_arch, sizeof(g->declared_arch), "%s", arch ? arch : "unspecified");
    snprintf(g->semantic_source, sizeof(g->semantic_source), "inferred_metadata_and_geometry");

    compute_fingerprint(g);
    return g;
}

void vbs_graph_free(vbs_graph_t *g) {
    if (!g) return;
    free(g->nodes);
    free(g);
}
