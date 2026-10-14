#include "vbs_backend.h"
#include "vbs_gguf.h"
#include "vbs_graph.h"
#include <stdint.h>
#include <stdbool.h>

typedef struct {
    vbs_device_t *device;
    const vbs_backend_ops_t *backend;

    // Device-resident activation buffers (Ping-pong)
    vbs_buffer_t *act_a;
    vbs_buffer_t *act_b;
    vbs_buffer_t *logits;
    vbs_buffer_t *token_io; // Holds current uint32_t token on GPU

    // VRAM KV-Cache
    vbs_buffer_t *kv_cache_k;
    vbs_buffer_t *kv_cache_v;

    // Double-buffered DMA windows for weights exceeding VRAM
    vbs_buffer_t *stage_window[2];
    vbs_buffer_t *priv_window[2];
    size_t window_bytes;
    size_t turn;

    uint32_t context_pos;
} vbs_gpu_runner_t;

vbs_gpu_runner_t *vbs_gpu_runner_init(
    vbs_device_t *dev,
    const vbs_backend_ops_t *backend,
    size_t embed_dim,
    size_t vocab_size,
    size_t max_context,
    size_t n_layers,
    size_t n_kv_heads,
    size_t head_dim,
    size_t window_bytes
) {
    vbs_gpu_runner_t *r = (vbs_gpu_runner_t *)calloc(1, sizeof(vbs_gpu_runner_t));
    r->device = dev;
    r->backend = backend;
    r->window_bytes = window_bytes;

    // 1. Allocate on-device activation buffers (Private VRAM)
    r->act_a  = backend->alloc_buffer(dev, embed_dim * sizeof(float), false);
    r->act_b  = backend->alloc_buffer(dev, embed_dim * sizeof(float), false);
    r->logits = backend->alloc_buffer(dev, vocab_size * sizeof(float), false);

    // 2. Token I/O: Shared buffer for zero-latency single-integer readback
    r->token_io = backend->alloc_buffer(dev, sizeof(uint32_t), true);

    // 3. VRAM KV Cache (Half Precision: FP16)
    size_t kv_size = n_layers * max_context * n_kv_heads * head_dim * sizeof(uint16_t);
    r->kv_cache_k = backend->alloc_buffer(dev, kv_size, false);
    r->kv_cache_v = backend->alloc_buffer(dev, kv_size, false);

    // 4. Double-Buffered Execution Windows (Strict Bounded Invariant)
    for (int i = 0; i < 2; i++) {
        r->stage_window[i] = backend->alloc_buffer(dev, window_bytes, true);  // Host Shared
        r->priv_window[i]  = backend->alloc_buffer(dev, window_bytes, false); // GPU Private
    }

    return r;
}

// Executes one autoregressive generation cycle strictly on the GPU
uint32_t vbs_gpu_step(
    vbs_gpu_runner_t *r,
    const vbs_graph_t *graph,
    const vbs_gguf_model_t *model,
    uint32_t input_token,
    float temperature,
    float min_p,
    uint32_t seed
) {
    // 1. Stage initial token to GPU register buffer
    *((uint32_t *)r->token_io->host_ptr) = input_token;

    // 2. Dispatch operations strictly on device
    vbs_buffer_t *cur_in  = r->act_a;
    vbs_buffer_t *cur_out = r->act_b;

    for (size_t i = 0; i < graph->n_nodes; i++) {
        const vbs_node_t *node = &graph->nodes[i];

        switch (node->kind) {
            case VBS_OP_EMBEDDING:
                // GPU-side embedding table lookup: token_io -> cur_in
                break;

            case VBS_OP_NORMALIZE:
                // RMSNorm: cur_in -> cur_out (on-chip reduction)
                break;

            case VBS_OP_ROTARY:
                // RoPE in-place angle rotation on Q/K heads
                break;

            case VBS_OP_ATTENTION:
                // GQA Attention Kernel over VRAM KV Cache
                break;

            case VBS_OP_ACTIVATION:
                // SwiGLU: silu(gate) * up
                break;

            case VBS_OP_ELEMENTWISE:
                // Residual Add: cur_out += cur_in
                break;

            case VBS_OP_LINEAR:
                // Double-buffered 2D slice execution: streams weight tiles directly
                // through private window without storing intermediate weights
                break;

            default:
                break;
        }

        // Ping-pong activation pointers on device without copying bytes
        vbs_buffer_t *tmp = cur_in;
        cur_in = cur_out;
        cur_out = tmp;
    }

    // 3. GPU-Side Sampling Kernel: Evaluates logits directly in VRAM and
    // writes exactly 1 token ID to r->token_io
    // (Metal: sample_gumbel / min_p_mask / argmax_write_token)
    // (CUDA:  sample_gumbel / argmax_write_token PTX)

    // 4. Synchronize only for the single output token integer
    r->backend->synchronize_device(r->device);
    r->context_pos++;

    return *((uint32_t *)r->token_io->host_ptr);
}
