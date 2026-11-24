/* model_e2e_test.c — regression test for the model loader +
 * dequantizer + CPU forward pass, against a tiny synthetic model
 * built in memory.
 *
 * No model file is committed: the test writes a minimal
 * llama-arch model file (2 layers, emb 256, vocab 128) to a temp
 * file, loads it through vbs_cpu_open, runs a few inference
 * steps, and checks the logits are finite and bit-identical
 * across two runs. The tensor mix deliberately covers the quant
 * types the CPU path must convert at load (Q8_K, Q4_K, Q6_K,
 * Q5_K, Q5_0) as well as the natively supported ones (Q8_0, Q4_0,
 * F32), so a dequantizer layout regression fails loudly here
 * instead of silently producing garbage on a real model.
 *
 * Build: gcc -std=c11 -Wall -Wextra -O2 -Iinclude -D_GNU_SOURCE
 *        tests/model_e2e_test.c src/vbs_cpu.c src/vbs_model.c
 *        src/vbs_dequant.c -lm -o model_e2e_test
 */

#include "vbs_cpu.h"
#include "vbs_model.h"
#include "vbs_dequant.h"
#include "vbs_port.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* --- deterministic PRNG (xorshift32, canonical step in vbs_port.h) --- */
static uint32_t rng_state = 0x12345678u;

static int rand_range(int span, int lo) {
    return (int)(vbs_xorshift32_next(&rng_state) % (uint32_t)span) + lo;
}
static uint8_t rand_byte(void) {
    return (uint8_t)(vbs_xorshift32_next(&rng_state) & 0xFF);
}

/* --- growing byte buffer for the model file --- */
typedef struct {
    uint8_t *buf;
    size_t len, cap;
} writer_t;

static void w_append(writer_t *w, const void *p, size_t n) {
    if (w->len + n > w->cap) {
        w->cap = (w->len + n) * 2 + 256;
        w->buf = (uint8_t *)realloc(w->buf, w->cap);
    }
    memcpy(w->buf + w->len, p, n);
    w->len += n;
}
static void w_u32(writer_t *w, uint32_t v) { w_append(w, &v, 4); }
static void w_u64(writer_t *w, uint64_t v) { w_append(w, &v, 8); }
static void w_f32(writer_t *w, float v) { w_append(w, &v, 4); }
static void w_str(writer_t *w, const char *s) {
    w_u64(w, (uint64_t)strlen(s));
    w_append(w, s, strlen(s));
}
static void w_meta_u32(writer_t *w, const char *key, uint32_t v) {
    w_str(w, key);
    w_u32(w, 4);
    w_u32(w, v);
}
static void w_meta_f32(writer_t *w, const char *key, float v) {
    w_str(w, key);
    w_u32(w, 6);
    w_f32(w, v);
}
static void w_meta_str(writer_t *w, const char *key, const char *v) {
    w_str(w, key);
    w_u32(w, 8);
    w_str(w, v);
}

#define EMB 256
#define HEADS 4
#define KV_HEADS 2
#define HEAD_DIM 64
#define FFN 512
#define VOCAB 128
#define LAYERS 2
#define CTX 64

typedef struct {
    const char *name;
    uint32_t type;
    uint64_t dims[2];
    uint32_t n_dims;
    uint8_t *data; /* packed tensor bytes */
    size_t data_len;
} synth_tensor_t;

static synth_tensor_t tensors[64];
static size_t n_tensors;

/* Store a half-precision scale at the head of a block. */
static void put_scale(uint8_t *b, float v) {
    uint16_t h = vbs_f32_to_f16(v);
    memcpy(b, &h, 2);
}

/* Fill one packed block of `type` with deterministic pseudo-random
 * quants and small scales, so dequantized weights stay in a sane
 * range. One function per block layout; the RNG consumption order
 * matches the original writer so the synthetic model is unchanged. */
static void fill_q8_0(uint8_t *b) {
    put_scale(b, 0.02f);
    for (int i = 0; i < 32; i++) b[2 + i] = (uint8_t)(int8_t)rand_range(201, -100);
}
static void fill_q4_0(uint8_t *b) {
    put_scale(b, 0.02f);
    for (int i = 0; i < 16; i++) b[2 + i] = rand_byte();
}
static void fill_q5_0(uint8_t *b) {
    put_scale(b, 0.02f);
    uint32_t qh = vbs_xorshift32_next(&rng_state);
    memcpy(b + 2, &qh, 4);
    for (int i = 0; i < 16; i++) b[6 + i] = rand_byte();
}
static void fill_q8_k(uint8_t *b) {
    float d = 0.02f;
    memcpy(b, &d, 4);
    for (int i = 0; i < 256; i++) b[4 + i] = (uint8_t)(int8_t)rand_range(201, -100);
    /* Block sums: the sum of each 16-element group, stored at +260. */
    for (int j = 0; j < 16; j++) {
        int sum = 0;
        for (int i = 0; i < 16; i++) sum += (int8_t)b[4 + j * 16 + i];
        int16_t s = (int16_t)sum;
        memcpy(b + 260 + j * 2, &s, 2);
    }
}
static void fill_q4_k(uint8_t *b) {
    put_scale(b, 0.05f);
    put_scale(b + 2, 0.01f);
    for (int i = 0; i < 12; i++) b[4 + i] = rand_byte();
    for (int i = 0; i < 128; i++) b[16 + i] = rand_byte();
}
static void fill_q5_k(uint8_t *b) {
    put_scale(b, 0.05f);
    put_scale(b + 2, 0.01f);
    for (int i = 0; i < 12; i++) b[4 + i] = rand_byte();
    for (int i = 0; i < 32; i++) b[16 + i] = rand_byte();
    for (int i = 0; i < 128; i++) b[48 + i] = rand_byte();
}
static void fill_q6_k(uint8_t *b) {
    for (int i = 0; i < 128; i++) b[i] = rand_byte();
    for (int i = 0; i < 64; i++) b[128 + i] = rand_byte();
    for (int i = 0; i < 16; i++) b[192 + i] = (uint8_t)(int8_t)rand_range(61, -30);
    put_scale(b + 208, 0.02f);
}

static void fill_block(vbs_model_type_t type, uint8_t *b) {
    switch (type) {
        case VBS_MODEL_TYPE_Q8_0: fill_q8_0(b); break;
        case VBS_MODEL_TYPE_Q4_0: fill_q4_0(b); break;
        case VBS_MODEL_TYPE_Q5_0: fill_q5_0(b); break;
        case VBS_MODEL_TYPE_Q8_K: fill_q8_k(b); break;
        case VBS_MODEL_TYPE_Q4_K: fill_q4_k(b); break;
        case VBS_MODEL_TYPE_Q5_K: fill_q5_k(b); break;
        case VBS_MODEL_TYPE_Q6_K: fill_q6_k(b); break;
        default: {
            vbs_block_geometry_t g;
            vbs_get_geometry(type, &g);
            memset(b, 0, g.bytes_per_block);
            break;
        }
    }
}

static void add_tensor(const char *name, vbs_model_type_t type,
                       uint64_t cols, uint64_t rows) {
    synth_tensor_t *t = &tensors[n_tensors++];
    t->name = name;
    t->type = (uint32_t)type;
    t->dims[0] = cols;
    t->dims[1] = rows;
    t->n_dims = rows > 1 ? 2 : 1;
    if (type == VBS_MODEL_TYPE_F32) {
        t->data_len = (size_t)(cols * rows * 4);
        t->data = (uint8_t *)malloc(t->data_len);
        /* Norm weights near 1.0. */
        float *f = (float *)t->data;
        for (uint64_t i = 0; i < cols * rows; i++)
            f[i] = 0.9f + 0.2f * (float)(vbs_xorshift32_next(&rng_state) % 1000) / 1000.0f;
    } else {
        vbs_block_geometry_t g;
        vbs_get_geometry(type, &g);
        uint64_t blocks = (cols / g.elements_per_block) * rows;
        t->data_len = (size_t)(blocks * g.bytes_per_block);
        t->data = (uint8_t *)malloc(t->data_len);
        for (uint64_t i = 0; i < blocks; i++)
            fill_block(type, t->data + (size_t)(i * g.bytes_per_block));
    }
}

/* Per-layer tensor set, in file order. */
typedef struct {
    const char *suffix;
    vbs_model_type_t type;
    uint64_t cols, rows;
} layer_tensor_t;

static const layer_tensor_t k_layer_tensors[] = {
    { "attn_norm.weight",   VBS_MODEL_TYPE_F32,  EMB, 1 },
    { "attn_q.weight",      VBS_MODEL_TYPE_Q8_K, EMB, EMB },
    { "attn_k.weight",      VBS_MODEL_TYPE_Q4_K, EMB, KV_HEADS * HEAD_DIM },
    { "attn_v.weight",      VBS_MODEL_TYPE_Q6_K, EMB, KV_HEADS * HEAD_DIM },
    { "attn_output.weight", VBS_MODEL_TYPE_Q5_0, EMB, EMB },
    { "ffn_norm.weight",    VBS_MODEL_TYPE_F32,  EMB, 1 },
    { "ffn_gate.weight",    VBS_MODEL_TYPE_Q4_0, EMB, FFN },
    { "ffn_up.weight",      VBS_MODEL_TYPE_Q8_0, EMB, FFN },
    { "ffn_down.weight",    VBS_MODEL_TYPE_Q5_K, FFN, EMB },
};

static char *build_model(size_t *out_len) {
    add_tensor("token_embd.weight", VBS_MODEL_TYPE_Q8_0, EMB, VOCAB);
    add_tensor("output_norm.weight", VBS_MODEL_TYPE_F32, EMB, 1);
    /* No output.weight: the loader must tie output to the embedding. */
    for (size_t l = 0; l < LAYERS; l++) {
        for (size_t k = 0; k < sizeof(k_layer_tensors) / sizeof(k_layer_tensors[0]); k++) {
            char namebuf[128];
            snprintf(namebuf, sizeof(namebuf), "blk.%zu.%s", l, k_layer_tensors[k].suffix);
            add_tensor(strdup(namebuf), k_layer_tensors[k].type,
                       k_layer_tensors[k].cols, k_layer_tensors[k].rows);
        }
    }

    writer_t w = {0};
    /* "GGUF" magic — the format's file signature (kept interface name). */
    w_append(&w, "GGUF", 4);
    w_u32(&w, 3); /* version */
    w_u64(&w, (uint64_t)n_tensors);
    w_u64(&w, 12); /* metadata count */
    w_meta_str(&w, "general.architecture", "llama");
    w_meta_u32(&w, "llama.block_count", LAYERS);
    w_meta_u32(&w, "llama.context_length", CTX);
    w_meta_u32(&w, "llama.embedding_length", EMB);
    w_meta_u32(&w, "llama.feed_forward_length", FFN);
    w_meta_u32(&w, "llama.attention.head_count", HEADS);
    w_meta_u32(&w, "llama.attention.head_count_kv", KV_HEADS);
    w_meta_f32(&w, "llama.rope.freq_base", 10000.0f);
    w_meta_f32(&w, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    w_meta_u32(&w, "llama.vocab_size", VOCAB);
    w_meta_u32(&w, "llama.rope.dimension_count", HEAD_DIM);
    w_meta_u32(&w, "tokenizer.ggml.eos_token_id", 2);

    /* Tensor infos first, then data (each tensor 32-byte aligned). */
    uint64_t rel = 0;
    uint64_t rels[64];
    for (size_t i = 0; i < n_tensors; i++) {
        rel = (rel + 31) & ~(uint64_t)31;
        rels[i] = rel;
        rel += tensors[i].data_len;
    }
    for (size_t i = 0; i < n_tensors; i++) {
        w_str(&w, tensors[i].name);
        w_u32(&w, tensors[i].n_dims);
        for (uint32_t d = 0; d < tensors[i].n_dims; d++) w_u64(&w, tensors[i].dims[d]);
        w_u32(&w, tensors[i].type);
        w_u64(&w, rels[i]);
    }
    /* Data section starts at the next 32-byte boundary; each tensor
     * is 32-aligned too, matching the rels[] computed above. */
    while (w.len % 32) {
        uint8_t z = 0;
        w_append(&w, &z, 1);
    }
    for (size_t i = 0; i < n_tensors; i++) {
        while ((w.len % 32) != 0) {
            uint8_t z = 0;
            w_append(&w, &z, 1);
        }
        w_append(&w, tensors[i].data, tensors[i].data_len);
    }
    *out_len = w.len;
    return (char *)w.buf;
}

static int run_logits(vbs_cpu_model_t *m, const uint32_t *toks, size_t n, float *out) {
    char err[512] = {0};
    vbs_cpu_reset(m);
    for (size_t i = 0; i < n; i++) {
        if (!vbs_cpu_step_ex(m, toks[i], i + 1 == n, err, sizeof(err))) {
            fprintf(stderr, "model-e2e: step failed: %s\n", err);
            return 0;
        }
    }
    memcpy(out, vbs_cpu_logits(m), vbs_cpu_vocab_size(m) * sizeof(float));
    return 1;
}

int main(void) {
    size_t model_len = 0;
    char *model = build_model(&model_len);
    char path[] = "/tmp/vbs_model_e2e_XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) {
        perror("model-e2e: mkstemp");
        return 1;
    }
    if (write(fd, model, model_len) != (ssize_t)model_len) {
        perror("model-e2e: write");
        close(fd);
        unlink(path);
        return 1;
    }
    close(fd);

    int fails = 0;
    char err[512] = {0};
    vbs_cpu_model_t *m = vbs_cpu_open(path, 32, err, sizeof(err));
    if (!m) {
        printf("model-e2e: LOAD FAIL: %s\n", err);
        unlink(path);
        return 1;
    }
    printf("model-e2e: load OK (vocab=%zu, eos=%u)\n",
           vbs_cpu_vocab_size(m), vbs_cpu_eos_id(m));
    if (vbs_cpu_vocab_size(m) != VOCAB || vbs_cpu_eos_id(m) != 2) {
        printf("model-e2e: FAIL: wrong vocab/eos\n");
        fails++;
    }

    const uint32_t prompt[] = {1, 5, 10, 20, 42};
    float *logits1 = (float *)malloc(VOCAB * sizeof(float));
    float *logits2 = (float *)malloc(VOCAB * sizeof(float));
    if (!run_logits(m, prompt, 5, logits1)) fails++;
    if (!run_logits(m, prompt, 5, logits2)) fails++;
    if (!fails) {
        int finite = 1, identical = 1;
        float top = logits1[0];
        size_t argmax = 0;
        for (size_t i = 0; i < VOCAB; i++) {
            if (!isfinite(logits1[i])) finite = 0;
            if (memcmp(&logits1[i], &logits2[i], sizeof(float)) != 0) identical = 0;
            if (logits1[i] > top) {
                top = logits1[i];
                argmax = i;
            }
        }
        printf("model-e2e: forward OK (finite=%s deterministic=%s argmax=%zu logit=%.4f)\n",
               finite ? "yes" : "no", identical ? "yes" : "no", argmax, (double)top);
        if (!finite || !identical) fails++;
    }
    free(logits1);
    free(logits2);
    vbs_cpu_close(m);
    unlink(path);
    printf("model-e2e: %s\n", fails ? "FAILURES" : "ALL PASS");
    return fails ? 1 : 0;
}
