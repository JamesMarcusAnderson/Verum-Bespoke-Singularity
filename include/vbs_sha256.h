/* vbs_sha256.h — SHA-256 (FIPS 180-4), static inline, zero
 * dependencies beyond <stdint.h>/<stddef.h>/<string.h>.
 *
 * The round constants and initial hash values are the standard's
 * own tables (cube roots of the first 64 primes, square roots of
 * the first 8 primes); they are data from the specification, not
 * implementation code, so they appear here verbatim as the
 * standard publishes them.
 */

#ifndef VBS_SHA256_H
#define VBS_SHA256_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VBS_SHA256_BLOCK_SIZE 64
#define VBS_SHA256_DIGEST_SIZE 32

typedef struct {
    uint32_t state[8];
    uint64_t count;
    uint8_t  buffer[64];
} vbs_sha256_ctx_t;

static inline uint32_t vbs_rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

/* The six SHA-256 boolean functions (FIPS 180-4 section 4.1.2). */
static inline uint32_t sha_ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}
static inline uint32_t sha_maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}
static inline uint32_t sha_big0(uint32_t x) {
    return vbs_rotr(x, 2) ^ vbs_rotr(x, 13) ^ vbs_rotr(x, 22);
}
static inline uint32_t sha_big1(uint32_t x) {
    return vbs_rotr(x, 6) ^ vbs_rotr(x, 11) ^ vbs_rotr(x, 25);
}
static inline uint32_t sha_small0(uint32_t x) {
    return vbs_rotr(x, 7) ^ vbs_rotr(x, 18) ^ (x >> 3);
}
static inline uint32_t sha_small1(uint32_t x) {
    return vbs_rotr(x, 17) ^ vbs_rotr(x, 19) ^ (x >> 10);
}

/* Round constants K[0..63] (FIPS 180-4 section 4.2.2). */
static const uint32_t g_vbs_sha256_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

/* Compress one 64-byte block into the state. The eight working
 * variables live in an array that rotates one slot per round,
 * which is the same recurrence as the textbook a..h form. */
static inline void vbs_sha256_transform(vbs_sha256_ctx_t *ctx,
                                        const uint8_t *data) {
    uint32_t w[64];
    uint32_t v[8];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)data[i * 4] << 24) |
               ((uint32_t)data[i * 4 + 1] << 16) |
               ((uint32_t)data[i * 4 + 2] << 8) |
               ((uint32_t)data[i * 4 + 3]);
    }
    for (int i = 16; i < 64; i++)
        w[i] = sha_small1(w[i - 2]) + w[i - 7] +
               sha_small0(w[i - 15]) + w[i - 16];

    for (int i = 0; i < 8; i++) v[i] = ctx->state[i];
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = v[7] + sha_big1(v[4]) + sha_ch(v[4], v[5], v[6]) +
                      g_vbs_sha256_k[i] + w[i];
        uint32_t t2 = sha_big0(v[0]) + sha_maj(v[0], v[1], v[2]);
        for (int j = 7; j > 0; j--) v[j] = v[j - 1];
        v[4] += t1; /* e (now at index 4 after the shift) gains t1 */
        v[0] = t1 + t2;
    }
    for (int i = 0; i < 8; i++) ctx->state[i] += v[i];
}

static inline void vbs_sha256_init(vbs_sha256_ctx_t *ctx) {
    /* Initial hash values H0 (FIPS 180-4 section 5.3.3). */
    static const uint32_t h0[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    for (int i = 0; i < 8; i++) ctx->state[i] = h0[i];
    ctx->count = 0;
}

static inline void vbs_sha256_update(vbs_sha256_ctx_t *ctx, const void *data,
                                     size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    size_t buffered = (size_t)(ctx->count & 0x3F);
    ctx->count += len;

    /* Finish a partial block left over from the previous call. */
    if (buffered > 0) {
        size_t need = 64 - buffered;
        if (len < need) {
            memcpy(ctx->buffer + buffered, p, len);
            return;
        }
        memcpy(ctx->buffer + buffered, p, need);
        vbs_sha256_transform(ctx, ctx->buffer);
        p += need;
        len -= need;
    }
    /* Whole blocks straight from the input. */
    while (len >= 64) {
        vbs_sha256_transform(ctx, p);
        p += 64;
        len -= 64;
    }
    /* Whatever is left starts the next partial block. */
    if (len > 0) memcpy(ctx->buffer, p, len);
}

static inline void vbs_sha256_final(vbs_sha256_ctx_t *ctx, uint8_t digest[32]) {
    uint64_t total_bits = ctx->count * 8;
    size_t buffered = (size_t)(ctx->count & 0x3F);
    /* Padding: 0x80, then zeros until 8 bytes remain in the final
     * block for the big-endian bit length. */
    size_t pad_len = (buffered < 56) ? (56 - buffered) : (120 - buffered);
    uint8_t pad[64] = { 0x80 };
    vbs_sha256_update(ctx, pad, pad_len);
    uint8_t len_bytes[8];
    for (int i = 0; i < 8; i++)
        len_bytes[i] = (uint8_t)(total_bits >> ((7 - i) * 8));
    vbs_sha256_update(ctx, len_bytes, 8);

    for (int i = 0; i < 8; i++) {
        digest[i * 4]     = (uint8_t)(ctx->state[i] >> 24);
        digest[i * 4 + 1] = (uint8_t)(ctx->state[i] >> 16);
        digest[i * 4 + 2] = (uint8_t)(ctx->state[i] >> 8);
        digest[i * 4 + 3] = (uint8_t)(ctx->state[i]);
    }
}

#ifdef __cplusplus
}
#endif

#endif /* VBS_SHA256_H */
