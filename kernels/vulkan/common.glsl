// common.glsl — shared helpers for the Verum Vulkan compute kernels.
//
// Every kernel that reads packed weights binds them as a uint array named
// `w_words` at set 0, binding 0. Byte-addressed loads go through these
// helpers so block layouts stay byte-exact with the GGUF spec and with
// vbs_dequant.c (the CPU reference). No 8/16-bit storage extensions are
// needed: f16 values are unpacked from uint words, int8/nibbles from bytes.
//
// Included with `#include "common.glsl"` (glslangValidator -I kernels/vulkan).

// Unsigned byte at byte index `b` in w_words (little-endian words).
uint load_u8(uint b) {
    uint w = w_words[b >> 2];
    return (w >> ((b & 3u) * 8u)) & 0xFFu;
}

// Signed byte at byte index `b`.
int load_i8(uint b) {
    uint u = load_u8(b);
    return (u >= 128u) ? int(u) - 256 : int(u);
}

// f16 (little-endian, 2 bytes) at byte index `b`, converted to f32.
float load_f16(uint b) {
    uint bits = load_u8(b) | (load_u8(b + 1u) << 8);
    return unpackHalf2x16(bits).x;
}

// f32 (little-endian, 4 bytes) at byte index `b`.
float load_f32(uint b) {
    uint bits = load_u8(b) | (load_u8(b + 1u) << 8) |
                (load_u8(b + 2u) << 16) | (load_u8(b + 3u) << 24);
    return uintBitsToFloat(bits);
}

// Q4_K dequant of one element. Block layout (144 B):
//   f16 d; f16 dmin; u8 scales[12]; u8 qs[128]
// The super-block scale/min come FIRST in ggml's block_q4_K (unlike
// Q2_K/Q3_K/Q6_K, where they come last). Mirrors deq_q4_k() in
// src/vbs_dequant.c exactly.
float deq_q4k(uint base, uint e) {
    uint g = e >> 6;
    uint w = e & 63u;
    uint byte = load_u8(base + 16u + g * 32u + (w & 31u));
    uint nib = (w < 32u) ? (byte & 15u) : (byte >> 4u);
    uint j = e >> 5; // 0..7 sub-block
    // unpack_scale_min_k4(scales, j)
    uint sc, mn;
    if (j < 4u) {
        sc = load_u8(base + 4u + j) & 63u;
        mn = load_u8(base + 4u + j + 4u) & 63u;
    } else {
        sc = (load_u8(base + 4u + j + 4u) & 15u) |
             ((load_u8(base + 4u + j - 4u) >> 6) << 4);
        mn = (load_u8(base + 4u + j + 4u) >> 4) |
             ((load_u8(base + 4u + j) >> 6) << 4);
    }
    float d = load_f16(base + 0u);
    float dmn = load_f16(base + 2u);
    return d * float(sc) * float(nib) - dmn * float(mn);
}

// Q6_K dequant of one element. Block layout (210 B):
//   u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d
// Mirrors deq_q6_k() in src/vbs_dequant.c exactly.
float deq_q6k(uint base, uint e) {
    uint h = e >> 7;
    uint q = (e & 127u) >> 5;
    uint l = e & 31u;
    uint qi = (q < 2u) ? (h * 64u + q * 32u + l)
                       : (h * 64u + (q - 2u) * 32u + l);
    uint shift = (q < 2u) ? 0u : 4u;
    uint lo = (load_u8(base + qi) >> shift) & 15u;
    uint hi = (load_u8(base + 128u + h * 32u + l) >> (q * 2u)) & 3u;
    int sc = load_i8(base + 192u + h * 8u + q * 2u + (l >> 4));
    float d = load_f16(base + 208u);
    return d * float(sc) * float(int(lo | (hi << 4)) - 32);
}
