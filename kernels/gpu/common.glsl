// common.glsl — byte-level weight readers and block unpackers shared
// by the Verum GPU compute kernels.
//
// Packed weights arrive as an array of 32-bit words named `w_words`
// (set 0, binding 0). The readers below pull bytes, half floats, and
// floats out of those words, and the unpackers turn one packed block
// element back into a float. Block layouts follow the model file
// format specification exactly, so these agree bit-for-bit with the
// CPU reference dequantizers in src/vbs_dequant.c.
//
// No 16-bit storage features are used anywhere: half floats are
// unpacked from words with the standard unpack builtin, and int8
// values and nibbles are extracted from bytes.
//
// Included with `#include "common.glsl"` (the build passes
// -Ikernels/gpu to the GLSL compiler).

// One unsigned byte at byte offset `at` (words are little-endian).
uint byte_at(uint at) {
    uint word = w_words[at >> 2];
    uint shift = (at & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

// One signed byte at byte offset `at`, sign-extended to int.
int sbyte_at(uint at) {
    uint raw = byte_at(at);
    return raw < 128u ? int(raw) : int(raw) - 256;
}

// One half float (2 bytes, little-endian) at byte offset `at`.
float half_at(uint at) {
    uint packed = byte_at(at) | (byte_at(at + 1u) << 8);
    return unpackHalf2x16(packed).x;
}

// One float (4 bytes, little-endian) at byte offset `at`.
float float_at(uint at) {
    uint packed = byte_at(at) | (byte_at(at + 1u) << 8) |
                  (byte_at(at + 2u) << 16) | (byte_at(at + 3u) << 24);
    return uintBitsToFloat(packed);
}

// Q8_0 element `e` of the block starting at byte `base`.
// Block (34 bytes): half scale, then 32 signed bytes.
float unpack_q8_0(uint base, uint e) {
    return half_at(base) * float(sbyte_at(base + 2u + e));
}

// Q4_0 element `e` of the block starting at byte `base`.
// Block (18 bytes): half scale, then 16 bytes holding 32 nibbles —
// elements 0..15 in the low nibbles, 16..31 in the high nibbles.
// Each nibble is biased by 8.
float unpack_q4_0(uint base, uint e) {
    float scale = half_at(base);
    uint packed = byte_at(base + 2u + (e & 15u));
    uint nibble = e < 16u ? (packed & 15u) : (packed >> 4u);
    return scale * (float(nibble) - 8.0);
}

// Q4_1 element `e` of the block starting at byte `base`.
// Block (20 bytes): half scale, half offset, then 16 nibble bytes
// laid out like Q4_0. Value is scale * nibble + offset.
float unpack_q4_1(uint base, uint e) {
    float scale = half_at(base);
    float offset = half_at(base + 2u);
    uint packed = byte_at(base + 4u + (e & 15u));
    uint nibble = e < 16u ? (packed & 15u) : (packed >> 4u);
    return scale * float(nibble) + offset;
}

// Pull one 6-bit field out of the Q4_K scale table. `lo` holds the
// low 4 bits (in its low or high nibble) and `hi` holds the top 2
// bits (in its top two bits).
uint field6(uint lo, uint hi, bool upper) {
    uint low_part = upper ? (lo >> 4) : (lo & 15u);
    return low_part | ((hi >> 6) << 4);
}

// Q4_K element `e` of the block starting at byte `base`.
// Block (144 bytes): half scale, half min-scale, 12 scale-table
// bytes, then 128 nibble bytes in four 32-byte groups. The block
// scale and min-scale come FIRST here (unlike Q2_K/Q3_K/Q6_K, where
// they come last). Value is d * sc * nibble - dmin * mn.
float unpack_q4k(uint base, uint e) {
    uint group = e >> 6;              // 0..3, selects a 64-element group
    uint within = e & 63u;            // position inside the group
    uint packed = byte_at(base + 16u + group * 32u + (within & 31u));
    uint nibble = within < 32u ? (packed & 15u) : (packed >> 4u);

    // Sub-block j (0..7) owns one (scale, min) pair of 6-bit fields.
    uint j = e >> 5;
    uint sc, mn;
    if (j < 4u) {
        sc = byte_at(base + 4u + j) & 63u;
        mn = byte_at(base + 8u + j) & 63u;
    } else {
        uint mid = byte_at(base + 8u + j);
        sc = field6(mid, byte_at(base + j), false);
        mn = field6(mid, byte_at(base + 4u + j), true);
    }
    float d = half_at(base);
    float dmin = half_at(base + 2u);
    return d * float(sc) * float(nibble) - dmin * float(mn);
}

// Q6_K element `e` of the block starting at byte `base`.
// Block (210 bytes): 128 low-nibble bytes, 64 high-bit bytes,
// 16 signed scale bytes, half scale last. Each element is a 6-bit
// value (4 low bits + 2 high bits) biased by 32, times its
// sub-block scale and the block scale.
float unpack_q6k(uint base, uint e) {
    uint half = e >> 7;               // 0..1, selects a 128-element half
    uint quarter = (e & 127u) >> 5;   // 0..3, selects a 32-element run
    uint lane = e & 31u;
    uint low_index = half * 64u + (quarter & 1u) * 32u + lane;
    uint low_shift = quarter < 2u ? 0u : 4u;
    uint low_bits = (byte_at(base + low_index) >> low_shift) & 15u;
    uint high_bits = (byte_at(base + 128u + half * 32u + lane) >> (quarter * 2u)) & 3u;
    int scale = sbyte_at(base + 192u + half * 8u + quarter * 2u + (lane >> 4));
    float d = half_at(base + 208u);
    int value = int(low_bits | (high_bits << 4)) - 32;
    return d * float(scale) * float(value);
}
