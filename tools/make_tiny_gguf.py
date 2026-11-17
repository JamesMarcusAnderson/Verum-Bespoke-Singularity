#!/usr/bin/env python3
"""make_tiny_gguf.py — generate a small, valid GGUF v3 llama model for testing.

The weights are random (scaled small so logits stay finite); this is a
functional test fixture, not a trained model. It exists so the CPU inference
path, the recursive loop, and the fused-kernel builder can be exercised
end-to-end with no real model file present.

Layout matches what vbs_gguf.c parses and what vbs_cpu.c requires:
  - GGUF v3 header, metadata (llama.* keys), then 32-byte-aligned tensor data
  - Q8_0 weights, F32 norms; tensor names blk.<L>.* as the CPU loader expects

Usage:
  python3 tools/make_tiny_gguf.py [--out tiny.gguf] [--emb 512] [--ffn 1408]
                                  [--layers 2] [--heads 8] [--vocab 256]
                                  [--ctx 128] [--seed 1234]
"""
import argparse
import math
import os
import random
import struct
import sys

M_U8, M_I8, M_U16, M_I16, M_U32, M_I32, M_F32, M_BOOL, M_STRING = range(9)

GGML_Q8_0 = 8
GGML_F32 = 0


def f32_to_f16(v):
    """Round-to-nearest-even float32 -> float16 bits (matches vbs_cpu.c)."""
    bits = struct.unpack("<I", struct.pack("<f", v))[0]
    sign = (bits >> 16) & 0x8000
    exp = ((bits >> 23) & 0xFF) - 127
    mant = bits & 0x7FFFFF
    if exp > 15:
        return sign | 0x7C00
    if exp >= -14:
        h = ((exp + 15) << 10) | (mant >> 13)
        rem = mant & 0x1FFF
        if rem > 0x1000 or (rem == 0x1000 and (h & 1)):
            h += 1
        return sign | h
    if exp < -25:
        return sign
    mant |= 0x800000
    shift = -exp - 1
    h = mant >> shift
    rem = mant & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    if rem > half or (rem == half and (h & 1)):
        h += 1
    return sign | h


def quantize_q8_0(values):
    """Pack a flat float list into GGML Q8_0 blocks (32 elems, 34 bytes)."""
    assert len(values) % 32 == 0
    out = bytearray()
    for b in range(0, len(values), 32):
        blk = values[b:b + 32]
        amax = max(abs(x) for x in blk)
        d = amax / 127.0 if amax else 0.0
        out += struct.pack("<H", f32_to_f16(d))
        inv = 1.0 / d if d else 0.0
        for x in blk:
            q = int(round(x * inv))
            q = max(-127, min(127, q))
            out += struct.pack("<b", q)
    return bytes(out)


def gguf_string(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def meta_u32(key, val):
    return gguf_string(key) + struct.pack("<I", M_U32) + struct.pack("<I", val)


def meta_f32(key, val):
    return gguf_string(key) + struct.pack("<I", M_F32) + struct.pack("<f", val)


def meta_str(key, val):
    return gguf_string(key) + struct.pack("<I", M_STRING) + gguf_string(val)


def build(path, emb, ffn, layers, heads, vocab, ctx, seed):
    rng = random.Random(seed)
    kv_heads = heads

    def rand_weights(n, scale=0.02):
        return [rng.gauss(0.0, scale) for _ in range(n)]

    tensors = []  # (name, type, dims[list], data bytes)

    def add_q8(name, rows, cols):
        tensors.append((name, GGML_Q8_0, [cols, rows],
                        quantize_q8_0(rand_weights(rows * cols))))

    def add_f32_1d(name, n, fill=None):
        vals = fill if fill is not None else [1.0] * n
        tensors.append((name, GGML_F32, [n], struct.pack("<%df" % n, *vals)))

    add_q8("token_embd.weight", vocab, emb)
    add_f32_1d("output_norm.weight", emb)
    add_q8("output.weight", vocab, emb)
    for L in range(layers):
        add_f32_1d("blk.%d.attn_norm.weight" % L, emb)
        add_q8("blk.%d.attn_q.weight" % L, emb, emb)
        add_q8("blk.%d.attn_k.weight" % L, kv_heads * (emb // heads), emb)
        add_q8("blk.%d.attn_v.weight" % L, kv_heads * (emb // heads), emb)
        add_q8("blk.%d.attn_output.weight" % L, emb, emb)
        add_f32_1d("blk.%d.ffn_norm.weight" % L, emb)
        add_q8("blk.%d.ffn_gate.weight" % L, ffn, emb)
        add_q8("blk.%d.ffn_up.weight" % L, ffn, emb)
        add_q8("blk.%d.ffn_down.weight" % L, emb, ffn)

    metadata = b"".join([
        meta_str("general.architecture", "llama"),
        meta_u32("general.alignment", 32),
        meta_u32("llama.block_count", layers),
        meta_u32("llama.context_length", ctx),
        meta_u32("llama.embedding_length", emb),
        meta_u32("llama.feed_forward_length", ffn),
        meta_u32("llama.attention.head_count", heads),
        meta_u32("llama.attention.head_count_kv", kv_heads),
        meta_f32("llama.rope.freq_base", 10000.0),
        meta_f32("llama.attention.layer_norm_rms_epsilon", 1e-5),
        meta_u32("llama.vocab_size", vocab),
        meta_u32("tokenizer.ggml.eos_token_id", vocab - 1),
    ])

    header = struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), 12)

    # Tensor infos first (offsets relative to aligned data start).
    info = bytearray()
    data_blobs = []
    rel = 0
    for name, typ, dims, data in tensors:
        if rel % 32:
            pad = 32 - (rel % 32)
            data_blobs.append(b"\x00" * pad)
            rel += pad
        info += gguf_string(name)
        info += struct.pack("<I", len(dims))
        for d in dims:
            info += struct.pack("<Q", d)
        info += struct.pack("<I", typ)
        info += struct.pack("<Q", rel)
        data_blobs.append(data)
        rel += len(data)

    pre_data = header + metadata + bytes(info)
    data_start = (len(pre_data) + 31) // 32 * 32
    with open(path, "wb") as f:
        f.write(pre_data)
        f.write(b"\x00" * (data_start - len(pre_data)))
        for blob in data_blobs:
            f.write(blob)
    return os.path.getsize(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tiny.gguf")
    ap.add_argument("--emb", type=int, default=512)
    ap.add_argument("--ffn", type=int, default=1408)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    if a.emb % a.heads or a.emb % 32 or a.ffn % 32:
        sys.exit("emb must be divisible by heads and 32; ffn by 32")
    size = build(a.out, a.emb, a.ffn, a.layers, a.heads, a.vocab, a.ctx, a.seed)
    print("wrote %s (%d bytes) emb=%d ffn=%d layers=%d heads=%d vocab=%d ctx=%d"
          % (a.out, size, a.emb, a.ffn, a.layers, a.heads, a.vocab, a.ctx))


if __name__ == "__main__":
    main()
