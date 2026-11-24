#!/usr/bin/env python3
"""make_tiny_model.py — generate a small, valid model file (GGUF v3 format) for testing.

The weights are random (scaled small so logits stay finite); this is a
functional test fixture, not a trained model. It exists so the CPU
inference path, the recursive loop, and the fused-kernel builder can be
exercised end-to-end with no real model file present.

Layout matches what vbs_model.c parses and what vbs_cpu.c requires:
  - model file header (GGUF v3), metadata (llama.* keys), then
    32-byte-aligned tensor data
  - Q8_0 weights, F32 norms; tensor names blk.<L>.* as the CPU loader expects

Usage:
  python3 tools/make_tiny_model.py [--out tiny.gguf] [--emb 512] [--ffn 1408]
                                  [--layers 2] [--heads 8] [--vocab 256]
                                  [--ctx 128] [--seed 1234]
"""
import argparse
import os
import random
import struct
import sys

# Metadata value kinds, as stored in the file.
META_U8, META_I8, META_U16, META_I16, META_U32, META_I32, META_F32, META_BOOL, META_STR = range(9)

# Format type codes (the format's vocabulary; values are the spec's).
FMT_F32 = 0
FMT_Q8_0 = 8


def float_to_half_bits(value):
    """Round-to-nearest-even float32 -> float16 bits (matches vbs_cpu.c)."""
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    sign = (bits >> 16) & 0x8000
    exponent = ((bits >> 23) & 0xFF) - 127
    mantissa = bits & 0x7FFFFF
    if exponent > 15:
        return sign | 0x7C00
    if exponent >= -14:
        half = ((exponent + 15) << 10) | (mantissa >> 13)
        remainder = mantissa & 0x1FFF
        if remainder > 0x1000 or (remainder == 0x1000 and (half & 1)):
            half += 1
        return sign | half
    if exponent < -25:
        return sign
    mantissa |= 0x800000
    shift = -exponent - 1
    half = mantissa >> shift
    remainder = mantissa & ((1 << shift) - 1)
    halfway = 1 << (shift - 1)
    if remainder > halfway or (remainder == halfway and (half & 1)):
        half += 1
    return sign | half


def pack_q8_0(values):
    """Pack a flat float list into Q8_0 blocks (32 elements, 34 bytes each)."""
    assert len(values) % 32 == 0
    packed = bytearray()
    for start in range(0, len(values), 32):
        block = values[start:start + 32]
        peak = max(abs(x) for x in block)
        scale = peak / 127.0 if peak else 0.0
        packed += struct.pack("<H", float_to_half_bits(scale))
        inverse = 1.0 / scale if scale else 0.0
        for x in block:
            q = int(round(x * inverse))
            q = max(-127, min(127, q))
            packed += struct.pack("<b", q)
    return bytes(packed)


def encode_string(text):
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def meta_u32(key, value):
    return encode_string(key) + struct.pack("<I", META_U32) + struct.pack("<I", value)


def meta_f32(key, value):
    return encode_string(key) + struct.pack("<I", META_F32) + struct.pack("<f", value)


def meta_str(key, value):
    return encode_string(key) + struct.pack("<I", META_STR) + encode_string(value)


def build(path, emb, ffn, layers, heads, vocab, ctx, seed):
    rng = random.Random(seed)
    kv_heads = heads
    tensors = []  # (name, type code, dims, data bytes)

    def random_weights(count, scale=0.02):
        return [rng.gauss(0.0, scale) for _ in range(count)]

    def add_quantized(name, rows, cols):
        tensors.append((name, FMT_Q8_0, [cols, rows],
                        pack_q8_0(random_weights(rows * cols))))

    def add_plain(name, count, fill=None):
        values = fill if fill is not None else [1.0] * count
        tensors.append((name, FMT_F32, [count],
                        struct.pack("<%df" % count, *values)))

    add_quantized("token_embd.weight", vocab, emb)
    add_plain("output_norm.weight", emb)
    add_quantized("output.weight", vocab, emb)
    for layer in range(layers):
        add_plain("blk.%d.attn_norm.weight" % layer, emb)
        add_quantized("blk.%d.attn_q.weight" % layer, emb, emb)
        add_quantized("blk.%d.attn_k.weight" % layer, kv_heads * (emb // heads), emb)
        add_quantized("blk.%d.attn_v.weight" % layer, kv_heads * (emb // heads), emb)
        add_quantized("blk.%d.attn_output.weight" % layer, emb, emb)
        add_plain("blk.%d.ffn_norm.weight" % layer, emb)
        add_quantized("blk.%d.ffn_gate.weight" % layer, ffn, emb)
        add_quantized("blk.%d.ffn_up.weight" % layer, ffn, emb)
        add_quantized("blk.%d.ffn_down.weight" % layer, emb, ffn)

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
        # "tokenizer.ggml.eos_token_id" is a metadata key stored in model
        # files — a kept interface name.
        meta_u32("tokenizer.ggml.eos_token_id", vocab - 1),
    ])

    # b"GGUF" is the format's magic bytes — a kept interface name.
    header = struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), 12)

    # Tensor records first (offsets relative to the aligned data start),
    # each tensor's data padded to a 32-byte boundary.
    records = bytearray()
    blobs = []
    offset = 0
    for name, kind, dims, data in tensors:
        if offset % 32:
            pad = 32 - (offset % 32)
            blobs.append(b"\x00" * pad)
            offset += pad
        records += encode_string(name)
        records += struct.pack("<I", len(dims))
        for dim in dims:
            records += struct.pack("<Q", dim)
        records += struct.pack("<I", kind)
        records += struct.pack("<Q", offset)
        blobs.append(data)
        offset += len(data)

    prefix = header + metadata + bytes(records)
    data_start = (len(prefix) + 31) // 32 * 32
    with open(path, "wb") as handle:
        handle.write(prefix)
        handle.write(b"\x00" * (data_start - len(prefix)))
        for blob in blobs:
            handle.write(blob)
    return os.path.getsize(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="tiny.gguf")
    parser.add_argument("--emb", type=int, default=512)
    parser.add_argument("--ffn", type=int, default=1408)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--vocab", type=int, default=256)
    parser.add_argument("--ctx", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.emb % args.heads or args.emb % 32 or args.ffn % 32:
        sys.exit("emb must be divisible by heads and 32; ffn by 32")
    size = build(args.out, args.emb, args.ffn, args.layers, args.heads,
                 args.vocab, args.ctx, args.seed)
    print("wrote %s (%d bytes) emb=%d ffn=%d layers=%d heads=%d vocab=%d ctx=%d"
          % (args.out, size, args.emb, args.ffn, args.layers, args.heads,
             args.vocab, args.ctx))


if __name__ == "__main__":
    main()
