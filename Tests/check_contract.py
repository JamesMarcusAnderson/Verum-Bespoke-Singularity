#!/usr/bin/env python3
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
src = "\n".join(p.read_text() for p in sorted((root / "Sources").glob("*.[mh]")))

required = {
    "mmap-backed model": "mmap(NULL" in src,
    "two-dimensional tile API": "rowStart" in src and "columnStart" in src,
    "bounded double windows": "_stage[2]" in src and "_private[2]" in src,
    "runtime MSL generation": "newLibraryWithSource" in src,
    "remote peer view": "newRemoteBufferViewForDevice" in src,
    "explicit destination fallback": "return out" in src,
    "metadata graph override": "verum.graph.json" in src,
    "ambiguity rejection": "Unresolved matrix semantics" in src,
}

for forbidden in ("VBSModelPlan", "encodeLayer", "attn_q.weight", "RMS_EPS", "EOS_TOKEN"):
    required[f"no hard-coded {forbidden}"] = forbidden not in src

# Canonical GGML serialization IDs used by GGUF.  These assertions catch the
# exact corruption present in the uploaded loader.
enum = (root / "Sources" / "VBSGGUF.h").read_text()
for name, value in {"Q4_0":2,"Q5_0":6,"Q8_0":8,"Q2_K":10,"Q3_K":11,"Q4_K":12,"Q5_K":13,"Q6_K":14,"BF16":30}.items():
    required[f"canonical type {name}={value}"] = bool(re.search(rf"VBSGGMLType{name}\s*=\s*{value}\b", enum))

failed = [name for name, ok in required.items() if not ok]
for name, ok in required.items():
    print(("PASS" if ok else "FAIL"), name)
if failed:
    raise SystemExit(f"{len(failed)} contract checks failed")
print(f"PASS all {len(required)} architecture contracts")

