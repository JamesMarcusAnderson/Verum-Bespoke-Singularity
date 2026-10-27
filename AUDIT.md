# Audit - 2026-10-27, verified by running make test on Linux, CPU, no GPU

## What was checked, and what passed
- Build: make, clang/gcc, C11, -Wall -Wextra -O3 - PASS, binary 215K
- --help: exit 0 (fixed, was 64) - PASS
- --backends: CUDA compiled yes, available no; HIP compiled yes, available no; Metal compiled no (Linux) - PASS, honest
- --learn: stores prompt->correct in TSV, no model needed - PASS
- --train-adapter: loss 6.9078 -> 0.0622, p(target) 0.0010 -> 0.9397, 50 epochs - PASS
- --train-block: loss 1.4507 -> 0.0030, 200 epochs, gradient-checked diff <0.00003 - PASS
- --train-tiny: loss 2.0747 -> 0.0010, generates 1 2 3 5 1 2 3 5 1 2 3 5 - PASS
- --autonomous-c: final loss 0.0098, generates pattern, no epochs/lr given - PASS
- --train-text: loss 3.0450 -> 0.0003, generates 'hello world hello world' exactly, 500 epochs - PASS (added in audit)
- --autotune: CPU, 2 cores, 7935 MiB RAM, window 991 MiB, ~49000 epochs/s, second run reuses config - PASS
- Daemon: 2 cycles, autotune saved then reused, autonomous-c each cycle - PASS (previous run)
- Self-edit: add/update/remove, test-gated, backup in .self_backup - PASS (previous run)

## What was NOT verified, and why
- --generate / --inspect / --compile / --benchmark / --evolve / --autonomy / --prune
  with a real GGUF model: no model file is in the repo, and none was downloaded
  for this audit. Without a model, --generate correctly fails with
  "Cannot open GGUF file: model.gguf", exit 1. That is not hidden.
- HIP embedding: still a stub in vbs_backend_hip.c ("HIP backend is a stub: no embedding kernels").
  Do not claim HIP works fully.
- Windows: Makefile uses -ldl -lpthread, which is MinGW/MSYS2 only, not native MSVC.
  PORTABLE.md now says that explicitly. Do not claim native Windows MSVC support.
- Singularity / AGI: NOT achieved, NOT claimed here. Today Verum is:
  a GGUF inference engine + a 1-layer trainable transformer + memory/adapter
  learning + autonomous tuning/editing. That is a foundation, not singularity.

## What was fixed in this audit
- --help exit 64 -> 0
- vbs_autotune.c misleading-indentation warning fixed
- Added make test / test_verum.py, covering all model-free modes
- README updated: line count 4630 -> ~5970, new modes listed, Windows/MinGW clarified,
  GGUF modes marked as requiring a model file

## Roadmap to be first to singularity, honestly
1. Multi-layer transformer training (stack blocks, backprop through layers) - not yet done
2. Train on real text data, not just a repeating pattern - not yet done
3. Large-scale training: data, compute, distributed backends - not yet done
4. Evaluation on real benchmarks, not just loss on a toy pattern - not yet done
5. Safety / alignment for autonomous self-editing at scale - test-gated now, needs more
Until 1-4 are done, do not claim singularity, AGI, or "best in the world" as a fact.
