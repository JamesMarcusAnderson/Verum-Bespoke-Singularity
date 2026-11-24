# Changelog

## Unreleased

### Fixed
- MSVC build (first run 2026-11-17, CI job `windows-msvc`): the Vulkan
  SPIR-V kernel table initialized `n_words` from the `vbs_spv_*_words`
  variables, which are `static const size_t` — not constant expressions
  in C, so MSVC rejected the file-scope initializer (error C2099).
  GCC/Clang accept them as an extension, which is why only MSVC failed.
  The table now computes word counts with `sizeof`, a true constant
  expression on every compiler. One error, one line.
- MSVC link (second failure, same first run): `main.c` called POSIX
  `strtok_r`, which does not exist on MSVC (link error LNK2019). It now
  uses `vbs_strtok_r` from `vbs_port.h`, the engine's own reentrant
  tokenizer — the same one `vbs_cpu_main.c` already used. The two
  identical prompt-id parsing blocks in `main.c` (--loop and --fuse)
  are now one helper, `parse_prompt_ids`.

## v0.10.0 — 2026-11-17 — GGUF proven, Windows target, universal GPU path, recursive self-optimization loop

### Added
- Real GGUF inference end-to-end on CPU (SmolLM-135M Q8_0 + Q4_K_M
  verified: "The capital of France is Paris."). Q8_K dequant bit-exact.
  Five real bugs fixed, including the Q4_K/Q5_K dequant layout being
  backwards.
- Windows target: MinGW cross-compile verified (PE32+ runs under Wine,
  all modes identical); `vbs_port.h` carries every platform difference;
  CMake MSVC path documented (untested on real Windows).
- Universal GPU path: Vulkan/SPIR-V backend (21 precompiled kernels)
  with automatic CPU fallback; HIP ported behind the old interface;
  backend selection logged (Vulkan > CUDA > HIP > Metal > CPU). Verified
  on Lavapipe + CPU fallback; real-GPU execution still needs hardware.
- Recursive self-optimization loop: structured logs → hotspot analysis →
  generate C11 candidate → compile → benchmark-verify in a forked child
  → hot-swap between tokens → rollback on regression. Proven on a
  synthetic model and a real one (avx2 accepted; wrong/slow/broken
  candidates each rejected by their own gate; rollback proven twice).
- Two novel mechanisms: model-specialized fused FFN kernel at load time
  (1.53x on the FFN section, 1.15–1.18x end-to-end on the tiny model,
  bit-identical output) and per-machine autotune profiles embedded in
  the binary (`.vbs_tune` section, fingerprint-gated).
- `make test` now also runs `model_e2e_test` + `--backend-test` (78/78
  with Vulkan, 44/44 with no ICD); `make test-d` runs the loop demo
  (6 sections).

### Verified
- `make` clean (0 errors, 4 pre-existing warnings); `make test` and
  `make test-d` pass.
- Real inference reproduced: Q8_0, 30.4 tok/s prefill, 18.5 tok/s
  decode on a shared 1.5-vCPU VM.

### Not yet verified — the v1.0.0 gates
- GPU backends on real hardware (Vulkan verified on Lavapipe only).
- MSVC build on a real Windows machine.
- LLM-authored C11 candidates (the loop's candidate list is the plug-in
  point); the native-emitter rewrite still blocks `vbs_compile()`-based
  JIT.

## v0.9.0 — 2026-11-17 — First official audited release

Audited 2026-11-17: every file in this release was hashed against the live
Dropbox masters and the GitHub head, and the full build was verified clean.
Only code that compiles and passes its checks is in this release.

### Added
- C11 compiler frontend: lexer, parser, typechecker, driver
  (`src/compiler/`, `include/vbs_compiler.h`) with GCN, MSL, PTX, and SPIR-V
  emitters plus validators. Inert until wired into the build.
- Baremetal phase-2 GPU driver (`baremetal/phase2/gpu.c`).
- Baremetal phase-1 boot (`baremetal/phase1/boot.c`, fixed: missing
  `<stddef.h>` include).
- Standalone `vbs-cpu` CLI (`src/vbs_cpu_main.c`): reads prompt token IDs
  from stdin, writes one JSON result. Unwired until a Makefile target is added.
- `Makefile` gains a real `test` target (`make test` runs the model-free
  self-checks).
- `README.md` corrected: 8,119 lines of C, 20 modes, accurate CI description.

### Verified
- `make` clean (gcc 13, `-std=c11 -Wall -Wextra -O3`), `make test` passes.
- Model-free modes pass: `--dequant` and `--sample` ALL PASS;
  `--autonomous-c` final loss 0.0098; `--train-tiny`, `--train-text`,
  `--autotune` pass.

### Held — not in this release
- HIP backend rewrite, JIT (`vbs_jit.c`) + native AArch64/x86-64 emitters,
  optimizer: all fail to compile against the current backend interface.
  Tracked for the adapter port (old `vbs_backend.h` interface stays canonical).

### Not yet verified — the v1.0.0 gates
- GGUF inference end-to-end against a real model file.
- GPU backends on real hardware.
- Windows build target.
