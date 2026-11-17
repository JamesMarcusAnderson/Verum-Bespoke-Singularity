# Changelog

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
