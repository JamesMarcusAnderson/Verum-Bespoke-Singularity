# Verum-Bespoke-Singularity
Pure-C GPU LLM inference engine.
Verum builds one binary, verum-slice, that reads model files (GGUF format) and tokenizer.json. It has the following modes — the usage lines printed by `./verum-slice --help`: --backends, --backend-test, --inspect, --compile, --benchmark, --generate, --cpu, --dequant, --sample, --learn, --train-adapter, --train-block, --train-tiny, --autonomous-c, --autotune, --train-text, --evolve, --autonomy, and --prune, plus two --generate variants that load an adapter file (--adapter) or learned memory (--memory). The full usage list is in the Run section below.
Latest release: [v0.10.0](https://github.com/JamesMarcusAnderson/Verum-Bespoke-Singularity/releases/tag/v0.10.0).
## Backends
Single backend abstraction (vbs_backend_ops_t) covering Vulkan, Metal, CUDA, HIP, and CPU. Vulkan (SPIR-V) is the universal GPU path — one backend for AMD, NVIDIA, and Intel GPUs (and Apple via MoltenVK) — with automatic CPU fallback when no GPU or Vulkan driver is present. The same C source is rebuilt per OS, with the GPU backend selected at runtime where a driver is present. Vulkan, CUDA, and HIP are loaded at runtime (dlopen), not link-time locked; the engine builds and runs with no GPU libraries installed.
## Backend status
Vulkan is the universal path: 21 precompiled SPIR-V kernels (GLSL sources in kernels/gpu/, regenerated with kernels/build-kernels.sh) cover linear matvec (F32/F16/Q8_0/Q4_0/Q4_1/Q4_K/Q6_K/LNS8), embedding (F32/F16/Q8_0/Q4_0/Q4_1/Q4_K/Q6_K), and all seven primitive ops. HIP embedding now computes (host-staged) instead of stubbing. Metal and CUDA are the fuller paths. Backend selection is explicit and logged (`--backends` prints the probe report; `VBS_BACKEND=<name>` forces one); `./verum-slice --backend-test` runs the per-backend correctness suite (78 checks on Vulkan, 44 on CPU-only) and is part of `make test`.
## From-scratch GPU drivers

`src/driver/` holds from-scratch C11 GPU drivers behind a shared driver
framework (`vbs_drv.c` / `vbs_drv.h`), with a native backend
(`vbs_backend_native.c`) that plugs into the engine's backend interface.
Vendor coverage: Turing (NVIDIA), RDNA2 (AMD), and Alchemist (Intel),
each with its own encoder and shader assembler under `src/driver/shader/`.

## Why from-scratch

November 2026 made the from-scratch case in public, at scale — and
gave Verum a standard to measure itself against.

**Strata 0.2.16 (Nov 18, 2026) dropped llama.cpp for its own
engine.** Strata, the open-source local AI app (about 45,700 GitHub
stars), replaced llama.cpp/ggml matrix kernels with an inference
engine it wrote from scratch, covering NVIDIA, AMD, Intel Arc and
Apple Silicon. Its headline claim is token-for-token identical output
to 0.2.15 across nine different GPUs, backed by published equivalence
tables and long-run reliability data. The fear with any engine
rewrite is silently different output; Strata claims to have avoided
it, in public, at large scale. That equivalence-testing approach —
same tokens out, across GPUs — is the standard Verum's own
verification should meet or beat. Verum's differentiator against
Strata stands: pure C11 (not C++/CUDA), its own GPU drivers, and a
recursive self-optimization loop.

**The same weeks produced a wave of independent from-scratch
engines.** tinyinfer (about 2,200 lines of C++ and CUDA, no PyTorch,
no cuBLAS) claims 235 tokens/s running TinyLlama-1.1B at 4-bit on a
6 GB laptop RTX 3050. TinyCoder, a from-scratch C++ and CUDA engine
reading GGUF directly, claims a 35B mixture-of-experts model running
at 53.6 tokens/s on an 11 GB RTX 2080 Ti against 17.1 for llama.cpp.
Garlic LLM runs Qwen3.5-9B entirely offline on a jailbroken PS4 —
there is no compute API for PS4 homebrew, so the engine hand-builds
about 400 PM4 GPU command packets per token; it is, in effect, the
driver, at a claimed 31–39 tokens/s. All three sets of numbers are
author-reported, with no independent verification found; they are
claims, not benchmarks. But the pattern is the point: hand-written
kernels matching or beating llama.cpp on consumer and legacy hardware
is now a visible movement, not one project. Garlic LLM's
command-packet approach is the closest cousin to Verum's own
from-scratch drivers in `src/driver/` — GPU control at the packet
level, because no driver was shipped for the hardware.

**The ecosystem Verum builds on also steadied itself.** llama.cpp
hit 1.0.0 on Nov 13, 2026 — its first versioned release, with
semantic versioning for the public C API — and the GGUF format itself
did not change (still version 3), which matters because Verum loads
GGUF models. Then prerelease b12669 (Nov 24, 2026) added
`--target-size` and `--target-bpw` to `llama-quantize`: instead of
picking a fixed preset, you state a file-size or bits-per-weight
budget and an optimizer picks the per-tensor quantization mix (238
comparisons across 34 models reported: 195 wins, 26 ties, 16 losses
against standard quantization on KL divergence). The budget-based
quantizer is a feature idea for Verum's loader, not a shipped
feature.

**Honest limits stand.** None of the above changes what Verum has
and has not proven: real-GPU execution is still unverified (Vulkan
execution was verified on Lavapipe, a software device, not on GPU
hardware), and LLM-authored candidates are still not done. See
"What doesn't yet work" below. The November wave validates the
from-scratch bet; it does not validate this engine's hardware
results, which remain to be run on real GPUs.

Sources: LinuxCompatible on Strata 0.2.16, Nov 19, 2026 — https://www.linuxcompatible.org/story/strata-0216-runs-its-own-engine-drops-llamacpp-claims-identical-output/ ;
tinyinfer write-up, Nov 14, 2026 — https://medium.com/@klaadithya/how-i-built-an-llm-inference-engine-from-scratch-in-cuda-235-tokens-s-on-a-6-gb-laptop-gpu-f5ac062950a6 ;
TinyCoder coverage, Nov 22, 2026 — https://news.hamidun.com/en/news/17866/yandex-grant-funds-an-llm-engine-35b-model-on-rtx-2080-ti-ru ;
Garlic LLM coverage, Nov 16–19, 2026 — https://repo-explainer.com/spinzar/garlic-llm-ps4 and https://thatfeed.com/2026/11/19/2573/hacked-ps4-ran-a-9-billion-parameter-ai-model-at-up-to-43-tokens-per-second ;
llama.cpp 1.0.0 — https://www.ozercemkelahmet.com/articles/llama-cpp-1-0-local-inference ;
llama.cpp b12669 target-size quantization, Nov 24, 2026 — https://ingoampt.com/llama-cpp-b12669-custom-gguf-size-targets/ .

## Codebase
18,126 lines across src/*.c and include/*.h, C11.

## Build

Build from the repository root with the Makefile:

```sh
make
```

This builds the `verum-slice` binary. The Makefile uses clang if available, otherwise gcc, otherwise cc, with C11, `-Wall -Wextra -O3`, `-Iinclude`, and `-D_GNU_SOURCE`, and links `-ldl -lm -lpthread`. CUDA and HIP are enabled with `-DVBS_ENABLE_CUDA=1` and `-DVBS_ENABLE_HIP=1`. Metal is added only on Darwin: the Makefile adds `-DVBS_ENABLE_METAL=1`, the Foundation, Metal, and CoreGraphics frameworks, and `-lobjc`, which enables the Metal section of `src/vbs_gpu_vendors.c`. To remove build outputs, run:

```sh
make clean
```

Windows (MinGW-w64) — cross-compile from Linux, or natively under MSYS2/MinGW:

```sh
make mingw
```

This builds `verum-slice.exe` with `x86_64-w64-mingw32-gcc`, linking only `-lm` (dynamic loading is LoadLibraryA; the engine uses no pthreads) and `-Wl,--export-all-symbols` so the JIT specializer can link its DLLs against the exe. All platform differences live in `include/vbs_port.h`; see PORTABLE.md for the verified platform list.

CMake is also supported (same sources and flags), mainly for MSVC on Windows:

```sh
cmake -S . -B build && cmake --build build
```

MSVC needs Visual Studio 2019+ (`/std:c11`); the MSVC build is covered by CI — see Tests/CI.

## Run

The model modes require a model file argument (running them bare prints the usage text and exits with status 64):

```sh
./verum-slice --inspect model.gguf
./verum-slice --compile model.gguf
./verum-slice --benchmark model.gguf
./verum-slice --generate --tokenizer tokenizer.json --prompt "Hello" model.gguf
```

Full usage, identical to the output of `./verum-slice --help`:

```text
./verum-slice --backends
./verum-slice --inspect model.gguf
./verum-slice --compile [--window-mib N] model.gguf
./verum-slice --benchmark [--tokens N] [--window-mib N] model.gguf
./verum-slice --generate --tokenizer tokenizer.json --prompt "Hello" [--tokens N] model.gguf
./verum-slice --cpu --tokenizer tokenizer.json --prompt "Hello" [--tokens N] model.gguf
./verum-slice --dequant
./verum-slice --sample [--temperature 0.7] [--top-k 40] [--top-p 0.9] [--seed 12345]
./verum-slice --learn --prompt "What is X?" --correct "X is Y" [--memory verum-memory.tsv]
./verum-slice --train-adapter --target-token 1234 [--vocab 32000] [--epochs 100] [--lr 0.5] --adapter-out adapter.bin
./verum-slice --generate --adapter adapter.bin ...
./verum-slice --train-block [--epochs 500] [--lr 0.05]
./verum-slice --train-tiny [--epochs 500] [--lr 0.05] [--block-out tiny.bin]
./verum-slice --autonomous-c
./verum-slice --autotune [--autotune-out verum-autotune.json]
./verum-slice --train-text --prompt "hello world " [--epochs 500]
./verum-slice --generate --memory verum-memory.tsv --tokenizer tokenizer.json --prompt "What is X?" model.gguf
./verum-slice --evolve --tokenizer tokenizer.json --prompt "Hello" model.gguf
./verum-slice --autonomy --tokenizer tokenizer.json --prompt "Hello" [--tokens N] model.gguf
./verum-slice --prune --tokenizer tokenizer.json --prompt "The capital of France is Paris" model.gguf
./verum-slice --loop [--prompt-ids 1,2,3] [--tokens N] [--iterations K] [--loop-poison] model.gguf
./verum-slice --hotbench [--tune-out verum-tune.json]
./verum-slice --fuse [--cache-dir DIR] [--tokens N] model.gguf
./verum-slice --tune-show
```

## Recursive loop (Phase D)

The CPU inference hot path (quantized matvec) runs through a versioned
kernel table (`include/vbs_hotpath.h`) instead of being called directly.
`--loop` runs a live generation session and, between tokens, improves that
hot path unsupervised: it reads the structured profiling counters to find
the hotspot, compiles the next candidate kernel with the system compiler,
benchmark-verifies it in a forked child (a crashing candidate kills only
the child), and hot-swaps it in only when it is correct AND measurably
faster (≥2%) than the module currently running. After a swap the next
token window is measured live; a regression rolls back to the previous
module. Every decision is written to a structured JSONL log
(`vbs-loop.jsonl` by default). The daemon runs the loop each cycle with
`--loop-model model.gguf`.

Two more mechanisms ship with it:

- `--fuse` builds a model-specialized fused FFN kernel (gate + up + SiLU
  in one pass over the shared Q8 activations, dimensions baked in as
  compile-time constants), compiles it once, caches the `.so` by model
  SHA-256 + machine fingerprint, and verifies it numerically against the
  reference path before use.
- `--hotbench` measures every kernel variant and writes a per-machine
  tune profile (`--tune-out`); `tools/vbs_tune_embed.sh` embeds that
  profile inside a copy of the binary (ELF `.vbs_tune` section), so the
  tune travels with the binary and is applied only when the machine
  fingerprint matches. `--tune-show` reads it back.

`make test-d` runs the whole Phase D demonstration end to end on a
synthetic tiny model (`tools/make_tiny_model.py` generates it; no real
model file needed).

## Tests/CI

The workflow in `.github/workflows/ci.yml` runs on pushes and pull requests to `main` and `master`. CI runs five jobs: Ubuntu gcc, Ubuntu clang, macOS clang, Windows cross-compile (MinGW) with a Wine run, and Windows native (MSVC via CMake) — all green on the latest run. It checks out the code, installs the compiler on Linux, runs `make clean` and `make CC=<compiler>`, runs `./verum-slice --help || ./verum-slice || true`, and runs a Python check that the `verum-slice` binary exists. The workflow also defines a `make test` step and two Windows jobs (`windows-mingw` cross-compiles with MinGW on ubuntu-latest and runs the model-free self-checks under Wine; `windows-msvc` builds natively with MSVC via CMake on windows-latest and runs the same checks). `make test` is a local target: it builds the binary and runs the model-free self-checks: `./verum-slice --help` must exit 0, `--dequant` and `--sample` must each print ALL PASS, `--backend-test` must print ALL PASS (twice: once normally, once with no Vulkan ICD to prove the CPU fallback), and the `model_e2e_test` regression test (loader + dequantizer + CPU forward on a tiny synthetic model file built in memory, covering Q8_K/Q4_K/Q6_K/Q5_K/Q5_0 load-time conversion) must print ALL PASS. The workflow does not run a benchmark, and the self-checks need no model file.

## What works / What doesn't

What works: the single `verum-slice` binary, model file and `tokenizer.json` input, the shared Vulkan, Metal, CUDA, HIP, and CPU backend abstraction, the universal Vulkan/SPIR-V GPU path with automatic CPU fallback, and the fuller Metal and CUDA paths. The CPU reference path (`--cpu`, and the standalone `vbs-cpu` CLI built by `make`) runs real llama-arch model files end to end on CPU: weights stored as Q5_0/Q5_1, Q2_K–Q8_K, Q8_1, or BF16 are dequantized to Q8_0 once at load (measured ~10x faster than an F16 target, at reference-level logit accuracy), while F32/F16/Q4_0/Q4_1/Q8_0 run natively. The tokenizer is a GPT-2-style byte-level BPE implementation verified token-for-token against the HuggingFace `tokenizers` reference, including chat special tokens (`<|im_start|>`, `<|im_end|>`).

What doesn't yet work: real-GPU execution and GPU performance numbers are unverified (Vulkan execution was verified on Lavapipe, a software device, not on GPU hardware). The engine's own SPIR-V emitter produces a structural shell with no body codegen, so real dispatch uses precompiled SPIR-V kernels. The JIT specializer's in-exe compile step on Windows needs a MinGW compiler on PATH (verified under Wine with a Windows MinGW toolchain installed). CI verifies the build and checks the existence of the binary; it does not run a benchmark. The `make test` self-checks run locally.

## License

MIT. See `LICENSE`.
