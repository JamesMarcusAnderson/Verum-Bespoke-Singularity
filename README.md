# Verum-Bespoke-Singularity
Pure-C GPU LLM inference engine.
Verum builds one binary, verum-slice, that reads GGUF models and tokenizer.json. It has 21 modes — the usage lines printed by `./verum-slice --help`: --backends, --backend-test, --inspect, --compile, --benchmark, --generate, --cpu, --dequant, --sample, --learn, --train-adapter, --train-block, --train-tiny, --autonomous-c, --autotune, --train-text, --evolve, --autonomy, and --prune, plus two --generate variants that load an adapter file (--adapter) or learned memory (--memory). The full usage list is in the Run section below.
## Backends
Single backend abstraction (vbs_backend_ops_t) covering Vulkan, Metal, CUDA, HIP, and CPU. Vulkan (SPIR-V) is the universal GPU path — one backend for AMD, NVIDIA, and Intel GPUs (and Apple via MoltenVK) — with automatic CPU fallback when no GPU or Vulkan driver is present. The same C source is rebuilt per OS, with the GPU backend selected at runtime where a driver is present. Vulkan, CUDA, and HIP are loaded at runtime (dlopen), not link-time locked; the engine builds and runs with no GPU libraries installed.
## Backend status
Vulkan is the universal path: 21 precompiled SPIR-V kernels (GLSL sources in kernels/vulkan/, regenerated with kernels/build-spirv.sh) cover linear matvec (F32/F16/Q8_0/Q4_0/Q4_1/Q4_K/Q6_K/LNS8), embedding (F32/F16/Q8_0/Q4_0/Q4_1/Q4_K/Q6_K), and all seven primitive ops. HIP embedding now computes (host-staged) instead of stubbing. Metal and CUDA are the fuller paths. Backend selection is explicit and logged (`--backends` prints the probe report; `VBS_BACKEND=<name>` forces one); `./verum-slice --backend-test` runs the per-backend correctness suite (78 checks on Vulkan, 44 on CPU-only) and is part of `make test`.
## Codebase
8,620 lines across src/*.c and include/*.h, C11.

## Build

Build from the repository root with the Makefile:

```sh
make
```

This builds the `verum-slice` binary. The Makefile uses clang if available, otherwise gcc, otherwise cc, with C11, `-Wall -Wextra -O3`, `-Iinclude`, and `-D_GNU_SOURCE`, and links `-ldl -lm -lpthread`. CUDA and HIP are enabled with `-DVBS_ENABLE_CUDA=1` and `-DVBS_ENABLE_HIP=1`. Metal is added only on Darwin: the Makefile adds `-DVBS_ENABLE_METAL=1`, the Foundation, Metal, and CoreGraphics frameworks, `-lobjc`, and `src/vbs_backend_metal.c`. To remove build outputs, run:

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

MSVC needs Visual Studio 2019+ (`/std:c11`); the MSVC path is documented but untested on a real Windows machine — the windows-msvc CI job is its first run. MinGW is the supported Windows build until that job goes green.

## Run

The model modes require a GGUF model file argument (running them bare prints the usage text and exits with status 64):

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
synthetic tiny model (`tools/make_tiny_gguf.py` generates it; no real
model file needed).

## Tests/CI

The workflow in `.github/workflows/ci.yml` runs on pushes and pull requests to `main` and `master`, on ubuntu-latest with gcc, ubuntu-latest with clang, and macos-latest with clang. It checks out the code, installs the compiler on Linux, runs `make clean` and `make CC=<compiler>`, runs `./verum-slice --help || ./verum-slice || true`, and runs a Python check that the `verum-slice` binary exists. The workflow file in this tree also defines a `make test` step and two Windows jobs (`windows-mingw` cross-compiles with MinGW on ubuntu-latest and runs the model-free self-checks under Wine; `windows-msvc` builds natively with MSVC via CMake on windows-latest and runs the same checks) — those take effect once the workflow file is updated on GitHub. Note: workflow files cannot be pushed via the GitHub App (403) — the ci.yml changes need James in the browser. `make test` is a local target: it builds the binary and runs the model-free self-checks: `./verum-slice --help` must exit 0, `--dequant` and `--sample` must each print ALL PASS, `--backend-test` must print ALL PASS (twice: once normally, once with no Vulkan ICD to prove the CPU fallback), and the `gguf_e2e_test` regression test (loader + dequantizer + CPU forward on a tiny synthetic GGUF built in memory, covering Q8_K/Q4_K/Q6_K/Q5_K/Q5_0 load-time conversion) must print ALL PASS. The workflow does not run a benchmark, and the self-checks need no model file.

## What works / What doesn't

What works: the single `verum-slice` binary, GGUF model and `tokenizer.json` input, the shared Vulkan, Metal, CUDA, HIP, and CPU backend abstraction, the universal Vulkan/SPIR-V GPU path with automatic CPU fallback, and the fuller Metal and CUDA paths. The CPU reference path (`--cpu`, and the standalone `vbs-cpu` CLI built by `make`) runs real llama-arch GGUF models end to end on CPU: weights stored as Q5_0/Q5_1, Q2_K–Q8_K, Q8_1, or BF16 are dequantized to Q8_0 once at load (measured ~10x faster than an F16 target, at llama.cpp-level logit accuracy), while F32/F16/Q4_0/Q4_1/Q8_0 run natively. The tokenizer is a GPT-2-style byte-level BPE implementation verified token-for-token against the HuggingFace `tokenizers` reference, including chat special tokens (`<|im_start|>`, `<|im_end|>`).

What doesn't yet work: real-GPU execution and GPU performance numbers are unverified (Vulkan execution was verified on Lavapipe, a software device, not on GPU hardware). The engine's own SPIR-V emitter produces a structural shell with no body codegen, so real dispatch uses precompiled SPIR-V kernels. The MSVC build path is documented but untested on a real Windows machine (CI's windows-msvc job is its first run). The JIT specializer's in-exe compile step on Windows needs a MinGW compiler on PATH (verified under Wine with a Windows MinGW toolchain installed). CI verifies the build and checks the existence of the binary; it does not run a benchmark. The `make test` self-checks run locally.

## License

MIT. See `LICENSE`.
