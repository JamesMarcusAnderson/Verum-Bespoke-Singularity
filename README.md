# Verum-Bespoke-Singularity
Pure-C GPU LLM inference engine.
Verum builds one binary, verum-slice, that reads GGUF models and tokenizer.json, with commands --inspect, --compile, --benchmark, and --generate.
## Backends
Single backend abstraction (vbs_backend_ops_t) covering Metal, CUDA, HIP, and CPU. The same C source is rebuilt per OS, with the GPU backend selected at runtime where a driver is present. CUDA and HIP are loaded at runtime, not link-time locked.
## Backend status
Metal and CUDA are the fuller paths. HIP embedding is currently a stub and is work in progress.
## Codebase
About 5,970 lines across src/*.c and include/*.h, C11 (2026-10-27 audit).

## Build

Build from the repository root with the Makefile:

```sh
make
```

This builds the `verum-slice` binary. The Makefile uses clang if available, otherwise gcc, otherwise cc, with C11, `-Wall -Wextra -O3`, `-Iinclude`, and `-D_GNU_SOURCE`, and links `-ldl -lm -lpthread`. CUDA and HIP are enabled with `-DVBS_ENABLE_CUDA=1` and `-DVBS_ENABLE_HIP=1`. Metal is added only on Darwin: the Makefile adds `-DVBS_ENABLE_METAL=1`, the Foundation, Metal, and CoreGraphics frameworks, `-lobjc`, and `src/vbs_backend_metal.c`. Windows: MinGW/MSYS2 only (Makefile uses -ldl -lpthread). No native MSVC target. To remove build outputs, run:

```sh
make clean
```

## Run

Model-free modes (no GGUF needed, tested by `make test`):
```sh
./verum-slice --backends
./verum-slice --learn --prompt "What is X?" --correct "X is Y"
./verum-slice --train-adapter --target-token 1234 --vocab 32000 --epochs 100 --lr 0.5 --adapter-out adapter.bin
./verum-slice --train-block --epochs 500 --lr 0.05
./verum-slice --train-tiny --epochs 500 --lr 0.05
./verum-slice --autonomous-c
./verum-slice --autotune
python3 verum-daemon.py --cycles 3 --interval 2
python3 self_edit.py
```

GGUF modes (require a model file + tokenizer.json, not in repo):
```sh
./verum-slice --inspect model.gguf
./verum-slice --compile model.gguf
./verum-slice --benchmark model.gguf
./verum-slice --generate --tokenizer tokenizer.json --prompt "Hello" model.gguf
```

## Tests

```sh
make test
```
Runs test_verum.py: build, --help, --backends, --learn, --train-adapter,
--train-block, --train-tiny, --autonomous-c, --autotune (including reuse).
GGUF modes are not tested here, because no model is in the repo - see AUDIT.md.


## Tests/CI

The workflow in `.github/workflows/ci.yml` runs on pushes and pull requests to `main` and `master`, on ubuntu-latest with gcc, ubuntu-latest with clang, and macos-latest with clang. It checks out the code, installs the compiler on Linux, runs `make clean` and `make CC=<compiler>`, runs `./verum-slice --help || ./verum-slice || true`, and runs a Python check that the `verum-slice` binary exists. The workflow does not run a benchmark or a counted test suite.

## What works / What doesn't

What works: the single `verum-slice` binary, GGUF model and `tokenizer.json` input, the shared Metal, CUDA, HIP, and CPU backend abstraction, and the fuller Metal and CUDA paths.

What doesn't yet work: HIP embedding is a stub and is work in progress. Windows is MinGW/MSYS2 only, and CI now runs `make test` in addition to the build check (see .github/workflows/ci.yml).

## License

MIT. See `LICENSE`.
