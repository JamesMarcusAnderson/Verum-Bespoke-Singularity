# Verum-Bespoke-Singularity
Pure-C GPU LLM inference engine.
Verum builds one binary, verum-slice, that reads GGUF models and tokenizer.json, with commands --inspect, --compile, --benchmark, and --generate.
## Backends
Single backend abstraction (vbs_backend_ops_t) covering Metal, CUDA, HIP, and CPU. The same C source is rebuilt per OS, with the GPU backend selected at runtime where a driver is present. CUDA and HIP are loaded at runtime, not link-time locked.
## Backend status
Metal and CUDA are the fuller paths. HIP embedding is currently a stub and is work in progress.
## Codebase
About 4,630 lines across src/*.c and include/*.h, C11.

## Build

Build from the repository root with the Makefile:

```sh
make
```

This builds the `verum-slice` binary. The Makefile uses clang if available, otherwise gcc, otherwise cc, with C11, `-Wall -Wextra -O3`, `-Iinclude`, and `-D_GNU_SOURCE`, and links `-ldl -lm -lpthread`. CUDA and HIP are enabled with `-DVBS_ENABLE_CUDA=1` and `-DVBS_ENABLE_HIP=1`. Metal is added only on Darwin: the Makefile adds `-DVBS_ENABLE_METAL=1`, the Foundation, Metal, and CoreGraphics frameworks, `-lobjc`, and `src/vbs_backend_metal.c`. There is no Windows build target in the Makefile. To remove build outputs, run:

```sh
make clean
```

## Run

```sh
./verum-slice --inspect
./verum-slice --compile
./verum-slice --benchmark
./verum-slice --generate
```

## Tests/CI

The workflow in `.github/workflows/ci.yml` runs on pushes and pull requests to `main` and `master`, on ubuntu-latest with gcc, ubuntu-latest with clang, and macos-latest with clang. It checks out the code, installs the compiler on Linux, runs `make clean` and `make CC=<compiler>`, runs `./verum-slice --help || ./verum-slice || true`, and runs a Python check that the `verum-slice` binary exists. The workflow does not run a benchmark or a counted test suite.

## What works / What doesn't

What works: the single `verum-slice` binary, GGUF model and `tokenizer.json` input, the shared Metal, CUDA, HIP, and CPU backend abstraction, and the fuller Metal and CUDA paths.

What doesn't yet work: HIP embedding is a stub and is work in progress. There is no Windows build target in the Makefile, and CI verifies the build and the existence of the binary rather than running a benchmark.

## License

MIT. See `LICENSE`.
