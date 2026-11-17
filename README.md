# Verum-Bespoke-Singularity
Pure-C GPU LLM inference engine.
Verum builds one binary, verum-slice, that reads GGUF models and tokenizer.json. It has 20 modes — the usage lines printed by `./verum-slice --help`: --backends, --inspect, --compile, --benchmark, --generate, --cpu, --dequant, --sample, --learn, --train-adapter, --train-block, --train-tiny, --autonomous-c, --autotune, --train-text, --evolve, --autonomy, and --prune, plus two --generate variants that load an adapter file (--adapter) or learned memory (--memory). The full usage list is in the Run section below.
## Backends
Single backend abstraction (vbs_backend_ops_t) covering Metal, CUDA, HIP, and CPU. The same C source is rebuilt per OS, with the GPU backend selected at runtime where a driver is present. CUDA and HIP are loaded at runtime, not link-time locked.
## Backend status
Metal and CUDA are the fuller paths. HIP embedding is currently a stub and is work in progress.
## Codebase
8,119 lines across src/*.c and include/*.h, C11.

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
./verum-slice --generate --tokenizer tokenizer.json --prompt "Hello" model.gguf
./verum-slice --cpu --tokenizer tokenizer.json --prompt "Hello" model.gguf
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
```

## Tests/CI

The workflow in `.github/workflows/ci.yml` runs on pushes and pull requests to `main` and `master`, on ubuntu-latest with gcc, ubuntu-latest with clang, and macos-latest with clang. It checks out the code, installs the compiler on Linux, runs `make clean` and `make CC=<compiler>`, runs `./verum-slice --help || ./verum-slice || true`, and runs a Python check that the `verum-slice` binary exists. `make test` is a local target: it builds the binary and runs the model-free self-checks: `./verum-slice --help` must exit 0, and `--dequant` and `--sample` must each print ALL PASS. The workflow does not run a benchmark, and the self-checks need no model file.

## What works / What doesn't

What works: the single `verum-slice` binary, GGUF model and `tokenizer.json` input, the shared Metal, CUDA, HIP, and CPU backend abstraction, and the fuller Metal and CUDA paths.

What doesn't yet work: HIP embedding is a stub and is work in progress. There is no Windows build target in the Makefile. CI verifies the build and checks the existence of the binary; it does not run a benchmark. The `make test` self-checks run locally.

## License

MIT. See `LICENSE`.
