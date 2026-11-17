# Portable: works on any computer
- No GPU required. CUDA/HIP are runtime-loaded (dlopen / LoadLibrary),
  not link-locked. If no GPU is present, backends report available:no
  and the CPU path runs.
- Metal is only added on Darwin (macOS). Linux/Windows builds skip it.
- Build: `make` with cc/clang/gcc. That's it.
- Self-learn: `./verum-slice --learn --prompt X --correct Y`
- Train: `./verum-slice --train-adapter ...` and `--train-block ...`
- Autonomous: `python3 self_improve.py --iterations 3`
  That loop edits only src/self_tune.h, rebuilds, tests, and reverts
  any edit that does not improve loss. It never edits core engine files
  without a test gate, and it keeps a .bak backup during each try.

## Platforms (verified 2026-11-17)
- Linux x86-64, gcc 13.3 and clang: `make`, `make test` pass.
  Model-free modes verified byte-identical before/after the portability
  work (only benchmark timing numbers vary run to run).
- macOS: clang via CI (macos-latest). Metal backend only on Darwin.
- Windows x86-64, MinGW-w64: `make mingw` cross-compiles verum-slice.exe
  from Linux and runs under Wine (`--help`, `--backends`, `--dequant`,
  `--sample`, `--autonomous-c`, `--train-tiny`, `--autotune`, `--learn`
  all verified; GGUF open/parse error paths verified with fixture files).
  Natively on Windows: build under MSYS2/MinGW with `make mingw`, or
  `cmake -S . -B build && cmake --build build`.
- Windows x86-64, MSVC: CMake build documented (Visual Studio 2019+,
  `/std:c11`), NOT yet run on a real Windows machine — the windows-msvc
  CI job is the first MSVC run. The MinGW target above is the supported
  Windows build until that job goes green.
- All platform differences live in one header, `include/vbs_port.h`
  (monotonic clock, dynamic loading, file mapping, mkdir/stat, string
  helpers, aligned alloc, temp paths). No .c file calls a POSIX-only
  function directly, except the Darwin-only Metal backend and the
  freestanding baremetal/ code (not part of the host build).

## Windows notes
- The engine is pure C11. Windows needs no POSIX layer: file mapping is
  CreateFileMapping, GPU libraries load with LoadLibraryA
  (nvcuda.dll / amdhip64.dll), timers use QueryPerformanceCounter.
- Temp files go to %TEMP% (GetTempPathA); override with VBS_CACHE_DIR
  and VBS_SWAP_PATH as on Linux. GGUF parsing requires a little-endian
  host (all x86-64/ARM Windows machines are).
- The JIT specializer compiles its generated C to a .dll with MinGW gcc
  (or a MinGW-targeting clang) found on PATH, linked against the exe's
  import library (libverum-slice.dll.a, built next to the exe with
  -Wl,--export-all-symbols -Wl,--out-implib — ship it next to the exe)
  so the bridge symbols resolve at load time. The full round-trip
  (in-exe compile via a Windows MinGW toolchain, DLL load, entry-point
  calls back into the exe) is verified under Wine. Without a compiler
  on PATH, model modes that need specialization fail cleanly with
  "Specializer compilation failed".
- Python helpers (verum-daemon.py, self_edit.py, self_improve.py) run on
  Windows: they find verum-slice.exe, use mingw32-make if plain make is
  absent, and use the system temp dir. Start the daemon with
  `python verum-daemon.py --cycles 0 --interval 300` (run-daemon.sh is
  a Unix shell script; the Python command is the cross-platform form).

## Train-tiny end-to-end
./verum-slice --train-tiny --epochs 500 --lr 0.05 --block-out tiny.bin
Learns pattern 1,2,3,5 repeating, then greedy-generates:
  1 2 3 5 1 2 3 5 1 2 3 5
Loss 2.07 -> 0.0004. Gradient-checked: numeric vs analytic diff < 0.00003
for Wq/Wk/Wv/Wo/W1/W2/Wout, LayerNorm gammas, and biases.
A LayerNorm backward bug (missing /d) was found by that check and fixed.

## Automated daemon
Start once, it learns on its own:
  ./run-daemon.sh 300   # background, 300s/cycle, log daemon.log
  python3 verum-daemon.py --cycles 3 --interval 2   # test run
Each cycle: watches verum-memory.tsv for new corrections (AUTO-LEARN),
runs --autonomous-c, every 3rd cycle runs self_edit.py, writes daemon_state.json.

## Hardware autotune: best performance for any computer
  ./verum-slice --autotune
Detects cores, RAM, GPU (CUDA/HIP/Metal, else CPU), chooses backend,
threads=cores, window=RAM/8 clamped 64-1024 MiB, benchmarks the tiny
transformer block, saves verum-autotune.json with a hardware fingerprint.
Next run, if fingerprint matches, it reuses the learned config instead
of re-tuning. If hardware changes, it re-tunes automatically.
Daemon runs autotune every cycle, so it adapts to the machine it is on.
