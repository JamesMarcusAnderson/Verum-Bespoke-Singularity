# Portable: works on any computer
- No GPU required. CUDA/HIP are runtime-loaded (dlopen), not link-locked.
  If no GPU is present, backends report available:no and the CPU path runs.
- Metal is only added on Darwin (macOS). Linux/Windows builds skip it.
- Build: `make` with cc/clang/gcc. That's it.
- Self-learn: `./verum-slice --learn --prompt X --correct Y`
- Train: `./verum-slice --train-adapter ...` and `--train-block ...`
- Autonomous: `python3 self_improve.py --iterations 3`
  That loop edits only src/self_tune.h, rebuilds, tests, and reverts
  any edit that does not improve loss. It never edits core engine files
  without a test gate, and it keeps a .bak backup during each try.

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
