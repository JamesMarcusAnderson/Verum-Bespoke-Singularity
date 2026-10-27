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
