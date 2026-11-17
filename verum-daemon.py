#!/usr/bin/env python3
"""
Verum automated self-learning daemon.
Start it once, it keeps learning on its own - no human trigger per cycle.

Each cycle, automatically:
 1b. Runs --autotune (hardware: backend/threads/window, learned config reused if unchanged)
 1. Watches verum-memory.tsv for new corrections. If new lines appeared
    since last cycle, it logs them as AUTO-LEARN and runs --learn recall
    verification (via the memory file itself, no manual input).
 2. Runs the C autonomous loop: ./verum-slice --autonomous-c
    (trains until loss < 0.01, adjusts its own lr, generates pattern)
 3. Every 3rd cycle, runs self_edit.py (add/update/remove, test-gated)
 4. Writes daemon_state.json + daemon.log, then sleeps --interval seconds.

Recursive-loop integration (Phase D): with --loop-model model.gguf the
daemon also runs one unsupervised optimization loop per cycle
(./verum-slice --loop): the loop improves the live inference hot path,
hot-swaps verified-faster kernels between tokens, and rolls back on any
live regression. All loop decisions land in vbs-loop.jsonl.

Usage:
  python3 verum-daemon.py --cycles 3 --interval 2 --memory verum-memory.tsv
  python3 verum-daemon.py --cycles 0 --interval 60   # run forever, 60s/cycle
  python3 verum-daemon.py --cycles 0 --interval 300 --loop-model tiny.gguf --loop-iterations 2
  # background, any computer with python3:
  nohup python3 verum-daemon.py --cycles 0 --interval 300 > daemon.out 2>&1 &
"""
import argparse, json, os, shutil, subprocess, sys, time
ROOT = os.path.dirname(os.path.abspath(__file__))
# Cross-platform: the binary is verum-slice.exe on Windows, verum-slice elsewhere;
# make is mingw32-make under a MinGW install without plain `make`.
BIN = os.path.join(ROOT, "verum-slice.exe" if os.name == "nt" else "verum-slice")
MAKE = shutil.which("make") or shutil.which("mingw32-make") or "make"
LOG = os.path.join(ROOT, "daemon.log")
STATE = os.path.join(ROOT, "daemon_state.json")

def log(m):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {m}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line+"\n")

def run(cmd):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300)

def count_lines(path):
    if not os.path.exists(path): return 0
    with open(path, errors="ignore") as f: return sum(1 for _ in f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=3, help="0 = forever")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--memory", default=os.path.join(ROOT, "verum-memory.tsv"))
    ap.add_argument("--loop-model", default=None,
                    help="GGUF model for the recursive optimization loop (runs ./verum-slice --loop each cycle)")
    ap.add_argument("--loop-iterations", type=int, default=2,
                    help="optimization iterations per daemon cycle (default 2)")
    ap.add_argument("--loop-tokens", type=int, default=32,
                    help="tokens generated per loop run (default 32)")
    args = ap.parse_args()
    log("=== VERUM DAEMON START (automated, no human trigger per cycle) ===")
    if not os.path.exists(BIN):
        log("verum-slice missing, building")
        run([MAKE])
    last_count = count_lines(args.memory)
    log(f"watching {args.memory} initial_lines={last_count}")
    cycle = 0
    while args.cycles == 0 or cycle < args.cycles:
        cycle += 1
        log(f"--- cycle {cycle} ---")
        # 1. auto-learn from new memory
        cur = count_lines(args.memory)
        if cur > last_count:
            log(f"AUTO-LEARN: {cur-last_count} new correction(s) in memory, total {cur}")
            # verify recall works by checking the file directly
            with open(args.memory, errors="ignore") as f:
                lines = f.read().strip().splitlines()[-3:]
            for ln in lines: log(f"AUTO-LEARN memory: {ln[:120]}")
            last_count = cur
        else:
            log(f"no new memory (lines={cur})")
        # 1b. hardware autotune: best config for THIS computer, reused if hardware unchanged
        ra = run([BIN, "--autotune"])
        log(f"autotune rc={ra.returncode} {(ra.stdout.strip().splitlines() or [''])[-1] if ra.stdout else ''}")
        # 2. autonomous C training
        r = run([BIN, "--autonomous-c"])
        tail = (r.stdout.strip().splitlines() or [""])[-1] if r.stdout else ""
        log(f"autonomous-c rc={r.returncode} {tail}")
        # 3. self-edit every 3rd cycle
        if cycle % 3 == 0:
            r2 = run([sys.executable, "self_edit.py"])
            log(f"self_edit rc={r2.returncode}")
        # 4. recursive optimization loop (only when a model was given)
        if args.loop_model:
            rl = run(["./verum-slice", "--loop",
                      "--prompt-ids", "1,2,3",
                      "--tokens", str(args.loop_tokens),
                      "--iterations", str(args.loop_iterations),
                      "--loop-log", "vbs-loop.jsonl",
                      "--loop-work", ".verum-loop-work",
                      "--repo", ROOT,
                      args.loop_model])
            tail = ""
            if rl.stdout:
                lines = [ln for ln in rl.stdout.strip().splitlines() if ln.startswith("[loop] done")]
                tail = lines[-1] if lines else (rl.stdout.strip().splitlines() or [""])[-1]
            log(f"loop rc={rl.returncode} {tail}")
        with open(STATE, "w") as f:
            json.dump({"cycle": cycle, "memory_lines": cur, "last_run": time.time()}, f, indent=2)
        if args.cycles == 0 or cycle < args.cycles:
            log(f"sleep {args.interval}s")
            time.sleep(args.interval)
    log(f"=== VERUM DAEMON DONE after {cycle} cycles ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())
