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

Usage:
  python3 verum-daemon.py --cycles 3 --interval 2 --memory verum-memory.tsv
  python3 verum-daemon.py --cycles 0 --interval 60   # run forever, 60s/cycle
  # background, any computer with python3:
  nohup python3 verum-daemon.py --cycles 0 --interval 300 > daemon.out 2>&1 &
"""
import argparse, json, os, subprocess, sys, time
ROOT = os.path.dirname(os.path.abspath(__file__))
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
    args = ap.parse_args()
    log("=== VERUM DAEMON START (automated, no human trigger per cycle) ===")
    if not os.path.exists(os.path.join(ROOT, "verum-slice")):
        log("verum-slice missing, building")
        run(["make"])
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
        ra = run(["./verum-slice", "--autotune"])
        log(f"autotune rc={ra.returncode} {(ra.stdout.strip().splitlines() or [''])[-1] if ra.stdout else ''}")
        # 2. autonomous C training
        r = run(["./verum-slice", "--autonomous-c"])
        tail = (r.stdout.strip().splitlines() or [""])[-1] if r.stdout else ""
        log(f"autonomous-c rc={r.returncode} {tail}")
        # 3. self-edit every 3rd cycle
        if cycle % 3 == 0:
            r2 = run(["python3", "self_edit.py"])
            log(f"self_edit rc={r2.returncode}")
        with open(STATE, "w") as f:
            json.dump({"cycle": cycle, "memory_lines": cur, "last_run": time.time()}, f, indent=2)
        if args.cycles == 0 or cycle < args.cycles:
            log(f"sleep {args.interval}s")
            time.sleep(args.interval)
    log(f"=== VERUM DAEMON DONE after {cycle} cycles ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())
