#!/usr/bin/env python3
"""Verum automated self-learning daemon.

Start it once, it keeps learning on its own — no human trigger per cycle.

Each cycle, automatically:
 1. Watches verum-memory.tsv for new corrections. If new lines appeared
    since last cycle, it logs them as AUTO-LEARN and shows the latest
    ones (recall verification via the memory file itself, no manual input).
 2. Runs --autotune (hardware: backend/threads/window; the learned
    config is reused if the hardware is unchanged).
 3. Runs the C autonomous loop: ./verum-slice --autonomous-c
    (trains until loss < 0.01, adjusts its own lr, generates pattern).
 4. Every 3rd cycle, runs self_edit.py (add/update/remove, test-gated).
 5. Writes daemon_state.json + daemon.log, then sleeps --interval seconds.

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
import argparse
import json
import os
import sys
import time

from vbs_common import BIN, MAKE, ROOT, make_logger, run

LOG = os.path.join(ROOT, "daemon.log")
STATE = os.path.join(ROOT, "daemon_state.json")
log = make_logger(LOG)


def count_lines(path):
    """Lines in a file (0 if it does not exist)."""
    if not os.path.exists(path):
        return 0
    with open(path, errors="ignore") as handle:
        return sum(1 for _ in handle)


def last_line(output):
    """Last non-empty line of a command's stdout ("" if none)."""
    if not output:
        return ""
    lines = output.strip().splitlines()
    return lines[-1] if lines else ""


def check_memory(memory_path, known_count):
    """Log any corrections added to the memory file since last cycle.

    Returns the new line count.
    """
    current = count_lines(memory_path)
    if current <= known_count:
        log("no new memory (lines=%d)" % current)
        return current
    log("AUTO-LEARN: %d new correction(s) in memory, total %d"
        % (current - known_count, current))
    with open(memory_path, errors="ignore") as handle:
        recent = handle.read().strip().splitlines()[-3:]
    for line in recent:
        log("AUTO-LEARN memory: %s" % line[:120])
    return current


def run_loop(args):
    """One recursive optimization loop run; logs its outcome."""
    result = run(["./verum-slice", "--loop",
                  "--prompt-ids", "1,2,3",
                  "--tokens", str(args.loop_tokens),
                  "--iterations", str(args.loop_iterations),
                  "--loop-log", "vbs-loop.jsonl",
                  "--loop-work", ".verum-loop-work",
                  "--repo", ROOT,
                  args.loop_model])
    tail = ""
    if result.stdout:
        done = [ln for ln in result.stdout.strip().splitlines()
                if ln.startswith("[loop] done")]
        tail = done[-1] if done else last_line(result.stdout)
    log("loop rc=%d %s" % (result.returncode, tail))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=3, help="0 = forever")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--memory", default=os.path.join(ROOT, "verum-memory.tsv"))
    parser.add_argument("--loop-model", default=None,
                        help="model file for the recursive optimization loop "
                             "(runs ./verum-slice --loop each cycle)")
    parser.add_argument("--loop-iterations", type=int, default=2,
                        help="optimization iterations per daemon cycle (default 2)")
    parser.add_argument("--loop-tokens", type=int, default=32,
                        help="tokens generated per loop run (default 32)")
    args = parser.parse_args()

    log("=== VERUM DAEMON START (automated, no human trigger per cycle) ===")
    if not os.path.exists(BIN):
        log("verum-slice missing, building")
        run([MAKE])

    seen = count_lines(args.memory)
    log("watching %s initial_lines=%d" % (args.memory, seen))

    cycle = 0
    while args.cycles == 0 or cycle < args.cycles:
        cycle += 1
        log("--- cycle %d ---" % cycle)

        seen = check_memory(args.memory, seen)

        tuned = run([BIN, "--autotune"])
        log("autotune rc=%d %s" % (tuned.returncode, last_line(tuned.stdout)))

        trained = run([BIN, "--autonomous-c"])
        log("autonomous-c rc=%d %s" % (trained.returncode,
                                       last_line(trained.stdout)))

        if cycle % 3 == 0:
            edited = run([sys.executable, "self_edit.py"])
            log("self_edit rc=%d" % edited.returncode)

        if args.loop_model:
            run_loop(args)

        state = {"cycle": cycle, "memory_lines": seen,
                 "last_run": time.time()}
        with open(STATE, "w") as handle:
            json.dump(state, handle, indent=2)

        if args.cycles == 0 or cycle < args.cycles:
            log("sleep %ds" % args.interval)
            time.sleep(args.interval)

    log("=== VERUM DAEMON DONE after %d cycles ===" % cycle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
