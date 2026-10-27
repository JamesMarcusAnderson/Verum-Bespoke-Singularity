#!/usr/bin/env python3
"""Audit test suite for Verum - model-free modes only.
Run: python3 test_verum.py  (or make test)
Fails loudly if any claimed mode does not work. No GGUF model needed.
GGUF-dependent modes (--generate/--inspect/--compile/--benchmark/--evolve/--autonomy/--prune)
are NOT tested here, because no model file is in the repo - that is documented, not hidden.
"""
import subprocess, sys, os, re
ROOT = os.path.dirname(os.path.abspath(__file__))
def run(args):
    r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=300)
    return r
def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'} {name} {detail}")
    if not cond: sys.exit(1)

r = run(["make"])
check("build", r.returncode==0)
r = run(["./verum-slice", "--help"]); check("--help exit 0", r.returncode==0)
r = run(["./verum-slice", "--backends"]); check("--backends", r.returncode==0 and "Backend capability report" in r.stdout)
r = run(["./verum-slice", "--learn", "--prompt", "Test Q?", "--correct", "Test A", "--memory", "/tmp/test-memory.tsv"]); check("--learn", r.returncode==0 and "LEARNED" in r.stdout)
r = run(["./verum-slice", "--train-adapter", "--target-token", "42", "--vocab", "1000", "--epochs", "50", "--lr", "0.5", "--adapter-out", "/tmp/test-adapter.bin"])
check("--train-adapter loss down", r.returncode==0 and "p(target)=0.93" in r.stdout, r.stdout[-200:])
r = run(["./verum-slice", "--train-block", "--epochs", "200", "--lr", "0.05"]); check("--train-block", r.returncode==0 and "loss=0.003" in r.stdout, r.stdout[-200:])
r = run(["./verum-slice", "--train-tiny", "--epochs", "200", "--lr", "0.05"]); check("--train-tiny generates pattern", r.returncode==0 and "1 2 3 5 1 2 3 5" in r.stdout, r.stdout[-300:])
r = run(["./verum-slice", "--train-text", "--prompt", "hello world hello world ", "--epochs", "200", "--lr", "0.05"]); check("--train-text generates text", r.returncode==0 and "hello world hello world" in r.stdout, r.stdout[-300:])
r = run(["./verum-slice", "--autonomous-c"]); check("--autonomous-c", r.returncode==0 and "final loss=0.0098" in r.stdout, r.stdout[-300:])
r = run(["./verum-slice", "--autotune", "--autotune-out", "/tmp/test-autotune.json"]); check("--autotune", r.returncode==0 and "backend=" in r.stdout)
r = run(["./verum-slice", "--autotune", "--autotune-out", "/tmp/test-autotune.json"]); check("--autotune reuse", r.returncode==0 and "reused" in r.stdout)
print("ALL AUDIT TESTS PASS")
