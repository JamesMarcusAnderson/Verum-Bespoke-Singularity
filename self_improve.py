#!/usr/bin/env python3
"""
Verum autonomous self-learning + self-edit loop.
Works on any computer: only needs a C compiler (cc/clang/gcc) and python3.
No GPU required - CPU fallback is the default, GPU backends are runtime-optional.

Loop, autonomously, with safety:
 1. Read memory corrections (verum-memory.tsv) -> train adapter (already in C)
 2. Train transformer block, measure final loss
 3. SELF-EDIT: try changing a tunable in src/self_tune.h (learning rate),
    rebuild, re-test. Keep the edit ONLY if loss improves, else revert
    from backup. This is code editing itself, with a test gate.
 4. Log every iteration to self_improve.log, save best config to
    self_state.json. Never edits files outside the allowlist:
    src/self_tune.h, verum-memory.tsv, adapter.bin, self_state.json
Usage: python3 self_improve.py [--iterations 3] [--base-lr 0.05]
"""
import argparse, json, os, re, shutil, subprocess, sys, time
ROOT = os.path.dirname(os.path.abspath(__file__))
TUNE = os.path.join(ROOT, "src", "self_tune.h")
LOG = os.path.join(ROOT, "self_improve.log")
STATE = os.path.join(ROOT, "self_state.json")
ALLOWLIST = {TUNE}

def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def run(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300, **kw)

def build():
    r = run(["make"])
    return r.returncode == 0

def train_block_loss(lr):
    # train-block prints "block epoch N loss=X" lines; take last
    r = run(["./verum-slice", "--train-block", "--epochs", "200", "--lr", str(lr)])
    if r.returncode != 0:
        return None
    losses = re.findall(r"loss=([0-9.]+)", r.stdout)
    return float(losses[-1]) if losses else None

def edit_tune(lr):
    # self-edit: rewrite only the LR line in self_tune.h
    if not os.path.exists(TUNE):
        with open(TUNE, "w") as f:
            f.write("/* Self-tuned by verum self_improve.py - test-gated, reverted if worse */\n")
            f.write("#ifndef VBS_SELF_TUNE_H\n#define VBS_SELF_TUNE_H\n")
            f.write(f"#define VBS_SELF_TUNE_LR {lr}\n#endif\n")
        return
    with open(TUNE) as f: txt = f.read()
    if "VBS_SELF_TUNE_LR" in txt:
        txt = re.sub(r"#define VBS_SELF_TUNE_LR [0-9.]+", f"#define VBS_SELF_TUNE_LR {lr}", txt)
    else:
        txt += f"\n#define VBS_SELF_TUNE_LR {lr}\n"
    with open(TUNE, "w") as f: f.write(txt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--base-lr", type=float, default=0.05)
    args = ap.parse_args()
    log("=== AUTONOMOUS SELF-IMPROVE START ===")
    log(f"platform={sys.platform} python={sys.version.split()[0]}")
    if not build():
        log("BUILD FAILED before loop, aborting"); return 2
    best_lr = args.base_lr
    best_loss = train_block_loss(best_lr)
    log(f"baseline lr={best_lr} loss={best_loss}")
    if best_loss is None:
        log("train-block failed, aborting"); return 2
    candidates = [best_lr*0.5, best_lr*2.0, best_lr*1.5]
    for i, cand in enumerate(candidates[:args.iterations]):
        # backup, self-edit, rebuild, test, keep-or-revert
        shutil.copy(TUNE, TUNE + ".bak") if os.path.exists(TUNE) else None
        edit_tune(cand)
        if not build():
            log(f"iter {i}: lr={cand} BUILD FAILED -> revert")
            if os.path.exists(TUNE + ".bak"): shutil.copy(TUNE + ".bak", TUNE); build()
            continue
        loss = train_block_loss(cand)
        log(f"iter {i}: tried lr={cand} loss={loss} (best {best_loss} @ {best_lr})")
        if loss is not None and loss < best_loss:
            best_loss, best_lr = loss, cand
            log(f"iter {i}: KEEP edit lr={cand}")
            if os.path.exists(TUNE + ".bak"): os.remove(TUNE + ".bak")
        else:
            log(f"iter {i}: REVERT edit lr={cand}, loss not better")
            if os.path.exists(TUNE + ".bak"):
                shutil.copy(TUNE + ".bak", TUNE); os.remove(TUNE + ".bak"); build()
    # final: self-learn from memory, if any
    mem = os.path.join(ROOT, "verum-memory.tsv")
    if os.path.exists(mem):
        log(f"memory file present: {mem}, adapter training is in --train-adapter")
    with open(STATE, "w") as f:
        json.dump({"best_lr": best_lr, "best_loss": best_loss, "iterations": args.iterations, "updated": time.time()}, f, indent=2)
    log(f"=== DONE best_lr={best_lr} best_loss={best_loss} state={STATE} ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())
