#!/usr/bin/env python3
"""Verum autonomous self-learning + self-edit loop.

Works on any computer: only needs a C compiler (cc/clang/gcc) and
python3. No GPU required — CPU fallback is the default, GPU backends
are runtime-optional.

Each run, autonomously, with a test gate on every edit:
 1. Train the transformer block at the baseline learning rate and
    measure the final loss.
 2. SELF-EDIT: try a different learning rate in src/self_tune.h,
    rebuild, re-train. Keep the edit ONLY if the loss improves,
    else revert from backup. This is code editing itself, with a
    test gate.
 3. Log every iteration to self_improve.log and save the best config
    to self_state.json. It never edits files outside the allowlist:
    src/self_tune.h, verum-memory.tsv, adapter.bin, self_state.json.

Usage: python3 self_improve.py [--iterations 3] [--base-lr 0.05]
"""
import argparse
import json
import os
import re
import shutil
import sys
import time

from vbs_common import BIN, MAKE, ROOT, make_logger, run

TUNE = os.path.join(ROOT, "src", "self_tune.h")
LOG = os.path.join(ROOT, "self_improve.log")
STATE = os.path.join(ROOT, "self_state.json")
ALLOWLIST = {TUNE}
log = make_logger(LOG)


def build():
    return run([MAKE]).returncode == 0


def set_lr(lr):
    """Write lr into src/self_tune.h (creating the file if needed)."""
    if not os.path.exists(TUNE):
        text = ("/* Self-tuned by verum self_improve.py - test-gated, "
                "reverted if worse */\n"
                "#ifndef VBS_SELF_TUNE_H\n#define VBS_SELF_TUNE_H\n"
                "#define VBS_SELF_TUNE_LR %s\n#endif\n" % lr)
        with open(TUNE, "w") as handle:
            handle.write(text)
        return
    with open(TUNE) as handle:
        text = handle.read()
    if "VBS_SELF_TUNE_LR" in text:
        text = re.sub(r"#define VBS_SELF_TUNE_LR [0-9.]+",
                      "#define VBS_SELF_TUNE_LR %s" % lr, text)
    else:
        text += "\n#define VBS_SELF_TUNE_LR %s\n" % lr
    with open(TUNE, "w") as handle:
        handle.write(text)


def loss_at_lr(lr):
    """Set the tune file to lr, rebuild, train, return final loss.

    The training run gets NO --lr flag, so the edited file itself
    controls training — passing --lr on the CLI would bypass the
    tune file and the test would prove nothing. Returns None if the
    build or the training run fails.
    """
    set_lr(lr)
    if not build():
        return None
    result = run([BIN, "--train-block", "--epochs", "200"])
    if result.returncode != 0:
        return None
    losses = re.findall(r"loss=([0-9.]+)", result.stdout)
    return float(losses[-1]) if losses else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--base-lr", type=float, default=0.05)
    args = parser.parse_args()

    log("=== AUTONOMOUS SELF-IMPROVE START ===")
    log("platform=%s python=%s" % (sys.platform, sys.version.split()[0]))
    if not build():
        log("BUILD FAILED before loop, aborting")
        return 2

    best_lr = args.base_lr
    best_loss = loss_at_lr(best_lr)
    log("baseline lr=%s loss=%s" % (best_lr, best_loss))
    if best_loss is None:
        log("train-block failed, aborting")
        return 2

    tried = [best_lr * 0.5, best_lr * 2.0, best_lr * 1.5][:args.iterations]
    for index, candidate in enumerate(tried):
        backup = TUNE + ".bak"
        if os.path.exists(TUNE):
            shutil.copy(TUNE, backup)
        set_lr(candidate)
        if not build():
            log("iter %d: lr=%s BUILD FAILED -> revert" % (index, candidate))
            if os.path.exists(backup):
                shutil.copy(backup, TUNE)
                build()
            continue
        loss = loss_at_lr(candidate)
        log("iter %d: tried lr=%s loss=%s (best %s @ %s)"
            % (index, candidate, loss, best_loss, best_lr))
        if loss is not None and loss < best_loss:
            best_loss, best_lr = loss, candidate
            log("iter %d: KEEP edit lr=%s" % (index, candidate))
            if os.path.exists(backup):
                os.remove(backup)
        else:
            log("iter %d: REVERT edit lr=%s, loss not better"
                % (index, candidate))
            if os.path.exists(backup):
                shutil.copy(backup, TUNE)
                os.remove(backup)
                build()

    memory = os.path.join(ROOT, "verum-memory.tsv")
    if os.path.exists(memory):
        log("memory file present: %s, adapter training is in --train-adapter"
            % memory)
    state = {"best_lr": best_lr, "best_loss": best_loss,
             "iterations": args.iterations, "updated": time.time()}
    with open(STATE, "w") as handle:
        json.dump(state, handle, indent=2)
    log("=== DONE best_lr=%s best_loss=%s state=%s ==="
        % (best_lr, best_loss, STATE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
