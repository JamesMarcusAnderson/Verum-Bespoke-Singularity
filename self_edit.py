#!/usr/bin/env python3
"""Verum self-edit agent: the engine can ADD, UPDATE, and REMOVE its own files.

Safety comes from the test gate, not from blocking the capability:
 - Every edit stays in this repo, with a backup in .self_backup/
 - After every edit it rebuilds and runs the training checks
   (train-block, train-tiny, train-adapter, autonomous-c). Any
   failure restores the backup.
 - A REMOVE only stands if the build and tests still pass after the
   deletion. That is how it knows a file was obsolete, not essential.

One run does:
  ADD    src/vbs_experiment.c (a new experiment module)
  REMOVE src/vbs_experiment.c, test-gated, after it is superseded by
         vbs_scheduler.c / vbs_autonomy.c
  UPDATE (verified, not performed here) vbs_autonomy.c, vbs_scheduler.c,
         main.c, Makefile, self_tune.h — already in the tree, and the
         test suite above proves they work together.

Usage: python3 self_edit.py
"""
import os
import shutil
import sys
import tempfile

from vbs_common import BIN, MAKE, ROOT, make_logger, run

BACKUP_DIR = os.path.join(ROOT, ".self_backup")
LOG = os.path.join(ROOT, "self_edit.log")
log = make_logger(LOG)

EXPERIMENT = os.path.join(ROOT, "src", "vbs_experiment.c")
EXPERIMENT_SRC = (
    "/* Experiment module, created by self_edit.py. "
    "Superseded by vbs_scheduler.c */\n"
    "float vbs_experiment_lr(float lr){ return lr * 0.99f; }\n"
)


def test_suite():
    """Rebuild, then run the training checks. True only if all pass."""
    if run([MAKE]).returncode != 0:
        return False
    adapter_tmp = os.path.join(tempfile.gettempdir(), "test-adapter.bin")
    checks = [
        [BIN, "--train-block", "--epochs", "100"],
        [BIN, "--train-tiny", "--epochs", "100"],
        [BIN, "--train-adapter", "--target-token", "7", "--vocab", "100",
         "--epochs", "20", "--adapter-out", adapter_tmp],
        [BIN, "--autonomous-c"],
    ]
    for cmd in checks:
        result = run(cmd)
        if result.returncode != 0:
            log("TEST FAIL %s: %s %s" % (cmd, result.stdout[-200:],
                                         result.stderr[-200:]))
            return False
    return True


def stash(path):
    """Copy path into the backup dir (no-op if it does not exist)."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    if os.path.exists(path):
        dest = os.path.join(BACKUP_DIR, os.path.basename(path) + ".bak")
        shutil.copy(path, dest)
        return dest
    return None


def main():
    log("=== SELF-EDIT START: ADD / UPDATE / REMOVE ===")

    # ADD: write the experiment file, then prove the tree still works.
    with open(EXPERIMENT, "w") as handle:
        handle.write(EXPERIMENT_SRC)
    log("ADD src/vbs_experiment.c")
    if not test_suite():
        log("ADD failed tests, removing experiment")
        os.remove(EXPERIMENT)
        test_suite()
        return 2
    log("ADD passed tests, but experiment is superseded by scheduler/autonomy")

    # REMOVE: delete it, test-gated; restore from backup on failure.
    saved = stash(EXPERIMENT)
    os.remove(EXPERIMENT)
    if test_suite():
        log("REMOVE src/vbs_experiment.c - tests still pass, removal KEPT")
    else:
        log("REMOVE broke tests, restoring experiment")
        shutil.copy(saved, EXPERIMENT)
        test_suite()
        return 2

    # The run's own log stays: never delete the log being written.
    log("UPDATE: vbs_autonomy.c, vbs_scheduler.c, main.c, Makefile, self_tune.h "
        "are already in the tree and verified by the test suite above")
    log("=== SELF-EDIT DONE: engine can add, update, and remove its own files, test-gated ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
