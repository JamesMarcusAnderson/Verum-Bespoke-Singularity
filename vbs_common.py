#!/usr/bin/env python3
"""Shared runtime helpers for the Verum Python tools (vbs_common.py).

One home for the setup every Verum Python script needs: the repo root,
the verum-slice binary path, the make command, running a subprocess in
the repo root, and timestamped logging to a per-script log file.

Usage from any script that lives in the repo root:

    from vbs_common import ROOT, BIN, MAKE, run, make_logger

    LOG = os.path.join(ROOT, "my_script.log")
    log = make_logger(LOG)

ROOT is derived from this file's own location, so it works no matter
where the script is launched from, and Python puts the launching
script's directory (the repo root) on sys.path, so the import
always resolves.
"""
import os
import shutil
import subprocess
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

# The binary is verum-slice.exe on Windows, verum-slice elsewhere.
BIN_NAME = "verum-slice.exe" if os.name == "nt" else "verum-slice"
BIN = os.path.join(ROOT, BIN_NAME)


def _find_make():
    # Plain `make` first; a MinGW install without it may only have
    # mingw32-make. Fall back to the bare name and let the shell fail
    # loudly if neither exists.
    for candidate in ("make", "mingw32-make"):
        found = shutil.which(candidate)
        if found:
            return found
    return "make"


MAKE = _find_make()


def run(cmd, **kw):
    """Run cmd in the repo root, capturing output (5-minute timeout)."""
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          timeout=300, **kw)


def make_logger(log_path):
    """Return a log(msg) that prints a timestamped line and appends it to log_path."""
    def log(msg):
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = stamp + " " + msg
        print(line, flush=True)
        with open(log_path, "a") as handle:
            handle.write(line + "\n")
    return log
