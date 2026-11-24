#!/bin/sh
# run-daemon.sh — start the self-learning daemon in the background.
# No human trigger is needed after this; the daemon keeps cycling
# on its own until killed.
# Usage: ./run-daemon.sh [interval_seconds]
INTERVAL=${1:-300}
cd "$(dirname "$0")"
nohup python3 verum-daemon.py --cycles 0 --interval "$INTERVAL" > daemon.out 2>&1 &
echo "Verum daemon started pid $! interval ${INTERVAL}s log daemon.out"
