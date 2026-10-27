#!/bin/sh
# Automated start: no human trigger after this.
# Usage: ./run-daemon.sh [interval_seconds]
INTERVAL=${1:-300}
cd "$(dirname "$0")"
nohup python3 verum-daemon.py --cycles 0 --interval "$INTERVAL" > daemon.out 2>&1 &
echo "Verum daemon started pid $! interval ${INTERVAL}s log daemon.log"
