#!/bin/sh
# Portable build for any computer: Linux, macOS, Windows (MinGW/MSYS2)
# No GPU, no extra deps. GPU backends are runtime-optional.
set -e
CC=${CC:-$(command -v clang || command -v gcc || command -v cc)}
echo "Building with $CC on $(uname -s 2>/dev/null || echo Windows)"
make CC="$CC"
./verum-slice --backends || true
echo "Build OK: ./verum-slice"
