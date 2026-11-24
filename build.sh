#!/bin/sh
# build.sh — one-command build for any computer (Linux, macOS,
# Windows under MinGW/MSYS2). Needs only a C compiler and make;
# no GPU and no extra dependencies. GPU backends are runtime-optional.
set -e
CC=${CC:-$(command -v clang || command -v gcc || command -v cc)}
OS=$(uname -s 2>/dev/null || echo unknown)
echo "Building with $CC on $OS"
make CC="$CC"
./verum-slice --backends || true
echo "Build OK: ./verum-slice"
