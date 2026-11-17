#!/bin/sh
# vbs_tune_embed.sh — embed a tune profile JSON into a copy of the binary.
#
# Usage: tools/vbs_tune_embed.sh <binary> <profile.json> [output-binary]
#
# The profile is stored as a read-only ELF section named .vbs_tune, which
# the engine reads back at startup from /proc/self/exe (see vbs_tune.c).
# The input binary is never modified in place: the output is a new file
# (default: <binary>.tuned). Copying the output binary carries the profile
# with it — that is the point.
#
# Requires objcopy (binutils). Linux ELF64 only.
set -e
if [ $# -lt 2 ]; then
    echo "usage: $0 <binary> <profile.json> [output-binary]" >&2
    exit 64
fi
BIN="$1"
PROFILE="$2"
OUT="${3:-$BIN.tuned}"
if [ ! -f "$BIN" ]; then echo "no such binary: $BIN" >&2; exit 1; fi
if [ ! -f "$PROFILE" ]; then echo "no such profile: $PROFILE" >&2; exit 1; fi
command -v objcopy >/dev/null 2>&1 || { echo "objcopy not found (binutils)" >&2; exit 1; }
grep -q '"format": "vbs-tune-profile-v1"' "$PROFILE" || {
    echo "not a vbs tune profile: $PROFILE" >&2; exit 1
}
cp "$BIN" "$OUT"
# Drop any previous embedded profile first, so re-embedding replaces it.
objcopy --remove-section .vbs_tune "$OUT" 2>/dev/null || true
objcopy --add-section .vbs_tune="$PROFILE" \
        --set-section-flags .vbs_tune=readonly,data \
        "$OUT"
chmod +x "$OUT"
echo "embedded $PROFILE into $OUT (.vbs_tune section)"
echo "verify: $OUT --tune-show"
