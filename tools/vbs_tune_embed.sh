#!/bin/sh
# vbs_tune_embed.sh — embed a tune profile JSON into a copy of a binary.
#
# Usage: tools/vbs_tune_embed.sh <binary> <profile.json> [output-binary]
#
# The profile rides as a read-only ELF section named .vbs_tune, which
# the engine reads back at startup from /proc/self/exe (see vbs_tune.c).
# The input binary is never modified in place: the output is a new file
# (default: <binary>.tuned). Copying the output binary carries the
# profile with it — that is the point.
#
# Requires objcopy (binutils). Linux ELF64 only.
set -e

fail() { echo "$*" >&2; exit 1; }

[ $# -ge 2 ] || { echo "usage: $0 <binary> <profile.json> [output-binary]" >&2; exit 64; }
BIN="$1"
PROFILE="$2"
OUT="${3:-$BIN.tuned}"

[ -f "$BIN" ] || fail "no such binary: $BIN"
[ -f "$PROFILE" ] || fail "no such profile: $PROFILE"
command -v objcopy >/dev/null 2>&1 || fail "objcopy not found (binutils)"
grep -q '"format": "vbs-tune-profile-v1"' "$PROFILE" ||
    fail "not a vbs tune profile: $PROFILE"

cp "$BIN" "$OUT"
# Drop any previous embedded profile first, so re-embedding replaces it.
objcopy --remove-section .vbs_tune "$OUT" 2>/dev/null || true
objcopy --add-section .vbs_tune="$PROFILE" \
        --set-section-flags .vbs_tune=readonly,data \
        "$OUT"
chmod +x "$OUT"
echo "embedded $PROFILE into $OUT (.vbs_tune section)"
echo "verify: $OUT --tune-show"
