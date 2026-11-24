#!/bin/sh
# run_phase_d_demo.sh — end-to-end Phase D demonstration + checks.
# Run from the repo root:  make test-d
#
# Builds a synthetic tiny model, then exercises:
#   1. --hotbench (all kernel variants, interleaved ratios)
#   2. --loop (recursive loop, >=3 unsupervised iterations, live session)
#   3. --loop --loop-poison (rollback demonstration)
#   4. --fuse twice (cache MISS then HIT)
#   5. tune profile embed + --tune-show on a copied binary
#   6. daemon dry run (2 cycles with the loop enabled)
#
# Exits non-zero if any check fails. All artifacts land in .verum-demo/.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

DEMO=.verum-demo
MODEL="$DEMO/tiny.gguf"
FAILURES=0

# expect <label> <command...>: run the command, print PASS/FAIL, count it.
expect() {
    label="$1"; shift
    if "$@" >/dev/null 2>&1; then
        echo "PASS: $label"
    else
        echo "FAIL: $label"
        FAILURES=1
    fi
}

# expect_text <label> <file> <pattern>: the file must contain the pattern.
expect_text() {
    expect "$1" grep -q "$3" "$2"
}

# expect_count <label> <file> <pattern> <min>: the pattern must appear
# at least <min> times in the file.
expect_count() {
    expect "$1" sh -c "test \$(grep -c '$3' '$2') -ge $4"
}

section() { echo "=== $* ==="; }

rm -rf "$DEMO"
mkdir -p "$DEMO"

section "[0] synthetic tiny model"
python3 tools/make_tiny_model.py --out "$MODEL" --emb 512 --ffn 1408 \
    --layers 2 --heads 8 --vocab 256 --ctx 128
expect "tiny model exists" test -s "$MODEL"

section "[1] hotbench"
./verum-slice --hotbench --tune-out "$DEMO/tune.json" > "$DEMO/hotbench.txt" 2>&1
cat "$DEMO/hotbench.txt"
expect_text "hotbench: avx2 correct" "$DEMO/hotbench.txt" '^avx2 .*yes'
expect_text "hotbench: unroll4 correct" "$DEMO/hotbench.txt" '^unroll4 .*yes'
expect_text "hotbench: wrong-result rejected" "$DEMO/hotbench.txt" '^wrong-result .*NO'
expect_text "hotbench: broken-syntax compile failed" "$DEMO/hotbench.txt" '^broken-syntax .*COMPILE FAILED'
expect "hotbench: tune profile written" test -s "$DEMO/tune.json"

section "[2] recursive loop (live session, unsupervised)"
./verum-slice --loop --prompt-ids 1,2,3 --tokens 64 --iterations 6 \
    --loop-log "$DEMO/loop.jsonl" --loop-work "$DEMO/loop-work" --repo "$ROOT" \
    "$MODEL" > "$DEMO/loop.txt" 2>&1
cat "$DEMO/loop.txt"
expect_count "loop: >=3 iterations" "$DEMO/loop.jsonl" '"event":"candidate_start"' 3
expect_text "loop: at least one swap" "$DEMO/loop.jsonl" '"event":"candidate_decision".*"decision":"accepted"'
expect_text "loop: wrong candidate rejected" "$DEMO/loop.jsonl" '"candidate":"wrong-result".*"reason":"incorrect"'
expect_text "loop: broken candidate rejected" "$DEMO/loop.jsonl" '"candidate":"broken-syntax".*"reason":"compile_failed"'
expect_text "loop: slow candidate rejected" "$DEMO/loop.jsonl" '"candidate":"slow-wasteful".*"reason":"no_improvement"'
expect_text "loop: session generated tokens" "$DEMO/loop.jsonl" '"tokens_generated":[1-9]'

section "[3] rollback (poison fault injection)"
./verum-slice --loop --prompt-ids 1,2,3 --tokens 64 --iterations 2 --loop-poison \
    --loop-log "$DEMO/loop-poison.jsonl" --loop-work "$DEMO/loop-work-poison" --repo "$ROOT" \
    "$MODEL" > "$DEMO/loop-poison.txt" 2>&1
cat "$DEMO/loop-poison.txt"
expect_text "poison: injection logged" "$DEMO/loop-poison.jsonl" '"event":"poison_inject"'
expect_text "poison: rollback happened" "$DEMO/loop-poison.jsonl" '"event":"rollback"'

section "[4] fused FFN kernel (cache MISS then HIT)"
./verum-slice --fuse --cache-dir "$DEMO/fuse-cache" --tokens 32 "$MODEL" > "$DEMO/fuse1.txt" 2>&1
cat "$DEMO/fuse1.txt"
expect_text "fuse: first run is a cache MISS" "$DEMO/fuse1.txt" 'cache MISS'
expect_text "fuse: token sequences identical" "$DEMO/fuse1.txt" 'token sequences IDENTICAL'
./verum-slice --fuse --cache-dir "$DEMO/fuse-cache" --tokens 32 "$MODEL" > "$DEMO/fuse2.txt" 2>&1
cat "$DEMO/fuse2.txt"
expect_text "fuse: second run is a cache HIT" "$DEMO/fuse2.txt" 'cache HIT'

section "[5] tune profile travels with the binary"
sh tools/vbs_tune_embed.sh ./verum-slice "$DEMO/tune.json" "$DEMO/verum-slice-tuned"
mkdir -p "$DEMO/copy-dir"
cp "$DEMO/verum-slice-tuned" "$DEMO/copy-dir/verum-slice-tuned-copy"
expect "tuned copy: embedded profile matches" sh -c "$DEMO/copy-dir/verum-slice-tuned-copy --tune-show | grep -q 'MATCHES this machine'"
"$DEMO/copy-dir/verum-slice-tuned-copy" --tune-show

section "[6] daemon dry run (2 cycles, loop enabled)"
python3 verum-daemon.py --cycles 2 --interval 1 --loop-model "$MODEL" --loop-iterations 2 \
    > "$DEMO/daemon.txt" 2>&1 || true
cat "$DEMO/daemon.txt"
expect_text "daemon: ran loop cycles" "$DEMO/daemon.txt" 'loop rc=0'

echo "=== demo artifacts in $DEMO ==="
ls "$DEMO"
if [ "$FAILURES" -eq 0 ]; then
    echo "PHASE D DEMO: ALL CHECKS PASSED"
else
    echo "PHASE D DEMO: FAILURES PRESENT"
fi
exit "$FAILURES"
