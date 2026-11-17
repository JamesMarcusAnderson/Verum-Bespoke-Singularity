#!/bin/sh
# run_phase_d_demo.sh — end-to-end Phase D demonstration + checks.
# Run from the repo root:  make test-d
# Builds a synthetic tiny model, then exercises:
#   1. --hotbench (all kernel variants, interleaved ratios)
#   2. --loop (recursive loop, >=3 unsupervised iterations, live session)
#   3. --loop --loop-poison (rollback demonstration)
#   4. --fuse twice (cache MISS then HIT)
#   5. tune profile embed + --tune-show on a copied binary
#   6. daemon dry run (2 cycles with the loop enabled)
# Exits non-zero if any check fails. All artifacts land in .verum-demo/.
set -e
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
DEMO=.verum-demo
rm -rf "$DEMO"
mkdir -p "$DEMO"
MODEL="$DEMO/tiny.gguf"
FAIL=0
check() { # check <description> <condition...>
    desc="$1"; shift
    if "$@"; then echo "PASS: $desc";
    else echo "FAIL: $desc"; FAIL=1; fi
}

echo "=== [0] synthetic tiny model ==="
python3 tools/make_tiny_gguf.py --out "$MODEL" --emb 512 --ffn 1408 --layers 2 --heads 8 --vocab 256 --ctx 128
check "tiny model exists" test -s "$MODEL"

echo "=== [1] hotbench ==="
./verum-slice --hotbench --tune-out "$DEMO/tune.json" > "$DEMO/hotbench.txt" 2>&1
cat "$DEMO/hotbench.txt"
check "hotbench: avx2 correct" grep -q '^avx2 .*yes' "$DEMO/hotbench.txt"
check "hotbench: unroll4 correct" grep -q '^unroll4 .*yes' "$DEMO/hotbench.txt"
check "hotbench: wrong-result rejected" grep -q '^wrong-result .*NO' "$DEMO/hotbench.txt"
check "hotbench: broken-syntax compile failed" grep -q '^broken-syntax .*COMPILE FAILED' "$DEMO/hotbench.txt"
check "hotbench: tune profile written" test -s "$DEMO/tune.json"

echo "=== [2] recursive loop (live session, unsupervised) ==="
./verum-slice --loop --prompt-ids 1,2,3 --tokens 64 --iterations 6 \
    --loop-log "$DEMO/loop.jsonl" --loop-work "$DEMO/loop-work" --repo "$ROOT" \
    "$MODEL" > "$DEMO/loop.txt" 2>&1
cat "$DEMO/loop.txt"
check "loop: >=3 iterations" sh -c "grep -c '\"event\":\"candidate_start\"' $DEMO/loop.jsonl | awk '{exit !(\$1>=3)}'"
check "loop: at least one swap" grep -q '"event":"candidate_decision".*"decision":"accepted"' "$DEMO/loop.jsonl"
check "loop: wrong candidate rejected" grep -q '"candidate":"wrong-result".*"reason":"incorrect"' "$DEMO/loop.jsonl"
check "loop: broken candidate rejected" grep -q '"candidate":"broken-syntax".*"reason":"compile_failed"' "$DEMO/loop.jsonl"
check "loop: slow candidate rejected" grep -q '"candidate":"slow-wasteful".*"reason":"no_improvement"' "$DEMO/loop.jsonl"
check "loop: session generated tokens" grep -q '"tokens_generated":[1-9]' "$DEMO/loop.jsonl"

echo "=== [3] rollback (poison fault injection) ==="
./verum-slice --loop --prompt-ids 1,2,3 --tokens 64 --iterations 2 --loop-poison \
    --loop-log "$DEMO/loop-poison.jsonl" --loop-work "$DEMO/loop-work-poison" --repo "$ROOT" \
    "$MODEL" > "$DEMO/loop-poison.txt" 2>&1
cat "$DEMO/loop-poison.txt"
check "poison: injection logged" grep -q '"event":"poison_inject"' "$DEMO/loop-poison.jsonl"
check "poison: rollback happened" grep -q '"event":"rollback"' "$DEMO/loop-poison.jsonl"

echo "=== [4] fused FFN kernel (cache MISS then HIT) ==="
./verum-slice --fuse --cache-dir "$DEMO/fuse-cache" --tokens 32 "$MODEL" > "$DEMO/fuse1.txt" 2>&1
cat "$DEMO/fuse1.txt"
check "fuse: first run is a cache MISS" grep -q 'cache MISS' "$DEMO/fuse1.txt"
check "fuse: token sequences identical" grep -q 'token sequences IDENTICAL' "$DEMO/fuse1.txt"
./verum-slice --fuse --cache-dir "$DEMO/fuse-cache" --tokens 32 "$MODEL" > "$DEMO/fuse2.txt" 2>&1
cat "$DEMO/fuse2.txt"
check "fuse: second run is a cache HIT" grep -q 'cache HIT' "$DEMO/fuse2.txt"

echo "=== [5] tune profile travels with the binary ==="
sh tools/vbs_tune_embed.sh ./verum-slice "$DEMO/tune.json" "$DEMO/verum-slice-tuned"
mkdir -p "$DEMO/copy-dir"
cp "$DEMO/verum-slice-tuned" "$DEMO/copy-dir/verum-slice-tuned-copy"
check "tuned copy: embedded profile matches" sh -c "$DEMO/copy-dir/verum-slice-tuned-copy --tune-show | grep -q 'MATCHES this machine'"
"$DEMO/copy-dir/verum-slice-tuned-copy" --tune-show

echo "=== [6] daemon dry run (2 cycles, loop enabled) ==="
python3 verum-daemon.py --cycles 2 --interval 1 --loop-model "$MODEL" --loop-iterations 2 \
    > "$DEMO/daemon.txt" 2>&1 || true
cat "$DEMO/daemon.txt"
check "daemon: ran loop cycles" grep -q 'loop rc=0' "$DEMO/daemon.txt"

echo "=== demo artifacts in $DEMO ==="
ls "$DEMO"
if [ "$FAIL" -eq 0 ]; then echo "PHASE D DEMO: ALL CHECKS PASSED"; else echo "PHASE D DEMO: FAILURES PRESENT"; fi
exit $FAIL
