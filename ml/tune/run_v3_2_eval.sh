#!/usr/bin/env bash
# Fuse the v3.2 adapter and score it on every suite, sequentially.
#
# Same suite selection, flags and order as run_v3_corrected_eval.sh; only the
# fused model path and the tag differ, so the v2, previous-v3 and corrected-v3
# results all stay in place for paired comparison.
#
# Tag "v3-2" writes:
#   eval_v3/results/v3-2.json        146 cases, cleaned labels
#   eval/v3-2.json                   frozen 329, v1 weak labels
#   multitopic/results/v3-2.json     85 cases
#   smoke/results/v3-2/              20 cases
#   eval_v3/behaviour/v3-2.json      SPLIT/MERGE/STOP on the 52 two-topic cases
#
# Usage: bash ml/tune/run_v3_2_eval.sh

set -uo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"
PY=".venv-mlx/bin/python"
ADAPTER="ml/data/tune/adapters/qwen3-8b-lora-v3-2"
FUSED="ml/data/tune/adapters/qwen3-8b-lora-v3-2-fused"
TAG="v3-2"
LOGDIR="ml/data/tune/v3/logs"
mkdir -p "$LOGDIR"

echo "[$(date '+%H:%M:%S')] checking training is complete ..."
if ! "$PY" -m ml.tune.training_state --dir "$ADAPTER" --iters 4854 \
      --save-every 500 --log "$LOGDIR/v3_2_train.log" >/dev/null 2>&1; then
  echo "REFUSING to fuse: the adapter is not a complete 4854-iteration run."
  echo "Fusing a partial adapter would produce a model whose score means nothing."
  exit 1
fi

echo "[$(date '+%H:%M:%S')] fuse -> $FUSED"
"$PY" -m ml.tune.fuse_v3 --adapter "$ADAPTER" --out "$FUSED" \
    > "$LOGDIR/fuse_v3_2.log" 2>&1 || { echo "fuse FAILED"; tail -20 "$LOGDIR/fuse_v3_2.log"; exit 1; }
"$PY" - << 'PYEOF'
import json
m = json.load(open("ml/data/tune/adapters/qwen3-8b-lora-v3-2-fused/fuse_manifest.json"))
print(f"  config_matches_v2: {m['config_matches_v2']}")
print(f"  fused modules    : {m['fused_modules']}")
print(f"  inherited        : {m['inherited_sha256']}")
if not m["config_matches_v2"]:
    raise SystemExit("config.json differs from the v2 artifact; arms are not comparable")
PYEOF

echo "[$(date '+%H:%M:%S')] behaviour 52 (SPLIT/MERGE/STOP) ..."
"$PY" -m ml.tune.behavior_v3 --model "$FUSED" --tag "$TAG" --max-tokens 800 \
    > "$LOGDIR/behavior_v3_2.log" 2>&1
tail -6 "$LOGDIR/behavior_v3_2.log"

echo "[$(date '+%H:%M:%S')] eval_v3 146 ..."
"$PY" -m ml.tune.run_eval_v3 --model "$FUSED" --tag "$TAG" --max-tokens 800 \
    > "$LOGDIR/eval_v3_v3_2.log" 2>&1
echo "[$(date '+%H:%M:%S')] multitopic 85 ..."
"$PY" -m ml.tune.run_eval_multitopic --mode lora --model "$FUSED" --tag "$TAG" --max-tokens 1024 \
    > "$LOGDIR/eval_mt_v3_2.log" 2>&1
echo "[$(date '+%H:%M:%S')] frozen 329 ..."
"$PY" -m ml.tune.run_eval --mode lora --model "$FUSED" --tag "$TAG" --max-tokens 1024 \
    > "$LOGDIR/eval_frozen_v3_2.log" 2>&1
echo "[$(date '+%H:%M:%S')] smoke 20 ..."
"$PY" -m ml.tune.run_smoke --model "$FUSED" --tag "$TAG" \
    > "$LOGDIR/smoke_v3_2.log" 2>&1

echo "[$(date '+%H:%M:%S')] verifying no other arm moved ..."
.venv/bin/python -m ml.tune.artifact_sha --verify

echo "[$(date '+%H:%M:%S')] all evals done"