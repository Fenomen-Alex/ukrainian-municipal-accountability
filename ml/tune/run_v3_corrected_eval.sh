#!/usr/bin/env bash
# Score the corrected v3 treatment arm on every suite, sequentially.
#
# Identical suite selection and flags to run_v3_eval.sh; only the fused model
# path and the result tag differ, so the previous v3-treatment results and the
# v2 baselines are preserved for a paired comparison.
#
# Tag "v3-corrected" is what gates_v3.py --arm v3-corrected reads:
#   eval_v3/results/v3-corrected.json          (146 cases, cleaned labels)
#   eval/v3-corrected.json                     (frozen 329, v1 weak labels)
#   multitopic/results/v3-corrected.json       (85 cases)
#   smoke/results/v3-corrected/                (20 cases)

set -uo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"
PY=".venv-mlx/bin/python"
FUSED="ml/data/tune/adapters/qwen3-8b-lora-v3-corrected-fused"
TAG="v3-corrected"
LOGDIR="ml/data/tune/v3/logs"
mkdir -p "$LOGDIR"

echo "[$(date '+%H:%M:%S')] eval_v3 146 ..."
"$PY" -m ml.tune.run_eval_v3 --model "$FUSED" --tag "$TAG" --max-tokens 800 \
    > "$LOGDIR/eval_v3_corrected.log" 2>&1
echo "[$(date '+%H:%M:%S')] multitopic 85 ..."
"$PY" -m ml.tune.run_eval_multitopic --mode lora --model "$FUSED" --tag "$TAG" --max-tokens 1024 \
    > "$LOGDIR/eval_mt_corrected.log" 2>&1
echo "[$(date '+%H:%M:%S')] frozen 329 ..."
"$PY" -m ml.tune.run_eval --mode lora --model "$FUSED" --tag "$TAG" --max-tokens 1024 \
    > "$LOGDIR/eval_frozen_corrected.log" 2>&1
echo "[$(date '+%H:%M:%S')] smoke 20 ..."
"$PY" -m ml.tune.run_smoke --model "$FUSED" --tag "$TAG" \
    > "$LOGDIR/smoke_corrected.log" 2>&1
echo "[$(date '+%H:%M:%S')] all evals done"
