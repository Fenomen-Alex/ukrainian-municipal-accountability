#!/usr/bin/env bash
# Score the fused v3-treatment arm on every suite, sequentially. Logs land in
# ml/data/tune/v3/logs/. Run detached (tmux) so an unplugged terminal cannot
# kill the evaluations.
set -uo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"
PY=".venv-mlx/bin/python"
FUSED="ml/data/tune/adapters/qwen3-8b-lora-v3-fused"
LOGDIR="ml/data/tune/v3/logs"
mkdir -p "$LOGDIR"
echo "[$(date '+%H:%M:%S')] frozen 329 ..."
"$PY" -m ml.tune.run_eval --mode lora --model "$FUSED" --tag v3-treatment --max-tokens 1024 > "$LOGDIR/eval_frozen_treatment.log" 2>&1
echo "[$(date '+%H:%M:%S')] multitopic 85 ..."
"$PY" -m ml.tune.run_eval_multitopic --mode lora --model "$FUSED" --tag v3-treatment --max-tokens 1024 > "$LOGDIR/eval_mt_treatment.log" 2>&1
echo "[$(date '+%H:%M:%S')] smoke 20 ..."
"$PY" -m ml.tune.run_smoke --model "$FUSED" --tag v3-treatment > "$LOGDIR/smoke_treatment.log" 2>&1
echo "[$(date '+%H:%M:%S')] all evals done"