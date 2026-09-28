#!/usr/bin/env bash
# Run the remaining v3 experiment unattended, one GPU job at a time.
#
# Why a driver rather than a sequence of manual commands: the two arms are
# ~5h of training each, and the 16GB of unified memory means a second MLX process
# cannot run alongside the first. Every step below therefore has to be serialised,
# and the ordering is not obvious from the file names -- the treatment must not
# start until the control has been evaluated, or the control's scores would be
# collected on a machine that is also training.
#
# It waits for the already-running control training rather than starting it, so
# this script is safe to run while that job is in flight.
#
# Usage:  bash ml/tune/run_v3_experiment.sh
# Logs:   ml/data/tune/v3/logs/

set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$PWD"

MLX=".venv-mlx/bin/python"
PY=".venv/bin/python"
ADAPTERS="ml/data/tune/adapters"
LOGS="ml/data/tune/v3/logs"
BASE="mlx-community/Qwen3-8B-4bit"

CONTROL_ADAPTER="$ADAPTERS/qwen3-8b-lora-v3-control"
CONTROL_FUSED="$ADAPTERS/qwen3-8b-lora-v3-control-fused"
TREAT_ADAPTER="$ADAPTERS/qwen3-8b-lora-v3-treatment"
TREAT_FUSED="$ADAPTERS/qwen3-8b-lora-v3-treatment-fused"
TREAT_DATA="ml/data/tune/v3/treatment"

# The control was launched before this driver existed, so its log lives where it
# was started. training_state needs it: checkpoints alone cannot tell a finished
# run from a crash between the last periodic save and the end of training.
CONTROL_LOG="${CONTROL_LOG:-/var/folders/bc/8s_rgxsx0xx6hnv5xt2p_0cc0000gn/T/opencode/control_train.log}"

mkdir -p "$LOGS"

step() { printf '\n=== [%s] %s ===\n' "$(date +%H:%M:%S)" "$*" >&2; }

# Log a command to its own file, then echo a one-line outcome. A step that dies
# leaves its log intact, which is the only thing that makes a 10-hour run
# debuggable after the fact.
run() {
  local name="$1"; shift
  step "$name"
  if "$@" >"$LOGS/$name.log" 2>&1; then
    printf '  ok: %s -> %s\n' "$name" "$LOGS/$name.log" >&2
  else
    printf '  FAILED: %s -- see %s\n' "$name" "$LOGS/$name.log" >&2
    tail -20 "$LOGS/$name.log" >&2
    return 1
  fi
}

# mlx_lm rewrites adapters.safetensors at *every* checkpoint, so its presence does
# not mean the run finished -- waiting on it would fuse a 100-iteration adapter and
# call it the finished arm. ml.tune.training_state checks the numbered checkpoint
# sequence and the "Saved final weights to" line the trainer only prints after its
# loop; see that module for why both are required. The timeout is generous because
# the run is unattended: a short timeout would abort a healthy job, a long one only
# wastes a poll.
wait_for_training() {
  local dir="$1" log="$2" limit="${3:-43200}" elapsed=0
  step "waiting for training: $dir (limit ${limit}s)"
  while ! "$PY" -m ml.tune.training_state --dir "$dir" --iters 800 \
                 --save-every 100 --log "$log"; do
    if [ "$elapsed" -ge "$limit" ]; then
      printf '  timed out after %ss waiting for %s\n' "$limit" "$dir" >&2
      "$PY" -m ml.tune.training_state --dir "$dir" --iters 800 \
        --save-every 100 --log "$log" --describe >&2 || true
      return 1
    fi
    sleep 60; elapsed=$((elapsed + 60))
    if [ $((elapsed % 1800)) -eq 0 ]; then
      printf '  %ss: %s\n' "$elapsed" "$(ls "$dir" 2>/dev/null | tr '\n' ' ')" >&2
    fi
  done
  printf '  training complete after %ss\n' "$elapsed" >&2
  "$PY" -m ml.tune.training_state --dir "$dir" --iters 800 \
    --save-every 100 --log "$log" --describe >&2
}

# Every arm is scored by the same four runners under the same settings, and each
# gets its own tag so the results files cannot overwrite one another.
score_arm() {
  local fused="$1" tag="$2"
  run "$tag-eval-v3"     $MLX -m ml.tune.run_eval_v3 --model "$fused" --tag "$tag"
  run "$tag-eval-329"    $MLX -m ml.tune.run_eval --mode lora --model "$fused" --tag "$tag" --max-tokens 1024
  run "$tag-eval-mt"     $MLX -m ml.tune.run_eval_multitopic --mode lora --model "$fused" --tag "$tag" --max-tokens 1024
  run "$tag-smoke"       $MLX -m ml.tune.run_smoke --model "$fused" --tag "$tag"
}

# ---- 1. control -------------------------------------------------------------
wait_for_training "$CONTROL_ADAPTER" "$CONTROL_LOG"
run control-fuse $MLX -m ml.tune.fuse_v3 --adapter "$CONTROL_ADAPTER" --out "$CONTROL_FUSED"
score_arm "$CONTROL_FUSED" v3-control

# ---- 2. treatment -----------------------------------------------------------
# Same recipe as the control and as v2; only the data and the output path move.
step "training treatment (800 iters, expect ~5h)"
$MLX -m ml.tune.run_train \
  --iters 800 --batch-size 2 --grad-accumulation-steps 2 \
  --num-layers 16 --max-seq-length 2048 --learning-rate 1e-4 \
  --steps-per-eval 50 --val-batches 10 --steps-per-report 25 --save-every 100 --seed 42 \
  --data "$TREAT_DATA" --adapter-path "$TREAT_ADAPTER" >"$LOGS/treatment-train.log" 2>&1
run treatment-fuse $MLX -m ml.tune.fuse_v3 --adapter "$TREAT_ADAPTER" --out "$TREAT_FUSED"
score_arm "$TREAT_FUSED" v3-treatment

# ---- 3. analysis ------------------------------------------------------------
run compare $PY -m ml.tune.compare_arms --arms v2 v3-control v3-treatment \
  --markdown --out "$LOGS/comparison.md"

step "done"
printf 'comparison: %s\n' "$LOGS/comparison.md" >&2
