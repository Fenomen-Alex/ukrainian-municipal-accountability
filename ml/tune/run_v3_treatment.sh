# Unattended driver for the full-corpus v3 treatment training run.
#
# Purpose: the short 800-step v3 run covered only ~18.8 % of the corpus
# (1600 of 8519 examples) and its control arm failed validation (V3_REPORT.md).
# This driver trains the treatment on a budget that shows the model every
# training example: iters = ceil(train_rows / batch_size) = one full epoch.
#
# Reliability:
#   * every save_every iterations a numbered {step}_adapters.safetensors
#     checkpoint is written (adapters only, ~38 MB each, not full models);
#   * if the process dies for an infrastructure reason, the next loop iteration
#     resumes from the highest numbered checkpoint via --resume-adapter-file;
#     the seeded batch order makes the replayed prefix identical, so the
#     continuation is the same sequence that would have followed the checkpoint;
#   * completion is decided by ml.tune.training_state (numbered checkpoints run
#     continuously to the end AND the log contains the trainer's final marker),
#     never by process exit alone;
#   * three consecutive restarts with no log growth abort instead of spinning.
#
# Usage:
#   nohup bash ml/tune/run_v3_treatment.sh > ml/data/tune/v3/logs/treatment_driver.log 2>&1 &
#
# The training log is written to ml/data/tune/v3/logs/treatment_train.log
# (both paths are gitignored).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv-mlx/bin/python"
DATA="$ROOT/ml/data/tune/v3/treatment"
ADAPTERS="$ROOT/ml/data/tune/adapters"
OUT="$ADAPTERS/qwen3-8b-lora-v3-treatment"
LOGDIR="$ROOT/ml/data/tune/v3/logs"
LOG="$LOGDIR/treatment_train.log"
mkdir -p "$LOGDIR"

BATCH=2
SEED=42
LR=1e-4
NUM_LAYERS=16
GRAD_ACC=2
MAX_SEQ=2048
SAVE_EVERY=500

N_ROWS=$(wc -l < "$DATA/train.jsonl")
N_BATCHES=$(( (N_ROWS - BATCH) / BATCH + 1 ))
ITERS=$(( (N_ROWS + BATCH - 1) / BATCH ))
COVERAGE_PCT=$(( ITERS * 100 / N_BATCHES ))

echo "=== v3 treatment full-corpus run ==="
echo "train rows         : $N_ROWS"
echo "batch size         : $BATCH"
echo "iterations         : $ITERS"
echo "grad accumulation  : $GRAD_ACC (effective batch $((BATCH * GRAD_ACC)))"
echo "unique batches     : $N_BATCHES"
echo "corpus coverage    : ${COVERAGE_PCT}% (one full pass, every train example once)"
echo "model              : mlx-community/Qwen3-8B-4bit"
echo "LoRA               : rank 8, scale 20, dropout 0, layers $NUM_LAYERS"
echo "learning rate      : $LR (adamw)"
echo "seed               : $SEED"
echo "max seq            : $MAX_SEQ, mask_prompt=True, grad-checkpoint"
echo "save_every         : $SAVE_EVERY (~38 MB adapter ckpt each, $((ITERS / SAVE_EVERY)) mid-run)"
echo "adapters out       : $OUT"
echo "estimates          : ~20 s/iter -> ~$(( ITERS * 20 / 3600 )) h run (measured 18-22 s/iter)"
echo "start              : $(date '+%Y-%m-%d %H:%M:%S')"

rm -rf "$OUT"
mkdir -p "$OUT"

wait_done() {
  if "$PY" -m ml.tune.training_state --dir "$OUT" --iters "$ITERS" \
      --save-every "$SAVE_EVERY" --log "$LOG" >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

run_or_resume() {
  local latest resume_args=()
  latest=$(ls "$OUT"/*_adapters.safetensors 2>/dev/null | sort -V | tail -n1 || true)
  if [ -n "$latest" ]; then
    resume_args=(--resume-adapter-file "$latest")
    echo "[$(date '+%H:%M:%S')] resuming from $latest"
  else
    echo "[$(date '+%H:%M:%S')] starting fresh"
  fi
  "$PY" -m ml.tune.run_train \
      --model mlx-community/Qwen3-8B-4bit \
      --data "$DATA" \
      --adapter-path "$OUT" \
      --fine-tune-type lora --num-layers "$NUM_LAYERS" \
      --batch-size "$BATCH" --iters "$ITERS" --val-batches 10 \
      --learning-rate "$LR" --max-seq-length "$MAX_SEQ" \
      --grad-checkpoint --grad-accumulation-steps "$GRAD_ACC" \
      --steps-per-report 25 --steps-per-eval 50 \
      --save-every "$SAVE_EVERY" --seed "$SEED" \
      ${resume_args[@]+"${resume_args[@]}"} >> "$LOG" 2>&1 || true
}

fast_fail=0
while true; do
  if wait_done; then
    echo "=== training COMPLETE at $(date '+%Y-%m-%d %H:%M:%S') ==="
    "$PY" -m ml.tune.training_state --dir "$OUT" --iters "$ITERS" \
        --save-every "$SAVE_EVERY" --log "$LOG" --describe
    exit 0
  fi
  local_before=$(stat -f%m "$LOG" 2>/dev/null || echo 0)
  run_or_resume
  local_after=$(stat -f%m "$LOG" 2>/dev/null || echo 0)
  if [ "$local_after" -le "$local_before" ]; then
    fast_fail=$((fast_fail + 1))
    echo "[$(date '+%H:%M:%S')] run produced no log growth ($fast_fail/3)"
    [ "$fast_fail" -ge 3 ] && { echo "ABORT: 3 consecutive silent failures"; exit 1; }
  else
    fast_fail=0
  fi
  if wait_done; then continue; fi
  sleep 60
done