# Unattended driver for the corrected full-corpus v3 treatment training run.
#
# Identical recipe to run_v3_treatment.sh (same base model, LoRA, batch,
# grad-accum, lr, seed, layers, max-seq, save-every, val-batches, reporting
# cadence, one epoch at iters = ceil(rows / batch) = 4724). The ONLY change is
# the output directory and log path, so the previous v3-treatment adapter/fused
# model and the canonical v2 artifact are left untouched and can be compared
# against this arm.
#
# Corpus: ml/data/tune/v3/treatment/train.jsonl @ e8f336c (R1/R2/R3 corrected),
# 9447 rows = 5157 single + 1872 terse + 2418 multitopic, sha256
# 22e85d80b7a7abc12d0629ffc41fd125a2b718125e01f62fe9cbcd92815ef9f2.
#
# Usage:
#   nohup bash ml/tune/run_v3_corrected.sh \
#     > ml/data/tune/v3/logs/corrected_driver.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv-mlx/bin/python"
DATA="$ROOT/ml/data/tune/v3/treatment"
ADAPTERS="$ROOT/ml/data/tune/adapters"
OUT="$ADAPTERS/qwen3-8b-lora-v3-corrected"
LOGDIR="$ROOT/ml/data/tune/v3/logs"
LOG="$LOGDIR/corrected_train.log"
mkdir -p "$LOGDIR"

BATCH=2
SEED=42
LR=1e-4
NUM_LAYERS=16
GRAD_ACC=2
MAX_SEQ=2048
SAVE_EVERY=500

N_ROWS=$(wc -l < "$DATA/train.jsonl")
ITERS=$(( (N_ROWS + BATCH - 1) / BATCH ))

echo "=== v3 CORRECTED treatment full-corpus run ==="
echo "train rows         : $N_ROWS"
echo "batch size         : $BATCH"
echo "iterations         : $ITERS"
echo "grad accumulation  : $GRAD_ACC (effective batch $((BATCH * GRAD_ACC)))"
echo "model              : mlx-community/Qwen3-8B-4bit"
echo "LoRA               : rank 8, scale 20, dropout 0, layers $NUM_LAYERS"
echo "learning rate      : $LR (adamw)"
echo "seed               : $SEED"
echo "max seq            : $MAX_SEQ, mask_prompt=True, grad-checkpoint"
echo "save_every         : $SAVE_EVERY"
echo "adapters out       : $OUT"
echo "git HEAD           : $(git rev-parse HEAD)"
echo "start              : $(date '+%Y-%m-%d %H:%M:%S')"
echo "corpus sha256      : $(shasum -a 256 "$DATA/train.jsonl" | cut -d' ' -f1)"

# Fresh run only: never clobber a running/incomplete correction.
if [ -d "$OUT" ] && ls "$OUT"/*_adapters.safetensors >/dev/null 2>&1; then
  echo "NOTE: $OUT already has numbered checkpoints; resuming rather than wiping."
else
  rm -rf "$OUT"
  mkdir -p "$OUT"
fi

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
