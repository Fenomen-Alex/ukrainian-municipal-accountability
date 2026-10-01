# Unattended driver for the v3.2 same-object treatment training run.
#
# This is byte-for-byte the corrected-v3 driver (run_v3_corrected.sh) with three
# path changes and nothing else: DATA, the adapter output directory, and the
# log path. Same base model, LoRA rank/scale/dropout/layers, batch, grad accum,
# lr, seed, max-seq, save-every, val-batches, reporting cadence, resume policy
# and one-epoch rule. That is deliberate: v3.2 is a single-change experiment,
# so the recipe must not drift along with the corpus.
#
# The iteration count is derived from the corpus at run time rather than
# hardcoded, which is what the corrected run's 9,447 rows got wrong
# (iters == 4,724 but batches == 4,723, so the last row was dropped and the
# wrap replayed a reshuffled second epoch). v3.2's 9,708 rows make
# iters == batches == 4,854 exactly.
#
# Corpus: ml/data/tune/v3/treatment_v3_2/train.jsonl, 9708 rows =
#   5157 single + 1872 terse + 2418 multitopic (C4) + 261 same-object,
# sha256 c304c374af6aa9b9da06a3c47bf47881cd27249ea9b389abd2e1ecdf43ac622e.
# The first 9,447 rows are byte-identical to corrected v3.
#
# Usage:
#   nohup bash ml/tune/run_v3_2.sh \
#     > ml/data/tune/v3/logs/v3_2_driver.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv-mlx/bin/python"
DATA="$ROOT/ml/data/tune/v3/treatment_v3_2"
ADAPTERS="$ROOT/ml/data/tune/adapters"
OUT="$ADAPTERS/qwen3-8b-lora-v3-2"
LOGDIR="$ROOT/ml/data/tune/v3/logs"
LOG="$LOGDIR/v3_2_train.log"
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
N_BATCHES=$(( N_ROWS / BATCH ))

echo "=== v3.2 SAME-OBJECT treatment full-corpus run ==="
echo "train rows         : $N_ROWS"
echo "batch size         : $BATCH"
echo "full batches       : $N_BATCHES"
echo "iterations         : $ITERS"
if [ "$ITERS" -eq "$N_BATCHES" ]; then
  echo "coverage           : exact (iters == batches, every row seen once)"
else
  echo "coverage           : WARNING iters != batches, a row will be dropped or replayed"
fi
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

# Fresh run only: never clobber a running/incomplete run.
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