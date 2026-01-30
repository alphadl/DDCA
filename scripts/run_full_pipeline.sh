#!/usr/bin/env bash
# One-click pipeline: prepare data -> (demo inference or VERL training) -> evaluate.
# Use small dataset by default so it runs quickly without GPU/VERL.
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

DATA_DIR="${DATA_DIR:-data/processed}"
TRAIN_SIZE="${TRAIN_SIZE:-200}"
VAL_SIZE="${VAL_SIZE:-50}"
USE_MATH="${USE_MATH:-0}"
BUILTIN_ONLY="${BUILTIN_ONLY:-0}"  # 1 = no HF download, use built-in samples (fast demo)
SEED="${SEED:-42}"
RESULTS_FILE="${RESULTS_FILE:-}"   # If set, skip demo and use this for evaluation

echo "=== DCA full pipeline (data_dir=$DATA_DIR, train=$TRAIN_SIZE, val=$VAL_SIZE) ==="

# Step 1: Prepare data (parquet + jsonl)
echo "[1/3] Preparing data..."
python scripts/prepare_data.py \
  --output_dir "$DATA_DIR" \
  --train_size "$TRAIN_SIZE" \
  --val_size "$VAL_SIZE" \
  --seed "$SEED" \
  $([ "$USE_MATH" = "1" ] && echo "--use_math") \
  $([ "$BUILTIN_ONLY" = "1" ] && echo "--builtin_only")

# Step 2: Inference / training
if [ -n "$RESULTS_FILE" ] && [ -f "$RESULTS_FILE" ]; then
  echo "[2/3] Using existing results: $RESULTS_FILE"
  EVAL_RESULTS="$RESULTS_FILE"
else
  echo "[2/3] Demo inference (no VERL); for real training run VERL with $DATA_DIR/train.parquet"
  EVAL_RESULTS="$DATA_DIR/results_demo.jsonl"
  python scripts/demo_inference.py \
    --input "$DATA_DIR/val.jsonl" \
    --output "$EVAL_RESULTS" \
    --seed "$SEED" \
    --correct_ratio 0.6 \
    --k_rollouts 1
fi

# Step 3: Evaluate
echo "[3/3] Evaluating..."
python scripts/evaluate.py --results "$EVAL_RESULTS" --k 1

echo "=== Pipeline done. ==="
echo "  Data: $DATA_DIR (train.parquet, val.jsonl, test_gsm8k.jsonl)"
echo "  To run with VERL: set DATA_DIR and run VERL; then run with RESULTS_FILE=/path/to/results.jsonl"
