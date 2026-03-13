#!/bin/bash
# Worker script for 4pt SFT evaluation (no reject_term_increase)
# Backtracking + proper stop

WORKER_ID=$1
N_WORKERS=$2
TOTAL_SAMPLES=$3

PYTHON="python"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

REMAINDER=$((TOTAL_SAMPLES % N_WORKERS))
BASE_SAMPLES=$((TOTAL_SAMPLES / N_WORKERS))

if [ $WORKER_ID -lt $REMAINDER ]; then
    START_IDX=$((WORKER_ID * (BASE_SAMPLES + 1)))
    END_IDX=$(((WORKER_ID + 1) * (BASE_SAMPLES + 1)))
else
    START_IDX=$((REMAINDER * (BASE_SAMPLES + 1) + (WORKER_ID - REMAINDER) * BASE_SAMPLES))
    END_IDX=$((START_IDX + BASE_SAMPLES))
fi

echo "Worker ${WORKER_ID}: evaluating 4pt samples ${START_IDX} to ${END_IDX}"

cd "${BASE_DIR}"

export PYTHONPATH="${BASE_DIR}/spinorhelicity:${BASE_DIR}/spinorhelicity/environment:${BASE_DIR}/src:${PYTHONPATH}"

$PYTHON -u src/eval_sft_4pt.py \
    --model "${BASE_DIR}/models/sft_4pt_500k/best_model.pt" \
    --test_data "${BASE_DIR}/data/paper_test_4pt.pkl" \
    --output "$BASE_DIR/output/eval_4pt/worker_${WORKER_ID}.pkl" \
    --n_point 4 \
    --max_terms 20 \
    --max_terms_action 10 \
    --max_brackets_per_term 8 \
    --max_brackets 12 \
    --max_steps 100 \
    --timeout 30 \
    --sample_timeout 120 \
    --embed_dim 64 \
    --num_heads 4 \
    --num_layers 3 \
    --ff_dim 128 \
    --features_dim 128 \
    --backtrack \
    --max_backtracks 10 \
    --proper_stop \
    --start_idx ${START_IDX} \
    --end_idx ${END_IDX}
