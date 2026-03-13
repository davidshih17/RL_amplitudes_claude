#!/bin/bash
# Worker script for 5pt SFT evaluation with reject_term_increase=5
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

echo "Worker ${WORKER_ID}: evaluating 5pt samples ${START_IDX} to ${END_IDX} (rti=5)"

cd "${BASE_DIR}"

export PYTHONPATH="${BASE_DIR}/spinorhelicity:${BASE_DIR}/spinorhelicity/environment:${BASE_DIR}/src:${PYTHONPATH}"

$PYTHON -u src/eval_sft_5pt.py \
    --model "${BASE_DIR}/models/sft_5pt_500k/best_model.pt" \
    --test_data "${BASE_DIR}/data/paper_test_5pt.pkl" \
    --output "$BASE_DIR/output/eval_5pt/worker_${WORKER_ID}.pkl" \
    --n_point 5 \
    --max_terms 25 \
    --max_terms_action 25 \
    --max_brackets_per_term 15 \
    --max_brackets 20 \
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
    --reject_term_increase 5 \
    --start_idx ${START_IDX} \
    --end_idx ${END_IDX}
