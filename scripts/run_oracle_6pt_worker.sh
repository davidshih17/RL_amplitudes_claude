#!/bin/bash
# Worker script for 6pt oracle data generation (term-specific action space)
# Uses action_space_6pt with term_slot semantics (0=whole expr, 1-30=specific term)
# 6-point: 30 brackets, 32 identities per bracket (6 Schouten + 10 Momentum + 16 Momentum_sq)

WORKER_ID=$1
N_WORKERS=1000
TOTAL_SAMPLES=500000

PYTHON="python"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${BASE_DIR}/data/oracle_6pt_500k"

cd "${BASE_DIR}"

export PYTHONPATH="${BASE_DIR}/spinorhelicity:${BASE_DIR}/spinorhelicity/environment:${BASE_DIR}/src:${PYTHONPATH}"

echo "Worker ${WORKER_ID}: generating oracle 6pt data"
echo "  n_point=6, max_terms=30"
echo "  Action space: 29760 actions (30 brackets x 31 term_slots x 32 identities)"
echo "  Total samples: ${TOTAL_SAMPLES}, Workers: ${N_WORKERS}"

$PYTHON -u src/generate_oracle_worker_6pt.py \
    --worker_id ${WORKER_ID} \
    --num_workers ${N_WORKERS} \
    --total_samples ${TOTAL_SAMPLES} \
    --output_dir ${OUTPUT_DIR} \
    --n_point 6 \
    --min_scrambles 1 \
    --max_scrambles 3 \
    --max_scale 2 \
    --max_terms_gen 3 \
    --max_terms 30 \
    --max_brackets 30 \
    --l_scale 0.75 \
    --seed_offset 3000000 \
    --use_full_scrambles
