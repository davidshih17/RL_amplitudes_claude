#!/bin/bash
# Worker script for 5pt oracle data generation (term-specific action space)
# Uses action_space_5pt with term_slot semantics (0=whole expr, 1-10=specific term)
# 5-point: 20 brackets, 19 identities per bracket (3 Schouten + 8 Momentum + 8 Momentum_sq)

WORKER_ID=$1
N_WORKERS=1000
TOTAL_SAMPLES=500000

PYTHON="python"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${BASE_DIR}/data/oracle_5pt_500k"

cd "${BASE_DIR}"

export PYTHONPATH="${BASE_DIR}/spinorhelicity:${BASE_DIR}/spinorhelicity/environment:${BASE_DIR}/src:${PYTHONPATH}"

echo "Worker ${WORKER_ID}: generating oracle 5pt data"
echo "  n_point=5, max_terms=25"
echo "  Action space: 9880 actions (20 brackets x 26 term_slots x 19 identities)"
echo "  Total samples: ${TOTAL_SAMPLES}, Workers: ${N_WORKERS}"

$PYTHON -u src/generate_oracle_worker_5pt.py \
    --worker_id ${WORKER_ID} \
    --num_workers ${N_WORKERS} \
    --total_samples ${TOTAL_SAMPLES} \
    --output_dir ${OUTPUT_DIR} \
    --n_point 5 \
    --min_scrambles 1 \
    --max_scrambles 3 \
    --max_scale 2 \
    --max_terms_gen 3 \
    --max_terms 25 \
    --l_scale 0.75 \
    --seed_offset 2000000 \
    --use_full_scrambles
