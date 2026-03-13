#!/bin/bash
# Worker script for v10 oracle data generation (term-specific action space)
# Uses action_space_4pt with term_slot semantics (0=whole expr, 1-10=specific term)
# Parameters matched to paper: max_scale=2, max_terms_gen=3, l_scale=0.75

WORKER_ID=$1
N_WORKERS=1000
TOTAL_SAMPLES=500000

PYTHON="python"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${BASE_DIR}/data/oracle_4pt_500k"

cd "${BASE_DIR}"

export PYTHONPATH="${BASE_DIR}/spinorhelicity:${BASE_DIR}/spinorhelicity/environment:${BASE_DIR}/src:${PYTHONPATH}"

echo "Worker ${WORKER_ID}: generating oracle v10 data with paper parameters"
echo "  max_scale=2, max_terms_gen=3, l_scale=0.75, max_terms=10"
echo "  Action space: 1452 actions (12 brackets x 11 term_slots x 11 identities)"
echo "  Total samples: ${TOTAL_SAMPLES}, Workers: ${N_WORKERS}"

$PYTHON -u src/generate_oracle_worker_4pt.py \
    --worker_id ${WORKER_ID} \
    --num_workers ${N_WORKERS} \
    --total_samples ${TOTAL_SAMPLES} \
    --output_dir ${OUTPUT_DIR} \
    --n_point 4 \
    --min_scrambles 1 \
    --max_scrambles 3 \
    --max_scale 2 \
    --max_terms_gen 3 \
    --max_terms 10 \
    --l_scale 0.75 \
    --seed_offset 1000000 \
    --use_full_scrambles
