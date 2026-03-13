"""
Orchestrator for parallel step-by-step simplification of large expressions.

Manages rounds of Condor worker submissions:
1. Encode terms with contrastive encoder, rank groupings
2. Submit batches of groupings to Condor workers
3. Collect results, pick the best improvement
4. If no improvement in a batch, try the next batch
5. Repeat until convergence

Usage:
    python eval_large_expr_orchestrator.py \
        --input shared/data/EYM_PPPPH_for_ML.txt \
        --n_point 5 \
        --model models/sft_5pt_500k/best_model.pt \
        --work_dir output/parallel \
        --batch_size 50 --n_workers 10
"""

import argparse
import os
import sys
import time
import pickle
import random
import subprocess
import re
from pathlib import Path

import numpy as np
import torch
import sympy as sp

# ============================================================
# Path setup (same as eval_large_expr.py)
# ============================================================
BASE_DIR = Path(__file__).parent.parent
SHARED_DIR = BASE_DIR.parent / 'shared'

sys.path.insert(0, str(SHARED_DIR / 'spinorhelicity'))
sys.path.insert(0, str(SHARED_DIR / 'spinorhelicity' / 'environment'))
sys.path.insert(0, str(BASE_DIR / 'src'))

import environment.bracket_env
sys.modules['bracket_env'] = sys.modules['environment.bracket_env']

from eval_large_expr import (
    extract_num_denom,
    count_numerator_terms,
    count_brackets,
    load_contrastive_encoder,
    encode_all_terms_batch,
    pairwise_cosine_similarity,
    rank_all_groupings,
    FUNC_DICT,
    MODEL_CONFIG,
)


# ============================================================
# PYTHONPATH for Condor workers
# ============================================================
WORKER_PYTHONPATH = ':'.join([
    str(SHARED_DIR / 'spinorhelicity'),
    str(SHARED_DIR / 'spinorhelicity' / 'environment'),
    str(BASE_DIR / 'src'),
])

PYTHON_BIN = sys.executable
WORKER_SCRIPT = str(BASE_DIR / 'src' / 'eval_large_expr_worker.py')


def generate_condor_submit(round_dir, n_workers, state_file):
    """Generate a Condor submit file for this batch of workers."""
    submit_content = f"""universe = vanilla
executable = {PYTHON_BIN}
arguments = -u {WORKER_SCRIPT} --state_file {state_file} --worker_id $(Process) --n_workers {n_workers} --result_dir {round_dir}

output = {round_dir}/worker_$(Process).out
error = {round_dir}/worker_$(Process).err
log = {round_dir}/worker.log

environment = "PYTHONPATH={WORKER_PYTHONPATH}"

request_cpus = 2
request_memory = 8GB
request_disk = 2GB

+IsGPUJob = false
+MaxRuntime = 600

queue {n_workers}
"""
    submit_path = os.path.join(round_dir, 'workers.sub')
    with open(submit_path, 'w') as f:
        f.write(submit_content)
    return submit_path


def submit_condor_jobs(submit_path):
    """Submit a Condor submit file and return the cluster ID."""
    result = subprocess.run(
        ['condor_submit', submit_path],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        print(f"  condor_submit failed: {result.stderr}", flush=True)
        return None

    # Parse cluster ID from output like "1 job(s) submitted to cluster 12345."
    match = re.search(r'submitted to cluster (\d+)', result.stdout)
    if match:
        cluster_id = match.group(1)
        return cluster_id

    print(f"  Could not parse cluster ID from: {result.stdout}", flush=True)
    return None


def poll_for_results(round_dir, n_workers, timeout=600, poll_interval=5):
    """Poll for worker result files. Returns when all present or timeout."""
    t0 = time.time()
    while True:
        n_done = sum(
            1 for w in range(n_workers)
            if os.path.exists(os.path.join(round_dir, f'result_{w}.pkl'))
        )
        if n_done >= n_workers:
            return n_done

        elapsed = time.time() - t0
        if elapsed > timeout:
            print(f"  Timeout after {elapsed:.0f}s: {n_done}/{n_workers} workers done",
                  flush=True)
            return n_done

        time.sleep(poll_interval)


def collect_results(round_dir, n_workers):
    """Read all available result files and return combined results."""
    all_results = []
    for w in range(n_workers):
        result_path = os.path.join(round_dir, f'result_{w}.pkl')
        if not os.path.exists(result_path):
            print(f"  WARNING: result_{w}.pkl missing (worker may have failed)", flush=True)
            continue
        try:
            with open(result_path, 'rb') as f:
                worker_results = pickle.load(f)
            all_results.extend(worker_results)
        except Exception as e:
            print(f"  WARNING: failed to read result_{w}.pkl: {e}", flush=True)
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Orchestrator for parallel large expression simplification')
    parser.add_argument('--input', required=True, help='Path to expression text file')
    parser.add_argument('--n_point', type=int, required=True, choices=[4, 5])
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--work_dir', required=True, help='Working directory for state/results')

    # Parallel settings
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Number of groupings per Condor batch')
    parser.add_argument('--n_workers', type=int, default=10,
                        help='Number of Condor workers per batch')
    parser.add_argument('--max_rounds', type=int, default=200,
                        help='Maximum number of outer rounds')
    parser.add_argument('--poll_interval', type=float, default=5,
                        help='Seconds between result file polls')
    parser.add_argument('--batch_timeout', type=float, default=600,
                        help='Max seconds to wait for a batch of workers')

    # Model/eval settings (passed to workers via state file)
    parser.add_argument('--max_group_size', type=int, default=10,
                        help='Max terms per grouping')
    parser.add_argument('--timeout', type=float, default=30.0,
                        help='Per-action timeout in seconds')
    parser.add_argument('--reject_term_increase', type=int, default=1)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save final expression')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.work_dir, exist_ok=True)

    # ============================================================
    # Load and parse expression
    # ============================================================
    print(f"Loading expression from {args.input}", flush=True)
    with open(args.input, 'r') as f:
        expr_str = f.read().strip()
    print(f"Expression string length: {len(expr_str):,} characters", flush=True)

    print(f"Parsing expression to sympy...", flush=True)
    t0 = time.time()
    full_expr = sp.sympify(expr_str, locals=FUNC_DICT)
    print(f"Parsed in {time.time() - t0:.1f}s", flush=True)

    # Initial metrics
    print(f"Extracting numerator terms and denominator...", flush=True)
    t0 = time.time()
    init_terms, init_denom = extract_num_denom(full_expr)
    print(f"Extracted in {time.time() - t0:.1f}s", flush=True)

    start_n_terms = len(init_terms)
    start_bk = count_brackets(full_expr)
    print(f"Initial: {start_n_terms} numerator terms, {start_bk} brackets", flush=True)

    # ============================================================
    # Load contrastive encoder (for term similarity)
    # ============================================================
    encoder_c, env_c = load_contrastive_encoder(args.n_point, device='cpu')

    # Absolute model path for workers
    model_path_abs = os.path.abspath(args.model)

    # ============================================================
    # Main loop
    # ============================================================
    current_expr = full_expr
    total_batches_submitted = 0
    total_start_time = time.time()

    print(f"\nStarting parallel step-by-step simplification:", flush=True)
    print(f"  batch_size={args.batch_size}, n_workers={args.n_workers}, "
          f"max_group_size={args.max_group_size}", flush=True)
    print(f"  batch_timeout={args.batch_timeout}s, max_rounds={args.max_rounds}", flush=True)

    for round_num in range(args.max_rounds):
        round_t0 = time.time()

        # Extract current state
        terms_num, denom = extract_num_denom(current_expr)
        n_terms = len(terms_num)
        current_bk = count_brackets(current_expr)

        print(f"\n{'='*60}", flush=True)
        print(f"Round {round_num}: {n_terms} terms, {current_bk} brackets", flush=True)
        print(f"{'='*60}", flush=True)

        if n_terms <= args.max_group_size:
            print(f"  Only {n_terms} terms left (<= {args.max_group_size}), "
                  f"would need direct evaluation — stopping parallel approach.",
                  flush=True)
            break

        # Encode terms and compute similarity
        print(f"  Encoding {n_terms} terms...", flush=True)
        t_enc = time.time()
        embeddings, valid_mask = encode_all_terms_batch(
            env_c, terms_num, encoder_c, const_blind=True)
        sim_matrix = pairwise_cosine_similarity(embeddings, valid_mask)
        all_groupings_ranked = rank_all_groupings(sim_matrix, args.max_group_size)
        print(f"  Encoded in {time.time() - t_enc:.1f}s: {len(all_groupings_ranked)} groupings "
              f"(best sim={all_groupings_ranked[0][2]:.4f}, "
              f"worst={all_groupings_ranked[-1][2]:.4f})", flush=True)

        # Convert groupings to serializable format with global indices
        all_groupings_indexed = [
            (i, ref_idx, group_indices, avg_sim)
            for i, (ref_idx, group_indices, avg_sim) in enumerate(all_groupings_ranked)
        ]

        # Try batches of groupings
        found_improvement = False
        batch_num = 0

        for batch_start in range(0, len(all_groupings_indexed), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(all_groupings_indexed))
            this_batch = all_groupings_indexed[batch_start:batch_end]

            if len(this_batch) == 0:
                break

            print(f"\n  Batch {batch_num}: groupings [{batch_start}:{batch_end}] "
                  f"({len(this_batch)} groupings, {args.n_workers} workers)", flush=True)

            # Create round/batch directory
            batch_dir = os.path.join(
                args.work_dir, f'round_{round_num:04d}_batch_{batch_num:03d}')
            os.makedirs(batch_dir, exist_ok=True)

            # Write state file
            state = {
                'terms_str': [str(t) for t in terms_num],
                'denom_str': str(denom),
                'n_point': args.n_point,
                'model_path': model_path_abs,
                'n_terms': n_terms,
                'current_bk': current_bk,
                'timeout_sec': args.timeout,
                'reject_term_increase': args.reject_term_increase,
                'all_groupings': this_batch,
            }

            state_file = os.path.join(batch_dir, 'state.pkl')
            with open(state_file, 'wb') as f:
                pickle.dump(state, f)

            # Determine effective number of workers (don't use more than groupings)
            effective_workers = min(args.n_workers, len(this_batch))

            # Generate and submit Condor jobs
            submit_path = generate_condor_submit(batch_dir, effective_workers, state_file)
            cluster_id = submit_condor_jobs(submit_path)

            if cluster_id is None:
                print(f"  Batch {batch_num}: submit failed, skipping", flush=True)
                batch_num += 1
                continue

            total_batches_submitted += 1
            print(f"  Submitted cluster {cluster_id} ({effective_workers} workers)", flush=True)

            # Poll for results
            t_poll = time.time()
            n_done = poll_for_results(
                batch_dir, effective_workers,
                timeout=args.batch_timeout,
                poll_interval=args.poll_interval)
            print(f"  {n_done}/{effective_workers} workers finished "
                  f"({time.time() - t_poll:.0f}s)", flush=True)

            # Collect results
            all_results = collect_results(batch_dir, effective_workers)

            n_tried = len(all_results)
            improved_results = [r for r in all_results if r.get('improved', False)]
            n_no_action = sum(1 for r in all_results
                              if r.get('reason') == 'no_valid_action')

            print(f"  Batch {batch_num} results: {n_tried} tried, "
                  f"{len(improved_results)} improved, "
                  f"{n_no_action} no valid action", flush=True)

            if improved_results:
                # Pick best: minimize n_terms first, then bk
                best = min(improved_results, key=lambda r: (r['new_n_terms'], r['new_bk']))

                print(f"  *** BEST: grouping {best['grouping_idx']} "
                      f"(ref={best['ref_idx']}, sim={best['avg_sim']:.4f}): "
                      f"sub {best['sub_bk_before']}->{best['sub_bk_after']} bk "
                      f"({best['sub_terms_before']}->{best['sub_terms_after']} tm), "
                      f"full {current_bk}->{best['new_bk']} bk "
                      f"({n_terms}->{best['new_n_terms']} tm)", flush=True)

                # Show runner-up if multiple improvements
                if len(improved_results) > 1:
                    sorted_imp = sorted(improved_results,
                                        key=lambda r: (r['new_n_terms'], r['new_bk']))
                    for ri, r in enumerate(sorted_imp[1:min(4, len(sorted_imp))], 2):
                        print(f"      #{ri}: grouping {r['grouping_idx']} "
                              f"(ref={r['ref_idx']}): "
                              f"{n_terms}->{r['new_n_terms']} tm, "
                              f"{current_bk}->{r['new_bk']} bk", flush=True)

                # Update expression
                current_expr = sp.sympify(best['new_expr_str'], locals=FUNC_DICT)
                found_improvement = True
                break

            print(f"  Batch {batch_num}: no improvement", flush=True)
            batch_num += 1

        if not found_improvement:
            print(f"\nRound {round_num}: no grouping improved across all batches, stopping.",
                  flush=True)
            break

        round_time = time.time() - round_t0
        new_bk = count_brackets(current_expr)
        new_terms = count_numerator_terms(current_expr)
        print(f"\n  Round {round_num} done: {n_terms}->{new_terms} terms, "
              f"{current_bk}->{new_bk} bk ({round_time:.1f}s)", flush=True)

    # ============================================================
    # Final summary
    # ============================================================
    total_time = time.time() - total_start_time
    final_bk = count_brackets(current_expr)
    final_terms = count_numerator_terms(current_expr)

    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Input: {args.input}", flush=True)
    print(f"n_point: {args.n_point}", flush=True)
    print(f"Initial: {start_n_terms} numerator terms, {start_bk} brackets", flush=True)
    print(f"Final: {final_terms} numerator terms, {final_bk} brackets", flush=True)
    print(f"Numerator term reduction: {start_n_terms} -> {final_terms} "
          f"({100 * (1 - final_terms / max(start_n_terms, 1)):.1f}%)", flush=True)
    print(f"Bracket reduction: {start_bk} -> {final_bk} "
          f"({100 * (1 - final_bk / max(start_bk, 1)):.1f}%)", flush=True)
    print(f"Total batches submitted: {total_batches_submitted}", flush=True)
    print(f"Total time: {total_time:.1f}s", flush=True)

    # Save output
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.',
                    exist_ok=True)
        with open(args.output, 'w') as f:
            f.write(str(current_expr))
        print(f"\nFinal expression saved to {args.output}", flush=True)


if __name__ == '__main__':
    main()
