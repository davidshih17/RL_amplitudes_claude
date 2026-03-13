"""
Condor worker for parallel step-by-step simplification.

Reads a state file (pickle), tries assigned groupings (one model action each),
evaluates improvement by recombining with full expression, writes results.

Usage:
    python eval_large_expr_worker.py \
        --state_file /path/to/state.pkl \
        --worker_id 3 \
        --n_workers 10 \
        --result_dir /path/to/round_dir/
"""

import argparse
import os
import sys
import time
import pickle
import traceback
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
    apply_single_model_step,
    load_model_and_action_space,
    FUNC_DICT,
)
from eval_sft_5pt import init_worker, shutdown_worker


def main():
    parser = argparse.ArgumentParser(description='Parallel worker for large expression simplification')
    parser.add_argument('--state_file', required=True, help='Path to state pickle file')
    parser.add_argument('--worker_id', type=int, required=True, help='Worker ID (0-indexed)')
    parser.add_argument('--n_workers', type=int, required=True, help='Total number of workers')
    parser.add_argument('--result_dir', required=True, help='Directory to write result pickle')
    args = parser.parse_args()

    t0 = time.time()
    print(f"Worker {args.worker_id}/{args.n_workers} starting", flush=True)

    # Load state
    print(f"Loading state from {args.state_file}", flush=True)
    with open(args.state_file, 'rb') as f:
        state = pickle.load(f)

    terms_str = state['terms_str']
    denom_str = state['denom_str']
    n_point = state['n_point']
    model_path = state['model_path']
    n_terms = state['n_terms']
    current_bk = state['current_bk']
    timeout_sec = state['timeout_sec']
    reject_term_increase = state['reject_term_increase']
    all_groupings = state['all_groupings']

    # Compute this worker's slice
    total = len(all_groupings)
    per_worker = total // args.n_workers
    remainder = total % args.n_workers

    # Distribute remainder evenly: first 'remainder' workers get one extra
    if args.worker_id < remainder:
        start = args.worker_id * (per_worker + 1)
        end = start + per_worker + 1
    else:
        start = remainder * (per_worker + 1) + (args.worker_id - remainder) * per_worker
        end = start + per_worker

    my_groupings = all_groupings[start:end]
    print(f"Worker {args.worker_id}: assigned groupings [{start}:{end}] "
          f"({len(my_groupings)} of {total})", flush=True)

    if len(my_groupings) == 0:
        # Nothing to do
        result_path = os.path.join(args.result_dir, f'result_{args.worker_id}.pkl')
        with open(result_path, 'wb') as f:
            pickle.dump([], f)
        print(f"Worker {args.worker_id}: no groupings assigned, exiting", flush=True)
        return

    # Parse terms and denominator
    print(f"Parsing {len(terms_str)} terms...", flush=True)
    t_parse = time.time()
    terms_num = np.array([sp.sympify(t, locals=FUNC_DICT) for t in terms_str], dtype=object)
    denom = sp.sympify(denom_str, locals=FUNC_DICT)
    print(f"Parsed in {time.time() - t_parse:.1f}s", flush=True)

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}", flush=True)
    rl_model, action_space, cfg = load_model_and_action_space(n_point, model_path, device)

    # Init worker subprocess
    init_worker(n_point, max_terms=cfg['max_terms_action'])

    results = []

    try:
        for idx, (global_idx, ref_idx, group_indices, avg_sim) in enumerate(my_groupings):
            t_start = time.time()

            # Form sub-expression from group terms
            mask = np.zeros(n_terms, dtype=bool)
            for gi in group_indices:
                if gi < n_terms:
                    mask[gi] = True

            group_terms = terms_num[mask]
            remaining_terms = terms_num[~mask]

            try:
                sub_expr = group_terms.sum() / denom
                sub_expr = sp.cancel(sub_expr)
            except Exception as e:
                results.append({
                    'grouping_idx': global_idx,
                    'ref_idx': ref_idx,
                    'avg_sim': avg_sim,
                    'improved': False,
                    'reason': f'cancel_failed: {e}',
                })
                continue

            sub_bk = count_brackets(sub_expr)
            sub_n_terms = count_numerator_terms(sub_expr)

            # Try multiple model actions, block ones that don't decrease terms
            blocked_actions = set()
            max_action_retries = 20
            new_sub_expr = None

            for attempt in range(max_action_retries):
                candidate, action_info = apply_single_model_step(
                    model=rl_model,
                    current_expr=sub_expr,
                    action_space=action_space,
                    n_point=n_point,
                    max_terms=cfg['max_terms'],
                    max_brackets_per_term=cfg['max_brackets_per_term'],
                    device=device,
                    timeout_sec=timeout_sec,
                    reject_term_increase=reject_term_increase,
                    blocked_actions=blocked_actions,
                    max_retries=20,
                    verbose=False,
                )

                if candidate is None:
                    break  # no more valid actions

                cand_terms = count_numerator_terms(candidate)
                if cand_terms < sub_n_terms:
                    new_sub_expr = candidate
                    break
                else:
                    blocked_actions.add(action_info['action'])

            if new_sub_expr is None:
                n_tried = len(blocked_actions)
                reason = 'no_valid_action' if n_tried == 0 else f'no_term_decrease_{n_tried}_actions'
                results.append({
                    'grouping_idx': global_idx,
                    'ref_idx': ref_idx,
                    'avg_sim': avg_sim,
                    'improved': False,
                    'reason': reason,
                })
                print(f"  [{idx+1}/{len(my_groupings)}] Grouping {global_idx} "
                      f"(ref={ref_idx}): {reason} "
                      f"({time.time()-t_start:.1f}s)", flush=True)
                continue

            new_sub_bk = count_brackets(new_sub_expr)
            new_sub_terms = count_numerator_terms(new_sub_expr)

            # Sub terms decreased — recombine to get new full expression
            try:
                if len(remaining_terms) > 0:
                    new_full_expr = new_sub_expr + remaining_terms.sum() / denom
                else:
                    new_full_expr = new_sub_expr

                new_full_expr = sp.cancel(new_full_expr)
            except Exception as e:
                results.append({
                    'grouping_idx': global_idx,
                    'ref_idx': ref_idx,
                    'avg_sim': avg_sim,
                    'improved': False,
                    'reason': f'recombine_failed: {e}',
                })
                continue

            new_bk = count_brackets(new_full_expr)
            new_n_terms = count_numerator_terms(new_full_expr)

            result = {
                'grouping_idx': global_idx,
                'ref_idx': ref_idx,
                'avg_sim': avg_sim,
                'improved': True,
                'new_n_terms': new_n_terms,
                'new_bk': new_bk,
                'sub_bk_before': sub_bk,
                'sub_bk_after': new_sub_bk,
                'sub_terms_before': sub_n_terms,
                'sub_terms_after': new_sub_terms,
                'new_expr_str': str(new_full_expr),
                'reason': 'improved',
            }
            results.append(result)

            print(f"  [{idx+1}/{len(my_groupings)}] Grouping {global_idx} "
                  f"(ref={ref_idx}, sim={avg_sim:.4f}): "
                  f"sub {sub_bk}->{new_sub_bk} bk ({sub_n_terms}->{new_sub_terms} tm), "
                  f"full {current_bk}->{new_bk} bk ({n_terms}->{new_n_terms} tm) — "
                  f"IMPROVED ({time.time()-t_start:.1f}s)", flush=True)

    except Exception as e:
        print(f"Worker {args.worker_id} error: {e}", flush=True)
        traceback.print_exc()
    finally:
        shutdown_worker()

    # Write results
    result_path = os.path.join(args.result_dir, f'result_{args.worker_id}.pkl')
    with open(result_path, 'wb') as f:
        pickle.dump(results, f)

    n_improved = sum(1 for r in results if r.get('improved', False))
    elapsed = time.time() - t0
    print(f"\nWorker {args.worker_id} done: {len(results)} groupings tried, "
          f"{n_improved} improved, {elapsed:.1f}s total", flush=True)


if __name__ == '__main__':
    main()
