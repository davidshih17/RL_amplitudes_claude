#!/usr/bin/env python
"""
Condor worker: apply ONE model step to a contrastive grouping sub-expression.

Reads a pickle file with sub-expression + metadata, applies apply_single_model_step
(20 retries, RTI check), writes result pickle.

Usage:
    python parallel_step_worker.py \
        --grouping_file /path/to/grouping_NNN.pkl \
        --result_file /path/to/result_NNN.pkl \
        --model /path/to/best_model.pt \
        --n_point 5 \
        --reject_term_increase -1
"""

import argparse
import os
import sys
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import sympy as sp

# Path setup
BASE_DIR = Path(__file__).parent.parent
SHARED_DIR = BASE_DIR.parent / 'shared'

sys.path.insert(0, str(SHARED_DIR / 'spinorhelicity'))
sys.path.insert(0, str(SHARED_DIR / 'spinorhelicity' / 'environment'))
sys.path.insert(0, str(BASE_DIR / 'src'))

import environment.bracket_env
sys.modules['bracket_env'] = sys.modules['environment.bracket_env']

from environment.bracket_env import ab, sb
from train_sft_transformer import TransformerPolicySFT
from eval_sft_5pt import (
    count_brackets,
    count_terms,
    init_worker,
    shutdown_worker,
)
from eval_large_expr import (
    apply_single_model_step,
    count_numerator_terms,
)

FUNC_DICT = {'ab': ab, 'sb': sb}

MODEL_CONFIG = {
    5: {
        'max_terms': 25,
        'max_terms_action': 25,
        'max_brackets_per_term': 15,
        'max_brackets': 20,
        'bracket_feature_dim': 12,
        'term_feature_dim': 183,
    },
}


def load_model_and_action_space(n_point, model_path, device):
    cfg = MODEL_CONFIG[n_point]

    from action_space_5pt import TermSpecificActionSpace5pt
    action_space = TermSpecificActionSpace5pt(
        n_point=n_point,
        max_terms=cfg['max_terms_action'],
        max_brackets=cfg['max_brackets'],
    )

    n_term_slots = 1 + cfg['max_terms_action']
    model = TransformerPolicySFT(
        max_terms=cfg['max_terms'],
        term_feature_dim=cfg['term_feature_dim'],
        n_actions=action_space.n_actions,
        max_brackets=cfg['max_brackets'],
        actions_per_bracket=n_term_slots * action_space.identities_per_bracket,
        embed_dim=64,
        num_heads=4,
        num_layers=3,
        ff_dim=128,
        dropout=0.1,
        features_dim=128,
        global_feature_dim=2,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, action_space, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grouping_file', required=True)
    parser.add_argument('--result_file', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--n_point', type=int, default=5)
    parser.add_argument('--reject_term_increase', type=int, default=-1)
    parser.add_argument('--max_retries', type=int, default=20)
    parser.add_argument('--timeout', type=float, default=30.0)
    args = parser.parse_args()

    t0 = time.time()

    # Load grouping data
    with open(args.grouping_file, 'rb') as f:
        grouping = pickle.load(f)

    sub_expr_str = grouping['sub_expr_str']
    grouping_idx = grouping['grouping_idx']
    ref_idx = grouping['ref_idx']
    group_indices = grouping['group_indices']
    avg_sim = grouping['avg_sim']

    # Parse the sub-expression
    sub_expr = sp.sympify(sub_expr_str, locals=FUNC_DICT)
    sub_bk = count_brackets(sub_expr)
    sub_terms = count_numerator_terms(sub_expr)

    print(f"Grouping {grouping_idx}: ref={ref_idx}, {len(group_indices)} terms, "
          f"avg_sim={avg_sim:.4f}, {sub_bk} bk, {sub_terms} terms", flush=True)

    # Load model
    device = 'cpu'
    model, action_space, cfg = load_model_and_action_space(args.n_point, args.model, device)

    # Init worker subprocess (for apply_action_with_timeout)
    init_worker(args.n_point, max_terms=cfg['max_terms_action'])

    # Apply single model step
    new_sub_expr, action_info = apply_single_model_step(
        model=model,
        current_expr=sub_expr,
        action_space=action_space,
        n_point=args.n_point,
        max_terms=cfg['max_terms'],
        max_brackets_per_term=cfg['max_brackets_per_term'],
        device=device,
        timeout_sec=args.timeout,
        reject_term_increase=args.reject_term_increase,
        max_retries=args.max_retries,
        verbose=True,
    )

    shutdown_worker()

    elapsed = time.time() - t0

    if new_sub_expr is not None:
        new_sub_terms = count_numerator_terms(new_sub_expr)
        new_sub_bk = count_brackets(new_sub_expr)
        success = new_sub_terms < sub_terms  # strict term reduction
        new_sub_expr_str = str(new_sub_expr)
    else:
        new_sub_terms = sub_terms
        new_sub_bk = sub_bk
        success = False
        new_sub_expr_str = sub_expr_str

    result_data = {
        'grouping_idx': grouping_idx,
        'ref_idx': ref_idx,
        'group_indices': group_indices,
        'avg_sim': avg_sim,
        'n_group_terms': len(group_indices),
        'sub_bk_before': sub_bk,
        'sub_terms_before': sub_terms,
        'sub_bk_after': new_sub_bk,
        'sub_terms_after': new_sub_terms,
        'terms_reduced': sub_terms - new_sub_terms,
        'success': success,
        'action_info': action_info,
        'new_sub_expr_str': new_sub_expr_str,
        'elapsed': elapsed,
    }

    status = "SUCCESS" if success else "FAIL"
    print(f"\n{status}: {sub_bk}->{new_sub_bk} bk, {sub_terms}->{new_sub_terms} tm, {elapsed:.1f}s",
          flush=True)

    with open(args.result_file, 'wb') as f:
        pickle.dump(result_data, f)

    print(f"Wrote {args.result_file}", flush=True)


if __name__ == '__main__':
    main()
