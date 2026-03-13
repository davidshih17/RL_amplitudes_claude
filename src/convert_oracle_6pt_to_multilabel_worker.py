"""
Worker script for parallel conversion of 6pt oracle data to multi-label transitions.
Uses TermSpecificActionSpace6pt with term_slot semantics.

Total action space: 30 brackets * 31 term_slots * 32 identities = 29,760 actions
"""

import argparse
import os
import sys
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import sympy as sp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'spinorhelicity'))

from spinhelicity_env import build_observation_from_expr
from action_space_6pt import TermSpecificActionSpace6pt, get_unique_brackets, get_term_bracket_powers


@dataclass
class MultiLabelTransition:
    """Transition with multiple correct actions."""
    obs: np.ndarray
    action_mask: np.ndarray
    expert_action: int
    equivalent_actions: np.ndarray  # Binary mask of all correct actions


def build_action_mask_6pt(
    expr,
    action_space: TermSpecificActionSpace6pt,
    numerator_only: bool = True
) -> np.ndarray:
    """
    Build action mask for 6pt term-specific action space.

    For each bracket present in the expression:
    - term_slot=0 (whole expression): always valid if bracket exists
    - term_slot=1-30 (specific term): valid if bracket exists in that term
    """
    brackets = get_unique_brackets(expr, numerator_only)
    mask = np.zeros(action_space.n_actions, dtype=np.float32)

    term_bracket_powers = get_term_bracket_powers(expr, numerator_only)
    n_terms = len(term_bracket_powers)

    for bracket in brackets:
        try:
            bracket_idx = action_space.get_bracket_idx(bracket)
        except ValueError:
            continue

        # Enable all identities for this bracket
        for identity_type_idx in range(action_space.identities_per_bracket):
            # term_slot=0: whole expression (always valid if bracket exists)
            action_idx = action_space.encode_action(bracket_idx, 0, identity_type_idx)
            mask[action_idx] = 1.0

            # term_slot=1-30: specific term (valid if bracket in that term)
            for term_idx in range(min(n_terms, action_space.max_terms)):
                if bracket in term_bracket_powers[term_idx]:
                    term_slot = term_idx + 1
                    action_idx = action_space.encode_action(bracket_idx, term_slot, identity_type_idx)
                    mask[action_idx] = 1.0

    return mask


def convert_chunk(
    oracle_data: List[Dict],
    n_point: int,
    max_terms_obs: int,
    max_terms_action: int,
    max_brackets_per_term: int,
    max_brackets: int,
) -> List[MultiLabelTransition]:
    """Convert a chunk of oracle data to multi-label transitions."""
    action_space = TermSpecificActionSpace6pt(
        n_point=n_point,
        max_terms=max_terms_action,
        max_brackets=max_brackets
    )
    n_actions = action_space.n_actions

    transitions = []

    for sample in oracle_data:
        n_pt = sample['n_point']

        for step in sample['trajectory']:
            state_expr = step['state']
            action = step['action']

            # Build observation
            obs = build_observation_from_expr(
                state_expr, n_pt, max_terms_obs, max_brackets_per_term
            )

            # Build action mask
            action_mask = build_action_mask_6pt(state_expr, action_space)

            # Get expert action
            expert_action = action['action_idx']

            # Get equivalent actions from pre-computed list
            equiv_indices = action.get('equivalent_action_indices', [expert_action])

            # Filter to valid actions (in action mask and in range)
            equiv_set = {idx for idx in equiv_indices if idx < n_actions and action_mask[idx] > 0}

            # Ensure expert action is included
            if expert_action < n_actions:
                equiv_set.add(expert_action)

            # Create binary mask
            equiv_mask = np.zeros(n_actions, dtype=np.float32)
            for idx in equiv_set:
                equiv_mask[idx] = 1.0

            transitions.append(MultiLabelTransition(
                obs=obs,
                action_mask=action_mask,
                expert_action=expert_action,
                equivalent_actions=equiv_mask,
            ))

    return transitions


def main():
    parser = argparse.ArgumentParser(description='Worker for 6pt oracle to transitions conversion')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing oracle chunk files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save transition chunks')
    parser.add_argument('--worker_id', type=int, required=True)
    parser.add_argument('--num_workers', type=int, required=True)
    parser.add_argument('--n_point', type=int, default=6)
    parser.add_argument('--max_terms_obs', type=int, default=35,
                        help='Max terms for observation encoding')
    parser.add_argument('--max_terms_action', type=int, default=30,
                        help='Max terms for action space (term_slots)')
    parser.add_argument('--max_brackets_per_term', type=int, default=20)
    parser.add_argument('--max_brackets', type=int, default=30,
                        help='Max brackets for action space (30 for 6-point)')
    args = parser.parse_args()

    # Create action space to get dimensions
    action_space = TermSpecificActionSpace6pt(
        n_point=args.n_point,
        max_terms=args.max_terms_action,
        max_brackets=args.max_brackets
    )

    print(f"Worker {args.worker_id}/{args.num_workers} starting...", flush=True)
    print(f"Action space: {args.max_brackets} brackets x {args.max_terms_action + 1} term_slots x {action_space.identities_per_bracket} identities = {action_space.n_actions} actions", flush=True)

    # Find all chunk files
    chunk_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.pkl')])
    print(f"Found {len(chunk_files)} chunk files", flush=True)

    # Determine which chunks this worker processes
    chunks_per_worker = len(chunk_files) // args.num_workers
    extra = len(chunk_files) % args.num_workers

    if args.worker_id < extra:
        start_idx = args.worker_id * (chunks_per_worker + 1)
        end_idx = start_idx + chunks_per_worker + 1
    else:
        start_idx = extra * (chunks_per_worker + 1) + (args.worker_id - extra) * chunks_per_worker
        end_idx = start_idx + chunks_per_worker

    my_chunks = chunk_files[start_idx:end_idx]
    print(f"Worker {args.worker_id}: processing chunks {start_idx}-{end_idx-1} ({len(my_chunks)} files)", flush=True)

    all_transitions = []

    for i, chunk_file in enumerate(my_chunks):
        chunk_path = os.path.join(args.input_dir, chunk_file)
        print(f"  [{i+1}/{len(my_chunks)}] Loading {chunk_file}...", flush=True)

        with open(chunk_path, 'rb') as f:
            oracle_data = pickle.load(f)

        transitions = convert_chunk(
            oracle_data,
            args.n_point,
            args.max_terms_obs,
            args.max_terms_action,
            args.max_brackets_per_term,
            args.max_brackets,
        )
        all_transitions.extend(transitions)
        print(f"    Converted {len(oracle_data)} samples -> {len(transitions)} transitions", flush=True)

    # Save output
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f'transitions_{args.worker_id:04d}.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(all_transitions, f)

    print(f"Worker {args.worker_id}: saved {len(all_transitions)} transitions to {output_path}", flush=True)


if __name__ == '__main__':
    main()
