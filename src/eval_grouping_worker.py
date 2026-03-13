#!/usr/bin/env python
"""
Condor worker: evaluate one contrastive grouping using full evaluate_sample_backtrack.

Reads a pickle file containing the sub-expression (numerator only, no denominator),
runs evaluate_sample_backtrack with full settings (RTI=1, anti-cycle, backtracking),
and writes the result to a pickle file.

Usage:
    python eval_grouping_worker.py \
        --grouping_file /path/to/grouping_NNN.pkl \
        --result_file /path/to/result_NNN.pkl \
        --model /path/to/best_model.pt \
        --n_point 5
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
    evaluate_sample_backtrack,
    count_brackets,
    count_terms,
    init_worker,
    shutdown_worker,
    apply_action_with_timeout,
    ActionTimeoutError,
    build_action_mask_5pt as build_action_mask,
    obs_to_hash,
)
from spinhelicity_env import build_observation_from_expr
from add_ons.numerical_evaluations import MOMENTA_DICTS

FUNC_DICT = {'ab': ab, 'sb': sb}


def numerical_validate(old_expr, new_expr):
    """Check that new_expr is numerically equal to old_expr at 2 kinematic points."""
    max_rdiff = 0.0
    for ki in range(2):
        md = MOMENTA_DICTS[ki]
        subs = {}
        for i in range(1, 6):
            for j in range(i + 1, 6):
                subs[ab(i, j)] = md[f'5ptab{i}{j}']
                subs[sb(i, j)] = md[f'5ptsb{i}{j}']
        v_old = complex(old_expr.subs(subs).evalf())
        v_new = complex(new_expr.subs(subs).evalf())
        if abs(v_old) < 1e-15:
            rdiff = abs(v_new - v_old)
        else:
            rdiff = abs((v_new - v_old) / v_old)
        max_rdiff = max(max_rdiff, rdiff)
    return max_rdiff < 1e-9, max_rdiff


def factor_out_common_brackets(expr):
    """Factor out common bracket factors from the numerator of an expression.

    For example: (ab(1,2)**2*sb(1,2)*X + ab(1,2)*sb(1,2)**2*Y) / denom
    -> common_factor = ab(1,2)*sb(1,2), reduced = (ab(1,2)*X + sb(1,2)*Y) / denom

    Returns (common_factor, reduced_expr) where expr = common_factor * reduced_expr.
    If no common factor, returns (1, expr).
    """
    num, denom = sp.fraction(sp.cancel(expr))

    # Get individual terms
    if isinstance(num, sp.Add):
        terms = list(num.args)
    elif num == 0:
        return sp.Integer(1), expr
    else:
        return sp.Integer(1), expr  # single term, nothing to factor

    if len(terms) < 2:
        return sp.Integer(1), expr

    # For each term, extract bracket powers: {bracket: power}
    def get_bracket_powers(term):
        powers = {}
        # Handle Mul
        if isinstance(term, sp.Mul):
            factors = term.args
        else:
            factors = [term]

        for f in factors:
            if isinstance(f, sp.Function) and f.func.__name__ in ('ab', 'sb'):
                powers[f] = powers.get(f, 0) + 1
            elif isinstance(f, sp.Pow):
                base, exp = f.args
                if isinstance(base, sp.Function) and base.func.__name__ in ('ab', 'sb'):
                    try:
                        powers[base] = powers.get(base, 0) + int(exp)
                    except (TypeError, ValueError):
                        pass
        return powers

    # Find minimum power of each bracket across all terms
    all_powers = [get_bracket_powers(t) for t in terms]

    # Common brackets: those appearing in ALL terms
    common_brackets = set(all_powers[0].keys())
    for p in all_powers[1:]:
        common_brackets &= set(p.keys())

    if not common_brackets:
        return sp.Integer(1), expr

    # For each common bracket, find minimum power across terms
    common_factor = sp.Integer(1)
    for bk in sorted(common_brackets, key=str):
        min_power = min(p.get(bk, 0) for p in all_powers)
        if min_power > 0:
            common_factor *= bk ** min_power

    if common_factor == 1:
        return sp.Integer(1), expr

    # Divide numerator by common factor
    reduced_num = sp.cancel(num / common_factor)
    reduced_expr = sp.cancel(reduced_num / denom)

    return common_factor, reduced_expr


def _apply_single_model_step(
    model, current_expr, action_space, n_point, max_terms,
    max_brackets_per_term, device, timeout_sec,
    reject_term_increase=5, max_retries=20, verbose=False,
    skip_n=0,
):
    """Apply ONE model action to the expression. Returns (new_expr, action_info) or (None, None)."""
    current_brackets = count_brackets(current_expr)
    current_terms = count_terms(current_expr)

    try:
        obs = build_observation_from_expr(current_expr, n_point, max_terms, max_brackets_per_term)
    except Exception:
        return None, None

    action_mask = build_action_mask(current_expr, action_space)
    if action_mask.sum() == 0:
        return None, None

    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(obs_tensor)

    FLOAT_MIN = -3.4e38
    mask_tensor = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0).to(device)
    inf_mask = torch.clamp(torch.log(mask_tensor + 1e-10), min=FLOAT_MIN)
    masked_logits = logits + inf_mask
    sorted_actions = torch.argsort(masked_logits[0], descending=True).cpu().numpy()

    for retry_idx, action in enumerate(sorted_actions[skip_n:skip_n + max_retries]):
        if action_mask[action] == 0:
            continue

        spec = action_space.get_action_spec(action)
        bracket_type, i, j = action_space.get_bracket_from_idx(spec.bracket_idx)

        try:
            result = apply_action_with_timeout(
                current_expr, bracket_type, (i, j),
                spec.identity, spec.params, spec.term_slot,
                n_point, timeout_sec
            )

            if result is None:
                if verbose:
                    print(f"        retry {retry_idx}: {spec.identity} on {bracket_type}({i},{j}) t={spec.term_slot} -> None", flush=True)
                continue

            new_brackets = count_brackets(result)
            new_terms = count_terms(result)

            if reject_term_increase is not None:
                if new_terms - current_terms > reject_term_increase:
                    if verbose:
                        print(f"        retry {retry_idx}: {spec.identity} on {bracket_type}({i},{j}) t={spec.term_slot} -> {new_brackets} bk, {new_terms} tm (RTI reject)", flush=True)
                    continue

            if verbose:
                print(f"        retry {retry_idx}: {spec.identity} on {bracket_type}({i},{j}) t={spec.term_slot} -> {new_brackets} bk, {new_terms} tm (ACCEPT)", flush=True)

            action_info = {
                'action': int(action),
                'identity': spec.identity,
                'bracket': f"{bracket_type}({i},{j})",
                'term_slot': spec.term_slot,
                'brackets_before': current_brackets,
                'brackets_after': new_brackets,
                'terms_before': current_terms,
                'terms_after': new_terms,
                'retry': retry_idx,
            }
            return result, action_info

        except ActionTimeoutError:
            if verbose:
                print(f"        retry {retry_idx}: {spec.identity} on {bracket_type}({i},{j}) t={spec.term_slot} -> timeout", flush=True)
            continue
        except Exception:
            continue

    return None, None


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
    """Load model and create action space."""
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


def _form_grouping_indices(sim_matrix, thresholds, max_group_size, max_terms):
    """Form grouping index lists from similarity matrix. Pure index math, no sympy."""
    seen_groups = set()
    groupings = []
    n_terms = sim_matrix.shape[0]

    for thresh in thresholds:
        for ref_idx in range(n_terms):
            neighbor_sims = []
            for j in range(n_terms):
                if j != ref_idx and sim_matrix[ref_idx, j] >= thresh:
                    neighbor_sims.append((j, sim_matrix[ref_idx, j]))
            if not neighbor_sims:
                continue
            neighbor_sims.sort(key=lambda x: -x[1])
            group_indices = [ref_idx] + [j for j, _ in neighbor_sims[:max_group_size - 1]]
            if len(group_indices) < 2 or len(group_indices) > max_terms:
                continue
            key = frozenset(group_indices)
            if key in seen_groups:
                continue
            seen_groups.add(key)
            avg_sim = float(np.mean([sim_matrix[ref_idx, j]
                                     for j in group_indices if j != ref_idx]))
            groupings.append((group_indices, avg_sim))

    return groupings


def run_encode_groupings(args):
    """Encode beam entry terms, compute similarity, form groupings, write pickles."""
    from eval_large_expr import (
        load_contrastive_encoder, encode_all_terms_batch, pairwise_cosine_similarity,
    )

    t0 = time.time()

    with open(args.encode_task, 'rb') as f:
        task = pickle.load(f)

    beam_idx = task['beam_idx']
    terms_num_str = task['terms_num_str']
    denom_str = task['denom_str']
    thresholds = task['thresholds']
    max_group_size = task['max_group_size']
    max_terms = task['max_terms']
    max_brackets_per_term = task['max_brackets_per_term']
    skip_n = task['skip_n']
    groupings_dir = task['groupings_dir']
    step_dir = task['step_dir']

    print(f"Encode task: beam_idx={beam_idx}, {len(terms_num_str)} terms", flush=True)

    # Load contrastive encoder
    encoder_c, env_c = load_contrastive_encoder(args.n_point, device='cpu')

    # Parse terms
    t_parse = time.time()
    terms_num = [sp.sympify(s, locals=FUNC_DICT) for s in terms_num_str]
    print(f"  Parsed {len(terms_num)} terms ({time.time() - t_parse:.1f}s)", flush=True)

    # Encode and compute similarity
    t_enc = time.time()
    embeddings, valid_mask = encode_all_terms_batch(env_c, terms_num, encoder_c, const_blind=True)
    sim_matrix_t = pairwise_cosine_similarity(embeddings, valid_mask)
    sim_matrix = sim_matrix_t.numpy() if isinstance(sim_matrix_t, torch.Tensor) else sim_matrix_t
    print(f"  Encoded ({time.time() - t_enc:.1f}s)", flush=True)

    # Form groupings
    groupings = _form_grouping_indices(sim_matrix, thresholds, max_group_size, max_terms)
    print(f"  Formed {len(groupings)} groupings", flush=True)

    # Write shared expression data
    shared_path = os.path.join(step_dir, f'shared_beam_{beam_idx}.pkl')
    all_terms_str = [str(t) for t in terms_num]
    shared_data = {
        'all_terms_str': all_terms_str,
        'denom_str': denom_str,
        'max_terms': max_terms,
        'max_brackets_per_term': max_brackets_per_term,
    }
    with open(shared_path, 'wb') as f:
        pickle.dump(shared_data, f)

    # Write grouping pickles with beam_idx prefix
    for local_idx, (group_indices, avg_sim) in enumerate(groupings):
        grouping_data = {
            'grouping_idx': -1,  # will be renumbered by orchestrator
            'ref_idx': group_indices[0],
            'group_indices': group_indices,
            'avg_sim': avg_sim,
            'n_terms': len(group_indices),
            'sub_expr_str': None,
            'threshold': thresholds[0],
            'shared_file': shared_path,
            'skip_n': skip_n,
        }
        pkl_path = os.path.join(groupings_dir, f'grouping_{beam_idx}_{local_idx}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(grouping_data, f)

    # Write encode result
    result = {
        'beam_idx': beam_idx,
        'n_groupings': len(groupings),
        'shared_file': shared_path,
    }
    with open(args.encode_result, 'wb') as f:
        pickle.dump(result, f)

    print(f"  Wrote {len(groupings)} grouping pickles + shared file ({time.time() - t0:.1f}s total)",
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grouping_file')
    parser.add_argument('--result_file')
    parser.add_argument('--model')
    parser.add_argument('--n_point', type=int, default=5)
    parser.add_argument('--single_step', action='store_true',
                        help='Apply ONE model action instead of full evaluate_sample_backtrack')
    parser.add_argument('--beam_expand', action='store_true',
                        help='Beam expansion: try top-N actions on full expression, return all valid candidates')
    parser.add_argument('--encode_task',
                        help='Encode mode: path to encode task pickle')
    parser.add_argument('--encode_result',
                        help='Encode mode: path to write encode result pickle')
    args = parser.parse_args()

    # ---- encode_groupings mode ----
    if args.encode_task:
        run_encode_groupings(args)
        return

    t0 = time.time()

    # Load data (pickle format depends on mode)
    with open(args.grouping_file, 'rb') as f:
        grouping = pickle.load(f)

    # ---- beam_expand mode: try top-N actions on full expression ----
    if args.beam_expand:
        beam_idx = grouping['beam_idx']
        expr_str = grouping['expr_str']
        n_branches = grouping.get('n_branches', 5)
        rti = grouping.get('reject_term_increase', 5)
        skip_n = grouping.get('skip_n', 0)

        expr = sp.sympify(expr_str, locals=FUNC_DICT)
        expr_terms = count_terms(expr)
        expr_bk = count_brackets(expr)
        print(f"Beam expand: beam_idx={beam_idx}, {expr_terms} terms, {expr_bk} bk, "
              f"branches={n_branches}, RTI={rti}, skip={skip_n}", flush=True)

        device = 'cpu'
        model, action_space, cfg = load_model_and_action_space(args.n_point, args.model, device)
        init_worker(args.n_point, max_terms=cfg['max_terms_action'])

        max_terms = cfg['max_terms']
        max_brackets_per_term = cfg['max_brackets_per_term']

        # Build observation and action mask
        candidates = []
        try:
            obs = build_observation_from_expr(expr, args.n_point, max_terms, max_brackets_per_term)
            action_mask = build_action_mask(expr, action_space)

            if action_mask.sum() > 0:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(obs_tensor)

                FLOAT_MIN = -3.4e38
                mask_tensor = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0).to(device)
                inf_mask = torch.clamp(torch.log(mask_tensor + 1e-10), min=FLOAT_MIN)
                masked_logits = logits + inf_mask
                log_probs = torch.log_softmax(masked_logits[0], dim=0).cpu().numpy()
                sorted_actions = torch.argsort(masked_logits[0], descending=True).cpu().numpy()

                max_tries = min(n_branches * 4, 40)
                for action in sorted_actions[skip_n:skip_n + max_tries]:
                    if len(candidates) >= n_branches:
                        break
                    if action_mask[action] == 0:
                        continue

                    spec = action_space.get_action_spec(action)
                    bracket_type, i, j = action_space.get_bracket_from_idx(spec.bracket_idx)

                    try:
                        result_expr = apply_action_with_timeout(
                            expr, bracket_type, (i, j),
                            spec.identity, spec.params, spec.term_slot,
                            args.n_point, 30.0
                        )
                        if result_expr is None:
                            continue

                        new_terms = count_terms(result_expr)
                        if rti is not None and new_terms - expr_terms > rti:
                            continue

                        new_bk = count_brackets(result_expr)

                        # Numerical validation
                        is_valid, max_rdiff = numerical_validate(expr, result_expr)
                        if not is_valid:
                            print(f"  NUMERICAL FAIL: {spec.identity} on "
                                  f"{bracket_type}({i},{j}) t={spec.term_slot} "
                                  f"rdiff={max_rdiff:.2e}, SKIPPING", flush=True)
                            continue

                        action_info = {
                            'identity': spec.identity,
                            'bracket': f"{bracket_type}({i},{j})",
                            'term_slot': spec.term_slot,
                            'terms_before': expr_terms,
                            'terms_after': new_terms,
                            'bk_before': expr_bk,
                            'bk_after': new_bk,
                        }
                        candidates.append({
                            'expr_str': str(result_expr),
                            'terms': new_terms,
                            'bk': new_bk,
                            'action_info': action_info,
                            'log_prob': float(log_probs[action]),
                        })
                        print(f"  Action {len(candidates)}: {spec.identity} on "
                              f"{bracket_type}({i},{j}) t={spec.term_slot} -> "
                              f"{new_terms} tm, {new_bk} bk (rdiff={max_rdiff:.1e})",
                              flush=True)

                    except ActionTimeoutError:
                        continue
                    except Exception:
                        continue
        except Exception as e:
            print(f"  Error building obs/mask: {e}", flush=True)

        shutdown_worker()
        elapsed = time.time() - t0

        result_data = {
            'beam_idx': beam_idx,
            'candidates': candidates,
            'elapsed': elapsed,
        }
        print(f"\nBeam expand: {len(candidates)} candidates, {elapsed:.1f}s", flush=True)

        with open(args.result_file, 'wb') as f:
            pickle.dump(result_data, f)
        print(f"Wrote {args.result_file}", flush=True)
        return

    sub_expr_str = grouping['sub_expr_str']
    grouping_idx = grouping['grouping_idx']
    ref_idx = grouping['ref_idx']
    group_indices = grouping['group_indices']
    avg_sim = grouping['avg_sim']
    n_terms_orig = grouping['n_terms']
    threshold = grouping.get('threshold', None)

    # Parse the sub-expression — either from pre-computed string or from
    # all_terms_str + group_indices (when orchestrator skips sp.cancel)
    if sub_expr_str is not None:
        sub_expr = sp.sympify(sub_expr_str, locals=FUNC_DICT)
    else:
        # Load shared expression data (may be inline or in a separate file)
        if 'all_terms_str' in grouping:
            all_terms_str = grouping['all_terms_str']
            full_denom_str = grouping['denom_str']
        else:
            shared_file = grouping['shared_file']
            with open(shared_file, 'rb') as sf:
                shared = pickle.load(sf)
            all_terms_str = shared['all_terms_str']
            full_denom_str = shared['denom_str']
        all_terms = [sp.sympify(s, locals=FUNC_DICT) for s in all_terms_str]
        full_denom = sp.sympify(full_denom_str, locals=FUNC_DICT)
        group_terms_sum = sum(all_terms[i] for i in group_indices)
        sub_expr = sp.cancel(group_terms_sum / full_denom)
        sub_expr_str = str(sub_expr)
    sub_bk = count_brackets(sub_expr)
    sub_terms = count_terms(sub_expr)

    print(f"Grouping {grouping_idx}: ref={ref_idx}, {len(group_indices)} terms, "
          f"avg_sim={avg_sim:.4f}, {sub_bk} bk, {sub_terms} terms", flush=True)

    # Factor out common brackets and denominator — give model just the reduced numerator
    num, denom = sp.fraction(sp.cancel(sub_expr))
    common_factor, reduced_expr = factor_out_common_brackets(sub_expr)
    reduced_num, _ = sp.fraction(reduced_expr)
    model_expr = reduced_num  # just the numerator polynomial, no denominator
    model_bk = count_brackets(model_expr)
    model_terms = count_terms(model_expr)
    if common_factor != 1:
        cf_bk = count_brackets(common_factor)
        print(f"  Factored out common factor: {common_factor} ({cf_bk} bk)", flush=True)
    print(f"  Model gets numerator only: {model_terms} terms, {model_bk} bk "
          f"(full expr was {sub_terms} terms, {sub_bk} bk)", flush=True)

    # Load model
    device = 'cpu'
    model, action_space, cfg = load_model_and_action_space(args.n_point, args.model, device)

    # Init worker subprocess (for apply_action_with_timeout)
    init_worker(args.n_point, max_terms=cfg['max_terms_action'])

    final_expr_str = str(sub_expr)
    final_bk = sub_bk
    final_terms = sub_terms
    solved = False
    steps_taken = 0
    best_bk = sub_bk
    action_info = None
    recombined = None
    recomb_terms = None
    recomb_bk = None

    if args.single_step:
        # ---- Single-step mode: apply ONE model action ----
        print(f"\n--- Single-step mode ---", flush=True)
        try:
            grouped_skip_n = grouping.get('skip_n', 0)
            new_model_expr, action_info = _apply_single_model_step(
                model=model,
                current_expr=model_expr,
                action_space=action_space,
                n_point=args.n_point,
                max_terms=cfg['max_terms'],
                max_brackets_per_term=cfg['max_brackets_per_term'],
                device=device,
                timeout_sec=30.0,
                reject_term_increase=5,
                max_retries=20,
                verbose=True,
                skip_n=grouped_skip_n,
            )

            if new_model_expr is not None:
                new_model_terms = count_terms(new_model_expr)
                new_model_bk = count_brackets(new_model_expr)
                print(f"  Model: {model_bk}->{new_model_bk} bk, "
                      f"{model_terms}->{new_model_terms} tm", flush=True)

                # Reassemble sub-expression: common_factor * model_result / denom
                r_num, r_denom = sp.fraction(new_model_expr)
                full_result = sp.cancel(common_factor * r_num / (r_denom * denom))
                final_expr_str = str(full_result)
                final_bk = count_brackets(full_result)
                final_terms = count_terms(full_result)
                steps_taken = 1
                best_bk = final_bk

                # Recombine on worker: load shared expression data
                all_terms_str = grouping.get('all_terms_str')
                full_denom_str = grouping.get('denom_str')
                if all_terms_str is None and 'shared_file' in grouping:
                    shared_file = grouping['shared_file']
                    with open(shared_file, 'rb') as sf:
                        shared = pickle.load(sf)
                    all_terms_str = shared['all_terms_str']
                    full_denom_str = shared['denom_str']
                if all_terms_str is not None and full_denom_str is not None:
                    print(f"  Recombining on worker ({len(all_terms_str)} total terms)...",
                          flush=True)
                    t_recomb = time.time()
                    full_denom = sp.sympify(full_denom_str, locals=FUNC_DICT)
                    all_terms = [sp.sympify(s, locals=FUNC_DICT) for s in all_terms_str]
                    consumed = set(group_indices)
                    remaining_indices = sorted(set(range(len(all_terms))) - consumed)
                    if remaining_indices:
                        remaining_sum = sum(all_terms[i] for i in remaining_indices) / full_denom
                        new_sub_expr = sp.sympify(final_expr_str, locals=FUNC_DICT)
                        recombined = sp.cancel(new_sub_expr + remaining_sum)
                    else:
                        recombined = sp.sympify(final_expr_str, locals=FUNC_DICT)
                        recombined = sp.cancel(recombined)
                    recomb_terms = count_terms(recombined)
                    recomb_bk = count_brackets(recombined)
                    # Numerical validation against original expression
                    original_expr = sum(all_terms[i] for i in range(len(all_terms))) / full_denom
                    is_valid, max_rdiff = numerical_validate(original_expr, recombined)
                    valid_tag = "" if is_valid else " *** NUMERICAL FAIL ***"
                    print(f"  Recombined: {recomb_terms} terms, {recomb_bk} bk "
                          f"(rdiff={max_rdiff:.1e}, {time.time() - t_recomb:.1f}s){valid_tag}",
                          flush=True)
                    if not is_valid:
                        print(f"  SKIPPING invalid recombination", flush=True)
                        recombined = None
                        recomb_terms = None

                if new_model_terms < model_terms:
                    print(f"  SUCCESS: {sub_terms}->{final_terms} terms "
                          f"(model: {model_terms}->{new_model_terms})", flush=True)
                else:
                    print(f"  No sub-term reduction ({model_terms}->{new_model_terms}), "
                          f"but saving result for recombination", flush=True)
            else:
                print(f"  No valid action found", flush=True)

        except Exception as e:
            print(f"  Error in single-step: {e}", flush=True)

    else:
        # ---- Full mode: evaluate_sample_backtrack with escalating RTI ----
        rti_values = [1, 3, 5, 10, 20]

        for rti in rti_values:
            print(f"\n--- Attempting RTI={rti} (from scratch) ---", flush=True)
            try:
                result = evaluate_sample_backtrack(
                    model=model,
                    initial_expr=model_expr,
                    target_expr=None,
                    action_space=action_space,
                    n_point=args.n_point,
                    max_terms=cfg['max_terms'],
                    max_brackets_per_term=cfg['max_brackets_per_term'],
                    max_steps=100,
                    device=device,
                    timeout_sec=30.0,
                    sample_timeout=120.0,
                    max_backtracks=10,
                    proper_stop=True,
                    reject_term_increase=rti,
                    verbose=True,
                )

                r_expr_str = result.get('final_expr', str(model_expr))
                r_expr = sp.sympify(r_expr_str, locals=FUNC_DICT)
                r_terms = count_terms(r_expr)
                r_solved = result.get('solved_source_rel', False) or result.get('solved_proper', False)
                r_steps = result.get('steps', 0)

                print(f"  RTI={rti} model result: {model_bk}->{count_brackets(r_expr)} bk "
                      f"({model_terms}->{r_terms} tm), solved={r_solved}, steps={r_steps}", flush=True)

                if r_terms < model_terms or r_solved:
                    # Success: recombine common_factor * model_result / denom
                    r_num, r_denom = sp.fraction(r_expr)
                    full_result = sp.cancel(common_factor * r_num / (r_denom * denom))
                    final_expr_str = str(full_result)
                    final_bk = count_brackets(full_result)
                    final_terms = count_terms(full_result)
                    solved = r_solved
                    steps_taken = r_steps
                    best_bk = final_bk
                    print(f"  SUCCESS at RTI={rti}: {sub_terms}->{final_terms} terms "
                          f"(model: {model_terms}->{r_terms})", flush=True)
                    break
                else:
                    print(f"  RTI={rti} did not reduce terms, trying next...", flush=True)

            except Exception as e:
                print(f"  Error at RTI={rti}: {e}", flush=True)
                continue

    shutdown_worker()

    elapsed = time.time() - t0

    # Build result
    result_data = {
        'grouping_idx': grouping_idx,
        'ref_idx': ref_idx,
        'group_indices': group_indices,
        'avg_sim': avg_sim,
        'n_group_terms': len(group_indices),
        'sub_bk_before': sub_bk,
        'sub_terms_before': sub_terms,
        'sub_bk_after': final_bk,
        'sub_terms_after': final_terms,
        'best_bk': best_bk,
        'solved': solved,
        'steps_taken': steps_taken,
        'bk_reduced': sub_bk - final_bk,
        'terms_reduced': sub_terms - final_terms,
        'final_expr_str': final_expr_str,
        'elapsed': elapsed,
        'threshold': threshold,
        'action_info': action_info,
    }

    # Add recombined full expression if worker did recombination
    if recombined is not None:
        result_data['recombined_expr_str'] = str(recombined)
        result_data['recombined_terms'] = recomb_terms
        result_data['recombined_bk'] = recomb_bk

        # Pre-compute hash and decomposition so orchestrator doesn't need sp.sympify
        try:
            max_terms = grouping.get('max_terms', 25)
            max_bpt = grouping.get('max_brackets_per_term', 20)
            n_point = args.n_point
            if recomb_terms <= max_terms:
                obs = build_observation_from_expr(recombined, n_point, max_terms, max_bpt)
                result_data['obs_hash'] = obs_to_hash(obs)
            # Decompose into terms_num strings and denom string
            numerator, denominator = sp.fraction(recombined)
            if isinstance(numerator, sp.Add):
                result_data['terms_num_str'] = [str(t) for t in numerator.args]
            else:
                result_data['terms_num_str'] = [str(numerator)]
            result_data['denom_str'] = str(denominator)
        except Exception as e:
            print(f"  Warning: pre-compute hash/decomp failed: {e}", flush=True)

    print(f"\nResult: {sub_bk}->{final_bk} bk ({sub_terms}->{final_terms} tm), "
          f"solved={solved}, steps={steps_taken}, {elapsed:.1f}s", flush=True)

    # Write result
    with open(args.result_file, 'wb') as f:
        pickle.dump(result_data, f)

    print(f"Wrote {args.result_file}", flush=True)


if __name__ == '__main__':
    main()
