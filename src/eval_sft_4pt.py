"""
Evaluation script for SFT-trained spinor-helicity simplification models.
Version 4: Uses term-specific action space (TermSpecificActionSpace).

Key features:
- Uses 1452 action space (12 brackets * 11 term_slots * 11 identities)
- 11 identities per bracket: 1 Schouten + 6 Momentum + 4 Momentum_sq
- term_slot=0: apply to ENTIRE expression, term_slot=1-10: single term
- Retry on timeout with next best action
"""

import argparse
import os
import sys
import pickle
import time
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import sympy as sp
from tqdm import tqdm


class ActionTimeoutError(Exception):
    pass


# Global worker process
_worker_process = None
_request_queue = None
_response_queue = None


def _persistent_worker(request_queue, response_queue, n_point, max_terms):
    """Persistent worker that handles action requests with term-specific support."""
    import sys
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.dirname(SCRIPT_DIR)
    sys.path.insert(0, os.path.join(REPO_ROOT, 'spinorhelicity'))
    sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))
    from environment.spin_helicity_env import SpinHelExpr
    from environment.bracket_env import ab, sb
    from action_space_4pt import apply_identity_to_term, get_unique_brackets
    import sympy as sp
    from sympy import Function

    def get_identity_result(bracket, identity, params, n_pt, func_dict):
        """Get the RHS of an identity."""
        from environment.utils import reorder_expr
        bk_type = bracket.func.__name__
        bk_args = list(bracket.args)

        if identity == 'schouten':
            pk, pl = params['pk'], params['pl']
            pi, pj = bk_args
            bk_func = func_dict[bk_type]
            ret_expr = (bk_func(pi, pl) * bk_func(pk, pj) / bk_func(pk, pl) +
                        bk_func(pi, pk) * bk_func(pj, pl) / bk_func(pk, pl))
            return reorder_expr(ret_expr)

        elif identity == 'momentum':
            pout1, pout2 = params['pout1'], params['pout2']
            bk_type_opp = 'ab' if bk_type == 'sb' else 'sb'
            larg = pout1 if bk_type == 'ab' else pout2
            rarg = pout2 if bk_type == 'ab' else pout1
            in_arg = [arg for arg in bk_args if arg != pout1][0]
            sign = 1 if ((bk_type == 'ab' and pout1 == bk_args[0]) or
                         (bk_type == 'sb' and pout1 == bk_args[1])) else -1
            ret_expr = 0
            denom_bk = (func_dict[bk_type_opp](in_arg, pout2) if bk_type == 'ab'
                        else func_dict[bk_type_opp](pout2, in_arg))
            for k in range(1, n_pt + 1):
                if k == in_arg or k == larg or k == rarg:
                    continue
                ret_expr -= func_dict['ab'](larg, k) * func_dict['sb'](k, rarg) / denom_bk
            ret_expr = sign * ret_expr
            return reorder_expr(ret_expr)

        elif identity == 'momentum_sq':
            mom_add = params.get('mom_add', [])
            bk_type_opp = 'ab' if bk_type == 'sb' else 'sb'
            plist_lhs = list(mom_add) + bk_args
            plist_rhs = [i for i in range(1, n_pt + 1) if i not in plist_lhs]
            ret_expr = 0
            denom_bk = func_dict[bk_type_opp](bk_args[0], bk_args[1])
            for j1 in plist_rhs:
                for j2 in plist_rhs:
                    if j2 > j1:
                        ret_expr += (func_dict['ab'](j1, j2) * func_dict['sb'](j1, j2)) / denom_bk
            for l1 in plist_lhs:
                for l2 in plist_lhs:
                    if l2 > l1 and (l1 not in bk_args or l2 not in bk_args):
                        ret_expr -= (func_dict['ab'](l1, l2) * func_dict['sb'](l1, l2)) / denom_bk
            return reorder_expr(ret_expr)

        raise ValueError(f"Unknown identity: {identity}")

    while True:
        try:
            request = request_queue.get()
            if request is None:
                break

            expr_str, bracket_type, bracket_args, identity, params, term_slot = request

            try:
                expr_env = SpinHelExpr(expr_str, n_point)
                func_dict = {'ab': ab, 'sb': sb}

                # Reconstruct the bracket
                bk_func = ab if bracket_type == 'ab' else sb
                bracket = bk_func(bracket_args[0], bracket_args[1])

                # Check if bracket exists in expression
                num, _ = sp.fraction(expr_env.sp_expr)
                brackets_in_expr = [b for b in num.atoms(Function) if b.func.__name__ in ['ab', 'sb']]

                bracket_found = None
                for b in brackets_in_expr:
                    if b.func.__name__ == bracket_type and set(b.args) == set(bracket_args):
                        bracket_found = b
                        break

                if bracket_found is None:
                    response_queue.put(None)
                    continue

                # Apply action based on term_slot
                if term_slot == 0:
                    # Whole expression application (term_slot=0)
                    if identity == 'schouten':
                        expr_env.schouten_single(bracket_found, params['pk'], params['pl'], numerator_only=True)
                    elif identity == 'momentum':
                        expr_env.momentum_single(bracket_found, params['pout1'], params['pout2'], numerator_only=True)
                    elif identity == 'momentum_sq':
                        expr_env.momentum_sq(bracket_found, params['mom_add'], numerator_only=True)
                    else:
                        response_queue.put(None)
                        continue
                    expr_env.cancel()
                    result = expr_env.sp_expr
                else:
                    # Term-specific application (term_slot=1-10 -> term_idx=0-9)
                    term_idx = term_slot - 1
                    identity_result = get_identity_result(bracket_found, identity, params, n_point, func_dict)
                    result = apply_identity_to_term(
                        expr_env.sp_expr, term_idx, bracket_found, identity_result,
                        power_to_apply=1, numerator_only=True
                    )

                response_queue.put(str(result))
            except Exception as e:
                response_queue.put(None)
        except Exception:
            break


def init_worker(n_point, max_terms=10):
    """Initialize the persistent worker process."""
    global _worker_process, _request_queue, _response_queue

    if _worker_process is not None and _worker_process.is_alive():
        return

    _request_queue = mp.Queue()
    _response_queue = mp.Queue()
    _worker_process = mp.Process(
        target=_persistent_worker,
        args=(_request_queue, _response_queue, n_point, max_terms)
    )
    _worker_process.start()


def shutdown_worker():
    """Shutdown the persistent worker process."""
    global _worker_process, _request_queue, _response_queue

    if _worker_process is not None:
        if _request_queue is not None:
            _request_queue.put(None)
        _worker_process.join(timeout=2)
        if _worker_process.is_alive():
            _worker_process.terminate()
            _worker_process.join(timeout=1)
            if _worker_process.is_alive():
                _worker_process.kill()
        _worker_process = None
        _request_queue = None
        _response_queue = None


def apply_action_with_timeout(expr, bracket_type, bracket_args, identity, params, term_slot, n_point, timeout_sec):
    """Apply action with timeout using persistent worker.

    Args:
        term_slot: 0 = whole expression, 1-10 = specific term (term_slot-1)
    """
    global _worker_process, _request_queue, _response_queue

    if _worker_process is None or not _worker_process.is_alive():
        init_worker(n_point)

    while not _response_queue.empty():
        try:
            _response_queue.get_nowait()
        except:
            break

    _request_queue.put((str(expr), bracket_type, bracket_args, identity, params, term_slot))

    try:
        result_str = _response_queue.get(timeout=timeout_sec)
    except:
        shutdown_worker()
        init_worker(n_point)
        raise ActionTimeoutError("Action timed out")

    if result_str is None:
        return None

    try:
        from environment.bracket_env import ab, sb
        result = sp.sympify(result_str, locals={'ab': ab, 'sb': sb})
        return result
    except:
        return None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, 'spinorhelicity'))

from spinhelicity_env import build_observation_from_expr
from train_sft_transformer import TransformerPolicySFT
from action_space_4pt import TermSpecificActionSpace, get_unique_brackets, get_term_bracket_powers
from environment.bracket_env import ab, sb


def count_brackets(sp_expr: sp.Expr) -> int:
    """Count total bracket occurrences with multiplicity from powers.

    Power-aware: ab(1,2)**3 counts as 3, not 1.
    Counts over both numerator and denominator.
    """
    if sp_expr == 0:
        return 0
    count = 0
    numer, denom = sp.fraction(sp_expr)

    def _count(e):
        nonlocal count
        if isinstance(e, sp.Function) and e.func.__name__ in ('ab', 'sb'):
            count += 1
        elif isinstance(e, sp.Pow):
            base, exp = e.args
            if isinstance(base, sp.Function) and base.func.__name__ in ('ab', 'sb'):
                try:
                    count += int(abs(exp))
                except (TypeError, ValueError):
                    count += 1
            else:
                _count(base)
        elif isinstance(e, (sp.Mul, sp.Add)):
            for arg in e.args:
                _count(arg)

    _count(numer)
    _count(denom)
    return count


def count_terms(sp_expr: sp.Expr) -> int:
    """Count number of terms in expression."""
    if sp_expr == 0:
        return 0
    # Expand to sum and count args
    expanded = sp.expand(sp_expr)
    if expanded.func == sp.Add:
        return len(expanded.args)
    else:
        return 1


def obs_to_hash(obs: np.ndarray) -> int:
    """Hash observation for cycle detection."""
    return hash(obs.tobytes())


def build_action_mask_v3(expr, action_space: TermSpecificActionSpace, numerator_only: bool = True):
    """Build action mask for term-specific action space.

    term_slot=0: whole expression (valid if bracket exists anywhere)
    term_slot=1-10: specific term (valid if bracket exists in that term)
    """
    brackets = get_unique_brackets(expr, numerator_only)
    mask = np.zeros(action_space.n_actions, dtype=np.float32)

    term_bracket_powers = get_term_bracket_powers(expr, numerator_only)
    num_terms = len(term_bracket_powers)

    for bracket in brackets:
        try:
            bracket_idx = action_space.get_bracket_idx(bracket)
        except ValueError:
            continue

        # term_slot=0: whole expression - valid if bracket exists anywhere
        for id_idx in range(action_space.identities_per_bracket):
            action_idx = action_space.encode_action(bracket_idx, 0, id_idx)
            mask[action_idx] = 1.0

        # term_slot=1-10: specific term - valid if bracket in that term
        for term_idx in range(min(num_terms, action_space.max_terms)):
            if bracket in term_bracket_powers[term_idx]:
                term_slot = term_idx + 1  # term_slot = term_idx + 1
                for id_idx in range(action_space.identities_per_bracket):
                    action_idx = action_space.encode_action(bracket_idx, term_slot, id_idx)
                    mask[action_idx] = 1.0

    return mask


def expressions_equal(e1: sp.Expr, e2: sp.Expr) -> bool:
    """Check if two expressions are equal."""
    if e1 == e2:
        return True
    try:
        diff = sp.cancel(e1 - e2)
        if diff == 0:
            return True
        diff = sp.expand(diff)
        if diff == 0:
            return True
    except:
        pass
    return False


def evaluate_sample(
    model,
    initial_expr,
    target_expr,
    action_space: TermSpecificActionSpace,
    n_point: int,
    max_terms: int,
    max_brackets_per_term: int,
    max_steps: int,
    device: str,
    timeout_sec: float = 30.0,
    max_retries: int = 3,
    sample_timeout: float = 120.0,
    verbose: bool = False,
    sample_idx: int = 0,
    use_anticycle: bool = True,
    random_policy: bool = False,
    proper_stop: bool = False,
    reject_term_increase: int = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate a single sample with verbose output and detailed metrics.

    Args:
        reject_term_increase: If set, reject actions that increase terms by more than this amount.
    """
    t0 = time.time()
    current_expr = initial_expr
    trajectory = []
    solved = False
    timeout_count = 0
    retry_count = 0

    start_brackets = count_brackets(initial_expr)
    target_brackets = count_brackets(target_expr)
    min_brackets = start_brackets

    # For anti-cycle: track visited states
    visited_states = set()
    start_obs = build_observation_from_expr(initial_expr, n_point, max_terms, max_brackets_per_term)
    visited_states.add(obs_to_hash(start_obs))

    start_terms = count_terms(initial_expr)
    target_terms = count_terms(target_expr)

    # Source-relative tracking: find best state that is strictly simpler than source
    achieved_source_rel = False
    best_src_rel_bk = float('inf')
    best_src_rel_tm = float('inf')
    best_src_rel_expr = None

    if verbose:
        print(f"  [Sample {sample_idx}] start={start_brackets} target={target_brackets} terms={start_terms}", flush=True)
        # Print initial expression
        expr_str = str(initial_expr)
        if len(expr_str) > 500:
            expr_str = expr_str[:500] + "..."
        print(f"    initial expr = {expr_str}", flush=True)
        # Print target expression
        target_str = str(target_expr)
        if len(target_str) > 500:
            target_str = target_str[:500] + "..."
        print(f"    target expr  = {target_str}", flush=True)

    for step in range(max_steps):
        step_t0 = time.time()

        # Check time limit
        elapsed = time.time() - t0
        if elapsed > sample_timeout:
            if verbose:
                print(f"    Step {step}: timeout after {elapsed:.1f}s, breaking", flush=True)
            break

        # Check if expression is zero
        if current_expr == 0:
            min_brackets = 0
            if verbose:
                print(f"    Step {step}: expr=0, breaking", flush=True)
            break

        t1 = time.time()
        current_brackets = count_brackets(current_expr)
        current_terms = count_terms(current_expr)
        t_count = time.time() - t1

        if current_brackets < min_brackets:
            min_brackets = current_brackets

        # Check if we reached target
        if proper_stop:
            # Proper stop: require both brackets AND terms at or below target
            if current_brackets <= target_brackets and current_terms <= target_terms:
                if verbose:
                    print(f"    Step {step}: reached proper target (brackets={current_brackets}<={target_brackets}, terms={current_terms}<={target_terms}), breaking", flush=True)
                break
        else:
            # Default: stop when brackets reach target
            if current_brackets <= target_brackets:
                if verbose:
                    print(f"    Step {step}: reached target ({current_brackets}<={target_brackets}), breaking", flush=True)
                break

        # Build observation
        t1 = time.time()
        try:
            obs = build_observation_from_expr(current_expr, n_point, max_terms, max_brackets_per_term)
        except Exception as e:
            if verbose:
                print(f"    Step {step}: failed to build obs, breaking", flush=True)
            break
        t_obs = time.time() - t1

        # Build action mask
        t1 = time.time()
        action_mask = build_action_mask_v3(current_expr, action_space)
        t_mask = time.time() - t1

        if action_mask.sum() == 0:
            if verbose:
                print(f"    Step {step}: no valid actions, breaking", flush=True)
            break

        # Get action ordering (model or random)
        t1 = time.time()
        if random_policy:
            # Random policy: shuffle valid actions
            valid_actions = np.where(action_mask == 1)[0]
            np.random.shuffle(valid_actions)
            sorted_actions = valid_actions
        else:
            # Model policy: use model predictions
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            state_hash = obs_to_hash(obs)
            with torch.no_grad():
                logits = model(obs_tensor)

            # Apply mask and get sorted actions
            FLOAT_MIN = -3.4e38
            mask_tensor = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0).to(device)
            inf_mask = torch.clamp(torch.log(mask_tensor + 1e-10), min=FLOAT_MIN)
            masked_logits = logits + inf_mask

            if temperature > 0:
                # Temperature sampling: stochastic action ordering
                probs = torch.softmax(masked_logits[0] / temperature, dim=-1)
                # Clamp to avoid numerical issues with multinomial
                probs = torch.clamp(probs, min=1e-10)
                probs = probs / probs.sum()
                n_valid = int((action_mask > 0).sum())
                n_sample = min(n_valid, max_retries + 20)
                sorted_actions = torch.multinomial(probs, num_samples=n_sample, replacement=False).cpu().numpy()
            else:
                # Greedy: deterministic action ordering
                sorted_actions = torch.argsort(masked_logits[0], descending=True).cpu().numpy()
        t_model = time.time() - t1

        t1 = time.time()
        action_succeeded = False
        chosen_identity = None
        chosen_bracket = None
        chosen_term_slot = None

        for retry_idx, action in enumerate(sorted_actions[:max_retries + 20]):
            if action_mask[action] == 0:
                continue

            # Decode action
            spec = action_space.get_action_spec(action)
            bracket_type, i, j = action_space.get_bracket_from_idx(spec.bracket_idx)

            try:
                result = apply_action_with_timeout(
                    current_expr, bracket_type, (i, j),
                    spec.identity, spec.params, spec.term_slot,
                    n_point, timeout_sec
                )

                if result is None:
                    continue

                # Check for anti-cycle
                if use_anticycle:
                    try:
                        candidate_obs = build_observation_from_expr(result, n_point, max_terms, max_brackets_per_term)
                        candidate_hash = obs_to_hash(candidate_obs)
                        if candidate_hash in visited_states:
                            continue
                        visited_states.add(candidate_hash)
                    except:
                        pass

                # Check for term explosion
                new_terms = count_terms(result)
                if reject_term_increase is not None:
                    term_increase = new_terms - current_terms
                    if term_increase > reject_term_increase:
                        continue

                new_brackets = count_brackets(result)
                trajectory.append({
                    'action': int(action),
                    'bracket': f"{bracket_type}({i},{j})",
                    'term_slot': spec.term_slot,
                    'is_whole_expression': spec.is_whole_expression,
                    'identity': spec.identity,
                    'params': spec.params,
                    'retry': retry_idx,
                    'brackets_before': current_brackets,
                    'brackets_after': new_brackets,
                })
                chosen_identity = spec.identity
                chosen_bracket = f"{bracket_type}({i},{j})"
                chosen_term_slot = spec.term_slot
                current_expr = result
                action_succeeded = True

                # Check source-relative condition
                is_src_rel = (new_brackets < start_brackets and new_terms <= start_terms) or \
                             (new_brackets <= start_brackets and new_terms < start_terms)
                if is_src_rel:
                    achieved_source_rel = True
                    if new_brackets < best_src_rel_bk or (new_brackets == best_src_rel_bk and new_terms < best_src_rel_tm):
                        best_src_rel_bk = new_brackets
                        best_src_rel_tm = new_terms
                        best_src_rel_expr = result

                if retry_idx > 0:
                    retry_count += 1
                break
            except ActionTimeoutError:
                timeout_count += 1
                continue
            except Exception as e:
                continue

        t_apply = time.time() - t1

        if not action_succeeded:
            if verbose:
                print(f"    Step {step}: no valid action found, breaking", flush=True)
            break

        step_time = time.time() - step_t0
        if verbose:
            new_brackets = count_brackets(current_expr)
            new_terms = count_terms(current_expr)
            ts_str = f"ts={chosen_term_slot}" if chosen_term_slot else ""
            retry_str = f" (retry={retry_idx})" if retry_idx > 0 else ""
            print(f"    Step {step}: {current_brackets}->{new_brackets} terms={new_terms} via {chosen_identity} {chosen_bracket} {ts_str}{retry_str}", flush=True)
            # Print the expression (truncated if too long)
            expr_str = str(current_expr)
            if len(expr_str) > 500:
                expr_str = expr_str[:500] + "..."
            print(f"      expr = {expr_str}", flush=True)

    # Final metrics
    final_brackets = count_brackets(current_expr)
    final_terms = count_terms(current_expr)
    # target_terms and start_terms already computed above
    if final_brackets < min_brackets:
        min_brackets = final_brackets

    # Also check source-rel on final state
    is_src_rel = (final_brackets < start_brackets and final_terms <= start_terms) or \
                 (final_brackets <= start_brackets and final_terms < start_terms)
    if is_src_rel:
        achieved_source_rel = True
        if final_brackets < best_src_rel_bk or (final_brackets == best_src_rel_bk and final_terms < best_src_rel_tm):
            best_src_rel_bk = final_brackets
            best_src_rel_tm = final_terms
            best_src_rel_expr = current_expr

    # Target-relative: both brackets AND terms at or below target in final state
    solved_proper = (final_brackets <= target_brackets) and (final_terms <= target_terms)
    # Source-relative: achieved at any point during evaluation (but not target-rel)
    solved_source_rel = achieved_source_rel and not solved_proper
    # Legacy field
    solved = solved_proper

    # For PARTIAL, report best source-rel state
    if solved_source_rel and best_src_rel_expr is not None:
        reported_bk = best_src_rel_bk
        reported_tm = best_src_rel_tm
        reported_expr = str(best_src_rel_expr)
    else:
        reported_bk = final_brackets
        reported_tm = final_terms
        reported_expr = str(current_expr)

    if verbose:
        if solved_proper:
            print(f"  [Sample {sample_idx}] SUCCESS: {start_brackets}->{final_brackets} bk, "
                  f"{start_terms}->{final_terms} tm (target: {target_brackets} bk, {target_terms} tm) "
                  f"in {len(trajectory)} steps", flush=True)
        elif solved_source_rel:
            print(f"  [Sample {sample_idx}] PARTIAL (source-rel): {start_brackets}->{reported_bk} bk, "
                  f"{start_terms}->{reported_tm} tm (target: {target_brackets} bk, {target_terms} tm) "
                  f"in {len(trajectory)} steps", flush=True)
        else:
            print(f"  [Sample {sample_idx}] FAILED: {start_brackets}->{final_brackets} bk, "
                  f"{start_terms}->{final_terms} tm (target: {target_brackets} bk, {target_terms} tm) "
                  f"in {len(trajectory)} steps", flush=True)

    return {
        'solved': solved,
        'solved_proper': solved_proper,
        'solved_source_rel': solved_source_rel,
        'start_brackets': start_brackets,
        'target_brackets': target_brackets,
        'final_brackets': reported_bk,
        'min_brackets': min_brackets,
        'start_terms': start_terms,
        'target_terms': target_terms,
        'final_terms': reported_tm,
        'steps': len(trajectory),
        'reduction': start_brackets - min_brackets,
        'reduction_ratio': (start_brackets - min_brackets) / max(start_brackets, 1),
        'trajectory': trajectory,
        'final_expr': reported_expr,
        'timeout_count': timeout_count,
        'retry_count': retry_count,
    }


def evaluate_sample_backtrack(
    model,
    initial_expr,
    target_expr,
    action_space: TermSpecificActionSpace,
    n_point: int,
    max_terms: int,
    max_brackets_per_term: int,
    max_steps: int,
    device: str,
    timeout_sec: float = 30.0,
    max_retries: int = 3,
    sample_timeout: float = 120.0,
    verbose: bool = False,
    sample_idx: int = 0,
    use_anticycle: bool = True,
    patience: int = 5,
    max_backtracks: int = 10,
    reject_term_increase: int = None,
    random_policy: bool = False,
    proper_stop: bool = False,
    checkpoint_by_terms: bool = False,
) -> Dict[str, Any]:
    """Evaluate with backtracking on dead ends and max_steps failures.

    Args:
        reject_term_increase: If set, reject actions that increase terms by more than this amount.

    Backtracking is triggered in two cases:
    1. Dead end: no valid action can be found (all actions blocked by anti-cycle)
    2. Max steps: hit step limit without solving - backtrack to best checkpoint and retry

    On backtrack, we return to the best checkpoint (lowest bracket count) and block
    the action that led away from it, then continue with additional steps.
    """
    t0 = time.time()
    current_expr = initial_expr
    trajectory = []
    timeout_count = 0
    retry_count = 0
    backtrack_count = 0

    start_brackets = count_brackets(initial_expr)
    target_brackets = count_brackets(target_expr)
    min_brackets = start_brackets
    start_terms = count_terms(initial_expr)
    target_terms = count_terms(target_expr)

    # Source-relative tracking: find best state that is strictly simpler than source
    achieved_source_rel = False
    best_src_rel_bk = float('inf')
    best_src_rel_tm = float('inf')
    best_src_rel_expr = None

    # For anti-cycle: track visited states
    visited_states = set()
    start_obs = build_observation_from_expr(initial_expr, n_point, max_terms, max_brackets_per_term)
    visited_states.add(obs_to_hash(start_obs))

    # Backtracking state: list of checkpoints
    # Each checkpoint: [expr, brackets, terms, blocked_actions, visited_states_copy, action_that_left]
    # action_that_left is the action that led away from this checkpoint (to block on backtrack)
    # blocked_actions accumulates actions that led to bad paths from this checkpoint
    checkpoints = [[initial_expr, start_brackets, start_terms, set(), visited_states.copy(), None]]
    blocked_actions_at_current = set()  # Actions to skip at current state
    last_progress_action = None  # Action that led to bracket decrease (to block on backtrack)

    if verbose:
        print(f"  [Sample {sample_idx}] start={start_brackets} target={target_brackets} terms={start_terms} (backtrack mode)", flush=True)
        expr_str = str(initial_expr)
        if len(expr_str) > 500:
            expr_str = expr_str[:500] + "..."
        print(f"    initial expr = {expr_str}", flush=True)
        target_str = str(target_expr)
        if len(target_str) > 500:
            target_str = target_str[:500] + "..."
        print(f"    target expr  = {target_str}", flush=True)

    step = 0
    solved_early = False
    hit_timeout = False
    while step < max_steps:
        # Check time limit
        elapsed = time.time() - t0
        if elapsed > sample_timeout:
            if verbose:
                print(f"    Step {step}: timeout after {elapsed:.1f}s, breaking", flush=True)
            hit_timeout = True
            break

        # Check if expression is zero
        if current_expr == 0:
            min_brackets = 0
            if verbose:
                print(f"    Step {step}: expr=0, breaking", flush=True)
            solved_early = True
            break

        current_brackets = count_brackets(current_expr)
        current_terms = count_terms(current_expr)

        if current_brackets < min_brackets:
            min_brackets = current_brackets

        # Check if we reached target
        if proper_stop:
            # Proper stop: require both brackets AND terms at or below target
            if current_brackets <= target_brackets and current_terms <= target_terms:
                if verbose:
                    print(f"    Step {step}: reached proper target (brackets={current_brackets}<={target_brackets}, terms={current_terms}<={target_terms}), breaking", flush=True)
                solved_early = True
                break
        else:
            # Default: stop when brackets reach target
            if current_brackets <= target_brackets:
                if verbose:
                    print(f"    Step {step}: reached target ({current_brackets}<={target_brackets}), breaking", flush=True)
                solved_early = True
                break

        # Build observation
        try:
            obs = build_observation_from_expr(current_expr, n_point, max_terms, max_brackets_per_term)
        except Exception as e:
            if verbose:
                print(f"    Step {step}: failed to build obs, breaking", flush=True)
            break

        # Build action mask
        action_mask = build_action_mask_v3(current_expr, action_space)

        if action_mask.sum() == 0:
            if verbose:
                print(f"    Step {step}: no valid actions, breaking", flush=True)
            break

        # Get action ordering (model or random)
        if random_policy:
            # Random policy: shuffle valid actions
            valid_actions = np.where(action_mask == 1)[0]
            np.random.shuffle(valid_actions)
            sorted_actions = valid_actions
        else:
            # Model policy: use model predictions
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(obs_tensor)

            FLOAT_MIN = -3.4e38
            mask_tensor = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0).to(device)
            inf_mask = torch.clamp(torch.log(mask_tensor + 1e-10), min=FLOAT_MIN)
            masked_logits = logits + inf_mask

            sorted_actions = torch.argsort(masked_logits[0], descending=True).cpu().numpy()

        action_succeeded = False
        chosen_identity = None
        chosen_bracket = None
        chosen_term_slot = None
        chosen_action = None
        chosen_retry_idx = 0

        for retry_idx, action in enumerate(sorted_actions[:max_retries + 20]):
            if action_mask[action] == 0:
                continue
            if action in blocked_actions_at_current:
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
                    continue

                # Check for anti-cycle
                if use_anticycle:
                    try:
                        candidate_obs = build_observation_from_expr(result, n_point, max_terms, max_brackets_per_term)
                        candidate_hash = obs_to_hash(candidate_obs)
                        if candidate_hash in visited_states:
                            continue
                        visited_states.add(candidate_hash)
                    except:
                        pass

                # Check for term explosion
                new_terms = count_terms(result)
                if reject_term_increase is not None:
                    term_increase = new_terms - current_terms
                    if term_increase > reject_term_increase:
                        continue

                new_brackets = count_brackets(result)
                trajectory.append({
                    'action': int(action),
                    'bracket': f"{bracket_type}({i},{j})",
                    'term_slot': spec.term_slot,
                    'is_whole_expression': spec.is_whole_expression,
                    'identity': spec.identity,
                    'params': spec.params,
                    'retry': retry_idx,
                    'brackets_before': current_brackets,
                    'brackets_after': new_brackets,
                    'terms_before': current_terms,
                    'terms_after': new_terms,
                })
                chosen_identity = spec.identity
                chosen_bracket = f"{bracket_type}({i},{j})"
                chosen_term_slot = spec.term_slot
                chosen_action = action
                chosen_retry_idx = retry_idx

                current_expr = result

                # Check source-relative condition
                is_src_rel = (new_brackets < start_brackets and new_terms <= start_terms) or \
                             (new_brackets <= start_brackets and new_terms < start_terms)
                if is_src_rel:
                    achieved_source_rel = True
                    if new_brackets < best_src_rel_bk or (new_brackets == best_src_rel_bk and new_terms < best_src_rel_tm):
                        best_src_rel_bk = new_brackets
                        best_src_rel_tm = new_terms
                        best_src_rel_expr = result

                # Check if we made progress (bracket reduction)
                if new_brackets < current_brackets:
                    # Save checkpoint at the NEW state (after the bracket reduction)
                    # action_that_left is None initially - will be set when we take an action from this state
                    checkpoints.append([result, new_brackets, new_terms, set(), visited_states.copy(), None])
                    blocked_actions_at_current = set()
                else:
                    # Update the last checkpoint's action_that_left if we're leaving a checkpoint state
                    # This tracks which action led away from the best state
                    if len(checkpoints) > 0 and checkpoints[-1][5] is None:
                        checkpoints[-1][5] = action
                action_succeeded = True
                if retry_idx > 0:
                    retry_count += 1
                break
            except ActionTimeoutError:
                timeout_count += 1
                continue
            except Exception as e:
                continue

        if not action_succeeded:
            # Dead end - try to backtrack
            if backtrack_count < max_backtracks and len(checkpoints) > 1:
                # Find best checkpoint (lowest bracket count, or lowest term count if checkpoint_by_terms)
                best_idx = 0
                if checkpoint_by_terms:
                    best_metric = checkpoints[0][2]  # terms
                    for i, cp in enumerate(checkpoints):
                        if cp[2] < best_metric:
                            best_metric = cp[2]
                            best_idx = i
                else:
                    best_metric = checkpoints[0][1]  # brackets
                    for i, cp in enumerate(checkpoints):
                        if cp[1] < best_metric:
                            best_metric = cp[1]
                            best_idx = i

                saved_expr, saved_brackets, saved_terms, saved_blocked, saved_visited, action_that_left = checkpoints[best_idx]

                # Block the action that led away from this checkpoint
                if action_that_left is not None:
                    saved_blocked.add(action_that_left)
                    checkpoints[best_idx][5] = None  # Reset action_that_left

                if verbose:
                    print(f"    Step {step}: BACKTRACK (dead end) to checkpoint with {saved_brackets} brackets, {saved_terms} terms "
                          f"(blocking action {action_that_left})", flush=True)

                current_expr = saved_expr
                visited_states = saved_visited.copy()
                blocked_actions_at_current = saved_blocked.copy()
                backtrack_count += 1
                continue

            if verbose:
                print(f"    Step {step}: no valid action found, no backtrack possible, breaking", flush=True)
            break

        if verbose:
            new_brackets = count_brackets(current_expr)
            new_terms = count_terms(current_expr)
            ts_str = f"ts={chosen_term_slot}" if chosen_term_slot else ""
            retry_str = f" (retry={chosen_retry_idx})" if chosen_retry_idx > 0 else ""
            bt_str = f" [bt={backtrack_count}]" if backtrack_count > 0 else ""
            print(f"    Step {step}: {current_brackets}->{new_brackets} terms={new_terms} via {chosen_identity} {chosen_bracket} {ts_str}{retry_str}{bt_str}", flush=True)
            expr_str = str(current_expr)
            if len(expr_str) > 500:
                expr_str = expr_str[:500] + "..."
            print(f"      expr = {expr_str}", flush=True)

        step += 1

    # If we hit max_steps or timeout without solving, try backtracking multiple times
    # Give backtracking a fresh time budget per attempt (30 seconds each)
    backtrack_timeout_per_attempt = 30
    while not solved_early and backtrack_count < max_backtracks and len(checkpoints) > 1:
        # Reset timer for each backtrack attempt (fresh time budget)
        backtrack_start = time.time()

        # Find best checkpoint (lowest bracket count, or lowest term count if checkpoint_by_terms)
        best_idx = 0
        if checkpoint_by_terms:
            best_metric = checkpoints[0][2]  # terms
            for i, cp in enumerate(checkpoints):
                if cp[2] < best_metric:
                    best_metric = cp[2]
                    best_idx = i
        else:
            best_metric = checkpoints[0][1]  # brackets
            for i, cp in enumerate(checkpoints):
                if cp[1] < best_metric:
                    best_metric = cp[1]
                    best_idx = i

        # Get the best checkpoint
        saved_expr, saved_brackets, saved_terms, saved_blocked, saved_visited, action_that_left = checkpoints[best_idx]

        # If the best checkpoint has an action to block, block it
        if action_that_left is not None:
            saved_blocked.add(action_that_left)
            checkpoints[best_idx][5] = None  # Reset action_that_left
        # Otherwise, we'll just retry from this checkpoint with its accumulated blocked actions
        # (the new path will naturally try different actions due to anti-cycle)

        if verbose:
            print(f"    BACKTRACK #{backtrack_count+1} to checkpoint with {saved_brackets} brackets, {saved_terms} terms "
                  f"(blocking action {action_that_left})", flush=True)

        current_expr = saved_expr
        visited_states = saved_visited.copy()
        blocked_actions_at_current = saved_blocked.copy()
        backtrack_count += 1

        # Continue from the checkpoint with limited extra steps (try many short attempts)
        extra_steps = min(20, max_steps)  # Only 20 extra steps per backtrack attempt
        reached_target = False
        for extra_step in range(extra_steps):
            # Use per-backtrack timeout instead of original sample_timeout
            backtrack_elapsed = time.time() - backtrack_start
            if backtrack_elapsed > backtrack_timeout_per_attempt:
                if verbose:
                    print(f"    Extra step {extra_step}: backtrack timeout after {backtrack_elapsed:.1f}s, breaking", flush=True)
                break

            if current_expr == 0:
                min_brackets = 0
                reached_target = True
                break

            current_brackets = count_brackets(current_expr)
            current_terms = count_terms(current_expr)

            if current_brackets < min_brackets:
                min_brackets = current_brackets

            if proper_stop:
                # Proper stop: require both brackets AND terms at or below target
                if current_brackets <= target_brackets and current_terms <= target_terms:
                    if verbose:
                        print(f"    Extra step {extra_step}: reached proper target (brackets={current_brackets}<={target_brackets}, terms={current_terms}<={target_terms}), breaking", flush=True)
                    reached_target = True
                    break
            else:
                # Default: stop when brackets reach target
                if current_brackets <= target_brackets:
                    if verbose:
                        print(f"    Extra step {extra_step}: reached target ({current_brackets}<={target_brackets}), breaking", flush=True)
                    reached_target = True
                    break

            try:
                obs = build_observation_from_expr(current_expr, n_point, max_terms, max_brackets_per_term)
            except:
                break

            action_mask = build_action_mask_v3(current_expr, action_space)
            if action_mask.sum() == 0:
                break

            # Get action ordering (model or random)
            if random_policy:
                valid_actions = np.where(action_mask == 1)[0]
                np.random.shuffle(valid_actions)
                sorted_actions = valid_actions
            else:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(obs_tensor)

                FLOAT_MIN = -3.4e38
                mask_tensor = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0).to(device)
                inf_mask = torch.clamp(torch.log(mask_tensor + 1e-10), min=FLOAT_MIN)
                masked_logits = logits + inf_mask
                sorted_actions = torch.argsort(masked_logits[0], descending=True).cpu().numpy()

            action_succeeded = False
            for retry_idx, action in enumerate(sorted_actions[:max_retries + 20]):
                if action_mask[action] == 0:
                    continue
                if action in blocked_actions_at_current:
                    continue

                spec = action_space.get_action_spec(action)
                bracket_type, ii, jj = action_space.get_bracket_from_idx(spec.bracket_idx)

                try:
                    result = apply_action_with_timeout(
                        current_expr, bracket_type, (ii, jj),
                        spec.identity, spec.params, spec.term_slot,
                        n_point, timeout_sec
                    )

                    if result is None:
                        continue

                    if use_anticycle:
                        try:
                            candidate_obs = build_observation_from_expr(result, n_point, max_terms, max_brackets_per_term)
                            candidate_hash = obs_to_hash(candidate_obs)
                            if candidate_hash in visited_states:
                                continue
                            visited_states.add(candidate_hash)
                        except:
                            pass

                    # Check for term explosion
                    if reject_term_increase is not None:
                        new_terms_check = count_terms(result)
                        term_increase = new_terms_check - current_terms
                        if term_increase > reject_term_increase:
                            continue

                    new_brackets = count_brackets(result)
                    new_terms = count_terms(result)
                    current_expr = result
                    action_succeeded = True

                    # Check source-relative condition
                    is_src_rel = (new_brackets < start_brackets and new_terms <= start_terms) or \
                                 (new_brackets <= start_brackets and new_terms < start_terms)
                    if is_src_rel:
                        achieved_source_rel = True
                        if new_brackets < best_src_rel_bk or (new_brackets == best_src_rel_bk and new_terms < best_src_rel_tm):
                            best_src_rel_bk = new_brackets
                            best_src_rel_tm = new_terms
                            best_src_rel_expr = result

                    if verbose:
                        ts_str = f"ts={spec.term_slot}" if spec.term_slot else ""
                        retry_str = f" (retry={retry_idx})" if retry_idx > 0 else ""
                        print(f"    Extra step {extra_step}: {current_brackets}->{new_brackets} terms={new_terms} "
                              f"via {spec.identity} {bracket_type}({ii},{jj}) {ts_str}{retry_str} [bt#{backtrack_count}]", flush=True)

                    if new_brackets < current_brackets:
                        # Save new checkpoint at the new state
                        checkpoints.append([result, new_brackets, new_terms,
                                           set(), visited_states.copy(), None])
                        blocked_actions_at_current = set()
                    else:
                        # Update the checkpoint we backtracked to with the action that left it
                        if checkpoints[best_idx][5] is None:
                            checkpoints[best_idx][5] = action

                    break
                except ActionTimeoutError:
                    continue
                except:
                    continue

            if not action_succeeded:
                break

        if reached_target:
            break  # Success! Exit the backtrack loop

    # Final metrics
    final_brackets = count_brackets(current_expr)
    final_terms = count_terms(current_expr)
    target_terms = count_terms(target_expr)
    if final_brackets < min_brackets:
        min_brackets = final_brackets

    # Also check source-rel on final state
    is_src_rel = (final_brackets < start_brackets and final_terms <= start_terms) or \
                 (final_brackets <= start_brackets and final_terms < start_terms)
    if is_src_rel:
        achieved_source_rel = True
        if final_brackets < best_src_rel_bk or (final_brackets == best_src_rel_bk and final_terms < best_src_rel_tm):
            best_src_rel_bk = final_brackets
            best_src_rel_tm = final_terms
            best_src_rel_expr = current_expr

    # Target-relative: both brackets AND terms at or below target in final state
    solved_proper = (final_brackets <= target_brackets) and (final_terms <= target_terms)
    # Source-relative: achieved at any point during evaluation (but not target-rel)
    solved_source_rel = achieved_source_rel and not solved_proper
    # Legacy field
    solved = solved_proper

    # For PARTIAL, report best source-rel state
    if solved_source_rel and best_src_rel_expr is not None:
        reported_bk = best_src_rel_bk
        reported_tm = best_src_rel_tm
        reported_expr = str(best_src_rel_expr)
    else:
        reported_bk = final_brackets
        reported_tm = final_terms
        reported_expr = str(current_expr)

    if verbose:
        if solved_proper:
            print(f"  [Sample {sample_idx}] SUCCESS: {start_brackets}->{final_brackets} bk, "
                  f"{start_terms}->{final_terms} tm (target: {target_brackets} bk, {target_terms} tm) "
                  f"in {len(trajectory)} steps (bt={backtrack_count})", flush=True)
        elif solved_source_rel:
            print(f"  [Sample {sample_idx}] PARTIAL (source-rel): {start_brackets}->{reported_bk} bk, "
                  f"{start_terms}->{reported_tm} tm (target: {target_brackets} bk, {target_terms} tm) "
                  f"in {len(trajectory)} steps (bt={backtrack_count})", flush=True)
        else:
            print(f"  [Sample {sample_idx}] FAILED: {start_brackets}->{final_brackets} bk, "
                  f"{start_terms}->{final_terms} tm (target: {target_brackets} bk, {target_terms} tm) "
                  f"in {len(trajectory)} steps (bt={backtrack_count})", flush=True)

    return {
        'solved': solved,
        'solved_proper': solved_proper,
        'solved_source_rel': solved_source_rel,
        'start_brackets': start_brackets,
        'target_brackets': target_brackets,
        'final_brackets': reported_bk,
        'min_brackets': min_brackets,
        'start_terms': start_terms,
        'target_terms': target_terms,
        'final_terms': reported_tm,
        'steps': len(trajectory),
        'reduction': start_brackets - min_brackets,
        'reduction_ratio': (start_brackets - min_brackets) / max(start_brackets, 1),
        'trajectory': trajectory,
        'final_expr': reported_expr,
        'timeout_count': timeout_count,
        'retry_count': retry_count,
        'backtrack_count': backtrack_count,
    }


def select_best_rollout(rollout_results: List[Dict]) -> Dict:
    """Select the best result from multiple rollouts.
    Priority: solved_proper > solved_source_rel > lower final_brackets > lower final_terms."""
    def score(r):
        return (
            r.get('solved_proper', False),
            r.get('solved_source_rel', False),
            -r['final_brackets'],
            -r.get('final_terms', float('inf')),
        )
    return max(rollout_results, key=score)


def evaluate_on_dataset(
    model,
    test_samples: List[Dict],
    action_space: TermSpecificActionSpace,
    n_point: int,
    max_terms: int,
    max_brackets_per_term: int,
    max_steps: int,
    device: str,
    timeout_sec: float = 30.0,
    sample_timeout: float = 120.0,
    verbose: bool = True,
    use_backtrack: bool = False,
    patience: int = 5,
    max_backtracks: int = 10,
    reject_term_increase: int = None,
    random_policy: bool = False,
    proper_stop: bool = False,
    checkpoint_by_terms: bool = False,
    n_rollouts: int = 1,
    temperature: float = 0.0,
    temperature_only: bool = False,
) -> Dict[str, Any]:
    """Evaluate model on test dataset with detailed metrics."""
    if model is not None:
        model.eval()

    results = []
    total_timeouts = 0
    total_retries = 0
    total_backtracks = 0

    init_worker(n_point)

    if n_rollouts > 1:
        desc = f"Evaluating (best-of-{n_rollouts}, T={temperature})"
    else:
        desc = "Evaluating"

    try:
        for sample_idx, sample in enumerate(tqdm(test_samples, desc=desc)):
            initial_expr = sample['scrambled_expression']
            target_expr = sample['simple_expression']

            if n_rollouts > 1:
                # Multiple rollouts: rollout 0 = greedy, rollouts 1..N-1 = temperature sampling
                rollout_results = []
                for rollout_idx in range(n_rollouts):
                    rollout_temp = temperature if temperature_only else (0.0 if rollout_idx == 0 else temperature)
                    rollout_verbose = verbose and (rollout_idx == 0)  # Only verbose for first rollout

                    if use_backtrack:
                        r = evaluate_sample_backtrack(
                            model, initial_expr, target_expr,
                            action_space, n_point, max_terms, max_brackets_per_term,
                            max_steps, device, timeout_sec,
                            sample_timeout=sample_timeout,
                            verbose=rollout_verbose,
                            sample_idx=sample_idx,
                            patience=patience,
                            max_backtracks=max_backtracks,
                            reject_term_increase=reject_term_increase,
                            random_policy=random_policy,
                            proper_stop=proper_stop,
                            checkpoint_by_terms=checkpoint_by_terms,
                        )
                    else:
                        r = evaluate_sample(
                            model, initial_expr, target_expr,
                            action_space, n_point, max_terms, max_brackets_per_term,
                            max_steps, device, timeout_sec,
                            sample_timeout=sample_timeout,
                            verbose=rollout_verbose,
                            sample_idx=sample_idx,
                            random_policy=random_policy,
                            proper_stop=proper_stop,
                            reject_term_increase=reject_term_increase,
                            temperature=rollout_temp,
                        )
                    rollout_results.append(r)

                    # Early stop: if this rollout achieved proper solution, no need for more
                    if r.get('solved_proper', False):
                        break

                # Select best rollout
                result = select_best_rollout(rollout_results)
                if verbose:
                    n_solved_rollouts = sum(r.get('solved_proper', r['solved']) for r in rollout_results)
                    print(f"  [Sample {sample_idx}] Best of {n_rollouts} rollouts: "
                          f"solved_proper={result.get('solved_proper', False)}, "
                          f"brackets={result['final_brackets']}, terms={result.get('final_terms', '?')}, "
                          f"({n_solved_rollouts}/{n_rollouts} rollouts solved)", flush=True)
                total_backtracks += sum(r.get('backtrack_count', 0) for r in rollout_results)
            else:
                # Single rollout (original behavior)
                if use_backtrack:
                    result = evaluate_sample_backtrack(
                        model, initial_expr, target_expr,
                        action_space, n_point, max_terms, max_brackets_per_term,
                        max_steps, device, timeout_sec,
                        sample_timeout=sample_timeout,
                        verbose=verbose,
                        sample_idx=sample_idx,
                        patience=patience,
                        max_backtracks=max_backtracks,
                        reject_term_increase=reject_term_increase,
                        random_policy=random_policy,
                        proper_stop=proper_stop,
                        checkpoint_by_terms=checkpoint_by_terms,
                    )
                    total_backtracks += result.get('backtrack_count', 0)
                else:
                    result = evaluate_sample(
                        model, initial_expr, target_expr,
                        action_space, n_point, max_terms, max_brackets_per_term,
                        max_steps, device, timeout_sec,
                        sample_timeout=sample_timeout,
                        verbose=verbose,
                        sample_idx=sample_idx,
                        random_policy=random_policy,
                        proper_stop=proper_stop,
                        reject_term_increase=reject_term_increase,
                        temperature=temperature if temperature_only else 0.0,
                    )

            results.append(result)
            total_timeouts += result['timeout_count']
            total_retries += result['retry_count']

    finally:
        shutdown_worker()

    # Aggregate metrics
    n_samples = len(results)
    n_solved = sum(r['solved'] for r in results)
    n_solved_proper = sum(r.get('solved_proper', r['solved']) for r in results)
    n_source_rel = sum(r.get('solved_source_rel', False) for r in results)
    n_reached_target = sum(r['min_brackets'] <= r['target_brackets'] for r in results)

    avg_reduction = np.mean([r['reduction'] for r in results])
    avg_reduction_ratio = np.mean([r['reduction_ratio'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])

    min_brackets_list = [r['min_brackets'] for r in results]
    target_brackets_list = [r['target_brackets'] for r in results]
    min_vs_target = [r['min_brackets'] - r['target_brackets'] for r in results]

    # Term metrics
    final_terms_list = [r.get('final_terms', 0) for r in results]
    target_terms_list = [r.get('target_terms', 0) for r in results]
    # Count term explosions (solved by brackets but terms > target)
    n_term_explosions = sum(
        r['solved'] and r.get('final_terms', 0) > r.get('target_terms', float('inf'))
        for r in results
    )

    # By starting complexity
    by_start_complexity = defaultdict(list)
    for r in results:
        by_start_complexity[r['start_brackets']].append(r)

    return {
        'n_samples': n_samples,
        'solve_rate': n_solved / n_samples if n_samples > 0 else 0,
        'solve_rate_proper': n_solved_proper / n_samples if n_samples > 0 else 0,
        'source_rel_rate': n_source_rel / n_samples if n_samples > 0 else 0,
        'reached_target_rate': n_reached_target / n_samples if n_samples > 0 else 0,
        'term_explosion_count': n_term_explosions,
        'term_explosion_rate': n_term_explosions / n_samples if n_samples > 0 else 0,
        'avg_min_brackets': np.mean(min_brackets_list),
        'avg_target_brackets': np.mean(target_brackets_list),
        'avg_min_vs_target': np.mean(min_vs_target),
        'avg_final_terms': np.mean(final_terms_list) if final_terms_list else 0,
        'avg_target_terms': np.mean(target_terms_list) if target_terms_list else 0,
        'avg_reduction': avg_reduction,
        'avg_reduction_ratio': avg_reduction_ratio,
        'avg_steps': avg_steps,
        'total_timeouts': total_timeouts,
        'total_retries': total_retries,
        'total_backtracks': total_backtracks,
        'by_complexity': {
            k: {
                'count': len(v),
                'solve_rate': sum(r['solved'] for r in v) / len(v),
                'solve_rate_proper': sum(r.get('solved_proper', r['solved']) for r in v) / len(v),
                'source_rel_rate': sum(r.get('solved_source_rel', False) for r in v) / len(v),
                'reached_target_rate': sum(r['min_brackets'] <= r['target_brackets'] for r in v) / len(v),
                'avg_min_brackets': np.mean([r['min_brackets'] for r in v]),
                'avg_target_brackets': np.mean([r['target_brackets'] for r in v]),
                'avg_reduction': np.mean([r['reduction'] for r in v]),
                'avg_final_terms': np.mean([r.get('final_terms', 0) for r in v]),
                'avg_target_terms': np.mean([r.get('target_terms', 0) for r in v]),
            }
            for k, v in sorted(by_start_complexity.items())
        },
        'detailed_results': results,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate SFT model with term-specific action space')
    parser.add_argument('--model', type=str, default=None, help='Path to model checkpoint (not required for --random_policy)')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data')
    parser.add_argument('--output', type=str, default=None, help='Output path for results')

    parser.add_argument('--n_point', type=int, default=4)
    parser.add_argument('--max_terms', type=int, default=20)
    parser.add_argument('--max_terms_action', type=int, default=10)
    parser.add_argument('--max_brackets_per_term', type=int, default=8)
    parser.add_argument('--max_brackets', type=int, default=12)
    parser.add_argument('--max_steps', type=int, default=50)
    parser.add_argument('--timeout', type=float, default=30.0)
    parser.add_argument('--sample_timeout', type=float, default=120.0,
                        help='Timeout per sample in seconds')

    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--ff_dim', type=int, default=128)
    parser.add_argument('--features_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--start_idx', type=int, default=None)
    parser.add_argument('--end_idx', type=int, default=None)
    parser.add_argument('--sample_indices', type=str, default=None,
                        help='Comma-separated list of sample indices to evaluate (overrides start/end)')
    parser.add_argument('--no_verbose', action='store_true', help='Disable detailed per-step output')

    # Backtracking options
    parser.add_argument('--backtrack', action='store_true', help='Enable backtracking mode')
    parser.add_argument('--patience', type=int, default=5, help='(deprecated) Steps without progress before backtracking')
    parser.add_argument('--max_backtracks', type=int, default=10, help='Maximum number of backtracks per sample')
    parser.add_argument('--reject_term_increase', type=int, default=None, help='If set, reject actions that increase terms by more than this amount')

    # Random policy baseline
    parser.add_argument('--random_policy', action='store_true',
                        help='Use random action selection instead of model (baseline)')

    # Proper stopping criterion
    parser.add_argument('--proper_stop', action='store_true',
                        help='Stop only when both brackets AND terms reach target (not just brackets)')

    # Checkpoint selection strategy
    parser.add_argument('--checkpoint_by_terms', action='store_true',
                        help='Select best checkpoint by lowest term count instead of lowest bracket count')

    # Temperature sampling and multiple rollouts
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Temperature for sampling (0=greedy, >0=stochastic). Used for rollouts 1..N-1.')
    parser.add_argument('--n_rollouts', type=int, default=1,
                        help='Number of rollouts per sample. Rollout 0 is always greedy, rest use temperature.')
    parser.add_argument('--temperature_only', action='store_true',
                        help='Use temperature for ALL rollouts including rollout 0 (skip greedy)')
    parser.add_argument('--save_detailed', action='store_true',
                        help='Save per-sample detailed results in the output pkl')

    args = parser.parse_args()

    # Validate: need either model or random_policy
    if not args.random_policy and args.model is None:
        parser.error("--model is required unless using --random_policy")

    print("=" * 70)
    if args.random_policy:
        print("Evaluating RANDOM POLICY BASELINE - Term-Specific Action Space")
    else:
        print("Evaluating SFT Model (v4) - Term-Specific Action Space")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create action space
    action_space = TermSpecificActionSpace(
        n_point=args.n_point,
        max_terms=args.max_terms_action,
        max_brackets=args.max_brackets
    )
    print(f"\nAction space: {action_space.identities_per_bracket} identities/bracket, "
          f"{action_space.n_term_slots} term_slots, {action_space.n_actions} total actions")

    if args.random_policy:
        # Random policy: no model needed
        model = None
        print("\nUsing RANDOM POLICY (no model)")
    else:
        # Compute dimensions
        bracket_feature_dim = 2 + 2 * args.n_point
        term_feature_dim = 2 + args.max_brackets_per_term * bracket_feature_dim + 1

        # Create model
        # n_term_slots = 1 (whole expr) + max_terms (individual terms)
        n_term_slots = 1 + args.max_terms_action
        model = TransformerPolicySFT(
            max_terms=args.max_terms,
            term_feature_dim=term_feature_dim,
            n_actions=action_space.n_actions,
            max_brackets=args.max_brackets,
            actions_per_bracket=n_term_slots * action_space.identities_per_bracket,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            ff_dim=args.ff_dim,
            dropout=args.dropout,
            features_dim=args.features_dim,
            global_feature_dim=2,
        ).to(device)

        # Load checkpoint
        print(f"\nLoading model from {args.model}")
        checkpoint = torch.load(args.model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load test data
    print(f"Loading test data from {args.test_data}")
    with open(args.test_data, 'rb') as f:
        test_samples = pickle.load(f)
    print(f"Loaded {len(test_samples)} test samples")

    # Select samples based on indices or range
    if args.sample_indices is not None:
        indices = [int(x.strip()) for x in args.sample_indices.split(',')]
        test_samples = [test_samples[i] for i in indices]
        print(f"Evaluating specific samples: {indices} ({len(test_samples)} samples)")
    elif args.start_idx is not None or args.end_idx is not None:
        start = args.start_idx or 0
        end = args.end_idx or len(test_samples)
        test_samples = test_samples[start:end]
        print(f"Evaluating samples {start} to {end} ({len(test_samples)} samples)")

    # Evaluate
    policy_str = "RANDOM POLICY" if args.random_policy else "MODEL"
    stop_str = " with PROPER_STOP (brackets+terms)" if args.proper_stop else ""
    rollout_str = f" with {args.n_rollouts} rollouts (T={args.temperature})" if args.n_rollouts > 1 else ""
    if args.backtrack:
        print(f"\nRunning {policy_str} evaluation with BACKTRACKING (max_backtracks={args.max_backtracks}){stop_str}{rollout_str}...")
    else:
        print(f"\nRunning {policy_str} evaluation{stop_str}{rollout_str}...")
    results = evaluate_on_dataset(
        model, test_samples, action_space,
        args.n_point, args.max_terms, args.max_brackets_per_term,
        args.max_steps, device, args.timeout,
        sample_timeout=args.sample_timeout,
        verbose=not args.no_verbose,
        use_backtrack=args.backtrack,
        patience=args.patience,
        max_backtracks=args.max_backtracks,
        reject_term_increase=args.reject_term_increase,
        random_policy=args.random_policy,
        proper_stop=args.proper_stop,
        checkpoint_by_terms=args.checkpoint_by_terms,
        n_rollouts=args.n_rollouts,
        temperature=args.temperature,
        temperature_only=args.temperature_only,
    )

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Samples: {results['n_samples']}")
    print(f"Target-relative (SUCCESS): {results.get('solve_rate_proper', results['solve_rate'])*100:.2f}%")
    print(f"Source-relative (PARTIAL): {results.get('source_rel_rate', 0)*100:.2f}%")
    print(f"FAILED: {(1 - results.get('solve_rate_proper', results['solve_rate']) - results.get('source_rel_rate', 0))*100:.2f}%")
    print(f"Term explosion count: {results.get('term_explosion_count', 0)} ({results.get('term_explosion_rate', 0)*100:.2f}%)")
    print(f"Average min brackets reached: {results['avg_min_brackets']:.2f}")
    print(f"Average target brackets: {results['avg_target_brackets']:.2f}")
    print(f"Average min - target: {results['avg_min_vs_target']:.2f}")
    print(f"Average final terms: {results.get('avg_final_terms', 0):.2f}")
    print(f"Average target terms: {results.get('avg_target_terms', 0):.2f}")
    print(f"Average reduction: {results['avg_reduction']:.2f} brackets")
    print(f"Average reduction ratio: {results['avg_reduction_ratio']*100:.2f}%")
    print(f"Average steps: {results['avg_steps']:.1f}")
    print(f"Total timeouts: {results['total_timeouts']}")
    print(f"Total retries: {results['total_retries']}")
    if args.backtrack:
        print(f"Total backtracks: {results['total_backtracks']}")

    print("\nBy starting complexity:")
    for k, v in results['by_complexity'].items():
        print(f"  {k} brackets: {v['count']} samples, "
              f"solve={v['solve_rate']*100:.1f}% (proper={v.get('solve_rate_proper', v['solve_rate'])*100:.1f}%), "
              f"avg_min={v['avg_min_brackets']:.1f}, "
              f"avg_terms={v.get('avg_final_terms', 0):.1f}/{v.get('avg_target_terms', 0):.1f}")

    # Save results
    if args.output:
        if args.save_detailed:
            save_results = results
        else:
            # Remove detailed_results before saving to reduce file size
            save_results = {k: v for k, v in results.items() if k != 'detailed_results'}
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, 'wb') as f:
            pickle.dump(save_results, f)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
