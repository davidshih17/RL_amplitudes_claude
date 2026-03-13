#!/usr/bin/env python
"""
Worker script for parallel generation of spinor-helicity oracle data.
Version 5: Uses term-specific action space (action_space_4pt).

This extends v4 to properly encode term-specific actions:
- Actions now encode (bracket_idx, term_slot, identity_type_idx)
- term_slot=0: Apply to ENTIRE expression (all occurrences)
- term_slot=1-10: Apply to ONE instance in term (term_slot-1)
- Total action space: 12 brackets * 11 term_slots * 11 identities = 1452 actions

Identity types per bracket (11 total):
- 1 Schouten (swapping pk/pl gives identical result)
- 6 Momentum (pout1 in bracket args, pout2 in {pout1} + other_momenta)
- 4 Momentum_sq (all subsets of other_momenta as mom_add)
"""

import argparse
import os
import sys
import pickle
import random
from itertools import combinations, permutations
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import sympy as sp

# Add paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, 'spinorhelicity'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

from environment.bracket_env import ab, sb
from environment.spin_helicity_env import SpinHelExpr
from environment.helicity_generator import generate_random_amplitude, generate_random_bk
from environment.utils import reorder_expr
from action_space_4pt import (
    TermSpecificActionSpace,
    get_term_bracket_powers,
    get_unique_brackets,
    apply_identity_to_term,
    apply_identity_to_whole_expression,
    IDENTITY_SCHOUTEN,
    IDENTITY_MOMENTUM,
    IDENTITY_MOMENTUM_SQ,
)


FUNC_DICT = {'ab': ab, 'sb': sb}


def expressions_equal(expr1: sp.Expr, expr2: sp.Expr) -> bool:
    """Check if two sympy expressions are equal."""
    if expr1 == expr2:
        return True
    try:
        diff = sp.cancel(expr1 - expr2)
        if diff == 0:
            return True
        diff = sp.expand(diff)
        if diff == 0:
            return True
    except:
        pass
    return False


def get_func_dict_from_expr(sp_expr: sp.Expr) -> Dict[str, type]:
    """Extract function dict from expression's bracket functions."""
    func_dict = {}
    for f in sp_expr.atoms(sp.Function):
        name = f.func.__name__
        if name in ['ab', 'sb'] and name not in func_dict:
            func_dict[name] = f.func
    if 'ab' not in func_dict:
        func_dict['ab'] = FUNC_DICT['ab']
    if 'sb' not in func_dict:
        func_dict['sb'] = FUNC_DICT['sb']
    return func_dict


def get_identity_result(bracket: sp.Function, identity: str, params: Dict[str, Any],
                        n_point: int, func_dict: Dict) -> sp.Expr:
    """Get the RHS of an identity applied to a bracket."""
    bk_type = bracket.func.__name__
    bk_args = list(bracket.args)

    if identity == IDENTITY_SCHOUTEN:
        pk, pl = params['pk'], params['pl']
        pi, pj = bk_args
        bk_func = func_dict[bk_type]
        ret_expr = (bk_func(pi, pl) * bk_func(pk, pj) / bk_func(pk, pl) +
                    bk_func(pi, pk) * bk_func(pj, pl) / bk_func(pk, pl))
        return reorder_expr(ret_expr)

    elif identity == IDENTITY_MOMENTUM:
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
        for k in range(1, n_point + 1):
            if k == in_arg or k == larg or k == rarg:
                continue
            ret_expr -= func_dict['ab'](larg, k) * func_dict['sb'](k, rarg) / denom_bk
        ret_expr = sign * ret_expr
        return reorder_expr(ret_expr)

    elif identity == IDENTITY_MOMENTUM_SQ:
        mom_add = params.get('mom_add', [])
        bk_type_opp = 'ab' if bk_type == 'sb' else 'sb'
        plist_lhs = list(mom_add) + bk_args
        plist_rhs = [i for i in range(1, n_point + 1) if i not in plist_lhs]
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


def apply_identity_full(sp_expr: sp.Expr, bracket: sp.Function, identity: str,
                        params: Dict[str, Any], n_point: int,
                        numerator_only: bool = True) -> sp.Expr:
    """Apply identity to all instances of bracket in expression."""
    func_dict = get_func_dict_from_expr(sp_expr)
    identity_result = get_identity_result(bracket, identity, params, n_point, func_dict)

    if numerator_only:
        num, denom = sp.fraction(sp_expr)
        num = num.subs(bracket, identity_result)
        return sp.cancel(num / denom)
    else:
        return sp.cancel(sp_expr.subs(bracket, identity_result))


def get_num_terms(sp_expr: sp.Expr, numerator_only: bool = True) -> int:
    """Get number of terms in expression."""
    if sp_expr == 0:
        return 0
    if numerator_only:
        num, _ = sp.fraction(sp_expr)
    else:
        num = sp_expr
    num_expanded = sp.expand(num)
    if isinstance(num_expanded, sp.Add):
        return len(num_expanded.args)
    else:
        return 1


def compute_unscrambling_action_v5(
    scramble_action: Dict[str, Any],
    state_after_scramble: sp.Expr,
    state_before_scramble: sp.Expr,
    n_point: int,
    action_space: TermSpecificActionSpace,
    rng: np.random.RandomState,
    numerator_only: bool = True,
) -> Optional[Dict[str, Any]]:
    """Compute the unscrambling action using term-specific action space."""
    identity = scramble_action['identity']
    orig_bracket_args = scramble_action['bracket_args']
    params = scramble_action['params']
    bk_type = scramble_action['bk_type']

    i, j = orig_bracket_args

    brackets = get_unique_brackets(state_after_scramble, numerator_only)
    if len(brackets) == 0:
        return None

    term_bracket_powers = get_term_bracket_powers(state_after_scramble, numerator_only)
    num_terms = len(term_bracket_powers)

    bracket_lookup = {}
    for bk in brackets:
        key = (bk.func.__name__, tuple(sorted(bk.args)))
        bracket_lookup[key] = bk

    func_dict = get_func_dict_from_expr(state_after_scramble)
    candidate_specs = []

    if identity == IDENTITY_SCHOUTEN:
        k, l = params['pk'], params['pl']
        scramble_type = scramble_action.get('scramble_type', '')

        if 'identity_mul' in scramble_type or 'zero_add' in scramble_type:
            all_momenta = set([i, j, k, l])
            for bk in brackets:
                if bk.func.__name__ != bk_type:
                    continue
                bk_args = set(bk.args)
                if bk_args.issubset(all_momenta) and len(bk_args) == 2:
                    remaining = sorted(all_momenta - bk_args)
                    if len(remaining) == 2:
                        new_pk, new_pl = remaining[0], remaining[1]
                        candidate_specs.append({
                            'bracket': bk,
                            'identity': IDENTITY_SCHOUTEN,
                            'params': {'pk': new_pk, 'pl': new_pl},
                        })
        else:
            cross_brackets = [(i, l), (i, k), (j, l), (j, k)]
            for cb_args in cross_brackets:
                key = (bk_type, tuple(sorted(cb_args)))
                if key not in bracket_lookup:
                    continue
                cb = bracket_lookup[key]
                remaining = sorted([x for x in [i, j, k, l] if x not in cb_args])
                if len(remaining) != 2:
                    continue
                new_pk, new_pl = remaining[0], remaining[1]
                candidate_specs.append({
                    'bracket': cb,
                    'identity': IDENTITY_SCHOUTEN,
                    'params': {'pk': new_pk, 'pl': new_pl},
                })

    elif identity == IDENTITY_MOMENTUM:
        pout1, pout2 = params['pout1'], params['pout2']
        scramble_type = scramble_action.get('scramble_type', '')

        if 'identity_mul' in scramble_type or 'zero_add' in scramble_type:
            for m in range(1, n_point + 1):
                if m != pout1:
                    search_args = tuple(sorted([pout1, m]))
                    for search_bk_type in ['ab', 'sb']:
                        key = (search_bk_type, search_args)
                        if key in bracket_lookup:
                            candidate_specs.append({
                                'bracket': bracket_lookup[key],
                                'identity': IDENTITY_MOMENTUM,
                                'params': {'pout1': pout1, 'pout2': pout2},
                            })
                if m != pout2:
                    search_args = tuple(sorted([m, pout2]))
                    for search_bk_type in ['ab', 'sb']:
                        key = (search_bk_type, search_args)
                        if key in bracket_lookup:
                            candidate_specs.append({
                                'bracket': bracket_lookup[key],
                                'identity': IDENTITY_MOMENTUM,
                                'params': {'pout1': pout2, 'pout2': pout1},
                            })
        else:
            in_arg = j if pout1 == i else i
            other_indices = [x for x in range(1, n_point + 1) if x not in [in_arg, pout1, pout2]]
            for other in other_indices:
                for search_bk_type in [bk_type, 'ab' if bk_type == 'sb' else 'sb']:
                    new_bracket_args = tuple(sorted([pout1, other]))
                    key = (search_bk_type, new_bracket_args)
                    if key in bracket_lookup:
                        if search_bk_type == bk_type:
                            candidate_specs.append({
                                'bracket': bracket_lookup[key],
                                'identity': IDENTITY_MOMENTUM,
                                'params': {'pout1': pout1, 'pout2': pout2},
                            })
                        else:
                            candidate_specs.append({
                                'bracket': bracket_lookup[key],
                                'identity': IDENTITY_MOMENTUM,
                                'params': {'pout1': pout2, 'pout2': pout1},
                            })

    elif identity == IDENTITY_MOMENTUM_SQ:
        mom_add = list(params['mom_add'])
        plist_lhs = mom_add + [i, j]
        plist_rhs = [k for k in range(1, n_point + 1) if k not in plist_lhs]
        scramble_type = scramble_action.get('scramble_type', '')

        if 'identity_mul' in scramble_type or 'zero_add' in scramble_type:
            all_momenta = set(range(1, n_point + 1))
            for bk in brackets:
                bk_args = set(bk.args)
                needed_mom_add = sorted(all_momenta - bk_args - set([i, j]))
                for try_mom_add in [needed_mom_add, []]:
                    candidate_specs.append({
                        'bracket': bk,
                        'identity': IDENTITY_MOMENTUM_SQ,
                        'params': {'mom_add': try_mom_add},
                    })
        else:
            for r1 in plist_rhs:
                for r2 in plist_rhs:
                    if r2 > r1:
                        new_bracket_args = (r1, r2)
                        new_mom_add = sorted([x for x in plist_rhs if x not in [r1, r2]])
                        for search_bk_type in [bk_type, 'ab' if bk_type == 'sb' else 'sb']:
                            key = (search_bk_type, new_bracket_args)
                            if key in bracket_lookup:
                                candidate_specs.append({
                                    'bracket': bracket_lookup[key],
                                    'identity': IDENTITY_MOMENTUM_SQ,
                                    'params': {'mom_add': new_mom_add},
                                })

            for l1 in plist_lhs:
                for l2 in plist_lhs:
                    if l2 > l1 and not (l1 in [i, j] and l2 in [i, j]):
                        new_bracket_args = (l1, l2)
                        new_mom_add = sorted([x for x in plist_lhs if x not in [l1, l2]])
                        for search_bk_type in [bk_type, 'ab' if bk_type == 'sb' else 'sb']:
                            key = (search_bk_type, new_bracket_args)
                            if key in bracket_lookup:
                                candidate_specs.append({
                                    'bracket': bracket_lookup[key],
                                    'identity': IDENTITY_MOMENTUM_SQ,
                                    'params': {'mom_add': new_mom_add},
                                })

    verified_actions = []

    for spec in candidate_specs:
        bracket = spec['bracket']
        id_type = spec['identity']
        id_params = spec['params']

        try:
            bracket_idx = action_space.get_bracket_idx(bracket)
        except ValueError:
            continue

        try:
            identity_result = get_identity_result(bracket, id_type, id_params, n_point, func_dict)

            result_full = apply_identity_full(state_after_scramble, bracket, id_type,
                                              id_params, n_point, numerator_only)

            # term_slot=0: whole expression
            if expressions_equal(result_full, state_before_scramble):
                verified_actions.append({
                    'bracket': bracket,
                    'bracket_idx': bracket_idx,
                    'term_slot': 0,  # 0 = whole expression
                    'identity': id_type,
                    'params': id_params,
                    'is_whole_expression': True,
                })
                continue

            # term_slot=1-10: specific term (term_slot-1)
            for term_idx in range(min(num_terms, action_space.max_terms)):
                if bracket not in term_bracket_powers[term_idx]:
                    continue

                result_term = apply_identity_to_term(
                    state_after_scramble, term_idx, bracket, identity_result,
                    power_to_apply=1, numerator_only=numerator_only
                )

                if expressions_equal(result_term, state_before_scramble):
                    verified_actions.append({
                        'bracket': bracket,
                        'bracket_idx': bracket_idx,
                        'term_slot': term_idx + 1,  # term_slot = term_idx + 1
                        'identity': id_type,
                        'params': id_params,
                        'is_whole_expression': False,
                    })
        except Exception:
            continue

    if len(verified_actions) == 0:
        return None

    all_equivalent_action_indices = []
    for action in verified_actions:
        try:
            action_idx = action_space.find_action_index(
                bracket=action['bracket'],
                term_slot=action['term_slot'],
                identity=action['identity'],
                params=action['params'],
            )
            all_equivalent_action_indices.append(action_idx)
        except ValueError:
            continue

    if len(all_equivalent_action_indices) == 0:
        return None

    chosen_idx = rng.choice(len(verified_actions))
    chosen_action = verified_actions[chosen_idx]
    chosen_action_idx = all_equivalent_action_indices[chosen_idx]

    equivalent_actions = []
    for action in verified_actions:
        action_spec = {
            'bracket': str(action['bracket']),
            'bracket_idx': action['bracket_idx'],
            'term_slot': action['term_slot'],
            'identity': action['identity'],
            'params': action['params'],
            'is_whole_expression': action['is_whole_expression'],
        }
        equivalent_actions.append(action_spec)

    result = {
        'action_idx': chosen_action_idx,
        'equivalent_action_indices': all_equivalent_action_indices,
        'equivalent_actions': equivalent_actions,
        'bracket_idx': chosen_action['bracket_idx'],
        'term_slot': chosen_action['term_slot'],
        'bracket': str(chosen_action['bracket']),
        'identity': chosen_action['identity'],
        'params': chosen_action['params'],
        'is_whole_expression': chosen_action['is_whole_expression'],
        'num_equivalent_actions': len(verified_actions),
        'num_candidate_actions': len(candidate_specs),
    }

    return result


def generate_sample_with_oracle_v5(
    n_point: int,
    max_scrambles: int,
    min_scrambles: int,
    max_scale: int,
    max_terms: int,
    l_scale: float,
    rng: np.random.RandomState,
    action_space: TermSpecificActionSpace,
    numerator_only: bool = True,
    use_full_scrambles: bool = True,
    all_momenta: bool = False,
) -> Optional[Dict[str, Any]]:
    """Generate a sample with oracle trajectory using term-specific action space."""
    try:
        simple_expr, n_pt_gen, _ = generate_random_amplitude(
            [n_point], rng,
            max_terms_scale=max_scale,
            max_components=max_terms,
            l_scale=l_scale,
            session='NotRequired',
            all_momenta=all_momenta,
        )
        if simple_expr is None:
            return None

        expr_env = SpinHelExpr(str(simple_expr), n_pt_gen)
        states = [expr_env.sp_expr]
        scramble_actions = []

        num_scrambles = rng.randint(min_scrambles, max_scrambles + 1)

        for i in range(num_scrambles):
            curr_expr = expr_env.sp_expr
            if curr_expr == 0:
                break
            if numerator_only and isinstance(sp.fraction(curr_expr)[0], sp.Integer):
                break

            if use_full_scrambles:
                act_num = rng.randint(1, 6)
            else:
                act_num = rng.randint(1, 4)

            if numerator_only:
                function_probed, _ = sp.fraction(curr_expr)
            else:
                function_probed = curr_expr

            fct_list = [f for f in function_probed.atoms(sp.Function)]
            success = False
            scramble_action = None

            if act_num <= 3:
                if len(fct_list) == 0:
                    continue
                rdm_bracket = rng.choice(fct_list)
                bk_type = rdm_bracket.func.__name__
                args_bk = list(rdm_bracket.args)

            if act_num == 1:  # Schouten
                valid_k = [j for j in range(1, n_pt_gen + 1) if j not in args_bk]
                if len(valid_k) >= 2:
                    arg3 = rng.choice(valid_k)
                    valid_l = [j for j in valid_k if j != arg3]
                    arg4 = rng.choice(valid_l)
                    expr_env.schouten_single(rdm_bracket, arg3, arg4, numerator_only=numerator_only)
                    pk_stored, pl_stored = (arg3, arg4) if arg3 < arg4 else (arg4, arg3)
                    scramble_action = {
                        'identity': IDENTITY_SCHOUTEN,
                        'scramble_type': 'schouten',
                        'bracket_args': tuple(args_bk),
                        'bk_type': bk_type,
                        'params': {'pk': pk_stored, 'pl': pl_stored},
                    }
                    success = True

            elif act_num == 2:  # Momentum
                out_arg = rng.choice(args_bk)
                in_arg = [arg for arg in args_bk if arg != out_arg][0]
                valid_out2 = [j for j in range(1, n_pt_gen + 1) if j != in_arg]
                if len(valid_out2) > 0:
                    out2_arg = rng.choice(valid_out2)
                    expr_env.momentum_single(rdm_bracket, out_arg, out2_arg, numerator_only=numerator_only)
                    scramble_action = {
                        'identity': IDENTITY_MOMENTUM,
                        'scramble_type': 'momentum',
                        'bracket_args': tuple(args_bk),
                        'bk_type': bk_type,
                        'params': {'pout1': out_arg, 'pout2': out2_arg},
                    }
                    success = True

            elif act_num == 3:  # Momentum squared
                valid_add = [j for j in range(1, n_pt_gen + 1) if j not in args_bk]
                arg_add = rng.randint(0, min(len(valid_add) + 1, n_pt_gen - 2))
                if arg_add > 0:
                    mom_add = list(rng.choice(valid_add, arg_add, replace=False))
                else:
                    mom_add = []
                expr_env.momentum_sq(rdm_bracket, mom_add, numerator_only=numerator_only)
                scramble_action = {
                    'identity': IDENTITY_MOMENTUM_SQ,
                    'scramble_type': 'momentum_sq',
                    'bracket_args': tuple(args_bk),
                    'bk_type': bk_type,
                    'params': {'mom_add': mom_add},
                }
                success = True

            elif act_num == 4:  # Identity multiplication
                bk_type_new = ab if rng.randint(0, 2) == 0 else sb
                new_bk = generate_random_bk(bk_type_new, n_pt_gen, rng)
                order_bk = rng.randint(0, 2)

                info_add = expr_env.identity_mul(rng, new_bk, order_bk, numerator_only=numerator_only)

                if info_add and len(info_add) > 0:
                    inner_action = info_add[0]
                    bk_args = tuple(new_bk.args)
                    bk_type_str = new_bk.func.__name__

                    if inner_action[0] == 'S':
                        ii, jj = bk_args
                        pk = int(inner_action[2])
                        pl = int(inner_action[3])
                        cross_bracket_args = tuple(sorted([ii, pl]))
                        scramble_action = {
                            'identity': IDENTITY_SCHOUTEN,
                            'scramble_type': 'identity_mul_schouten',
                            'bracket_args': cross_bracket_args,
                            'bk_type': bk_type_str,
                            'params': {'pk': pk, 'pl': jj},
                        }
                        success = True

                    elif inner_action[0] in ['M+', 'M-']:
                        out2_arg = int(inner_action[2])
                        if inner_action[0] == 'M+':
                            pout1 = bk_args[0] if bk_type_str == 'ab' else bk_args[1]
                        else:
                            pout1 = bk_args[1] if bk_type_str == 'ab' else bk_args[0]
                        pout2 = out2_arg
                        in_arg = bk_args[0] if pout1 == bk_args[1] else bk_args[1]

                        valid_m = [m for m in range(1, n_pt_gen + 1) if m not in [in_arg, pout1, pout2]]
                        if valid_m:
                            new_bracket_args = tuple(sorted([pout1, valid_m[0]]))
                            scramble_action = {
                                'identity': IDENTITY_MOMENTUM,
                                'scramble_type': 'identity_mul_momentum',
                                'bracket_args': new_bracket_args,
                                'bk_type': bk_type_str,
                                'params': {'pout1': pout1, 'pout2': pout2},
                            }
                            success = True

                    elif inner_action[0] == 'M':
                        mom_add = [int(x) for x in inner_action[2:]] if len(inner_action) > 2 else []
                        plist_lhs = list(mom_add) + list(bk_args)
                        plist_rhs = [k for k in range(1, n_pt_gen + 1) if k not in plist_lhs]

                        if len(plist_rhs) >= 2:
                            new_bracket_args = tuple(sorted(plist_rhs[:2]))
                            scramble_action = {
                                'identity': IDENTITY_MOMENTUM_SQ,
                                'scramble_type': 'identity_mul_momentum_sq',
                                'bracket_args': new_bracket_args,
                                'bk_type': bk_type_str,
                                'params': {'mom_add': []},
                            }
                            success = True

            elif act_num == 5:  # Addition of zero
                bk_type_new = ab if rng.randint(0, 2) == 0 else sb
                base_bk = generate_random_bk(bk_type_new, n_pt_gen, rng)
                sign_bk = rng.randint(0, 2)

                success_zero, info_add = expr_env.zero_add(rng, base_bk, int(2*(sign_bk - 0.5)),
                                                           'NotRequired', numerator_only=numerator_only)

                if success_zero and info_add and len(info_add) > 0:
                    inner_action = info_add[0]
                    bk_args = tuple(base_bk.args)
                    bk_type_str = base_bk.func.__name__

                    if inner_action[0] == 'S':
                        ii, jj = bk_args
                        pk = int(inner_action[2])
                        pl = int(inner_action[3])
                        cross_bracket_args = tuple(sorted([ii, pl]))
                        scramble_action = {
                            'identity': IDENTITY_SCHOUTEN,
                            'scramble_type': 'zero_add_schouten',
                            'bracket_args': cross_bracket_args,
                            'bk_type': bk_type_str,
                            'params': {'pk': pk, 'pl': jj},
                        }
                        success = True

                    elif inner_action[0] in ['M+', 'M-']:
                        out2_arg = int(inner_action[2])
                        if inner_action[0] == 'M+':
                            pout1 = bk_args[0] if bk_type_str == 'ab' else bk_args[1]
                        else:
                            pout1 = bk_args[1] if bk_type_str == 'ab' else bk_args[0]
                        pout2 = out2_arg
                        in_arg = bk_args[0] if pout1 == bk_args[1] else bk_args[1]

                        valid_m = [m for m in range(1, n_pt_gen + 1) if m not in [in_arg, pout1, pout2]]
                        if valid_m:
                            new_bracket_args = tuple(sorted([pout1, valid_m[0]]))
                            scramble_action = {
                                'identity': IDENTITY_MOMENTUM,
                                'scramble_type': 'zero_add_momentum',
                                'bracket_args': new_bracket_args,
                                'bk_type': bk_type_str,
                                'params': {'pout1': pout1, 'pout2': pout2},
                            }
                            success = True

                    elif inner_action[0] == 'M':
                        mom_add = [int(x) for x in inner_action[2:]] if len(inner_action) > 2 else []
                        plist_lhs = list(mom_add) + list(bk_args)
                        plist_rhs = [k for k in range(1, n_pt_gen + 1) if k not in plist_lhs]

                        if len(plist_rhs) >= 2:
                            new_bracket_args = tuple(sorted(plist_rhs[:2]))
                            scramble_action = {
                                'identity': IDENTITY_MOMENTUM_SQ,
                                'scramble_type': 'zero_add_momentum_sq',
                                'bracket_args': new_bracket_args,
                                'bk_type': bk_type_str,
                                'params': {'mom_add': []},
                            }
                            success = True

            if success and scramble_action is not None:
                if numerator_only:
                    expr_env.cancel()
                states.append(expr_env.sp_expr)
                scramble_actions.append(scramble_action)

        if len(states) < 2:
            return None

        simple_complexity = str(simple_expr).count('ab(') + str(simple_expr).count('sb(')
        final_complexity = str(states[-1]).count('ab(') + str(states[-1]).count('sb(')
        if final_complexity <= simple_complexity:
            return None

        for state in states:
            if get_num_terms(state, numerator_only) > action_space.max_terms:
                return None

        trajectory = []
        reversed_states = list(reversed(states))
        reversed_scramble_actions = list(reversed(scramble_actions))

        for i in range(len(reversed_states) - 1):
            state_from = reversed_states[i]
            state_to = reversed_states[i + 1]
            scramble_action = reversed_scramble_actions[i]

            action = compute_unscrambling_action_v5(
                scramble_action, state_from, state_to, n_pt_gen, action_space, rng, numerator_only
            )

            if action is None:
                return None

            trajectory.append({
                'state': state_from,
                'target': state_to,
                'action': action,
                'scramble_action': scramble_action,
            })

        return {
            'simple_expression': simple_expr,
            'scrambled_expression': states[-1],
            'n_point': n_pt_gen,
            'num_scrambles': len(states) - 1,
            'trajectory': trajectory,
        }
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description='Worker for oracle data generation with term-specific actions (v5)')
    parser.add_argument('--worker_id', type=int, required=True)
    parser.add_argument('--num_workers', type=int, default=100)
    parser.add_argument('--total_samples', type=int, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--n_point', type=int, default=4)
    parser.add_argument('--min_scrambles', type=int, default=1)
    parser.add_argument('--max_scrambles', type=int, default=3)
    parser.add_argument('--max_scale', type=int, default=2, help='Max scale for amplitude generation (paper: 2)')
    parser.add_argument('--max_terms_gen', type=int, default=3, help='Max terms for generation (paper: 3)')
    parser.add_argument('--max_terms', type=int, default=10, help='Max terms for action space')
    parser.add_argument('--max_brackets', type=int, default=12, help='Max brackets (12 for 4-point)')
    parser.add_argument('--l_scale', type=float, default=0.75, help='Scale factor for generation (paper: 0.75)')
    parser.add_argument('--all_momenta', action='store_true', default=False,
                        help='Use all momenta in generation (paper: False)')
    parser.add_argument('--seed_offset', type=int, default=0)
    parser.add_argument('--use_full_scrambles', action='store_true', default=True,
                        help='Include identity_mul and zero_add scrambles')
    args = parser.parse_args()

    seed = args.worker_id + args.seed_offset
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    action_space = TermSpecificActionSpace(
        n_point=args.n_point,
        max_terms=args.max_terms,
        max_brackets=args.max_brackets
    )

    samples_per_worker = args.total_samples // args.num_workers
    extra = args.total_samples % args.num_workers
    my_samples = samples_per_worker + (1 if args.worker_id < extra else 0)

    print(f"Worker {args.worker_id}/{args.num_workers}: generating {my_samples} samples "
          f"(n_point={args.n_point}, scrambles={args.min_scrambles}-{args.max_scrambles}, "
          f"action_space={action_space.n_actions}, seed={seed})", flush=True)

    samples = []
    attempts = 0
    max_attempts = my_samples * 20

    while len(samples) < my_samples and attempts < max_attempts:
        attempts += 1
        sample = generate_sample_with_oracle_v5(
            n_point=args.n_point,
            max_scrambles=args.max_scrambles,
            min_scrambles=args.min_scrambles,
            max_scale=args.max_scale,
            max_terms=args.max_terms_gen,
            l_scale=args.l_scale,
            rng=rng,
            action_space=action_space,
            numerator_only=True,
            use_full_scrambles=args.use_full_scrambles,
            all_momenta=args.all_momenta,
        )

        if sample is not None:
            samples.append(sample)
            if len(samples) % 100 == 0:
                print(f"  Worker {args.worker_id}: {len(samples)}/{my_samples} samples generated "
                      f"({attempts} attempts)", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f'oracle_v5_{args.worker_id:04d}.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(samples, f)

    print(f"Worker {args.worker_id}: saved {len(samples)} samples to {output_path} "
          f"({attempts} attempts, {100*len(samples)/attempts:.1f}% success rate)", flush=True)


if __name__ == '__main__':
    main()
