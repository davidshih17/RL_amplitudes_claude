"""
Iterative simplification of large spinor-helicity expressions using
CDS contrastive grouping + RL model with backtracking.

Strategy:
1. Parse expression -> extract numerator terms + denominator
2. Use CDS contrastive encoder to group similar terms (likely to cancel)
3. For each group, run RL model with backtracking to simplify
4. Iterate with decaying similarity cutoff until convergence

Uses same evaluation settings as paper evals:
  max_steps=100, timeout=30s, sample_timeout=120s,
  max_backtracks=10, proper_stop=True, reject_term_increase=1

Usage:
    python eval_large_expr.py \
        --input shared/data/EYM_PPPPH_for_ML.txt \
        --n_point 5 \
        --model models/sft_5pt_500k/best_model.pt \
        --verbose
"""

import argparse
import os
import sys
import time
import random
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import sympy as sp
from sympy import Function


# ============================================================
# Path setup
# ============================================================

BASE_DIR = Path(__file__).parent.parent
SHARED_DIR = BASE_DIR.parent / 'shared'

sys.path.insert(0, str(SHARED_DIR / 'spinorhelicity'))
sys.path.insert(0, str(SHARED_DIR / 'spinorhelicity' / 'environment'))
sys.path.insert(0, str(BASE_DIR / 'src'))

import environment.bracket_env
# CRITICAL: Force 'bracket_env' module alias to be the SAME as 'environment.bracket_env'.
# Without this, Python treats them as two separate modules (because both
# shared/spinorhelicity and shared/spinorhelicity/environment are on sys.path),
# creating two different ab/sb classes. This breaks isinstance() checks in the CDS
# contrastive encoder's sympy_to_prefix (SYMPY_OPERATORS uses environment.bracket_env.ab,
# but pickled data and sympify may use bracket_env.ab).
sys.modules['bracket_env'] = sys.modules['environment.bracket_env']

from environment.bracket_env import ab, sb
from spinhelicity_env import build_observation_from_expr
from train_sft_transformer import TransformerPolicySFT

# Import evaluation functions DIRECTLY from eval_sft_5pt.py
# to guarantee EXACTLY the same behavior as paper evals
from eval_sft_5pt import (
    evaluate_sample_backtrack,
    build_action_mask_5pt as build_action_mask,
    count_brackets,
    count_terms,
    obs_to_hash,
    init_worker,
    shutdown_worker,
    apply_action_with_timeout,
    ActionTimeoutError,
)

FUNC_DICT = {'ab': ab, 'sb': sb}

from add_ons.numerical_evaluations import MOMENTA_DICTS
from eval_grouping_worker import factor_out_common_brackets
import json


def numerical_eval_5pt(expr, kinematics_idx=0):
    """Evaluate a 5pt spinor-helicity expression at pre-computed kinematics."""
    md = MOMENTA_DICTS[kinematics_idx]
    subs = {}
    for i in range(1, 6):
        for j in range(i + 1, 6):
            subs[ab(i, j)] = md[f'5ptab{i}{j}']
            subs[sb(i, j)] = md[f'5ptsb{i}{j}']
    return complex(expr.subs(subs).evalf())


def numerical_validate(old_expr, new_expr, label=""):
    """Check two expressions are numerically equal at 2 kinematic points.
    Returns (is_valid, max_rel_diff).
    """
    max_rdiff = 0.0
    for ki in range(2):
        v_old = numerical_eval_5pt(old_expr, ki)
        v_new = numerical_eval_5pt(new_expr, ki)
        if abs(v_old) < 1e-15:
            rdiff = abs(v_new - v_old)
        else:
            rdiff = abs((v_new - v_old) / v_old)
        max_rdiff = max(max_rdiff, rdiff)
    is_valid = max_rdiff < 1e-9
    status = "PASS" if is_valid else "FAIL"
    if label:
        print(f"  Numerical validation [{label}]: {status} (max rel diff = {max_rdiff:.2e})",
              flush=True)
    return is_valid, max_rdiff


# ============================================================
# REMOVED: All duplicated functions (ActionTimeoutError, worker,
# count_brackets, count_terms, obs_to_hash, build_action_mask,
# evaluate_sample_backtrack) are now imported from eval_sft_5pt.py
# ============================================================

# ============================================================
# Common denominator extraction (from CDS simplifier_methods.py)
# ============================================================

def extract_num_denom(input_eq):
    """Extract numerator terms and denominator from a sympy expression.
    From CDS simplifier_methods.py - puts expression over common denominator.
    """
    f = input_eq.cancel()
    numerator, denominator = sp.fraction(f)
    if isinstance(numerator, sp.Add):
        terms = np.asarray(numerator.args, dtype=object)
    else:
        terms = np.asarray([numerator], dtype=object)
    return terms, denominator


def count_numerator_terms(expression):
    """Count number of terms in the numerator of an expression.
    From CDS simplifier_methods.py.
    """
    numerator, denominator = sp.fraction(expression)
    if isinstance(numerator, sp.Add):
        return len(numerator.args)
    else:
        return 1


# ============================================================
# String-level expression splitting (kept for parsing/output)
# ============================================================

def split_expression_string(expr_str: str) -> List[str]:
    """Split an expression string into additive terms at top-level +/- signs."""
    expr_str = expr_str.strip()
    if not expr_str:
        return []

    terms = []
    current = []
    depth = 0
    i = 0

    while i < len(expr_str):
        c = expr_str[i]

        if c == '(':
            depth += 1
            current.append(c)
        elif c == ')':
            depth -= 1
            current.append(c)
        elif c in ('+', '-') and depth == 0 and i > 0:
            term = ''.join(current).strip()
            if term:
                terms.append(term)
            if c == '-':
                current = ['-']
            else:
                current = []
        else:
            current.append(c)

        i += 1

    last = ''.join(current).strip()
    if last:
        terms.append(last)

    return terms


def join_term_strings(terms: List[str]) -> str:
    """Join term strings back into a single expression string."""
    if not terms:
        return "0"

    parts = []
    for i, t in enumerate(terms):
        t = t.strip()
        if i == 0:
            parts.append(t)
        elif t.startswith('-'):
            parts.append(' - ' + t[1:].strip())
        else:
            parts.append(' + ' + t)

    return ''.join(parts)


# ============================================================
# CDS Contrastive encoder loading
# ============================================================

def load_contrastive_encoder(n_point, device='cpu'):
    """Load the CDS contrastive encoder for term similarity computation.

    Returns:
        (encoder_c, env_c): contrastive encoder model and its environment
    """
    from environment.utils import AttrDict
    from environment import build_env
    from model.contrastive_learner import build_modules_contrastive
    import environment

    # Contrastive model paths
    embedding_paths = {
        5: str(SHARED_DIR / 'spinorhelicity' / 'n5_embedding.pth'),
    }

    if n_point not in embedding_paths:
        raise ValueError(f"No contrastive encoder available for n_point={n_point}. "
                         f"Available: {list(embedding_paths.keys())}")

    embedding_path = embedding_paths[n_point]
    print(f"Loading contrastive encoder from {embedding_path}", flush=True)

    # Set CUDA flag for the CDS environment
    use_cpu = (device == 'cpu')
    environment.utils.CUDA = not use_cpu

    # Parameters matching the CDS streamlit_app.py setup
    params_contrastive = AttrDict({
        'tasks': 'contrastive',
        'env_name': 'char_env',
        'npt_list': [n_point],
        'max_scale': 2,
        'max_terms': 3,
        'max_scrambles': 3,
        'min_scrambles': 1,
        'save_info_scr': False,
        'save_info_scaling': False,
        'numeral_decomp': True,
        'int_base': 10,
        'max_len': 2048,
        'l_scale': 0.75,
        'numerator_only': True,
        'reduced_voc': True,
        'all_momenta': False,
        'emb_dim': 512,
        'n_enc_layers': 2,
        'n_dec_layers': 2,
        'head_layers': 2,
        'n_heads': 8,
        'dropout': 0,
        'n_max_positions': 256,
        'attention_dropout': 0,
        'sinusoidal_embeddings': False,
        'share_inout_emb': True,
        'positional_encoding': True,
        'norm_ffn': 'None',
        'reload_model': embedding_path,
        'cpu': use_cpu,
        'lib_path': None,
        'mma_path': None,
        'numerical_check': 0,
    })

    # Build contrastive environment
    env_c = build_env(params_contrastive)

    # Build and load contrastive encoder
    modules_c = build_modules_contrastive(env_c, params_contrastive)
    encoder_c = modules_c['encoder_c']
    encoder_c.eval()

    n_params = sum(p.numel() for p in encoder_c.parameters())
    print(f"Contrastive encoder loaded: {n_params:,} parameters", flush=True)

    return encoder_c, env_c


# ============================================================
# CDS Contrastive term grouping
# ============================================================

def encode_term(env_c, term, encoder_c):
    """Encode a sympy term using the contrastive encoder.
    From contrastive_simplifier.py.
    """
    from environment.utils import to_cuda

    t_prefix = env_c.sympy_to_prefix(term)
    t_in = to_cuda(torch.LongTensor([env_c.eos_index] + [env_c.word2id[w] for w in t_prefix] +
                                    [env_c.eos_index]).view(-1, 1))[0]
    len_in = to_cuda(torch.LongTensor([len(t_in)]))[0]

    with torch.no_grad():
        encoded = encoder_c('fwd', x=t_in, lengths=len_in, causal=False)

    return encoded


def normalize_term(term_in):
    """Normalize a numerator term to have constant +/-1.
    From contrastive_simplifier.py.
    """
    if isinstance(term_in, sp.Integer):
        return term_in / abs(term_in)

    if isinstance(term_in, sp.Function) or isinstance(term_in, sp.Pow):
        return term_in

    if not isinstance(term_in, sp.Mul):
        raise ValueError('Expected a multiplicative term to normalize for {}'.format(term_in))

    const_vect = [term_mult for term_mult in term_in.args if isinstance(term_mult, sp.Integer)]

    if len(const_vect) > 1:
        raise ValueError('Found two constants in a numerator term when normalizing')

    const = 1 if len(const_vect) == 0 else abs(const_vect[0])
    return term_in / const


def masked_similarity_term(env_c, term_ref, terms_comp, encoder_c, const_blind=False, masked=True):
    """Compute cosine similarity between a reference term and a list of terms.
    From contrastive_simplifier.py.
    """
    metric_sim = nn.CosineSimilarity(dim=-1)
    similarity_vect = torch.zeros(len(terms_comp))

    for i, term in enumerate(terms_comp):
        if masked:
            newterm_ref, newterm_comp = sp.fraction(sp.cancel(term_ref / term))
        else:
            newterm_ref, newterm_comp = term_ref, term

        if const_blind:
            try:
                newterm_ref = normalize_term(newterm_ref)
                newterm_comp = normalize_term(newterm_comp)
            except ValueError:
                similarity_vect[i] = 0.0
                continue

        try:
            encoded_ref = encode_term(env_c, newterm_ref, encoder_c)
            encoded_comp = encode_term(env_c, newterm_comp, encoder_c)
            similarity_vect[i] = metric_sim(encoded_ref, encoded_comp)
        except Exception:
            similarity_vect[i] = 0.0

    return similarity_vect


def find_single_simplification_terms(similarity_vect, terms_comp, cutoff=0.9, denominator=None,
                                     short_search=False, max_group_size=None):
    """Given similarity vector, group terms and form sub-expression.

    If max_group_size is set, always pick the top-N most similar terms (ignoring cutoff).
    Otherwise, use the CDS cutoff-based approach.

    Returns:
        (terms_to_simplify, rest_terms): sub-expression to simplify and remaining numerator terms
    """
    if max_group_size is not None:
        # Expression-dependent grouping: pick top-N most similar terms
        k = min(max_group_size, len(similarity_vect))
        if k < 2:
            return None, None
        _, indices_k = torch.topk(similarity_vect, k=k)
        mask = torch.zeros_like(similarity_vect, dtype=torch.bool)
        mask[indices_k] = True
    else:
        # CDS cutoff-based approach
        mask = similarity_vect > cutoff

        # If no terms are close enough then we assume that we cannot simplify
        # Higher than 1 as we have similarity one with the reference
        if not torch.sum(mask).item() > 1:
            return None, None

        # If we do a short search we limit up to 4 terms at most
        if short_search:
            _, indices_k = torch.topk(similarity_vect, k=min(4, len(similarity_vect)))
            mask2 = torch.zeros_like(mask, dtype=torch.bool)
            mask2[indices_k] = True
            mask = mask & mask2

    # Retain only the terms that have a high chance of simplifying and combine them
    relevant_terms = terms_comp[mask]
    terms_to_simplify = relevant_terms.sum()

    # Also return the other non-combined terms
    rest_terms = terms_comp[~mask]

    # Add back the overall denominator
    if denominator is not None:
        terms_to_simplify = terms_to_simplify / denominator

    return terms_to_simplify, rest_terms


# ============================================================
# Contrastive-based simplification pass (follows CDS exactly)
# ============================================================

def contrastive_simplification_pass(
    input_equation,
    target_expr,
    rl_model,
    action_space,
    encoder_c,
    env_c,
    n_point: int,
    max_terms: int,
    max_brackets_per_term: int,
    max_steps: int,
    device: str,
    timeout_sec: float = 30.0,
    sample_timeout: float = 120.0,
    max_backtracks: int = 10,
    reject_term_increase: int = 1,
    cutoff: float = 0.99,
    const_blind: bool = False,
    max_group_size: int = None,
    rng_active: bool = False,
    rng_gen = None,
    cache_invalid: list = None,
    verbose: bool = False,
):
    """One pass over all numerator terms using contrastive grouping + RL simplification.
    If max_group_size is set, always picks top-N most similar terms (ignoring cutoff).
    """
    # Extract numerator terms and denominator (CDS exactly)
    terms_num, denom = extract_num_denom(input_equation)

    # Shuffle if rng active (following CDS)
    if rng_active and rng_gen is not None:
        rng_gen.shuffle(terms_num)

    # Retain variables for parsing
    terms_left = terms_num
    index_term = 0

    # Retain the added solution terms
    solution_generated = 0
    terms_simplified_num = 0
    num_simplification = 0
    short_search = False

    if cache_invalid is None:
        cache_invalid = []

    # Loop till we have considered all the terms
    while len(terms_left) > 0 and index_term < len(terms_left):

        if verbose:
            print(f"    Term {index_term}/{len(terms_left)}: computing similarity...", flush=True)

        # Find the terms most likely to cancel with the reference term (CDS exactly)
        try:
            sim_vect = masked_similarity_term(env_c, terms_left[index_term], terms_left, encoder_c,
                                              const_blind=const_blind)
        except Exception as e:
            if verbose:
                print(f"    Term {index_term}: similarity failed: {e}", flush=True)
            index_term += 1
            short_search = False
            continue

        # Find group: sums numerator terms and divides by denominator
        simplifier_t, rest_t = find_single_simplification_terms(
            sim_vect, terms_left, cutoff=cutoff, denominator=denom,
            short_search=short_search, max_group_size=max_group_size)

        # If we expect a simplification we try it out else we consider the next term
        if simplifier_t is not None:

            # Simplify initial term and count its length (CDS exactly)
            simplifier_t = sp.cancel(simplifier_t)
            num_t_attempt = len(terms_left) - len(rest_t)

            sub_brackets = count_brackets(simplifier_t)
            sub_num_terms = count_numerator_terms(simplifier_t)

            if verbose:
                print(f"    Term {index_term}: group of {num_t_attempt} -> "
                      f"{sub_brackets} bk, {sub_num_terms} num_terms", flush=True)

            # Check cache (don't check if rng active, following CDS)
            if simplifier_t not in cache_invalid or rng_active:

                # Run RL model with backtracking
                try:
                    result = evaluate_sample_backtrack(
                        model=rl_model,
                        initial_expr=simplifier_t,
                        target_expr=target_expr,
                        action_space=action_space,
                        n_point=n_point,
                        max_terms=max_terms,
                        max_brackets_per_term=max_brackets_per_term,
                        max_steps=max_steps,
                        device=device,
                        timeout_sec=timeout_sec,
                        sample_timeout=sample_timeout,
                        max_backtracks=max_backtracks,
                        proper_stop=True,
                        reject_term_increase=reject_term_increase,
                        verbose=verbose,
                    )
                except Exception as e:
                    if verbose:
                        print(f"    Term {index_term}: RL eval failed: {e}", flush=True)
                    result = {'solved_source_rel': False, 'solved_proper': False}

                solution_attempt = None
                if result['solved_source_rel'] or result['solved_proper']:
                    solution_attempt = sp.sympify(result['final_expr'], locals={'ab': ab, 'sb': sb})

                # If we simplified then we update the terms to consider
                if solution_attempt is not None:

                    # Update the number of terms left (CDS exactly)
                    terms_left = rest_t

                    # For logging purposes track the simplification
                    num_simplification += 1
                    num_terms_simple = count_numerator_terms(solution_attempt)
                    num_terms_complex = count_numerator_terms(simplifier_t)
                    terms_simplified_num += num_terms_simple
                    solution_generated = solution_generated + solution_attempt

                    print(f"    Step {num_simplification}: simplified {num_terms_complex} -> "
                          f"{num_terms_simple} num terms ({sub_brackets} -> "
                          f"{result['final_brackets']} bk)", flush=True)
                    if verbose:
                        print(f"    {len(terms_left) + terms_simplified_num} terms left "
                              f"in numerator", flush=True)

                    # Reset the search method (CDS exactly - do NOT reset index_term)
                    short_search = False

                # If we only searched for a long expression we allow to search the smaller one
                elif not short_search and num_t_attempt > 4:
                    short_search = True
                    if simplifier_t not in cache_invalid:
                        cache_invalid.append(simplifier_t)

                # If no simplification and already short search we consider the next term
                else:
                    index_term += 1
                    short_search = False
                    if simplifier_t not in cache_invalid:
                        cache_invalid.append(simplifier_t)
            else:
                # In cache
                if not short_search and num_t_attempt > 4:
                    short_search = True
                else:
                    index_term += 1
                    short_search = False

        # When we expect no simplification we move to the next term
        else:
            index_term += 1
            short_search = False

    terms_left_end = len(terms_left) + terms_simplified_num

    if verbose:
        print(f"    Pass done: {count_numerator_terms(input_equation)} -> "
              f"{terms_left_end} numerator terms "
              f"({num_simplification} groups simplified)", flush=True)

    # Craft back the expression from the terms left over (CDS exactly).
    # terms_left are numerator terms; divide by denom to make a fraction.
    # solution_generated already includes denominator (from find_single_simplification_terms).
    terms_left = terms_left.sum()
    terms_left = terms_left / denom

    # Return the expression to be considered in the next pass
    return solution_generated + terms_left, terms_left_end, num_simplification


# ============================================================
# Step-by-step contrastive: regroup after every model action
# ============================================================

def encode_all_terms_batch(env_c, terms, encoder_c, const_blind=True):
    """Encode all numerator terms and return embedding matrix [N, emb_dim].

    Terms that fail to encode get zero vectors (will have 0 similarity to everything).
    Returns: (embeddings [N, D], valid_mask [N] boolean)
    """
    embeddings = []
    valid = []
    for term in terms:
        try:
            t = normalize_term(term) if const_blind else term
            emb = encode_term(env_c, t, encoder_c)  # shape varies; squeeze to 1D
            emb = emb.squeeze()
            if emb.dim() == 0:
                emb = emb.unsqueeze(0)
            embeddings.append(emb)
            valid.append(True)
        except Exception:
            # Use zero vector as placeholder
            if embeddings:
                embeddings.append(torch.zeros_like(embeddings[-1]))
            else:
                embeddings.append(torch.zeros(512))  # CDS default emb_dim
            valid.append(False)

    return torch.stack(embeddings), torch.tensor(valid, dtype=torch.bool)


def pairwise_cosine_similarity(embeddings, valid_mask=None):
    """Compute pairwise cosine similarity matrix [N, N].

    Invalid entries (from valid_mask=False) get similarity 0.
    """
    norms = embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    normalized = embeddings / norms
    sim_matrix = torch.mm(normalized, normalized.t())

    # Zero out rows/cols for invalid terms
    if valid_mask is not None:
        invalid = ~valid_mask
        sim_matrix[invalid, :] = 0.0
        sim_matrix[:, invalid] = 0.0

    return sim_matrix


def pick_best_group(sim_matrix, max_group_size):
    """Find the term whose top-k neighbors have highest avg similarity.

    Returns: (best_ref_idx, group_indices list, avg_similarity)
    """
    N = sim_matrix.size(0)
    k = min(max_group_size, N)

    if k < 2:
        return 0, list(range(N)), 0.0

    # Set diagonal to -inf to exclude self
    sim_no_self = sim_matrix.clone()
    sim_no_self.fill_diagonal_(-float('inf'))

    # For each term, get top-(k-1) similarities (excluding self)
    topk_vals, topk_indices = torch.topk(sim_no_self, k=min(k - 1, N - 1), dim=1)
    avg_sims = topk_vals.mean(dim=1)  # [N]

    best_ref = avg_sims.argmax().item()

    # Group = reference + its top-(k-1) neighbors
    group_indices = [best_ref] + topk_indices[best_ref].tolist()

    return best_ref, group_indices, avg_sims[best_ref].item()


def rank_all_groupings(sim_matrix, max_group_size):
    """Rank all candidate groupings by average similarity (descending).

    Each term i gives a candidate grouping: i + its top-(k-1) neighbors.
    Returns list of (ref_idx, group_indices, avg_sim) sorted by avg_sim descending.
    """
    N = sim_matrix.size(0)
    k = min(max_group_size, N)

    if k < 2:
        return [(0, list(range(N)), 0.0)]

    sim_no_self = sim_matrix.clone()
    sim_no_self.fill_diagonal_(-float('inf'))

    topk_vals, topk_indices = torch.topk(sim_no_self, k=min(k - 1, N - 1), dim=1)
    avg_sims = topk_vals.mean(dim=1)  # [N]

    # Sort by avg_sim descending
    sorted_refs = torch.argsort(avg_sims, descending=True).tolist()

    results = []
    for ref in sorted_refs:
        group_indices = [ref] + topk_indices[ref].tolist()
        results.append((ref, group_indices, avg_sims[ref].item()))

    return results


def rank_all_groupings_threshold(sim_matrix, sim_threshold, max_group_size):
    """Rank candidate groupings using a similarity threshold.

    For each reference term i, collect all terms with similarity >= sim_threshold,
    then take the top max_group_size-1 by similarity (plus the reference term itself).
    Skip terms with no neighbors above threshold.
    Returns list of (ref_idx, group_indices, avg_sim) sorted by avg_sim descending.
    """
    N = sim_matrix.size(0)

    sim_no_self = sim_matrix.clone()
    sim_no_self.fill_diagonal_(-float('inf'))

    results = []
    for ref in range(N):
        # Find all neighbors above threshold
        sims = sim_no_self[ref]
        above_mask = sims >= sim_threshold
        n_above = above_mask.sum().item()
        if n_above == 0:
            continue  # no neighbors above threshold

        # Get indices and sims of neighbors above threshold
        neighbor_indices = torch.where(above_mask)[0]
        neighbor_sims = sims[neighbor_indices]

        # Take top max_group_size-1 by similarity
        k = min(max_group_size - 1, len(neighbor_indices))
        topk_sims, topk_local = torch.topk(neighbor_sims, k=k)
        topk_indices = neighbor_indices[topk_local]

        group_indices = [ref] + topk_indices.tolist()
        avg_sim = topk_sims.mean().item()
        results.append((ref, group_indices, avg_sim))

    # Sort by avg_sim descending
    results.sort(key=lambda x: -x[2])
    return results


def apply_single_model_step(
    model, current_expr, action_space, n_point, max_terms,
    max_brackets_per_term, device, timeout_sec,
    visited_states=None, blocked_actions=None,
    reject_term_increase=1, max_retries=20, verbose=False,
):
    """Apply ONE model action to the expression.

    Returns:
        (new_expr, action_info) if successful, or (None, None) if no valid action.
    """
    current_brackets = count_brackets(current_expr)
    current_terms = count_terms(current_expr)

    try:
        obs = build_observation_from_expr(current_expr, n_point, max_terms, max_brackets_per_term)
    except Exception:
        return None, None

    action_mask = build_action_mask(current_expr, action_space)
    if action_mask.sum() == 0:
        return None, None

    # Model prediction
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(obs_tensor)

    FLOAT_MIN = -3.4e38
    mask_tensor = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0).to(device)
    inf_mask = torch.clamp(torch.log(mask_tensor + 1e-10), min=FLOAT_MIN)
    masked_logits = logits + inf_mask
    sorted_actions = torch.argsort(masked_logits[0], descending=True).cpu().numpy()

    if blocked_actions is None:
        blocked_actions = set()
    if visited_states is None:
        visited_states = set()

    for retry_idx, action in enumerate(sorted_actions[:max_retries]):
        if action_mask[action] == 0:
            continue
        if action in blocked_actions:
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

            # Anti-cycle check on sub-expression
            if visited_states:
                try:
                    candidate_obs = build_observation_from_expr(
                        result, n_point, max_terms, max_brackets_per_term)
                    candidate_hash = obs_to_hash(candidate_obs)
                    if candidate_hash in visited_states:
                        if verbose:
                            print(f"        retry {retry_idx}: {spec.identity} on {bracket_type}({i},{j}) t={spec.term_slot} -> cycle", flush=True)
                        continue
                    visited_states.add(candidate_hash)
                except Exception:
                    pass

            new_brackets = count_brackets(result)
            new_terms = count_terms(result)

            # Reject term explosion
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


def contrastive_step_by_step_simplification(
    input_equation,
    target_expr,
    rl_model,
    action_space,
    encoder_c,
    env_c,
    n_point: int,
    max_terms: int,
    max_brackets_per_term: int,
    max_steps: int = 200,
    device: str = 'cpu',
    timeout_sec: float = 30.0,
    sample_timeout: float = 300.0,
    max_backtracks: int = 10,
    reject_term_increase: int = 1,
    max_group_size: int = 10,
    sim_threshold: float = None,
    verbose: bool = False,
    output_dir: str = None,
):
    """Step-by-step contrastive simplification (CDS-like multi-group per pass).

    Each pass:
    1. Extract numerator terms, encode with contrastive encoder
    2. Iterate through reference terms in order of max neighbor similarity
    3. For each ref term, find group of similar available terms (>= threshold)
    4. Try to simplify the group sub-expression with RL model
    5. On success: remove group from available terms, accumulate simplified piece
    6. Continue to next ref term (DON'T recombine full expression yet)
    7. At end of pass: recombine all simplified pieces + remaining terms

    Between passes: if no simplifications found, relax threshold (0.99 -> 0.95).

    Once terms <= max_group_size, switch to standard evaluate_sample_backtrack.
    """
    t0 = time.time()
    current_expr = sp.cancel(input_equation)  # canonical form for consistent bracket counting
    start_bk = count_brackets(current_expr)
    start_terms_n = count_numerator_terms(current_expr)
    best_expr = current_expr
    best_bk = start_bk
    best_terms = start_terms_n

    round_num = 0
    total_groupings_tried = 0

    print(f"\nStarting step-by-step contrastive simplification (multi-group per pass):", flush=True)
    print(f"  Start: {start_bk} brackets, {start_terms_n} numerator terms", flush=True)
    print(f"  Max group size: {max_group_size}", flush=True)

    # Setup checkpoint directory
    checkpoints_dir = None
    if output_dir:
        checkpoints_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)
        print(f"  Checkpoints dir: {checkpoints_dir}", flush=True)

    # Store original expression for numerical validation
    original_expr = current_expr

    # Threshold relaxation schedule
    threshold_schedule = [sim_threshold]
    if sim_threshold is not None and sim_threshold > 0.95:
        threshold_schedule.append(0.95)
    current_threshold = sim_threshold

    if sim_threshold is not None:
        print(f"  Similarity threshold: {sim_threshold} (relaxes to 0.95)", flush=True)

    rng = np.random.default_rng(42)

    while round_num < max_steps:
        elapsed = time.time() - t0
        if elapsed > sample_timeout:
            print(f"  Pass {round_num}: total timeout {elapsed:.1f}s", flush=True)
            break

        # Extract current numerator terms
        terms_num, denom = extract_num_denom(current_expr)
        n_terms = len(terms_num)
        current_bk = count_brackets(current_expr)

        if n_terms <= max_group_size:
            # Small enough for direct model evaluation
            print(f"  Pass {round_num}: {n_terms} terms <= {max_group_size}, "
                  f"switching to evaluate_sample_backtrack ({current_bk} bk)", flush=True)

            remaining_time = max(sample_timeout - elapsed, 60.0)

            result = evaluate_sample_backtrack(
                model=rl_model,
                initial_expr=current_expr,
                target_expr=target_expr,
                action_space=action_space,
                n_point=n_point,
                max_terms=max_terms,
                max_brackets_per_term=max_brackets_per_term,
                max_steps=max_steps,
                device=device,
                timeout_sec=timeout_sec,
                sample_timeout=remaining_time,
                max_backtracks=max_backtracks,
                proper_stop=True,
                reject_term_increase=reject_term_increase,
                verbose=verbose,
            )

            if result['solved_source_rel'] or result['solved_proper']:
                final_expr = sp.sympify(result['final_expr'], locals={'ab': ab, 'sb': sb})
                final_bk = count_brackets(final_expr)
                if final_bk < best_bk:
                    best_bk = final_bk
                    best_expr = final_expr
                    print(f"  Direct eval: {current_bk} -> {final_bk} bk", flush=True)
            else:
                print(f"  Direct eval: no improvement from {current_bk} bk", flush=True)
            break

        # Encode all terms and compute similarity matrix
        thresh_str = f", thresh={current_threshold}" if current_threshold is not None else ""
        print(f"  Pass {round_num}: {n_terms} terms, {current_bk} bk{thresh_str} — "
              f"encoding terms...", flush=True)

        embeddings, valid_mask = encode_all_terms_batch(env_c, terms_num, encoder_c, const_blind=True)
        sim_matrix_t = pairwise_cosine_similarity(embeddings, valid_mask)
        # Convert to numpy for easier indexing
        sim_matrix = sim_matrix_t.numpy() if isinstance(sim_matrix_t, torch.Tensor) else sim_matrix_t

        # ---- CDS-like multi-group pass ----
        # Track which term indices are still available
        available = set(range(n_terms))
        solution_generated = sp.Integer(0)  # accumulated simplified sub-expressions
        solution_terms_count = 0  # total terms from simplified pieces
        num_simplifications = 0
        pass_actions = []  # action log for this pass

        # Determine reference term ordering: rank by max neighbor similarity
        ref_order = list(range(n_terms))
        # Sort by max neighbor similarity (descending) — try most promising first
        max_neighbor_sims = []
        for i in range(n_terms):
            sims_i = sim_matrix[i].copy()
            sims_i[i] = -float('inf')  # exclude self
            max_neighbor_sims.append(sims_i.max())
        ref_order.sort(key=lambda i: -max_neighbor_sims[i])

        # Count eligible refs (those with at least 1 available neighbor above threshold)
        def count_remaining_eligible():
            count = 0
            for ri in ref_order:
                if ri not in available:
                    continue
                for j in available:
                    if j != ri and (current_threshold is None or sim_matrix[ri, j] >= current_threshold):
                        count += 1
                        break
            return count

        n_eligible = count_remaining_eligible()
        print(f"    {n_eligible} eligible ref terms (with neighbors above threshold)", flush=True)

        ref_counter = 0
        for ref_idx in ref_order:
            if ref_idx not in available:
                continue

            elapsed = time.time() - t0
            if elapsed > sample_timeout:
                break

            # Find group: available terms with sim >= threshold
            group_indices = [ref_idx]
            neighbor_sims = []
            for j in available:
                if j == ref_idx:
                    continue
                sim_val = sim_matrix[ref_idx, j]
                if current_threshold is not None and sim_val < current_threshold:
                    continue
                neighbor_sims.append((j, sim_val))

            if len(neighbor_sims) == 0:
                continue  # no similar terms for this ref

            ref_counter += 1

            # Sort neighbors by similarity descending, cap at max_group_size-1
            neighbor_sims.sort(key=lambda x: -x[1])
            for j, s in neighbor_sims[:max_group_size - 1]:
                group_indices.append(j)

            if len(group_indices) < 2:
                continue

            avg_sim = np.mean([sim_matrix[ref_idx, j] for j in group_indices if j != ref_idx])

            total_groupings_tried += 1

            # Form sub-expression from group terms
            group_terms_arr = terms_num[np.array(group_indices)]
            sub_expr = group_terms_arr.sum() / denom
            sub_expr = sp.cancel(sub_expr)

            sub_bk = count_brackets(sub_expr)
            sub_n_terms = count_numerator_terms(sub_expr)

            # Factor out common brackets and denominator — give model just the reduced numerator
            sub_num, sub_denom = sp.fraction(sp.cancel(sub_expr))
            common_factor, reduced_expr = factor_out_common_brackets(sub_expr)
            reduced_num, _ = sp.fraction(reduced_expr)
            model_expr = reduced_num  # just the numerator polynomial, no denominator
            model_bk = count_brackets(model_expr)
            model_terms = count_terms(model_expr)

            if verbose:
                cf_str = f" (cf={common_factor})" if common_factor != 1 else ""
                print(f"    [{ref_counter}/{n_eligible}] ref={ref_idx}, {len(group_indices)} terms, "
                      f"avg_sim={avg_sim:.4f}, sub={sub_bk} bk/{sub_n_terms} tm, "
                      f"model={model_bk} bk/{model_terms} tm{cf_str}",
                      flush=True)

            # Apply single model step on the reduced numerator
            new_model_expr, action_info = apply_single_model_step(
                model=rl_model,
                current_expr=model_expr,
                action_space=action_space,
                n_point=n_point,
                max_terms=max_terms,
                max_brackets_per_term=max_brackets_per_term,
                device=device,
                timeout_sec=timeout_sec,
                reject_term_increase=reject_term_increase,
                max_retries=20,
                verbose=verbose,
            )

            if new_model_expr is None:
                if verbose:
                    print(f"      no valid action", flush=True)
                continue

            new_model_terms = count_terms(new_model_expr)
            if new_model_terms >= model_terms:
                if verbose:
                    new_model_bk = count_brackets(new_model_expr)
                    print(f"      model: {model_bk}->{new_model_bk} bk, "
                          f"{model_terms}->{new_model_terms} tm — REJECTED (no term reduction)",
                          flush=True)
                continue

            # Recombine: common_factor * model_result / sub_denom
            r_num, r_denom = sp.fraction(new_model_expr)
            new_sub_expr = sp.cancel(common_factor * r_num / (r_denom * sub_denom))
            new_sub_bk = count_brackets(new_sub_expr)
            new_sub_terms = count_numerator_terms(new_sub_expr)

            # Success! Remove group indices from available, accumulate result
            for idx in group_indices:
                available.discard(idx)
            solution_generated = solution_generated + new_sub_expr
            solution_terms_count += new_sub_terms
            num_simplifications += 1

            # Log action info
            if action_info is not None:
                pass_actions.append({
                    'ref_idx': ref_idx,
                    'group_indices': group_indices,
                    'avg_sim': float(avg_sim),
                    'sub_terms': sub_n_terms,
                    'sub_bk': sub_bk,
                    'model_terms_before': model_terms,
                    'model_terms_after': new_model_terms,
                    'action': action_info,
                    'new_sub_terms': new_sub_terms,
                    'new_sub_bk': new_sub_bk,
                })

            # Recount eligible refs after consuming terms
            n_eligible = count_remaining_eligible()

            print(f"    Step {num_simplifications}: model {model_terms}->{new_model_terms} tm, "
                  f"sub {sub_n_terms}->{new_sub_terms} tm ({sub_bk}->{new_sub_bk} bk). "
                  f"{len(available) + solution_terms_count} terms remaining "
                  f"({n_eligible} eligible refs left). "
                  f"Action: {action_info['identity']} {action_info['bracket']}",
                  flush=True)

        # ---- End of pass: recombine ----
        if num_simplifications == 0:
            # No simplifications — try relaxing threshold
            if current_threshold is not None and len(threshold_schedule) > 1:
                threshold_schedule.pop(0)
                current_threshold = threshold_schedule[0]
                print(f"  Pass {round_num}: no simplifications at current threshold, "
                      f"relaxing to {current_threshold}", flush=True)
                round_num += 1
                continue
            print(f"  Pass {round_num}: no improvement possible, stopping.", flush=True)
            break
        else:
            # Recombine: simplified pieces + remaining terms / denom
            remaining_indices = sorted(available)
            if len(remaining_indices) > 0:
                remaining_terms = terms_num[np.array(remaining_indices)]
                new_full_expr = solution_generated + remaining_terms.sum() / denom
            else:
                new_full_expr = solution_generated

            new_full_expr = sp.cancel(new_full_expr)
            new_bk = count_brackets(new_full_expr)
            new_n_terms = count_numerator_terms(new_full_expr)

            print(f"  Pass {round_num} done: {num_simplifications} simplifications, "
                  f"{n_terms} -> {new_n_terms} terms, {current_bk} -> {new_bk} bk",
                  flush=True)

            # Numerical validation against original
            is_valid, max_rdiff = numerical_validate(
                original_expr, new_full_expr, label=f"pass {round_num}")

            if not is_valid:
                print(f"  *** CORRUPTION DETECTED at pass {round_num}! "
                      f"max_rdiff={max_rdiff:.2e} — STOPPING ***", flush=True)
                # Save corrupted checkpoint
                if checkpoints_dir:
                    with open(os.path.join(checkpoints_dir,
                              f'CORRUPTED_pass_{round_num:04d}.txt'), 'w') as f:
                        f.write(str(new_full_expr))
                    with open(os.path.join(checkpoints_dir,
                              f'CORRUPTED_pass_{round_num:04d}_info.json'), 'w') as f:
                        json.dump({
                            'pass': round_num,
                            'n_terms_before': n_terms,
                            'n_terms_after': new_n_terms,
                            'bk_before': current_bk,
                            'bk_after': new_bk,
                            'max_rdiff': max_rdiff,
                            'actions': pass_actions,
                        }, f, indent=2, default=str)
                break

            # Save versioned checkpoint
            if checkpoints_dir:
                with open(os.path.join(checkpoints_dir,
                          f'pass_{round_num:04d}.txt'), 'w') as f:
                    f.write(str(new_full_expr))
                with open(os.path.join(checkpoints_dir,
                          f'pass_{round_num:04d}_info.json'), 'w') as f:
                    json.dump({
                        'pass': round_num,
                        'n_terms_before': n_terms,
                        'n_terms_after': new_n_terms,
                        'bk_before': current_bk,
                        'bk_after': new_bk,
                        'max_rdiff': max_rdiff,
                        'num_simplifications': num_simplifications,
                        'actions': pass_actions,
                    }, f, indent=2, default=str)

            current_expr = new_full_expr
            if new_n_terms < best_terms or (new_n_terms == best_terms and new_bk < best_bk):
                best_terms = new_n_terms
                best_bk = new_bk
                best_expr = new_full_expr
                print(f"  *** NEW BEST: {new_n_terms} terms, {new_bk} bk "
                      f"({start_terms_n} -> {new_n_terms} tm, "
                      f"{start_bk} -> {new_bk} bk)", flush=True)

            # Reset threshold back to initial after success
            if sim_threshold is not None:
                current_threshold = sim_threshold
                threshold_schedule = [sim_threshold]
                if sim_threshold > 0.95:
                    threshold_schedule.append(0.95)

        round_num += 1

    exec_time = time.time() - t0
    final_bk = count_brackets(best_expr)
    final_terms = count_numerator_terms(best_expr)

    print(f"\n  Step-by-step result: {start_bk} -> {final_bk} bk, "
          f"{start_terms_n} -> {final_terms} terms "
          f"({round_num} passes, {total_groupings_tried} groupings tried, "
          f"{exec_time:.1f}s)", flush=True)

    stats = {
        'initial_brackets': start_bk,
        'final_brackets': final_bk,
        'initial_num_terms': start_terms_n,
        'final_num_terms': final_terms,
        'rounds': round_num,
        'groupings_tried': total_groupings_tried,
        'total_time': exec_time,
    }

    return best_expr, stats


# ============================================================
# Main outer loop: contrastive total simplification (follows CDS exactly)
# ============================================================

def contrastive_total_simplification(
    input_equation,
    target_expr,
    rl_model,
    action_space,
    encoder_c,
    env_c,
    n_point: int,
    max_terms: int,
    max_brackets_per_term: int,
    max_steps: int,
    device: str,
    timeout_sec: float = 30.0,
    sample_timeout: float = 120.0,
    max_backtracks: int = 10,
    reject_term_increase: int = 1,
    init_cutoff: float = 0.99,
    power_decay: float = 0.5,
    const_blind: bool = False,
    max_group_size: int = None,
    max_passes: int = 20,
    verbose: bool = False,
):
    """Full iterative simplification using contrastive grouping + RL model.
    If max_group_size is set, uses top-N grouping instead of cutoff threshold.
    """
    start_time = time.time()

    # Initialize loop parameters (CDS exactly)
    reducing = True
    rng_active = False
    num_simplification = 0
    rng_passes = 0
    cutoff = init_cutoff
    cache_invalid = []

    # Create rng generator for shuffling
    rng_gen = np.random.RandomState(42)

    # Load the equation and record the number of terms in the numerator
    len_init = count_numerator_terms(input_equation)
    len_in = len_init

    print(f"\nStarting contrastive simplification (CDS approach):", flush=True)
    print(f"  Initial numerator terms: {len_init}", flush=True)
    print(f"  Initial cutoff: {init_cutoff}", flush=True)
    print(f"  Power decay: {power_decay}", flush=True)
    print(f"  Max passes: {max_passes}", flush=True)
    if max_group_size is not None:
        print(f"  Max group size: {max_group_size} (top-N grouping)", flush=True)
    print(f"  Eval settings: max_steps={max_steps}, timeout={timeout_sec}s, "
          f"sample_timeout={sample_timeout}s, max_backtracks={max_backtracks}, "
          f"RTI={reject_term_increase}", flush=True)

    pass_count = 0
    simple_form = input_equation

    # Try to simplify as long as we hope for a simplification (CDS exactly)
    while reducing and pass_count < max_passes:
        pass_t0 = time.time()
        pass_count += 1

        if rng_active:
            print(f"\n  Pass {pass_count} (cutoff={cutoff:.4f}, shuffling {rng_passes}/5):", flush=True)
        else:
            print(f"\n  Pass {pass_count} (cutoff={cutoff:.4f}):", flush=True)

        # Do a simplification pass over all the terms in the expression
        simple_form, len_new, num_simple = contrastive_simplification_pass(
            input_equation=input_equation,
            target_expr=target_expr,
            rl_model=rl_model,
            action_space=action_space,
            encoder_c=encoder_c,
            env_c=env_c,
            n_point=n_point,
            max_terms=max_terms,
            max_brackets_per_term=max_brackets_per_term,
            max_steps=max_steps,
            device=device,
            timeout_sec=timeout_sec,
            sample_timeout=sample_timeout,
            max_backtracks=max_backtracks,
            reject_term_increase=reject_term_increase,
            cutoff=cutoff,
            const_blind=const_blind,
            max_group_size=max_group_size,
            rng_active=rng_active,
            rng_gen=rng_gen,
            cache_invalid=cache_invalid,
            verbose=verbose,
        )

        pass_time = time.time() - pass_t0
        print(f"  Pass {pass_count}: {len_in} -> {len_new} num terms "
              f"({num_simple} groups simplified) [{pass_time:.1f}s]", flush=True)

        # CDS logic exactly
        if len_in > len_new or num_simple > 0:
            num_simplification += num_simple
            input_equation = simple_form
            len_in = len_new

            # Keep rng only if active and not decreasing (CDS exactly)
            rng_active = False if len_in > len_new else rng_active
            rng_passes = rng_passes + 1 if rng_active else 0
            if rng_active:
                print(f"  {len_new} terms: shuffling pass {rng_passes}/5", flush=True)
            reducing = rng_passes < 5

        # If the expression is of size 1 it is maximally simplified
        elif len_new == 1:
            reducing = False

        # If it has not decreased we allow for extra loops with reshuffling (CDS exactly)
        else:
            rng_passes += 1
            print(f"  No obvious simplification at {len_new} terms: "
                  f"shuffling pass {rng_passes}/5", flush=True)
            rng_active = True
            reducing = rng_passes < 5

        # At each pass we lower the similarity cutoff (CDS exactly)
        cutoff = cutoff * (init_cutoff ** power_decay)

    # Return the final expression in a minimal form
    simple_form = sp.cancel(simple_form)
    exec_time = time.time() - start_time
    len_final = count_numerator_terms(simple_form)

    print(f"\n  Went from {len_init} to {len_final} numerator terms in "
          f"{num_simplification} simplification steps and {exec_time:.1f} seconds", flush=True)

    stats = {
        'initial_num_terms': len_init,
        'final_num_terms': len_final,
        'total_groups_simplified': num_simplification,
        'passes': pass_count,
        'total_time': exec_time,
    }

    return simple_form, stats


# ============================================================
# Model configuration per n_point
# ============================================================

MODEL_CONFIG = {
    4: {
        'max_terms': 20,
        'max_terms_action': 10,
        'max_brackets_per_term': 8,
        'max_brackets': 12,
        'bracket_feature_dim': 10,  # 2 + 2*4
        'term_feature_dim': 83,     # 2 + 8*10 + 1
    },
    5: {
        'max_terms': 25,
        'max_terms_action': 25,
        'max_brackets_per_term': 15,
        'max_brackets': 20,
        'bracket_feature_dim': 12,  # 2 + 2*5
        'term_feature_dim': 183,    # 2 + 15*12 + 1
    },
}


def load_model_and_action_space(n_point, model_path, device):
    """Load model and create action space for given n_point."""
    cfg = MODEL_CONFIG[n_point]

    # Import action space class
    if n_point == 4:
        from action_space_4pt import TermSpecificActionSpace
        action_space = TermSpecificActionSpace(
            n_point=n_point,
            max_terms=cfg['max_terms_action'],
            max_brackets=cfg['max_brackets'],
        )
    elif n_point == 5:
        from action_space_5pt import TermSpecificActionSpace5pt
        action_space = TermSpecificActionSpace5pt(
            n_point=n_point,
            max_terms=cfg['max_terms_action'],
            max_brackets=cfg['max_brackets'],
        )
    else:
        raise ValueError(f"Unsupported n_point: {n_point}")

    print(f"Action space: {action_space.identities_per_bracket} identities/bracket, "
          f"{action_space.n_term_slots} term_slots, {action_space.n_actions} total actions", flush=True)

    # Create model
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

    # Load checkpoint
    print(f"Loading model from {model_path}", flush=True)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}", flush=True)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    return model, action_space, cfg


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Large expression simplification with CDS contrastive grouping + RL backtracking')
    parser.add_argument('--input', required=True, help='Path to expression text file')
    parser.add_argument('--n_point', type=int, required=True, choices=[4, 5])
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')

    # Eval settings (matching paper evals)
    parser.add_argument('--max_steps', type=int, default=100)
    parser.add_argument('--sample_timeout', type=float, default=120.0)
    parser.add_argument('--timeout', type=float, default=30.0)
    parser.add_argument('--max_backtracks', type=int, default=10)
    parser.add_argument('--reject_term_increase', type=int, default=1)

    # Contrastive grouping settings
    parser.add_argument('--init_cutoff', type=float, default=0.99)
    parser.add_argument('--power_decay', type=float, default=0.5)
    parser.add_argument('--const_blind', action='store_true')
    parser.add_argument('--max_group_size', type=int, default=None,
                        help='Max terms per group.')
    parser.add_argument('--sim_threshold', type=float, default=None,
                        help='Similarity threshold for grouping. Collect all terms with '
                             'sim >= threshold, then take top max_group_size.')
    parser.add_argument('--max_passes', type=int, default=20)

    # Step-by-step mode: regroup after every action
    parser.add_argument('--step_by_step', action='store_true',
                        help='Regroup after every model action instead of CDS pass-based approach.')

    # General settings
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load expression
    print(f"Loading expression from {args.input}", flush=True)
    with open(args.input, 'r') as f:
        expr_str = f.read().strip()
    print(f"Expression string length: {len(expr_str):,} characters", flush=True)

    # Parse full expression to sympy
    print(f"Parsing expression to sympy...", flush=True)
    t0 = time.time()
    try:
        full_expr = sp.sympify(expr_str, locals=FUNC_DICT)
    except Exception:
        # Fallback: parse term by term and sum
        print(f"  Direct parse failed, falling back to term-by-term parsing...", flush=True)
        string_terms = split_expression_string(expr_str)
        sympy_terms = []
        for t_str in string_terms:
            try:
                t_expr = sp.sympify(t_str, locals=FUNC_DICT)
                if t_expr != 0:
                    sympy_terms.append(t_expr)
            except Exception:
                pass
        full_expr = sum(sympy_terms) if sympy_terms else sp.Integer(0)
    parse_time = time.time() - t0
    print(f"Parsed in {parse_time:.1f}s", flush=True)

    # Put over common denominator and count initial complexity
    print(f"Extracting numerator terms and denominator...", flush=True)
    t0 = time.time()
    init_num_terms, init_denom = extract_num_denom(full_expr)
    extract_time = time.time() - t0
    n_init_num_terms = len(init_num_terms)
    init_brackets = count_brackets(full_expr)
    denom_brackets = count_brackets(init_denom)
    print(f"Extracted in {extract_time:.1f}s: {n_init_num_terms} numerator terms, "
          f"{denom_brackets} denominator brackets, "
          f"{init_brackets} total brackets", flush=True)

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}", flush=True)

    # Load RL model and action space
    rl_model, action_space, cfg = load_model_and_action_space(args.n_point, args.model, device)

    # Load CDS contrastive encoder
    encoder_c, env_c = load_contrastive_encoder(args.n_point, device='cpu')

    # Initialize worker
    init_worker(args.n_point, max_terms=cfg['max_terms_action'])

    # Run simplification
    try:
        # For large expressions without known target, use sp.Integer(0)
        # (simplify as much as possible — proper_stop will prevent early exit)
        target = sp.Integer(0)

        if args.step_by_step:
            # Step-by-step mode: regroup after every action
            group_size = args.max_group_size if args.max_group_size is not None else 10
            # Determine output dir for checkpoints
            output_dir = None
            if args.output:
                output_dir = os.path.dirname(args.output)
                if not output_dir:
                    output_dir = '.'

            final_expr, stats = contrastive_step_by_step_simplification(
                input_equation=full_expr,
                target_expr=target,
                rl_model=rl_model,
                action_space=action_space,
                encoder_c=encoder_c,
                env_c=env_c,
                n_point=args.n_point,
                max_terms=cfg['max_terms'],
                max_brackets_per_term=cfg['max_brackets_per_term'],
                max_steps=args.max_steps,
                device=device,
                timeout_sec=args.timeout,
                sample_timeout=args.sample_timeout,
                max_backtracks=args.max_backtracks,
                reject_term_increase=args.reject_term_increase,
                max_group_size=group_size,
                sim_threshold=args.sim_threshold,
                verbose=args.verbose,
                output_dir=output_dir,
            )
        else:
            # CDS pass-based approach
            final_expr, stats = contrastive_total_simplification(
                input_equation=full_expr,
                target_expr=target,
                rl_model=rl_model,
                action_space=action_space,
                encoder_c=encoder_c,
                env_c=env_c,
                n_point=args.n_point,
                max_terms=cfg['max_terms'],
                max_brackets_per_term=cfg['max_brackets_per_term'],
                max_steps=args.max_steps,
                device=device,
                timeout_sec=args.timeout,
                sample_timeout=args.sample_timeout,
                max_backtracks=args.max_backtracks,
                reject_term_increase=args.reject_term_increase,
                init_cutoff=args.init_cutoff,
                power_decay=args.power_decay,
                const_blind=args.const_blind,
                max_group_size=args.max_group_size,
                max_passes=args.max_passes,
                verbose=args.verbose,
            )
    finally:
        shutdown_worker()

    # Final summary
    final_brackets = count_brackets(final_expr)
    final_num_terms = count_numerator_terms(final_expr)

    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Input: {args.input}", flush=True)
    print(f"n_point: {args.n_point}", flush=True)
    print(f"Mode: {'step-by-step' if args.step_by_step else 'CDS pass-based'}", flush=True)
    print(f"Initial: {n_init_num_terms} numerator terms, {init_brackets} brackets", flush=True)
    print(f"Final: {final_num_terms} numerator terms, {final_brackets} brackets", flush=True)
    print(f"Numerator term reduction: {n_init_num_terms} -> {final_num_terms} "
          f"({100 * (1 - final_num_terms / max(n_init_num_terms, 1)):.1f}%)", flush=True)
    print(f"Bracket reduction: {init_brackets} -> {final_brackets} "
          f"({100 * (1 - final_brackets / max(init_brackets, 1)):.1f}%)", flush=True)
    print(f"Total time: {stats['total_time']:.1f}s", flush=True)

    # Save output
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        with open(args.output, 'w') as f:
            f.write(str(final_expr))
        print(f"\nFinal expression saved to {args.output}", flush=True)


if __name__ == '__main__':
    main()
