#!/usr/bin/env python
"""
Beam search simplification for spinor-helicity expressions.

Maintains K candidate expressions. At each step, for each candidate,
tries top-N model actions, scores all K*N results, keeps best K.
Stops when any candidate reaches 1 term or max_steps is hit.

Usage:
    python beam_search_5pt.py \
        --input expression.txt \
        --n_point 5 \
        --model models/sft_5pt_500k/best_model.pt \
        --beam_width 20 \
        --branches 5 \
        --max_steps 200 \
        --output output_dir/
"""

import argparse
import os
import sys
import time
import json
import pickle
import subprocess
import glob as glob_mod
from pathlib import Path

import hashlib
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
from spinhelicity_env import build_observation_from_expr
from train_sft_transformer import TransformerPolicySFT
from eval_sft_5pt import (
    count_brackets,
    count_terms,
    init_worker,
    shutdown_worker,
    apply_action_with_timeout,
    ActionTimeoutError,
    build_action_mask_5pt as build_action_mask,
)
from add_ons.numerical_evaluations import MOMENTA_DICTS
from eval_large_expr import (
    extract_num_denom,
)

FUNC_DICT = {'ab': ab, 'sb': sb}


def _decompose_cancelled_expr(expr):
    """Decompose an already-cancelled expression into (terms_num, denom).

    Like extract_num_denom but skips sp.cancel since the expression is
    already in cancelled form (e.g., from a worker that ran sp.cancel).
    """
    numerator, denominator = sp.fraction(expr)
    if isinstance(numerator, sp.Add):
        terms = np.asarray(numerator.args, dtype=object)
    else:
        terms = np.asarray([numerator], dtype=object)
    return terms, denominator


def expr_str_hash(expr_str):
    """Deterministic hash from expression string. Unlike Python's hash(),
    this is consistent across processes (no PYTHONHASHSEED dependency)."""
    return int.from_bytes(hashlib.md5(expr_str.encode()).digest()[:8], 'little')


# Condor settings (same as run_full_simplify_parallel.py)
PYTHON = sys.executable
WORKER_SCRIPT = str(BASE_DIR / 'src' / 'eval_grouping_worker.py')
PYTHONPATH = ':'.join([
    str(SHARED_DIR / 'spinorhelicity'),
    str(SHARED_DIR / 'spinorhelicity' / 'environment'),
    str(BASE_DIR / 'src'),
])


def submit_condor_jobs(n_groupings, groupings_dir, results_dir, logs_dir,
                       model_path, n_point, work_dir):
    """Generate and submit condor jobs for single-step grouping evaluation."""
    sub_file = os.path.join(work_dir, 'beam_step.sub')
    sub_content = f"""universe = vanilla
executable = {PYTHON}
arguments = -u {WORKER_SCRIPT} --grouping_file {groupings_dir}/grouping_$(Process).pkl --result_file {results_dir}/result_$(Process).pkl --model {model_path} --n_point {n_point} --single_step

output = {logs_dir}/worker_$(Process).out
error = {logs_dir}/worker_$(Process).err
log = {work_dir}/condor.log

environment = "PYTHONPATH={PYTHONPATH}"

request_cpus = 1
request_memory = 4GB
request_disk = 1GB

+IsGPUJob = false
+MaxRuntime = 600

queue {n_groupings}
"""
    with open(sub_file, 'w') as f:
        f.write(sub_content)

    result = subprocess.run(
        ['condor_submit', sub_file],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"    condor_submit FAILED: {result.stderr.strip()}", flush=True)
        return None

    for line in result.stdout.strip().split('\n'):
        if 'submitted to cluster' in line:
            cluster_id = line.split('cluster')[-1].strip().rstrip('.')
            return cluster_id
    return None


def wait_for_condor(cluster_id, n_groupings, results_dir, poll_interval=10):
    """Wait for all condor jobs to complete."""
    t0 = time.time()
    while True:
        n_results = len(glob_mod.glob(os.path.join(results_dir, 'result_*.pkl')))
        elapsed = time.time() - t0

        if n_results >= n_groupings:
            print(f"    All {n_groupings} workers done ({elapsed:.0f}s)", flush=True)
            return True

        try:
            result = subprocess.run(
                ['condor_q', cluster_id, '-totals'],
                capture_output=True, text=True, timeout=30
            )
            if 'Total for query: 0 jobs' in result.stdout:
                n_results = len(glob_mod.glob(os.path.join(results_dir, 'result_*.pkl')))
                print(f"    Cluster done, {n_results}/{n_groupings} results ({elapsed:.0f}s)", flush=True)
                return True
        except Exception:
            pass

        time.sleep(poll_interval)


def collect_results(n_groupings, results_dir):
    """Collect all result pickles."""
    results = []
    for i in range(n_groupings):
        pkl_path = os.path.join(results_dir, f'result_{i}.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                results.append(pickle.load(f))
    return results


def numerical_eval_5pt(expr, kinematics_idx=0):
    md = MOMENTA_DICTS[kinematics_idx]
    subs = {}
    for i in range(1, 6):
        for j in range(i + 1, 6):
            subs[ab(i, j)] = md[f'5ptab{i}{j}']
            subs[sb(i, j)] = md[f'5ptsb{i}{j}']
    return complex(expr.subs(subs).evalf())


def numerical_validate(old_expr, new_expr):
    max_rdiff = 0.0
    for ki in range(2):
        v_old = numerical_eval_5pt(old_expr, ki)
        v_new = numerical_eval_5pt(new_expr, ki)
        if abs(v_old) < 1e-15:
            rdiff = abs(v_new - v_old)
        else:
            rdiff = abs((v_new - v_old) / v_old)
        max_rdiff = max(max_rdiff, rdiff)
    return max_rdiff < 1e-9, max_rdiff


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


def expand_beam_candidate(model, expr, action_space, n_point, max_terms,
                          max_brackets_per_term, device, timeout_sec,
                          n_branches, visited_hashes, reject_term_increase=5):
    """Try top-N model actions on expr. Return list of (new_expr, action_info, log_prob)."""
    current_bk = count_brackets(expr)
    current_terms = count_terms(expr)

    try:
        obs = build_observation_from_expr(expr, n_point, max_terms, max_brackets_per_term)
    except Exception:
        return []

    action_mask = build_action_mask(expr, action_space)
    if action_mask.sum() == 0:
        return []

    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(obs_tensor)

    FLOAT_MIN = -3.4e38
    mask_tensor = torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0).to(device)
    inf_mask = torch.clamp(torch.log(mask_tensor + 1e-10), min=FLOAT_MIN)
    masked_logits = logits + inf_mask

    # Get log-probs for scoring
    log_probs = torch.log_softmax(masked_logits[0], dim=0).cpu().numpy()
    sorted_actions = torch.argsort(masked_logits[0], descending=True).cpu().numpy()

    results = []
    # Try more actions than n_branches to account for failures
    max_tries = min(n_branches * 4, 40)

    for action in sorted_actions[:max_tries]:
        if len(results) >= n_branches:
            break
        if action_mask[action] == 0:
            continue

        spec = action_space.get_action_spec(action)
        bracket_type, i, j = action_space.get_bracket_from_idx(spec.bracket_idx)

        try:
            result_expr = apply_action_with_timeout(
                expr, bracket_type, (i, j),
                spec.identity, spec.params, spec.term_slot,
                n_point, timeout_sec
            )

            if result_expr is None:
                continue

            new_terms = count_terms(result_expr)

            # Reject term explosion
            if reject_term_increase is not None:
                if new_terms - current_terms > reject_term_increase:
                    continue

            # Cycle check using deterministic expr_str hash
            new_hash = expr_str_hash(str(result_expr))
            if new_hash in visited_hashes:
                continue

            new_bk = count_brackets(result_expr)
            action_info = {
                'identity': spec.identity,
                'bracket': f"{bracket_type}({i},{j})",
                'term_slot': spec.term_slot,
                'terms_before': current_terms,
                'terms_after': new_terms,
                'bk_before': current_bk,
                'bk_after': new_bk,
            }

            results.append((result_expr, action_info, float(log_probs[action]), new_hash))

        except ActionTimeoutError:
            continue
        except Exception:
            continue

    return results


def expand_beam_grouped_condor(grouped_entries, beam, encoder_c, env_c,
                                thresholds, max_group_size, max_terms,
                                max_brackets_per_term, n_point,
                                reject_term_increase, visited_hashes,
                                model_path, output_dir, step):
    """Expand all grouped beam entries via two Condor steps.

    Step 1 (Encode): Farm contrastive encoding + grouping formation to Condor.
                     One worker per beam entry. Workers encode terms, compute
                     similarity, form groupings, and write grouping pickles.
    Step 2 (Simplify): One worker per grouping applies model action + recombination.

    encoder_c and env_c are unused (encoding is done on Condor workers).
    """
    # Per-step working directory
    step_dir = os.path.join(output_dir, 'condor', f'step_{step:04d}')
    encode_dir = os.path.join(step_dir, 'encode')
    groupings_dir = os.path.join(step_dir, 'groupings')
    results_dir = os.path.join(step_dir, 'results')
    logs_dir = os.path.join(step_dir, 'logs')
    for d in [encode_dir, groupings_dir, results_dir, logs_dir]:
        os.makedirs(d, exist_ok=True)

    # Clean old files from any previous attempt
    for d in [encode_dir, groupings_dir, results_dir]:
        for f_path in glob_mod.glob(os.path.join(d, '*.pkl')):
            os.remove(f_path)

    # ---- Step 1: Write encode task pickles and submit to Condor ----
    t_encode_phase = time.time()
    n_encode_jobs = len(grouped_entries)

    for local_idx, (beam_idx, entry) in enumerate(grouped_entries):
        # Get terms as strings (no sympy on login node)
        if entry.get('terms_num_str') is not None:
            terms_num_str = entry['terms_num_str']
            denom_str = entry.get('denom_str', '')
        elif entry.get('terms_num') is not None:
            terms_num_str = [str(t) for t in entry['terms_num']]
            denom_str = str(entry['denom'])
        else:
            # Fallback: need to extract (should be rare — only initial expr)
            terms_num, denom = extract_num_denom(entry['expr'])
            terms_num_str = [str(t) for t in terms_num]
            denom_str = str(denom)

        encode_task = {
            'beam_idx': beam_idx,
            'terms_num_str': terms_num_str,
            'denom_str': denom_str,
            'thresholds': thresholds,
            'max_group_size': max_group_size,
            'max_terms': max_terms,
            'max_brackets_per_term': max_brackets_per_term,
            'skip_n': entry.get('n_expanded', 0),
            'groupings_dir': groupings_dir,
            'step_dir': step_dir,
        }
        task_path = os.path.join(encode_dir, f'encode_task_{local_idx}.pkl')
        with open(task_path, 'wb') as f:
            pickle.dump(encode_task, f)

    # Submit encode Condor jobs
    encode_sub_file = os.path.join(step_dir, 'beam_encode.sub')
    encode_sub_content = f"""universe = vanilla
executable = {PYTHON}
arguments = -u {WORKER_SCRIPT} --encode_task {encode_dir}/encode_task_$(Process).pkl --encode_result {encode_dir}/result_$(Process).pkl --n_point {n_point}

output = {logs_dir}/encode_$(Process).out
error = {logs_dir}/encode_$(Process).err
log = {step_dir}/condor_encode.log

environment = "PYTHONPATH={PYTHONPATH}"

request_cpus = 1
request_memory = 4GB
request_disk = 1GB

+IsGPUJob = false
+MaxRuntime = 120

queue {n_encode_jobs}
"""
    with open(encode_sub_file, 'w') as f:
        f.write(encode_sub_content)

    result = subprocess.run(['condor_submit', encode_sub_file], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    condor_submit (encode) FAILED: {result.stderr.strip()}", flush=True)
        return []

    encode_cluster_id = None
    for line in result.stdout.strip().split('\n'):
        if 'submitted to cluster' in line:
            encode_cluster_id = line.split('cluster')[-1].strip().rstrip('.')
    if encode_cluster_id is None:
        return []

    print(f"    Encode: {n_encode_jobs} beam entries, cluster {encode_cluster_id}", flush=True)

    # Wait for encode workers
    wait_for_condor(encode_cluster_id, n_encode_jobs, encode_dir)
    t_encode_phase = time.time() - t_encode_phase

    # ---- Collect encode results, renumber groupings for simplify step ----
    t_collect_encode = time.time()

    # Read encode results to know how many groupings per beam entry
    encode_results = {}
    for local_idx in range(n_encode_jobs):
        result_path = os.path.join(encode_dir, f'result_{local_idx}.pkl')
        if os.path.exists(result_path):
            with open(result_path, 'rb') as f:
                encode_results[local_idx] = pickle.load(f)

    # Scan grouping files written by encode workers (grouping_{beam_idx}_{local_idx}.pkl)
    # Renumber to grouping_0.pkl, grouping_1.pkl, ... for simplify step
    global_idx = 0
    grouping_meta = {}

    for local_idx, (beam_idx, entry) in enumerate(grouped_entries):
        enc_result = encode_results.get(local_idx)
        if enc_result is None:
            print(f"    WARNING: encode result missing for beam_idx={beam_idx}", flush=True)
            continue

        n_grp = enc_result['n_groupings']

        for grp_local in range(n_grp):
            src_path = os.path.join(groupings_dir, f'grouping_{beam_idx}_{grp_local}.pkl')
            if not os.path.exists(src_path):
                print(f"    WARNING: grouping file missing: {src_path}", flush=True)
                continue

            # Read, update index, write to final path
            with open(src_path, 'rb') as f:
                grp_data = pickle.load(f)
            grp_data['grouping_idx'] = global_idx

            dst_path = os.path.join(groupings_dir, f'grouping_{global_idx}.pkl')
            with open(dst_path, 'wb') as f:
                pickle.dump(grp_data, f)

            # Remove original (always different name: grouping_X_Y vs grouping_N)
            if src_path != dst_path:
                os.remove(src_path)

            grouping_meta[global_idx] = {
                'beam_idx': beam_idx,
                'group_indices': grp_data['group_indices'],
                'full_terms': entry['terms'],
                'full_bk': entry['bk'],
                'entry': entry,
            }
            global_idx += 1

    n_groupings = global_idx
    t_collect_encode = time.time() - t_collect_encode

    if n_groupings == 0:
        return []

    print(f"    Encoded: {n_groupings} groupings across {len(grouped_entries)} beam entries "
          f"(encode={t_encode_phase:.1f}s, renumber={t_collect_encode:.1f}s)", flush=True)

    # ---- Step 2: Submit simplify Condor jobs ----
    cluster_id = submit_condor_jobs(
        n_groupings, groupings_dir, results_dir, logs_dir,
        model_path, n_point, step_dir,
    )
    if cluster_id is None:
        return []

    print(f"    Submitted simplify cluster {cluster_id}", flush=True)

    # Wait
    wait_for_condor(cluster_id, n_groupings, results_dir)

    # Collect results — workers already did recombination, no sp.cancel here
    _t_collect = time.time()
    results = collect_results(n_groupings, results_dir)
    _t_collect = time.time() - _t_collect
    n_recombined = sum(1 for r in results if r.get('recombined_expr_str') is not None)
    print(f"    Collected {len(results)} results, {n_recombined} with recombined expressions "
          f"(collect={_t_collect:.1f}s)",
          flush=True)

    candidates = []
    n_rejected_visited = 0
    for r in results:
        gidx = r['grouping_idx']
        if gidx not in grouping_meta:
            continue
        if r.get('action_info') is None:
            continue  # no valid action was found
        if r.get('recombined_expr_str') is None:
            continue  # worker didn't produce recombined result

        meta = grouping_meta[gidx]
        full_terms = meta['full_terms']
        full_bk = meta['full_bk']
        entry = meta['entry']
        group_indices = meta['group_indices']

        new_full_terms = r['recombined_terms']
        new_full_bk = r['recombined_bk']

        # Compute deterministic hash from expression string (not worker's obs_hash,
        # which uses Python's hash() and varies across processes due to PYTHONHASHSEED)
        new_hash = expr_str_hash(r['recombined_expr_str'])
        if new_hash in visited_hashes:
            n_rejected_visited += 1
            continue

        action_info = {
            'identity': r.get('action_info', {}).get('identity', '?') if r.get('action_info') else '?',
            'bracket': r.get('action_info', {}).get('bracket', '?') if r.get('action_info') else '?',
            'term_slot': r.get('action_info', {}).get('term_slot', -1) if r.get('action_info') else -1,
            'terms_before': full_terms,
            'terms_after': new_full_terms,
            'bk_before': full_bk,
            'bk_after': new_full_bk,
            'grouped': True,
            'group_size': len(group_indices),
            'sub_terms_before': r.get('sub_terms_before', 0),
            'sub_terms_after': r.get('sub_terms_after', 0),
        }

        # Defer ALL sympy parsing — only parse for the ~20 candidates that survive into beam
        candidates.append({
            'expr_str': r['recombined_expr_str'],
            'expr': None,  # lazy — parsed only if kept in beam
            'terms': new_full_terms,
            'bk': new_full_bk,
            'cum_log_prob': entry['cum_log_prob'],
            'actions': entry['actions'] + [action_info],
            'hash': new_hash,
            'parent_idx': -1,
            'terms_num': None,  # lazy — parsed only if kept in beam
            'denom': None,  # lazy
            'terms_num_str': r.get('terms_num_str'),
            'denom_str': r.get('denom_str'),
            'n_expanded': 0,
        })

    print(f"    {len(candidates)} valid candidates after collection"
          f" ({n_rejected_visited} rejected by visited_hashes)", flush=True)

    # Clean up step dir to reduce file I/O accumulation
    import shutil
    shutil.rmtree(step_dir, ignore_errors=True)

    return candidates, n_rejected_visited


def expand_beam_direct_condor(direct_entries, beam, max_terms, max_brackets_per_term,
                               n_point, n_branches, reject_term_increase, visited_hashes,
                               model_path, output_dir, step):
    """Expand direct (<=max_terms) beam entries via Condor workers.

    One Condor job per beam entry. Each worker tries top-N actions and returns
    all valid candidates. No grouping or recombination needed.
    """
    step_dir = os.path.join(output_dir, 'condor', f'step_{step:04d}')
    groupings_dir = os.path.join(step_dir, 'direct_groupings')
    results_dir = os.path.join(step_dir, 'direct_results')
    logs_dir = os.path.join(step_dir, 'direct_logs')
    for d in [groupings_dir, results_dir, logs_dir]:
        os.makedirs(d, exist_ok=True)

    # Clean old files
    for d in [groupings_dir, results_dir]:
        for f_path in glob_mod.glob(os.path.join(d, '*.pkl')):
            os.remove(f_path)

    # Write one pickle per beam entry
    entry_meta = {}
    for local_idx, (beam_idx, entry) in enumerate(direct_entries):
        pkl_data = {
            'beam_idx': beam_idx,
            'expr_str': str(entry['expr']),
            'n_branches': n_branches,
            'reject_term_increase': reject_term_increase,
            'skip_n': entry.get('n_expanded', 0),
        }
        pkl_path = os.path.join(groupings_dir, f'grouping_{local_idx}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(pkl_data, f)
        entry_meta[local_idx] = {'beam_idx': beam_idx, 'entry': entry}

    n_jobs = len(direct_entries)
    if n_jobs == 0:
        return []

    # Submit with --beam_expand flag
    sub_file = os.path.join(step_dir, 'beam_direct.sub')
    sub_content = f"""universe = vanilla
executable = {PYTHON}
arguments = -u {WORKER_SCRIPT} --grouping_file {groupings_dir}/grouping_$(Process).pkl --result_file {results_dir}/result_$(Process).pkl --model {model_path} --n_point {n_point} --beam_expand

output = {logs_dir}/direct_$(Process).out
error = {logs_dir}/direct_$(Process).err
log = {step_dir}/condor_direct.log

environment = "PYTHONPATH={PYTHONPATH}"

request_cpus = 1
request_memory = 4GB
request_disk = 1GB

+IsGPUJob = false
+MaxRuntime = 600

queue {n_jobs}
"""
    with open(sub_file, 'w') as f:
        f.write(sub_content)

    result = subprocess.run(['condor_submit', sub_file], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    condor_submit (direct) FAILED: {result.stderr.strip()}", flush=True)
        return []

    cluster_id = None
    for line in result.stdout.strip().split('\n'):
        if 'submitted to cluster' in line:
            cluster_id = line.split('cluster')[-1].strip().rstrip('.')
    if cluster_id is None:
        return []

    print(f"    Direct: {n_jobs} beam entries, cluster {cluster_id}", flush=True)

    # Wait
    wait_for_condor(cluster_id, n_jobs, results_dir)

    # Collect and flatten
    results = collect_results(n_jobs, results_dir)
    candidates = []
    n_rejected_visited = 0
    for r in results:
        beam_idx = r['beam_idx']
        meta = None
        for k, v in entry_meta.items():
            if v['beam_idx'] == beam_idx:
                meta = v
                break
        if meta is None:
            continue

        entry = meta['entry']
        for c in r.get('candidates', []):
            new_expr = sp.sympify(c['expr_str'], locals=FUNC_DICT)
            new_terms = c['terms']
            new_bk = c['bk']

            # Cycle check using deterministic expr_str hash
            new_hash = expr_str_hash(c['expr_str'])
            if new_hash in visited_hashes:
                n_rejected_visited += 1
                continue

            child_terms_num, child_denom = _decompose_cancelled_expr(new_expr)
            candidates.append({
                'expr': new_expr,
                'terms': new_terms,
                'bk': new_bk,
                'cum_log_prob': entry['cum_log_prob'] + c['log_prob'],
                'actions': entry['actions'] + [c['action_info']],
                'hash': new_hash,
                'parent_idx': beam_idx,
                'terms_num': child_terms_num,
                'denom': child_denom,
                'n_expanded': 0,
            })

    n_total = sum(len(r.get('candidates', [])) for r in results)
    print(f"    Collected {len(results)} results, {n_total} raw candidates, "
          f"{len(candidates)} after cycle check"
          f" ({n_rejected_visited} rejected by visited_hashes)", flush=True)

    # Clean up step dir to reduce file I/O accumulation
    import shutil
    shutil.rmtree(step_dir, ignore_errors=True)

    return candidates, n_rejected_visited


def beam_search(model, initial_expr, action_space, n_point, cfg, device,
                beam_width=20, n_branches=5, max_steps=200,
                timeout_sec=30.0, reject_term_increase=5, output_dir=None,
                encoder_c=None, env_c=None,
                group_thresholds=(0.6, 0.5, 0.4), max_group_size=25,
                model_path=None, patience=10, grouped_only=False,
                debug=False, resume_step=None):
    """Run beam search on expression.

    Direct entries (<=max_terms) are expanded locally.
    Grouped entries (>max_terms) are expanded via Condor workers.
    """

    max_terms = cfg['max_terms']
    max_brackets_per_term = cfg['max_brackets_per_term']

    initial_bk = count_brackets(initial_expr)
    initial_terms = count_terms(initial_expr)

    print(f"\nBeam search: width={beam_width}, branches={n_branches}, "
          f"max_steps={max_steps}, RTI={reject_term_increase}, patience={patience}",
          flush=True)
    print(f"Start: {initial_terms} terms, {initial_bk} brackets", flush=True)

    # Each beam entry: (expr, terms, brackets, cumulative_log_prob, action_history, hash)
    init_hash = expr_str_hash(str(initial_expr))

    # Cache terms_num/denom for grouped entries to avoid extract_num_denom
    init_terms_num, init_denom = extract_num_denom(initial_expr)

    beam = [{
        'expr': initial_expr,
        'terms': initial_terms,
        'bk': initial_bk,
        'cum_log_prob': 0.0,
        'actions': [],
        'hash': init_hash,
        'terms_num': init_terms_num,
        'denom': init_denom,
        'n_expanded': 0,
    }]

    # Global visited set to prevent cycles across the entire beam
    visited_hashes = set()
    if init_hash is not None:
        visited_hashes.add(init_hash)

    best_terms = initial_terms
    best_bk = initial_bk
    best_expr = initial_expr
    best_actions = []
    steps_since_improvement = 0

    checkpoints_dir = None
    if output_dir:
        checkpoints_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)

    start_step = 0

    # Resume from checkpoint if requested
    if resume_step is not None and checkpoints_dir:
        ckpt_path = os.path.join(checkpoints_dir, f'beam_step_{resume_step:04d}.pkl')
        print(f"Resuming from checkpoint: {ckpt_path}", flush=True)
        with open(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)
        # Restore beam
        beam = []
        for bs in ckpt['beam']:
            entry = {
                'expr': sp.sympify(bs['expr_str'], locals=FUNC_DICT),
                'expr_str': bs['expr_str'],
                'terms': bs['terms'],
                'bk': bs['bk'],
                'hash': bs['hash'],
                'n_expanded': bs.get('n_expanded', 0),
                'cum_log_prob': bs.get('cum_log_prob', 0.0),
                'actions': bs.get('actions', []),
            }
            if bs.get('terms_num_str') is not None:
                entry['terms_num'] = [sp.sympify(s, locals=FUNC_DICT)
                                      for s in bs['terms_num_str']]
                entry['denom'] = sp.sympify(bs['denom_str'], locals=FUNC_DICT)
                entry['terms_num_str'] = bs['terms_num_str']
                entry['denom_str'] = bs['denom_str']
            else:
                entry['terms_num'], entry['denom'] = extract_num_denom(entry['expr'])
            beam.append(entry)
        # Restore visited hashes and best tracking
        visited_hashes = ckpt['visited_hashes']
        best_terms = ckpt['best_terms']
        best_bk = ckpt['best_bk']
        best_expr = sp.sympify(ckpt['best_expr_str'], locals=FUNC_DICT)
        steps_since_improvement = ckpt['steps_since_improvement']
        start_step = resume_step + 1
        print(f"Resumed: step={start_step}, beam_size={len(beam)}, "
              f"best={best_terms}t/{best_bk}bk, "
              f"visited_hashes={len(visited_hashes)}, "
              f"patience={steps_since_improvement}/{patience}",
              flush=True)

    t0 = time.time()

    for step in range(start_step, max_steps):
        t_step = time.time()

        # Save beam state before expansion (for debug logging)
        if debug:
            prev_beam_state = beam

        # Track candidates from each path for debug
        direct_candidates = []
        condor_candidates = []

        # Separate beam into direct and grouped entries
        if grouped_only:
            # Keep using contrastive groupings even below max_terms
            direct_entries = [(i, e) for i, e in enumerate(beam) if e['terms'] < 3]
            grouped_entries = [(i, e) for i, e in enumerate(beam) if e['terms'] >= 3]
        else:
            direct_entries = [(i, e) for i, e in enumerate(beam) if e['terms'] <= max_terms]
            grouped_entries = [(i, e) for i, e in enumerate(beam)
                              if e['terms'] > max_terms]

        all_candidates = []
        n_rejected_direct = 0
        n_rejected_grouped = 0

        # Direct entries: expand via Condor (one job per beam entry)
        if direct_entries and output_dir and model_path:
            direct_candidates, n_rejected_direct = expand_beam_direct_condor(
                direct_entries, beam, max_terms, max_brackets_per_term,
                n_point, n_branches, reject_term_increase, visited_hashes,
                model_path, output_dir, step,
            )
            all_candidates.extend(direct_candidates)

        # Grouped entries: expand via Condor (one job per grouping)
        if grouped_entries and output_dir and model_path:
            condor_candidates, n_rejected_grouped = expand_beam_grouped_condor(
                grouped_entries, beam, encoder_c, env_c,
                list(group_thresholds), max_group_size, max_terms,
                max_brackets_per_term, n_point,
                reject_term_increase, visited_hashes,
                model_path, output_dir, step,
            )
            all_candidates.extend(condor_candidates)

        # Track which beam indices were expanded this step
        expanded_indices = set(i for i, _ in direct_entries) | set(i for i, _ in grouped_entries)
        direct_max_tries = min(n_branches * 4, 40)

        # Always carry parents forward so we never lose the current best
        # Increment n_expanded for entries that were expanded this step
        for i, entry in enumerate(beam):
            n_exp = entry.get('n_expanded', 0)
            if i in expanded_indices:
                n_exp += direct_max_tries
            all_candidates.append({
                'expr': entry['expr'],
                'expr_str': entry.get('expr_str'),
                'terms': entry['terms'],
                'bk': entry['bk'],
                'cum_log_prob': entry['cum_log_prob'],
                'actions': entry['actions'],
                'hash': entry['hash'],
                'parent_idx': -1,
                'n_expanded': n_exp,
                'terms_num': entry.get('terms_num'),
                'denom': entry.get('denom'),
                'terms_num_str': entry.get('terms_num_str'),
                'denom_str': entry.get('denom_str'),
                '_is_parent': True,
            })

        if len(all_candidates) == len(beam):
            # Only parents, no children — truly stuck
            print(f"  Step {step}: no valid expansions, stopping.", flush=True)
            break

        # Score: primary = fewer terms, secondary = fewer brackets
        all_candidates.sort(key=lambda c: (c['terms'], c['bk']))

        # Deduplicate by hash, keeping best score
        seen_hashes_this_step = set()
        deduped = []
        for c in all_candidates:
            if c['hash'] is not None:
                if c['hash'] in seen_hashes_this_step:
                    continue
                seen_hashes_this_step.add(c['hash'])
                visited_hashes.add(c['hash'])
            deduped.append(c)

        # Keep top beam_width
        beam = deduped[:beam_width]

        # Materialize lazy expressions for beam entries that survived
        _t_mat = time.time()
        _n_mat_expr = 0
        _n_mat_terms = 0
        for entry in beam:
            if entry.get('expr') is None and entry.get('expr_str') is not None:
                entry['expr'] = sp.sympify(entry['expr_str'], locals=FUNC_DICT)
                _n_mat_expr += 1
            if entry.get('terms_num') is None and entry.get('terms_num_str') is not None:
                entry['terms_num'] = [sp.sympify(s, locals=FUNC_DICT) for s in entry['terms_num_str']]
                entry['denom'] = sp.sympify(entry['denom_str'], locals=FUNC_DICT)
                _n_mat_terms += 1
        _t_mat = time.time() - _t_mat
        if _n_mat_expr > 0 or _n_mat_terms > 0:
            print(f"    Materialized {_n_mat_expr} expr + {_n_mat_terms} terms_num ({_t_mat:.1f}s)",
                  flush=True)

        # Stats
        terms_list = [c['terms'] for c in beam]
        bk_list = [c['bk'] for c in beam]
        step_time = time.time() - t_step

        # Track best
        top = beam[0]
        new_best = False
        if top['terms'] < best_terms or (top['terms'] == best_terms and top['bk'] < best_bk):
            best_terms = top['terms']
            best_bk = top['bk']
            best_expr = top['expr']
            best_actions = top['actions']
            new_best = True

        if new_best:
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1

        best_str = " ***NEW BEST***" if new_best else ""
        print(f"  Step {step}: beam terms=[{min(terms_list)},{max(terms_list)}] "
              f"bk=[{min(bk_list)},{max(bk_list)}] "
              f"({len(all_candidates)} expanded, {len(deduped)} unique, "
              f"kept {len(beam)}) {step_time:.1f}s{best_str}",
              flush=True)

        # Debug logging
        if debug and output_dir:
            debug_record = {
                'step': step,
                'beam_before': [
                    {'hash': str(e['hash'])[:16] if e['hash'] is not None else None,
                     'terms': e['terms'],
                     'bk': e['bk'],
                     'n_expanded': e.get('n_expanded', 0)}
                    for e in prev_beam_state
                ],
                'n_candidates_direct': len(direct_candidates) if direct_entries else 0,
                'n_candidates_grouped': len(condor_candidates) if grouped_entries else 0,
                'n_rejected_visited_direct': n_rejected_direct,
                'n_rejected_visited_grouped': n_rejected_grouped,
                'n_total_visited_hashes': len(visited_hashes),
                'n_before_dedup': len(all_candidates),
                'n_after_dedup': len(deduped),
                'n_parents_in_beam': sum(1 for e in beam if e.get('_is_parent', False)),
                'n_children_in_beam': sum(1 for e in beam if not e.get('_is_parent', False)),
                'beam_after': [
                    {'hash': str(e['hash'])[:16] if e['hash'] is not None else None,
                     'terms': e['terms'],
                     'bk': e['bk'],
                     'n_expanded': e.get('n_expanded', 0),
                     'is_parent': e.get('_is_parent', False)}
                    for e in beam
                ],
                'best_terms': best_terms,
                'best_bk': best_bk,
                'steps_since_improvement': steps_since_improvement,
                'step_time': round(step_time, 1),
            }
            debug_log_path = os.path.join(output_dir, 'debug_log.jsonl')
            with open(debug_log_path, 'a') as f:
                f.write(json.dumps(debug_record, default=str) + '\n')

        # Save per-step checkpoint (if debug mode, save every step for restart)
        if debug and checkpoints_dir:
            beam_state = []
            for e in beam:
                bs = {
                    'expr_str': str(e['expr']) if e.get('expr') is not None else e.get('expr_str'),
                    'terms': e['terms'],
                    'bk': e['bk'],
                    'hash': e['hash'],
                    'n_expanded': e.get('n_expanded', 0),
                    'cum_log_prob': e['cum_log_prob'],
                    'actions': e['actions'],
                }
                if e.get('terms_num_str') is not None:
                    bs['terms_num_str'] = e['terms_num_str']
                    bs['denom_str'] = e.get('denom_str')
                elif e.get('terms_num') is not None:
                    bs['terms_num_str'] = [str(t) for t in e['terms_num']]
                    bs['denom_str'] = str(e['denom'])
                beam_state.append(bs)
            ckpt = {
                'step': step,
                'beam': beam_state,
                'visited_hashes': visited_hashes,
                'best_terms': best_terms,
                'best_bk': best_bk,
                'best_expr_str': str(best_expr),
                'steps_since_improvement': steps_since_improvement,
            }
            ckpt_path = os.path.join(checkpoints_dir, f'beam_step_{step:04d}.pkl')
            with open(ckpt_path, 'wb') as f:
                pickle.dump(ckpt, f)

        # Early stop if no improvement for `patience` steps
        if patience > 0 and steps_since_improvement >= patience:
            print(f"  No improvement for {patience} steps, stopping.", flush=True)
            break

        # Numerical validation on new best
        if new_best:
            is_valid, max_rdiff = numerical_validate(initial_expr, best_expr)
            valid_str = "PASS" if is_valid else "FAIL"
            print(f"    Numerical check: {valid_str} (max rel diff = {max_rdiff:.2e})",
                  flush=True)

        # Save checkpoint on improvement (always, not just debug)
        if new_best and checkpoints_dir:
            with open(os.path.join(checkpoints_dir, f'step_{step:04d}.txt'), 'w') as f:
                f.write(str(best_expr))
            with open(os.path.join(checkpoints_dir, f'step_{step:04d}_info.json'), 'w') as f:
                json.dump({
                    'step': step,
                    'terms': best_terms,
                    'bk': best_bk,
                    'n_actions': len(best_actions),
                    'last_action': best_actions[-1] if best_actions else None,
                }, f, indent=2, default=str)

        # Check for solved
        if best_terms <= 1:
            print(f"  SOLVED at step {step}! 1 term = Parke-Taylor form.", flush=True)
            break

    total_time = time.time() - t0

    # Numerical validation
    is_valid, max_rdiff = numerical_validate(initial_expr, best_expr)
    valid_str = "PASS" if is_valid else "FAIL"
    print(f"\nNumerical validation: {valid_str} (max rel diff = {max_rdiff:.2e})", flush=True)

    print(f"\nBeam search result: {initial_terms} -> {best_terms} terms, "
          f"{initial_bk} -> {best_bk} bk ({total_time:.1f}s, {step+1} steps)", flush=True)

    return best_expr, best_terms, best_bk, best_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--n_point', type=int, default=5)
    parser.add_argument('--model', required=True)
    parser.add_argument('--beam_width', type=int, default=20)
    parser.add_argument('--branches', type=int, default=5)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--timeout', type=float, default=30.0)
    parser.add_argument('--reject_term_increase', type=int, default=5)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--patience', type=int, default=10,
                        help='Stop after this many steps with no improvement (0=disabled)')
    parser.add_argument('--grouped_only', action='store_true',
                        help='Use contrastive groupings for all terms (no 25-term threshold)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable detailed per-step debug logging to debug_log.jsonl')
    parser.add_argument('--resume_step', type=int, default=None,
                        help='Resume from checkpoint at this step number')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load expression
    print(f"Loading expression from {args.input}", flush=True)
    with open(args.input, 'r') as f:
        expr_str = f.read().strip()
    full_expr = sp.sympify(expr_str, locals=FUNC_DICT)
    full_expr = sp.cancel(full_expr)

    # Setup
    device = 'cpu'
    cfg = MODEL_CONFIG[args.n_point]

    from action_space_5pt import TermSpecificActionSpace5pt
    action_space = TermSpecificActionSpace5pt(
        n_point=args.n_point,
        max_terms=cfg['max_terms_action'],
        max_brackets=cfg['max_brackets'],
    )
    print(f"Action space: {action_space.n_actions} total actions", flush=True)

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

    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded (epoch {checkpoint.get('epoch', '?')})", flush=True)

    # Init worker for apply_action_with_timeout
    init_worker(args.n_point, max_terms=cfg['max_terms_action'])

    # Contrastive encoding is done on Condor workers — no need to load on login node
    initial_terms = count_terms(full_expr)
    encoder_c, env_c = None, None

    output_dir = args.output
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        best_expr, best_terms, best_bk, best_actions = beam_search(
            model, full_expr, action_space, args.n_point, cfg, device,
            beam_width=args.beam_width,
            n_branches=args.branches,
            max_steps=args.max_steps,
            timeout_sec=args.timeout,
            reject_term_increase=args.reject_term_increase,
            output_dir=output_dir,
            encoder_c=encoder_c,
            env_c=env_c,
            model_path=args.model,
            patience=args.patience,
            grouped_only=args.grouped_only,
            debug=args.debug,
            resume_step=args.resume_step,
        )
    finally:
        shutdown_worker()

    # Save final
    if output_dir:
        final_path = os.path.join(output_dir, 'final_expression.txt')
        with open(final_path, 'w') as f:
            f.write(str(best_expr))
        print(f"Saved to {final_path}", flush=True)


if __name__ == '__main__':
    main()
