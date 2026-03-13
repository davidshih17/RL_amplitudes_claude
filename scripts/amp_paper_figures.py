#!/usr/bin/env python
"""
Generate amplitude paper figures:
  1. Solve rate vs source bracket count (for 4pt, 5pt, 6pt)
  2. Solve rate vs number of target terms (for 4pt, 5pt, 6pt)

First run computes data from raw files and saves a cache.
Subsequent runs load the cache and only remake plots (instant).
To force recomputation, delete the cache file.
"""

import re
import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
CACHE_PATH = os.path.join(SCRIPT_DIR, 'amp_paper_figures_cache.npz')
FIG_DIR = os.path.join(REPO_ROOT, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ============================================================
# Try to load from cache
# ============================================================
if os.path.exists(CACHE_PATH):
    print(f"Loading cached data from {CACHE_PATH}")
    cache = np.load(CACHE_PATH)
    merged = {}
    for npt in [4, 5, 6]:
        merged[npt] = {
            key: cache[f'{npt}_{key}']
            for key in ['tgt_terms', 'src_bk', 'ours_srcrel', 'ours_tgtrel', 'cheung_srcrel', 'cheung_tgtrel']
        }
        print(f"  {npt}pt: {len(merged[npt]['src_bk'])} samples")
else:
    # ============================================================
    # No cache — compute everything from scratch
    # ============================================================
    print("No cache found, computing from raw data...")

    sys.path.insert(0, os.path.join(REPO_ROOT, 'spinorhelicity'))
    import sympy as sp

    def count_brackets_sympy(expr_str):
        if expr_str.strip() == '0':
            return 0
        try:
            expr = sp.sympify(expr_str)
            count = 0
            bracket_re = re.compile(r'^(ab|sb)\d+$')
            def count_in_expr(e):
                nonlocal count
                if isinstance(e, sp.Symbol) and bracket_re.match(str(e)):
                    count += 1
                elif isinstance(e, sp.Pow):
                    base, exp = e.args
                    if isinstance(base, sp.Symbol) and bracket_re.match(str(base)):
                        try: count += int(abs(exp))
                        except: count += 1
                    else: count_in_expr(base)
                elif isinstance(e, (sp.Mul, sp.Add)):
                    for arg in e.args: count_in_expr(arg)
            numer, denom = sp.fraction(expr)
            count_in_expr(numer)
            count_in_expr(denom)
            return count
        except:
            total = 0
            for match in re.finditer(r'(ab|sb)\d+', expr_str):
                total += 1
            for match in re.finditer(r'(ab|sb)\d+\*\*(\d+)', expr_str):
                total += int(match.group(2)) - 1
            return total

    def count_terms(expr_str):
        if expr_str.strip() == '0':
            return 0

        try:
            expr = sp.sympify(expr_str)
            numer, denom = sp.fraction(expr)
            numer_expanded = sp.expand(numer)
            terms = sp.Add.make_args(numer_expanded)
            return len(terms)
        except:
            if '/' in expr_str:
                numer = expr_str.split('/')[0]
            else:
                numer = expr_str
            numer = numer.strip()
            if numer.startswith('(') and numer.endswith(')'):
                numer = numer[1:-1]
            return numer.count('+') + numer.count(' - ') + 1

    def parse_eval_file_full(filepath):
        equations = []
        current_eq = None
        with open(filepath, 'r') as f:
            for line in f:
                line = line.rstrip('\n')
                eq_match = re.match(r'Equation (\d+) \((\d+)/(\d+)\)', line)
                if eq_match:
                    if current_eq is not None:
                        equations.append(current_eq)
                    current_eq = {
                        'idx': int(eq_match.group(1)),
                        'n_valid': int(eq_match.group(2)),
                        'n_hyps': int(eq_match.group(3)),
                        'src': None, 'tgt': None, 'hyps': [],
                    }
                    continue
                if current_eq is None:
                    continue
                if line.startswith('src='):
                    current_eq['src'] = line[4:]
                    continue
                if line.startswith('tgt='):
                    current_eq['tgt'] = line[4:]
                    continue
                hyp_match = re.match(r'^([01])\s+(?:(-?\d+\.\d+e[+-]\d+)\s+)?(.+)$', line)
                if hyp_match:
                    current_eq['hyps'].append({
                        'valid': bool(int(hyp_match.group(1))),
                        'score': float(hyp_match.group(2)) if hyp_match.group(2) else None,
                        'expr': hyp_match.group(3),
                    })
        if current_eq is not None:
            equations.append(current_eq)
        return equations

    def parse_our_model_worker_out(out_path, worker_id, n_workers, total_samples):
        """Parse v2 format worker output."""
        remainder = total_samples % n_workers
        base_samples = total_samples // n_workers
        if worker_id < remainder:
            start_idx = worker_id * (base_samples + 1)
        else:
            start_idx = remainder * (base_samples + 1) + (worker_id - remainder) * base_samples
        results = []
        if not os.path.exists(out_path):
            return results
        with open(out_path, 'r') as f:
            content = f.read()
        # v2 format patterns
        success_pattern = re.compile(
            r'\[Sample (\d+)\] SUCCESS: (\d+)->(\d+) bk, (\d+)->(\d+) tm '
            r'\(target: (\d+) bk, (\d+) tm\)')
        partial_pattern = re.compile(
            r'\[Sample (\d+)\] PARTIAL \(source-rel\): (\d+)->(\d+) bk, (\d+)->(\d+) tm '
            r'\(target: (\d+) bk, (\d+) tm\)')
        fail_pattern = re.compile(
            r'\[Sample (\d+)\] FAILED: (\d+)->(\d+) bk, (\d+)->(\d+) tm '
            r'\(target: (\d+) bk, (\d+) tm\)')
        for m in success_pattern.finditer(content):
            local_idx = int(m.group(1))
            global_idx = start_idx + local_idx
            results.append({
                'global_idx': global_idx,
                'start_brackets': int(m.group(2)),
                'min_brackets': int(m.group(3)),
                'target_brackets': int(m.group(6)),
                'final_terms': int(m.group(5)),
                'target_terms': int(m.group(7)),
                'success_proper': True,
                'success_loose': True,
            })
        for m in partial_pattern.finditer(content):
            local_idx = int(m.group(1))
            global_idx = start_idx + local_idx
            results.append({
                'global_idx': global_idx,
                'start_brackets': int(m.group(2)),
                'min_brackets': int(m.group(3)),
                'target_brackets': int(m.group(6)),
                'final_terms': int(m.group(5)),
                'target_terms': int(m.group(7)),
                'success_proper': False,
                'success_loose': True,
            })
        for m in fail_pattern.finditer(content):
            local_idx = int(m.group(1))
            global_idx = start_idx + local_idx
            results.append({
                'global_idx': global_idx,
                'start_brackets': int(m.group(2)),
                'min_brackets': int(m.group(3)),
                'target_brackets': int(m.group(6)),
                'final_terms': int(m.group(5)),
                'target_terms': int(m.group(7)),
                'success_proper': False,
                'success_loose': False,
            })
        return results

    EVAL_BASE = os.path.join(REPO_ROOT, 'results', 'paper_model_eval')
    SCRATCH_RTI = os.path.join(REPO_ROOT, 'results', 'rti_study')
    EVAL_PATHS_B20 = {
        4: EVAL_BASE + "/eval_paper_4pt_beam20/beam20_4pt/eval.spin_hel.test.0",
        5: EVAL_BASE + "/eval_paper_5pt_beam20/beam20_5pt/eval.spin_hel.test.0",
        6: EVAL_BASE + "/eval_paper_6pt_beam20/beam20_6pt/eval.spin_hel.test.0",
    }
    NPT_INFO = {
        4: {"test_size": 10020, "n_workers": 1000},
        5: {"test_size": 10270, "n_workers": 1000},
        6: {"test_size": 9934, "n_workers": 1000},
    }

    # Step 1: Cheung B=20
    cheung_data = {}
    for npt in [4, 5, 6]:
        print(f"  Parsing {npt}pt Cheung data...", flush=True)
        t0 = time.time()
        equations = parse_eval_file_full(EVAL_PATHS_B20[npt])
        npt_data = {}
        for eq in equations:
            idx = eq['idx']
            src_bk = count_brackets_sympy(eq['src'])
            tgt_bk = count_brackets_sympy(eq['tgt'])
            src_t = count_terms(eq['src'])
            tgt_t = count_terms(eq['tgt'])
            valid_hyps = [h for h in eq['hyps'] if h['valid']]
            srcrel = False
            tgtrel = False
            if valid_hyps:
                for h in valid_hyps:
                    hyp_bk = count_brackets_sympy(h['expr'])
                    hyp_t = count_terms(h['expr'])
                    # Source-rel: 2D criterion (same as our model)
                    if not srcrel:
                        if (hyp_bk < src_bk and hyp_t <= src_t) or \
                           (hyp_bk <= src_bk and hyp_t < src_t):
                            srcrel = True
                    # Target-rel: any valid hyp with bk <= tgt AND tm <= tgt
                    if not tgtrel:
                        if hyp_bk <= tgt_bk and hyp_t <= tgt_t:
                            tgtrel = True
                    if srcrel and tgtrel:
                        break
            # Pathological = target does NOT satisfy source-rel relative to source
            is_src_rel = (tgt_bk < src_bk and tgt_t <= src_t) or \
                         (tgt_bk <= src_bk and tgt_t < src_t)
            npt_data[idx] = {
                'src_bk': src_bk, 'tgt_bk': tgt_bk, 'tgt_terms': tgt_t,
                'cheung_srcrel': srcrel, 'cheung_tgtrel': tgtrel,
                'pathological': not is_src_rel,
            }
        cheung_data[npt] = npt_data
        print(f"    {len(equations)} eqs, {time.time()-t0:.0f}s", flush=True)

    # Step 2: Our model
    our_data = {}
    for npt in [4, 5, 6]:
        total_samples = NPT_INFO[npt]["test_size"]
        n_workers = NPT_INFO[npt]["n_workers"]
        out_dir = os.path.join(SCRATCH_RTI, f"{npt}pt_rti1_v2")
        all_results = {}
        for wid in range(n_workers):
            out_path = os.path.join(out_dir, f"{wid}.out")
            worker_results = parse_our_model_worker_out(out_path, wid, n_workers, total_samples)
            for r in worker_results:
                all_results[r['global_idx']] = r
        our_data[npt] = all_results
        print(f"  {npt}pt: loaded {len(all_results)} / {total_samples}", flush=True)

    # Train/test overlap indices (state-only matches)
    overlap = {
        4: {0, 1, 24, 28, 48, 61, 64, 69, 82, 94, 106, 119, 684, 1161,
            2647, 2706, 2900, 2923, 3275, 3301, 3369, 4144, 4402, 4681,
            5798, 6120, 6911, 7279},
        5: {6, 22, 25, 27, 49, 71, 298, 791, 7070},
        6: set(),
    }

    # Step 3: Merge
    merged = {}
    for npt in [4, 5, 6]:
        cd = cheung_data[npt]
        od = our_data[npt]
        common = sorted(set(cd.keys()) & set(od.keys()))
        clean = [i for i in common if not cd[i]['pathological'] and i not in overlap[npt]]
        merged[npt] = {
            'tgt_terms': np.array([cd[i]['tgt_terms'] for i in clean]),
            'src_bk': np.array([od[i]['start_brackets'] for i in clean]),
            'ours_tgtrel': np.array([od[i]['success_proper'] for i in clean], dtype=bool),
            'ours_srcrel': np.array([od[i]['success_proper'] or od[i]['success_loose'] for i in clean], dtype=bool),
            'cheung_srcrel': np.array([cd[i]['cheung_srcrel'] for i in clean], dtype=bool),
            'cheung_tgtrel': np.array([cd[i]['cheung_tgtrel'] for i in clean], dtype=bool),
        }
        print(f"  {npt}pt: {len(clean)} clean samples", flush=True)

    # Save cache
    cache_data = {}
    for npt in [4, 5, 6]:
        for key in ['tgt_terms', 'src_bk', 'ours_srcrel', 'ours_tgtrel', 'cheung_srcrel', 'cheung_tgtrel']:
            cache_data[f'{npt}_{key}'] = merged[npt][key]
    np.savez(CACHE_PATH, **cache_data)
    print(f"Saved cache to {CACHE_PATH}")


# ============================================================
# PLOTTING (runs every time, instant from cache)
# ============================================================

def equal_bin_data(bk_values, bin_width):
    """Create equal-width bins for bracket counts."""
    bk_min = int(np.min(bk_values))
    bk_max = int(np.max(bk_values))
    result = []
    start = bk_min
    while start <= bk_max:
        end = min(start + bin_width - 1, bk_max)
        bin_vals = list(range(start, end + 1))
        mask = np.isin(bk_values, bin_vals)
        n = mask.sum()
        center = (start + end) / 2.0
        label = str(start) if start == end else f"{start}-{end}"
        if n > 0:
            result.append((center, label, mask, n))
        start += bin_width
    return result

MIN_SAMPLES_PER_BIN = 10  # Skip bins with fewer samples to avoid noisy spikes

def log_bin_data(bk_values, n_bins=15):
    """Create logarithmically-spaced bins for bracket counts.
    Returns (geo_center, mask, n) for each bin with n > 0."""
    bk_min = int(np.min(bk_values))
    bk_max = int(np.max(bk_values))
    edges = np.logspace(np.log10(bk_min), np.log10(bk_max + 1), n_bins + 1)
    edges = np.unique(np.round(edges).astype(int))
    result = []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1] - 1
        mask = (bk_values >= lo) & (bk_values <= hi)
        n = mask.sum()
        geo_center = np.sqrt(lo * max(hi, lo))  # geometric mean for log-scale position
        if n > 0:
            result.append((geo_center, mask, n))
    return result

# --- Figure 1: Solve rate vs source bracket count ---
print("\nPlotting solve rate vs source bracket count")
from matplotlib.ticker import ScalarFormatter
for npt in [4, 5, 6]:
    m = merged[npt]
    bk = m['src_bk']
    bin_data = [x for x in log_bin_data(bk, n_bins=20) if x[2] >= MIN_SAMPLES_PER_BIN]
    print(f"  {npt}pt: {len(bin_data)} log bins, range={int(np.min(bk))}-{int(np.max(bk))}")

    bin_centers, bin_counts = [], []
    rates = {k: [] for k in ['ours_srcrel', 'ours_tgtrel', 'cheung_srcrel', 'cheung_tgtrel']}
    errs = {k: [] for k in rates}

    for center, mask, n in bin_data:
        bin_centers.append(center)
        bin_counts.append(n)
        for k in rates:
            p = m[k][mask].mean()
            rates[k].append(p * 100)
            errs[k].append(np.sqrt(p * (1 - p) / n) * 100)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(bin_centers, rates['ours_tgtrel'], yerr=errs['ours_tgtrel'],
                fmt='b-o', markersize=4, capsize=2, label='Ours, target-rel')
    ax.errorbar(bin_centers, rates['ours_srcrel'], yerr=errs['ours_srcrel'],
                fmt='b--s', markersize=4, capsize=2, alpha=0.7, label='Ours, source-rel')
    ax.errorbar(bin_centers, rates['cheung_tgtrel'], yerr=errs['cheung_tgtrel'],
                fmt='r-o', markersize=4, capsize=2, label=f'CDS ($B$=20), target-rel')
    ax.errorbar(bin_centers, rates['cheung_srcrel'], yerr=errs['cheung_srcrel'],
                fmt='r--s', markersize=4, capsize=2, alpha=0.7, label=f'CDS ($B$=20), source-rel')

    ax.set_xscale('log')
    ax.set_xlabel('Source bracket count', fontsize=12)
    ax.set_ylabel('Solve rate (%)', fontsize=12)
    ax.set_title(f'{npt}-point amplitudes', fontsize=13)
    # Clean integer tick labels on log axis
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.legend(fontsize=9, loc='lower left')
    ax.set_ylim([70, 101])
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    # Bar widths proportional to bin width in log space
    bar_widths = []
    for i, c in enumerate(bin_centers):
        if i == 0:
            w = bin_centers[1] / bin_centers[0] if len(bin_centers) > 1 else 2
        else:
            w = bin_centers[i] / bin_centers[i-1]
        bar_widths.append(c * (w**0.4 - w**(-0.4)))
    ax2.bar(bin_centers, bin_counts, alpha=0.1, color='gray', width=bar_widths)
    ax2.set_ylabel('Number of samples', fontsize=10, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    plt.tight_layout()
    outpath = os.path.join(FIG_DIR, f'amp_solve_rate_vs_brackets_{npt}pt.pdf')
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath}")

# --- Figure 2: Solve rate vs target terms ---
print("\nPlotting solve rate vs target terms")
for npt in [4, 5, 6]:
    m = merged[npt]
    term_vals = sorted(np.unique(m['tgt_terms']))

    sample_counts = []
    rates = {k: [] for k in ['ours_srcrel', 'ours_tgtrel', 'cheung_srcrel', 'cheung_tgtrel']}
    errs = {k: [] for k in rates}

    for t in term_vals:
        mask = m['tgt_terms'] == t
        n = mask.sum()
        sample_counts.append(n)
        for k in rates:
            p = m[k][mask].mean()
            rates[k].append(p * 100)
            errs[k].append(np.sqrt(p * (1 - p) / n) * 100)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(term_vals, rates['ours_tgtrel'], yerr=errs['ours_tgtrel'],
                fmt='b-o', markersize=4, capsize=2, label='Ours, target-rel')
    ax.errorbar(term_vals, rates['ours_srcrel'], yerr=errs['ours_srcrel'],
                fmt='b--s', markersize=4, capsize=2, alpha=0.7, label='Ours, source-rel')
    ax.errorbar(term_vals, rates['cheung_tgtrel'], yerr=errs['cheung_tgtrel'],
                fmt='r-o', markersize=4, capsize=2, label=f'CDS ($B$=20), target-rel')
    ax.errorbar(term_vals, rates['cheung_srcrel'], yerr=errs['cheung_srcrel'],
                fmt='r--s', markersize=4, capsize=2, alpha=0.7, label=f'CDS ($B$=20), source-rel')

    ax.set_xlabel('Number of target terms', fontsize=12)
    ax.set_ylabel('Solve rate (%)', fontsize=12)
    ax.set_title(f'{npt}-point amplitudes', fontsize=13)
    ax.set_xticks(term_vals)
    ax.legend(fontsize=9, loc='lower left')
    ax.set_ylim([80, 101])
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.bar(term_vals, sample_counts, alpha=0.1, color='gray', width=0.6)
    ax2.set_ylabel('Number of samples', fontsize=10, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    plt.tight_layout()
    outpath = os.path.join(FIG_DIR, f'amp_solve_rate_vs_tgt_terms_{npt}pt.pdf')
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  Saved {outpath}")

# --- Summary stats ---
print("\nSummary:")
for npt in [4, 5, 6]:
    m = merged[npt]
    n = len(m['src_bk'])
    print(f"  {npt}pt (N={n}): ours_srcrel={m['ours_srcrel'].mean()*100:.2f}% ours_tgtrel={m['ours_tgtrel'].mean()*100:.2f}% cheung_srcrel={m['cheung_srcrel'].mean()*100:.2f}% cheung_tgtrel={m['cheung_tgtrel'].mean()*100:.2f}%")

print("\nDone!")
