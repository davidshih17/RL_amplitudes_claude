#!/usr/bin/env python
"""
Analyze the paper model's evaluation output and compute proper simplification metrics.

Parses the eval.spin_hel.test.0 output files and computes:
- Numerical equivalence (the paper's reported metric)
- Brackets <= target (our stricter metric)
- Terms <= target
- Proper simplification (brackets AND terms <= target)

Usage:
    python scripts/analyze_paper_eval_proper_metric.py --eval_file results/paper_model_eval/eval_paper_4pt/beam5/eval.spin_hel.test.0
    python scripts/analyze_paper_eval_proper_metric.py --eval_file results/paper_model_eval/eval_paper_5pt/beam5_5pt/eval.spin_hel.test.0
    python scripts/analyze_paper_eval_proper_metric.py --eval_file results/paper_model_eval/eval_paper_6pt/beam5_6pt/eval.spin_hel.test.0
"""

import re
import argparse
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, 'spinorhelicity'))

import sympy as sp
from environment.bracket_env import ab, sb


def count_brackets_sympy(expr_str):
    """Count total bracket occurrences (with multiplicity from powers) using sympy.

    The eval output uses composite symbol names like ab13, sb24 (not function form).
    We parse with sympy and walk the expression tree, counting symbols whose name
    matches ab\d+ or sb\d+, multiplied by their power.
    """
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
                    try:
                        count += int(abs(exp))
                    except (TypeError, ValueError):
                        count += 1
                else:
                    count_in_expr(base)
            elif isinstance(e, (sp.Mul, sp.Add)):
                for arg in e.args:
                    count_in_expr(arg)

        numer, denom = sp.fraction(expr)
        count_in_expr(numer)
        count_in_expr(denom)
        return count
    except Exception:
        # Regex fallback
        total = 0
        for match in re.finditer(r'(ab|sb)\d+', expr_str):
            total += 1
        for match in re.finditer(r'(ab|sb)\d+\*\*(\d+)', expr_str):
            total += int(match.group(2)) - 1
        return total


def count_terms(expr_str):
    """Count number of additive terms in the numerator."""
    if expr_str.strip() == '0':
        return 1  # "0" is one term
    try:
        local_dict = {'ab': ab, 'sb': sb}
        expr = sp.sympify(expr_str, locals=local_dict)
        numer, denom = sp.fraction(expr)
        # Count terms in expanded numerator
        numer_expanded = sp.expand(numer)
        terms = sp.Add.make_args(numer_expanded)
        return len(terms)
    except Exception:
        # Rough fallback
        if '/' in expr_str:
            numer = expr_str.split('/')[0]
        else:
            numer = expr_str
        # Remove outer parens
        numer = numer.strip()
        if numer.startswith('(') and numer.endswith(')'):
            numer = numer[1:-1]
        return numer.count('+') + numer.count(' - ') + 1


def parse_eval_file(filepath):
    """Parse the evaluation output file.

    Format:
        Equation N (V/T)
        src=<expr>
        tgt=<expr>
        <valid> [<score>] <hyp_expr>
        ...
        (blank line)
    """
    equations = []
    current_eq = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip('\n')

            # New equation
            eq_match = re.match(r'Equation (\d+) \((\d+)/(\d+)\)', line)
            if eq_match:
                if current_eq is not None:
                    equations.append(current_eq)
                current_eq = {
                    'idx': int(eq_match.group(1)),
                    'n_valid': int(eq_match.group(2)),
                    'n_hyps': int(eq_match.group(3)),
                    'src': None,
                    'tgt': None,
                    'hyps': [],
                }
                continue

            if current_eq is None:
                continue

            # Source
            if line.startswith('src='):
                current_eq['src'] = line[4:]
                continue

            # Target
            if line.startswith('tgt='):
                current_eq['tgt'] = line[4:]
                continue

            # Hypothesis: "1 -1.234e+00 <expr>" or "1 <expr>" or "0 <expr>"
            hyp_match = re.match(r'^([01])\s+(?:(-?\d+\.\d+e[+-]\d+)\s+)?(.+)$', line)
            if hyp_match:
                valid = int(hyp_match.group(1))
                score = float(hyp_match.group(2)) if hyp_match.group(2) else None
                hyp_expr = hyp_match.group(3)
                current_eq['hyps'].append({
                    'valid': bool(valid),
                    'score': score,
                    'expr': hyp_expr,
                })
                continue

    if current_eq is not None:
        equations.append(current_eq)

    return equations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_file', type=str, required=True,
                        help='Path to eval.spin_hel.test.0 file')
    args = parser.parse_args()

    print(f"Parsing {args.eval_file}...", flush=True)
    equations = parse_eval_file(args.eval_file)
    print(f"Found {len(equations)} equations", flush=True)

    # Compute metrics
    n_total = len(equations)
    n_equiv = 0  # numerically equivalent (paper's metric)
    n_brackets_ok = 0  # brackets <= target
    n_terms_ok = 0  # terms <= target
    n_proper = 0  # brackets AND terms <= target

    n_more_complex = 0  # hyp has more brackets than target
    n_same = 0  # hyp has same brackets as target
    n_simpler = 0  # hyp has fewer brackets than target

    failures = []

    for eq in equations:
        tgt = eq['tgt']
        tgt_brackets = count_brackets_sympy(tgt)
        tgt_terms = count_terms(tgt)

        # Find best valid hypothesis (first valid one = highest beam score)
        best_hyp = None
        for hyp in eq['hyps']:
            if hyp['valid']:
                best_hyp = hyp
                break

        if best_hyp is None:
            # No valid hypothesis found
            continue

        n_equiv += 1
        hyp_brackets = count_brackets_sympy(best_hyp['expr'])
        hyp_terms = count_terms(best_hyp['expr'])

        if hyp_brackets <= tgt_brackets:
            n_brackets_ok += 1
        if hyp_terms <= tgt_terms:
            n_terms_ok += 1
        if hyp_brackets <= tgt_brackets and hyp_terms <= tgt_terms:
            n_proper += 1

        if hyp_brackets > tgt_brackets:
            n_more_complex += 1
        elif hyp_brackets == tgt_brackets:
            n_same += 1
        else:
            n_simpler += 1

        # Track failures
        if hyp_brackets > tgt_brackets or hyp_terms > tgt_terms:
            failures.append({
                'idx': eq['idx'],
                'src': eq['src'],
                'tgt': tgt,
                'hyp': best_hyp['expr'],
                'tgt_brackets': tgt_brackets,
                'hyp_brackets': hyp_brackets,
                'tgt_terms': tgt_terms,
                'hyp_terms': hyp_terms,
            })

    print(f"\n{'='*60}", flush=True)
    print(f"RESULTS: {args.eval_file}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"\nTotal equations: {n_total}", flush=True)
    print(f"\nNumerical equivalence (paper's metric): {n_equiv}/{n_total} ({100*n_equiv/n_total:.2f}%)", flush=True)

    if n_equiv > 0:
        print(f"\nOf the {n_equiv} equivalent outputs:", flush=True)
        print(f"  Hyp simpler than target:      {n_simpler} ({100*n_simpler/n_equiv:.2f}%)", flush=True)
        print(f"  Hyp same complexity as target: {n_same} ({100*n_same/n_equiv:.2f}%)", flush=True)
        print(f"  Hyp MORE COMPLEX than target:  {n_more_complex} ({100*n_more_complex/n_equiv:.2f}%)", flush=True)

    print(f"\nProper simplification metrics (of {n_total} total):", flush=True)
    print(f"  Brackets <= target:               {n_brackets_ok}/{n_total} ({100*n_brackets_ok/n_total:.2f}%)", flush=True)
    print(f"  Terms <= target:                  {n_terms_ok}/{n_total} ({100*n_terms_ok/n_total:.2f}%)", flush=True)
    print(f"  Proper (brackets AND terms):      {n_proper}/{n_total} ({100*n_proper/n_total:.2f}%)", flush=True)

    if failures:
        print(f"\nFirst 5 failure examples:", flush=True)
        for f in failures[:5]:
            print(f"\n  Equation {f['idx']}:", flush=True)
            print(f"    tgt={f['tgt']}  (brackets={f['tgt_brackets']}, terms={f['tgt_terms']})", flush=True)
            print(f"    hyp={f['hyp']}  (brackets={f['hyp_brackets']}, terms={f['hyp_terms']})", flush=True)


if __name__ == '__main__':
    main()
