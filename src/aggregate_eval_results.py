"""
Aggregate evaluation results from parallel worker pkl files.

Usage:
    python aggregate_eval_results.py --input_dir data/eval_4pt/ \
        --output results/eval_4pt.pkl \
        --n_workers 1000
"""

import argparse
import os
import pickle
import numpy as np
from collections import defaultdict


def aggregate_results(input_dir, n_workers, verbose=True):
    """Load and aggregate results from worker pkl files."""

    # Load all worker results
    worker_results = []
    missing = []
    for i in range(n_workers):
        path = os.path.join(input_dir, f"worker_{i}.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
            worker_results.append((i, data))
        else:
            missing.append(i)

    if missing:
        print(f"WARNING: {len(missing)} missing worker files: {missing[:20]}{'...' if len(missing) > 20 else ''}")

    if not worker_results:
        print("ERROR: No worker results found!")
        return None

    # Aggregate totals using weighted sums
    total_samples = 0
    total_solved = 0
    total_solved_proper = 0
    total_reached_target = 0
    total_term_explosions = 0
    weighted_min_brackets = 0.0
    weighted_target_brackets = 0.0
    weighted_min_vs_target = 0.0
    weighted_final_terms = 0.0
    weighted_target_terms = 0.0
    weighted_reduction = 0.0
    weighted_reduction_ratio = 0.0
    weighted_steps = 0.0
    total_timeouts = 0
    total_retries = 0
    total_backtracks = 0

    # By complexity
    complexity_counts = defaultdict(int)
    complexity_solved = defaultdict(int)
    complexity_solved_proper = defaultdict(int)
    complexity_reached_target = defaultdict(int)
    complexity_min_brackets = defaultdict(float)
    complexity_target_brackets = defaultdict(float)
    complexity_reduction = defaultdict(float)
    complexity_final_terms = defaultdict(float)
    complexity_target_terms = defaultdict(float)

    for worker_id, data in worker_results:
        n = data['n_samples']
        if n == 0:
            continue
        total_samples += n
        total_solved += int(round(data['solve_rate'] * n))
        total_solved_proper += int(round(data['solve_rate_proper'] * n))
        total_reached_target += int(round(data['reached_target_rate'] * n))
        total_term_explosions += data['term_explosion_count']

        weighted_min_brackets += float(data['avg_min_brackets']) * n
        weighted_target_brackets += float(data['avg_target_brackets']) * n
        weighted_min_vs_target += float(data['avg_min_vs_target']) * n
        weighted_final_terms += float(data['avg_final_terms']) * n
        weighted_target_terms += float(data['avg_target_terms']) * n
        weighted_reduction += float(data['avg_reduction']) * n
        weighted_reduction_ratio += float(data['avg_reduction_ratio']) * n
        weighted_steps += float(data['avg_steps']) * n
        total_timeouts += data['total_timeouts']
        total_retries += data['total_retries']
        total_backtracks += data['total_backtracks']

        for c, stats in data.get('by_complexity', {}).items():
            cn = stats['count']
            complexity_counts[c] += cn
            complexity_solved[c] += int(round(stats['solve_rate'] * cn))
            complexity_solved_proper[c] += int(round(stats['solve_rate_proper'] * cn))
            complexity_reached_target[c] += int(round(stats['reached_target_rate'] * cn))
            complexity_min_brackets[c] += float(stats['avg_min_brackets']) * cn
            complexity_target_brackets[c] += float(stats['avg_target_brackets']) * cn
            complexity_reduction[c] += float(stats['avg_reduction']) * cn
            complexity_final_terms[c] += float(stats['avg_final_terms']) * cn
            complexity_target_terms[c] += float(stats['avg_target_terms']) * cn

    if total_samples == 0:
        print("ERROR: All workers had 0 samples!")
        return None

    # Compute aggregated metrics
    results = {
        'n_samples': total_samples,
        'n_workers_found': len(worker_results),
        'n_workers_missing': len(missing),
        'solve_rate': total_solved / total_samples,
        'solve_rate_proper': total_solved_proper / total_samples,
        'reached_target_rate': total_reached_target / total_samples,
        'term_explosion_count': total_term_explosions,
        'term_explosion_rate': total_term_explosions / total_samples,
        'avg_min_brackets': weighted_min_brackets / total_samples,
        'avg_target_brackets': weighted_target_brackets / total_samples,
        'avg_min_vs_target': weighted_min_vs_target / total_samples,
        'avg_final_terms': weighted_final_terms / total_samples,
        'avg_target_terms': weighted_target_terms / total_samples,
        'avg_reduction': weighted_reduction / total_samples,
        'avg_reduction_ratio': weighted_reduction_ratio / total_samples,
        'avg_steps': weighted_steps / total_samples,
        'total_timeouts': total_timeouts,
        'total_retries': total_retries,
        'total_backtracks': total_backtracks,
    }

    # By complexity
    by_complexity = {}
    for c in sorted(complexity_counts.keys()):
        cn = complexity_counts[c]
        by_complexity[c] = {
            'count': cn,
            'solve_rate': complexity_solved[c] / cn,
            'solve_rate_proper': complexity_solved_proper[c] / cn,
            'reached_target_rate': complexity_reached_target[c] / cn,
            'avg_min_brackets': complexity_min_brackets[c] / cn,
            'avg_target_brackets': complexity_target_brackets[c] / cn,
            'avg_reduction': complexity_reduction[c] / cn,
            'avg_final_terms': complexity_final_terms[c] / cn,
            'avg_target_terms': complexity_target_terms[c] / cn,
        }
    results['by_complexity'] = by_complexity

    if verbose:
        print(f"=" * 70)
        print(f"AGGREGATED RESULTS ({len(worker_results)}/{n_workers} workers)")
        print(f"=" * 70)
        print(f"Samples: {total_samples}")
        print(f"Solve rate (brackets only): {results['solve_rate']*100:.2f}%")
        print(f"Solve rate (proper, brackets+terms): {results['solve_rate_proper']*100:.2f}%")
        print(f"Term explosion count: {total_term_explosions} ({results['term_explosion_rate']*100:.2f}%)")
        print(f"Average min brackets reached: {results['avg_min_brackets']:.2f}")
        print(f"Average target brackets: {results['avg_target_brackets']:.2f}")
        print(f"Average min - target: {results['avg_min_vs_target']:.2f}")
        print(f"Average final terms: {results['avg_final_terms']:.2f}")
        print(f"Average target terms: {results['avg_target_terms']:.2f}")
        print(f"Average reduction: {results['avg_reduction']:.2f} brackets")
        print(f"Average reduction ratio: {results['avg_reduction_ratio']*100:.2f}%")
        print(f"Average steps: {results['avg_steps']:.1f}")
        print(f"Total timeouts: {total_timeouts}")
        print(f"Total retries: {total_retries}")
        print(f"Total backtracks: {total_backtracks}")
        print()
        print(f"By starting complexity:")
        for c in sorted(by_complexity.keys()):
            s = by_complexity[c]
            print(f"  {c} brackets: {s['count']} samples, "
                  f"solve={s['solve_rate']*100:.1f}% (proper={s['solve_rate_proper']*100:.1f}%), "
                  f"avg_min={s['avg_min_brackets']:.1f}, "
                  f"avg_terms={s['avg_final_terms']:.1f}/{s['avg_target_terms']:.1f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Aggregate parallel eval worker results")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing worker_*.pkl files')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for aggregated results pkl')
    parser.add_argument('--n_workers', type=int, default=1000,
                        help='Number of workers to expect')
    args = parser.parse_args()

    results = aggregate_results(args.input_dir, args.n_workers)

    if results is not None:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
