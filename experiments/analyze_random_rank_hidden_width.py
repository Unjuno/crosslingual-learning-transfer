import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def cumulative_best(frame):
    out = frame.sort_values('B_step').copy()
    out['best_nll'] = np.minimum.accumulate(out['B_nll'].to_numpy(dtype=float))
    return out


def first_crossing(frame, threshold):
    g = cumulative_best(frame)
    x = g['B_step'].to_numpy(dtype=float)
    y = g['best_nll'].to_numpy(dtype=float)
    if y[0] <= threshold:
        return float(x[0])
    for i in range(1, len(x)):
        if y[i] <= threshold:
            x0, x1 = x[i - 1], x[i]
            y0, y1 = y[i - 1], y[i]
            if y1 == y0:
                return float(x1)
            return float(x0 + (x1 - x0) * (y0 - threshold) / (y0 - y1))
    return float('nan')


def performance_aligned_effect(a_nat, target):
    a = cumulative_best(a_nat)
    t = cumulative_best(target)
    top = min(float(a['best_nll'].iloc[0]), float(t['best_nll'].iloc[0]))
    bottom = max(float(a['best_nll'].iloc[-1]), float(t['best_nll'].iloc[-1]))
    high = bottom + 0.8 * (top - bottom)
    low = bottom + 0.2 * (top - bottom)
    span_a = first_crossing(a, low) - first_crossing(a, high)
    span_t = first_crossing(t, low) - first_crossing(t, high)
    return span_t - span_a


def one_sided_sign_p(k, n):
    return sum(math.comb(n, j) for j in range(k, n + 1)) / (2 ** n)


def analyze_file(path):
    df = pd.read_csv(path)
    effects = {}
    for keep in sorted(df['keep_dim'].unique()):
        a = df[(df['keep_dim'] == keep) & (df['condition'] == 'A_nat')]
        t = df[(df['keep_dim'] == keep) & (df['condition'] == 'target25')]
        effects[int(keep)] = performance_aligned_effect(a, t)
    x = np.array(sorted(effects), dtype=float)
    y = np.array([effects[int(k)] for k in x], dtype=float)
    row = {
        'seed': int(df['seed'].iloc[0]),
        'hidden_width': int(df['hidden_width'].iloc[0]),
        'family': int(df['family'].iloc[0]),
        'rank_slope': float(np.polyfit(x, y, 1)[0]),
    }
    row.update({f'effect_keep{k}': effects[k] for k in sorted(effects)})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input_dir', type=Path)
    ap.add_argument('--family-out', type=Path, required=True)
    ap.add_argument('--summary-out', type=Path, required=True)
    args = ap.parse_args()
    files = sorted(args.input_dir.glob('family*_seed*_H*.csv'))
    if not files:
        raise SystemExit(f'no curve CSVs found in {args.input_dir}')
    family = pd.DataFrame(analyze_file(p) for p in files).sort_values(['seed', 'hidden_width', 'family'])
    summaries = []
    for (seed, width), g in family.groupby(['seed', 'hidden_width']):
        k = int((g['rank_slope'] < 0).sum())
        n = int(len(g))
        summaries.append({
            'seed': int(seed),
            'hidden_width': int(width),
            'mean_rank_slope': float(g['rank_slope'].mean()),
            'median_rank_slope': float(g['rank_slope'].median()),
            'n_negative': k,
            'n_families': n,
            'one_sided_sign_p': one_sided_sign_p(k, n),
            'min_rank_slope': float(g['rank_slope'].min()),
            'max_rank_slope': float(g['rank_slope'].max()),
        })
    summary = pd.DataFrame(summaries).sort_values(['seed', 'hidden_width'])
    args.family_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    family.to_csv(args.family_out, index=False)
    summary.to_csv(args.summary_out, index=False)


if __name__ == '__main__':
    main()
