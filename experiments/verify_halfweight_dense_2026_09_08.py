#!/usr/bin/env python3
"""Recompute a descriptive resolution check, not a replacement primary verdict."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def recompute(rows):
    keys = [(int(r['seed']), r['condition']) for r in rows]
    expected = {(s, c) for s in range(32000, 32010) for c in ('JA_TJA', 'JA_TEN')}
    if len(rows) != 20 or set(keys) != expected:
        raise ValueError('missing, duplicate or unknown dense measurement cells')
    by = dict(zip(keys, rows))
    for r in rows:
        vals = {k: float(r[k]) for k in ('interpolated_span', 'discrete_span', 'high_crossing', 'low_crossing')}
        if not all(math.isfinite(v) for v in vals.values()):
            raise ValueError('nonfinite dense measurement')
        if not 0 <= vals['high_crossing'] < vals['low_crossing'] <= 250:
            raise ValueError('invalid or censored crossing')
        if abs(vals['low_crossing'] - vals['high_crossing'] - vals['interpolated_span']) > 1e-10:
            raise ValueError('span differs from its recorded crossings')
        if not 0 < vals['discrete_span'] <= 250 or not vals['discrete_span'].is_integer():
            raise ValueError('invalid first-observed span')
    result = {}
    for field, prefix in [('interpolated_span', 'dense_interpolated'), ('discrete_span', 'dense_first_observed')]:
        savings = [1 - float(by[(s, 'JA_TEN')][field]) / float(by[(s, 'JA_TJA')][field]) for s in range(32000, 32010)]
        result[prefix + '_mean_saving'] = statistics.mean(savings)
        result[prefix + '_favorable'] = sum(x > 0 for x in savings)
    return result


def verify(root):
    folder = root / 'results'
    with (folder / 'natural_r2_halfweight_dense_spans_2026-09-08.csv').open(newline='', encoding='utf-8') as f:
        result = recompute(list(csv.DictReader(f)))
    report = json.loads((folder / 'natural_r2_measurement_diagnostics_2026-09-08.json').read_text(encoding='utf-8'))['halfweight_measurement']
    for key, value in result.items():
        if abs(value - report[key]) > 1e-12:
            raise ValueError(f'dense summary disagrees: {key}')
    return {'dense_record_verification': 'PASS', 'recomputed_diagnostic': result,
            'scope': 'Descriptive timing-record arithmetic; no new primary decision or training.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    print(json.dumps(verify(p.parse_args().root), indent=2, allow_nan=False))
