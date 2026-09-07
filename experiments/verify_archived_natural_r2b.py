#!/usr/bin/env python3
"""One-command, standard-library-only verification of the released R2b archive.

Extract only the forty allowlisted regular files, verify SHA256 for the archive
and every member, independently recompute the science gates, and compare with
the original locked adjudicator. A valid experimental FAIL is successful
verification; corrupt/missing inputs never become efficacy evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

from verify_natural_r2b_raw import SEEDS, require, verify


def expected_names():
    names = set()
    for seed in SEEDS:
        names.update(f'natural_s2_seed{seed}_{suffix}'
                     for suffix in ('curves.csv', 'summary.csv', 'audit.json'))
        names.add(f'seed{seed}_execution_trace.json')
    return names


def extract_verified(archive: Path, manifest: dict, destination: Path):
    require(hashlib.sha256(archive.read_bytes()).hexdigest() == manifest['archive_sha256'],
            'archive SHA256 mismatch')
    require(set(manifest['members']) == expected_names(), 'archive manifest file set mismatch')
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        require(len(members) == 40 and {m.name for m in members} == expected_names(),
                'archive member set mismatch')
        require(sum(m.size for m in members) <= 5_000_000, 'archive exceeds size limit')
        for item in members:
            require(item.isfile() and '/' not in item.name and '\\' not in item.name,
                    'unsafe archive member')
            require(0 < item.size < 1_000_000, 'invalid member size')
            reader = tar.extractfile(item)
            require(reader is not None, 'missing member data')
            with reader:
                data = reader.read(1_000_001)
            require(len(data) == item.size, 'incomplete archive member')
            require(hashlib.sha256(data).hexdigest() == manifest['members'][item.name],
                    'member SHA256 mismatch: ' + item.name)
            (destination / item.name).write_bytes(data)


def compare_adjudicators(raw_report: dict, original_report: dict):
    require(raw_report['verdict'] == original_report['verdict'], 'adjudicator verdict disagreement')
    require(raw_report['favorable_primary_signs'] == original_report['n_favorable_primary_sign'],
            'adjudicator sign-count disagreement')
    for a, b in [('mean_saving_vs_TJA', 'mean_saving_vs_TJA'),
                 ('exact_one_sided_sign_p', 'exact_one_sided_sign_p'),
                 ('mean_JA_TEN_nll', 'mean_JA_TEN_nll'),
                 ('mean_JA_TJA_nll', 'mean_JA_TJA_nll')]:
        left, right = raw_report[a], original_report[b]
        require(left is None and right is None or left is not None and right is not None
                and abs(left-right) <= 1e-9, 'adjudicator numeric disagreement: ' + a)


def main():
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', type=Path, default=repo/'results/natural_r2b_raw.tar.gz')
    ap.add_argument('--manifest', type=Path, default=repo/'results/natural_r2b_raw_manifest.json')
    ap.add_argument('--json-out', type=Path)
    args = ap.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory(prefix='verify_r2b_') as d:
            root = Path(d)
            extract_verified(args.archive, manifest, root)
            report = verify(root)
            original_out = root/'original_verdict.json'
            subprocess.run([sys.executable, str(repo/'experiments/adjudicate_natural_ja_en_s2_halfweight_tradeoff.py'),
                            str(root), '--json-out', str(original_out)],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            compare_adjudicators(report, json.loads(original_out.read_text()))
            report['archive_and_member_hashes_verified'] = True
            report['original_locked_adjudicator_agrees'] = True
            report['verification_success'] = True
            code = 0
    except (ValueError, KeyError, OSError, TypeError, tarfile.TarError,
            subprocess.CalledProcessError) as e:
        report = {'verdict': 'UNCERTAIN', 'verification_success': False, 'error': str(e)}
        code = 2
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding='utf-8')
    print(text)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
