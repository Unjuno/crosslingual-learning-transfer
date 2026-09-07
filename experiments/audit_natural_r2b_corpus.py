#!/usr/bin/env python3
"""Report exact-line/window overlap without changing the frozen natural corpus.

No source text is written to the report. Exact matches do not detect semantic
or near-duplicate leakage. A matched corpus hash is not an independence proof.
"""
import argparse
import hashlib
import json
from pathlib import Path

from verify_natural_r2b_raw import CORPUS, require


def audit(data_dir: Path) -> dict:
    languages = {}
    for lang in ('ja', 'en', 'ru'):
        train = (data_dir / f'{lang}_train.txt').read_bytes()
        evaluation = (data_dir / f'{lang}_eval.txt').read_bytes()
        for split, data in [('train', train), ('eval', evaluation)]:
            require(hashlib.sha256(data).hexdigest() == CORPUS[f'{lang}_{split}'],
                    'frozen corpus hash mismatch: ' + lang + '_' + split)
        tr_lines = train.decode('utf-8').splitlines()
        ev_lines = evaluation.decode('utf-8').splitlines()
        tr_line_set = set(tr_lines)
        # One training example has 128 context bytes plus one next-byte target.
        tr_windows = {train[i:i+129] for i in range(len(train)-128)}
        record = dict(train_bytes=len(train), eval_bytes=len(evaluation),
                      eval_lines=len(ev_lines),
                      eval_lines_exactly_present_in_train_after_locked_normalization=
                          sum(line in tr_line_set for line in ev_lines),
                      matching_129byte_eval_windows=
                          sum(evaluation[i:i+129] in tr_windows for i in range(len(evaluation)-128)),
                      total_129byte_eval_windows=len(evaluation)-128)
        if lang == 'ja':
            record['entirely_ascii_train_lines'] = sum(line.isascii() for line in tr_lines)
            record['train_lines'] = len(tr_lines)
        languages[lang] = record
    return {'kind': 'supplementary exact-overlap audit; no corpus modification',
            'outputs_match_frozen_manifest': True, 'languages': languages,
            'scope': 'Exact-match audit only. Shared ASCII and near/semantic duplicates remain possible.'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir', type=Path)
    ap.add_argument('--json-out', type=Path)
    args = ap.parse_args()
    report = audit(args.data_dir)
    text = json.dumps(report, indent=2, sort_keys=True) + '\n'
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
