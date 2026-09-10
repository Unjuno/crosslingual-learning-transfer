#!/usr/bin/env python3
"""Verify public records and packaging boundaries; never launch model training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = 'publication/files.sha256.json'
SNAPSHOT = 'publication/scientific-snapshot.json'
ALLOWED_EXTENSIONS = {'.md', '.py', '.json', '.csv', '.yml', '.yaml', '.txt', '.cff'}
ALLOWED_NAMES = {'LICENSE', '.gitignore', '.gitattributes'}
SECRET_PATTERNS = {
    'private-key': re.compile(r'-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----'),
    'github-token': re.compile(r'\bgh[pousr]_[A-Za-z0-9]{30,}\b'),
    'github-fine-grained-token': re.compile(r'\bgithub_pat_[A-Za-z0-9_]{60,}\b'),
    'cloud-access-key': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'api-key-like': re.compile(r'\bsk-[A-Za-z0-9_-]{40,}\b'),
    'credential-in-url': re.compile(r'https?://[^\s/:@]+:[^\s/@]+@'),
}
VERIFIERS = [
    'experiments/verify_final_synthetic_closure.py',
    'experiments/verify_natural_r1_confirmatory.py',
    'experiments/verify_halfweight_public_2026_09_08.py',
    'experiments/verify_halfweight_dense_2026_09_08.py',
    'research/archived_adaptation_locus_2026_09_09/experiments/verify_records.py',
]


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def read_json(path: Path):
    def invalid(value):
        raise ValueError(f'{path.name}: invalid JSON constant {value}')
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=invalid)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_path(root: Path, name: str) -> Path:
    p = PurePosixPath(name)
    require(bool(name) and not p.is_absolute() and '..' not in p.parts and '\\' not in name,
            f'unsafe manifest path: {name}')
    require(str(p) == name and name != '.', f'noncanonical manifest path: {name}')
    result = root / name
    require(result.resolve().is_relative_to(root.resolve()), f'path escapes root: {name}')
    require(not any((root / Path(*p.parts[:i])).is_symlink() for i in range(1, len(p.parts)+1)),
            f'symlink is not publishable: {name}')
    return result


def check_file_policy(name: str) -> None:
    p = PurePosixPath(name)
    require(not any(part in {'.git', '.venv', '__pycache__', 'data', 'checkpoints', 'dist', 'secrets'}
                    for part in p.parts), f'non-public path: {name}')
    require(not p.name.startswith('.env'), f'environment file: {name}')
    require(p.suffix in ALLOWED_EXTENSIONS or p.name in ALLOWED_NAMES,
            f'unreviewed file type: {name}')
    require(not re.match(r'(?:ja|en|ru)_(?:train|eval|test)\.txt$', p.name),
            f'corpus output is not publishable: {name}')


def check_text(name: str, text: str) -> None:
    for label, pattern in SECRET_PATTERNS.items():
        require(pattern.search(text) is None, f'{label} pattern detected in {name}; content withheld')
    require(('sandbox' + ':/') not in text, f'non-public sandbox link in {name}')


def local_links(root: Path, name: str, text: str) -> int:
    # Check inline Markdown file links, not remote URLs or heading fragments.
    text = re.sub(r'```.*?```', '', text, flags=re.S)
    count = 0
    for target in re.findall(r'\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)', text):
        target = target.strip('<>')
        if target.startswith('#') or urlsplit(target).scheme or target.startswith('//'):
            continue
        relative = unquote(urlsplit(target).path)
        if not relative:
            continue
        resolved = (root / name).parent / relative
        require(resolved.resolve().is_relative_to(root.resolve()), f'link escapes root: {name}')
        require(resolved.exists(), f'broken local Markdown link: {name} -> {target}')
        count += 1
    return count


def check_final_status(root: Path) -> None:
    final = read_json(root / 'results/FINAL_STATUS.json')
    release = read_json(root / 'publication/release.json')
    old = read_json(root / 'results/current_status_2026-09-09.json')
    r1 = read_json(root / 'results/natural_r1_confirmatory_adjudication.json')
    r2 = read_json(root / 'results/natural_r2_halfweight_verification_2026-09-08.json')
    require(final['project_status'] == 'FROZEN_RESEARCH_SNAPSHOT', 'research scope is not frozen')
    require(final['new_training_runs_in_publication_cleanup'] == 0, 'publication must not claim training')
    require(final['historical_decisions']['natural_R1_weight1'] == r1['verdict'] == 'FAIL', 'R1 verdict drift')
    require(final['historical_decisions']['natural_R2_weight_half'] == 'PASS_REGISTERED_GRID'
            and r2['verdict'] == 'PASS', 'R2 verdict drift')
    require(final['historical_decisions']['state_signal_seed18500'] == 'BLOCKED_NOT_COUNTED', 'blocked claim drift')
    require(final['natural_R2'] == old['natural_R2_halfweight'], 'R2 summary differs from scientific record')
    require(final['archived_body_head']['independent_new_seeds'] == 0, 'archival pseudoreplication')
    require(final['archived_body_head']['body_origin_effect_nats_per_byte'] ==
            old['archived_adaptation_locus_2026_09_09']['body_head_substitution']['mean_endpoint_body_origin_effect_nats_per_byte'],
            'body/head effect drift')
    require(release['frozen'] is True and release['prerelease'] is True, 'research release policy')
    require(re.fullmatch(r'v\d+\.\d+\.\d+-research(?:\.\d+)?', release['version']) is not None, 'invalid research version')
    require(final['release_version'] == release['version'], 'version mismatch')
    require(final['date_jst'] == release['date'], 'release date mismatch')
    citation = (root / 'CITATION.cff').read_text(encoding='utf-8')
    require(f'version: "{release["version"]}"' in citation, 'citation version mismatch')
    require('doi:' not in citation.lower(), 'unregistered DOI must not be added')
    required = {'target_unaware_Japanese_text_only', 'robust_population_saving_at_least_five_percent',
                'pure_learning_rate_improvement', 'independent_corpus_generalization',
                'post_English_Japanese_retention_guarantee', 'total_compute_saving',
                'human_learning', 'large_LLM_generalization', 'universal_language_distance'}
    require(required <= set(final['not_established']), 'lost claim boundary')


def check_workflow(name: str, text: str) -> None:
    require('contents: read' in text, f'no default read-only permissions: {name}')
    require('pull_request_target' not in text, f'privileged PR trigger: {name}')
    for action in re.findall(r'^\s*-?\s*uses:\s*(\S+)', text, re.M):
        require(re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+@[0-9a-f]{40}', action) is not None,
                f'unpinned external action: {name}')
    if 'contents: write' in text:
        require(name == '.github/workflows/publish-research-snapshot.yml', 'unexpected writer workflow')
        require("github.ref == 'refs/heads/main'" in text and 'pull_request:' not in text,
                'release writer must only run trusted main')


def check_snapshot(root: Path) -> int:
    locked = read_json(root / SNAPSHOT)
    for name, expected in locked['files'].items():
        p = safe_path(root, name)
        require(p.is_file() and digest(p) == expected, f'historical scientific file changed: {name}')
    require(locked['source_commit'] == '63a6a056f9575d027dde09d3d19be172e821cd95', 'evidence-base drift')
    return len(locked['files'])


def static_checks(root: Path) -> dict:
    inventory = read_json(root / INVENTORY)['files']
    require(INVENTORY not in inventory, 'inventory cannot hash itself')
    if (root / '.git').exists():
        tracked = subprocess.run(['git', '-C', str(root), 'ls-files', '-z'], check=True,
                                 capture_output=True).stdout.decode('utf-8').split('\0')
        require(set(filter(None, tracked)) - {INVENTORY} == set(inventory),
                'tracked file set differs from publication inventory; stage and deliberately refresh it')
    nlinks = 0
    for name, sha in inventory.items():
        path = safe_path(root, name)
        check_file_policy(name)
        require(path.is_file() and path.stat().st_size <= 2_000_000, f'missing/oversized public file: {name}')
        require(digest(path) == sha, f'publication hash mismatch: {name}')
        text = path.read_text(encoding='utf-8')
        check_text(name, text)
        if path.suffix == '.md':
            nlinks += local_links(root, name, text)
        elif path.suffix == '.json':
            read_json(path)
        elif path.suffix == '.py':
            compile(text, name, 'exec')  # syntax only; no imports or training
        if name.startswith('.github/workflows/'):
            check_workflow(name, text)
    nlocked = check_snapshot(root)
    check_final_status(root)
    return {'inventory_files_including_manifest': len(inventory)+1,
            'historical_scientific_files_unchanged': nlocked,
            'local_inline_markdown_links_checked': nlinks,
            'exposure_scan_scope': 'manifest-listed tracked snapshot only; heuristic; not git history'}


def run_checks(root: Path) -> list[dict]:
    commands = [[path] for path in VERIFIERS] + [
        ['-m', 'unittest', 'discover', '-s', 'experiments', '-p', 'test_*.py', '-v'],
        ['-m', 'unittest', 'discover', '-s', 'research/archived_adaptation_locus_2026_09_09/experiments',
         '-p', 'test_public_records.py', '-v'],
        ['-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py', '-v'],
    ]
    runs = []
    for command in commands:
        begin = time.monotonic()
        p = subprocess.run([sys.executable, *command], cwd=root, text=True, encoding='utf-8',
                           capture_output=True, timeout=120)
        runs.append({'command': ['python', *command], 'exit_code': p.returncode,
                     'elapsed_seconds': time.monotonic()-begin,
                     'stdout': p.stdout, 'stderr': p.stderr})
        print(('OK ' if p.returncode == 0 else 'ERROR ') + ' '.join(command), flush=True)
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--report', type=Path)
    ap.add_argument('--static-only', action='store_true')
    args = ap.parse_args()
    root = args.root.resolve()
    report_path = args.report or root / 'dist/publication-verification.json'
    result = {'status': 'FAIL', 'scope': 'record arithmetic and publication integrity; NOT scientific hypothesis or training validation',
              'python': sys.version, 'static_only': args.static_only}
    try:
        require(sys.version_info >= (3, 11), 'Python 3.11 or newer required; tested on 3.11–3.13')
        result['static'] = static_checks(root)
        if not args.static_only:
            result['runs'] = run_checks(root)
            require(all(r['exit_code'] == 0 for r in result['runs']), 'one or more record/test checks failed')
        result['status'] = 'STATIC_PASS_ONLY' if args.static_only else 'PASS'
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        result['error'] = str(exc)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'runs'}, indent=2, ensure_ascii=False))
    return 0 if result['status'] in {'PASS', 'STATIC_PASS_ONLY'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
