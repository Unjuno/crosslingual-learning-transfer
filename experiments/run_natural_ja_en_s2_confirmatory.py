#!/usr/bin/env python3
"""Run the locked R1 natural S2 confirmatory cohort without interactive seed peeking."""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

SEEDS = list(range(31000, 31010))
CONDITIONS = {"JA_nat", "JA_TJA", "JA_TEN", "JA_TRU"}
EXPECTED_HASHES = {
    "en_train": "74f59db34b5b8334cf463cf0bcf5f355e54cc7de98cc4eb35ef379ecb42e8258",
    "en_eval": "15221408b2c4c2fffee8a5f16feb651f2de5947e85906867f119fdf7210797da",
    "ja_train": "5a0d86cebeb5e9f71c86ea1bcf619c8a42bb21d04ed2d37ce35f5aee58bffd58",
    "ja_eval": "5a44582a08d6e3c48c610d663b22642c8b8612dc19f25a326b2ce906a504e7e4",
    "ru_train": "12a53e456480b56977c548021231c9adf9b4cee97902f6d1acb67e6cbb0d46ed",
    "ru_eval": "52a1cd50a20645577e988f6023dcf1853e856f5babd180a0e1d620029c689800",
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_corpus(data_dir):
    observed = {}
    for name, expected in EXPECTED_HASHES.items():
        path = data_dir / f"{name}.txt"
        if not path.exists():
            raise SystemExit(f"missing frozen corpus file: {path}")
        observed[name] = sha256(path)
        if observed[name] != expected:
            raise SystemExit(
                f"frozen corpus hash mismatch for {name}: "
                f"expected {expected}, got {observed[name]}"
            )
    return observed


def read_summary(path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 4 or {r["condition"] for r in rows} != CONDITIONS:
        raise ValueError(f"malformed summary: {path}")
    if any(r.get("smoke_only") != "False" for r in rows):
        raise ValueError(f"smoke output cannot resume confirmatory: {path}")
    if "validation_only" in rows[0] and any(
        r.get("validation_only") != "False" for r in rows
    ):
        raise ValueError(f"validation output cannot resume confirmatory: {path}")
    return rows


def verify_seed_output(outdir, seed, hashes):
    summary = outdir / f"natural_s2_seed{seed}_summary.csv"
    audit_path = outdir / f"natural_s2_seed{seed}_audit.json"
    curves = outdir / f"natural_s2_seed{seed}_curves.csv"
    if not (summary.exists() and audit_path.exists() and curves.exists()):
        return False
    read_summary(summary)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    required = (
        audit.get("seed") == seed
        and audit.get("scientific_evidence") is True
        and audit.get("smoke_only") is False
        and audit.get("validation_only") is False
        and audit.get("prior_result_seed_hits") == []
        and audit.get("corpus_hashes_match_frozen_manifest") is True
        and audit.get("corpus_sha256") == hashes
        and audit.get("student_phase0_sources") == {"ja": 400, "en": 0, "ru": 0}
        and audit.get("student_phase1_sources") == {"ja": 200, "en": 0, "ru": 0}
        and audit.get("teacher_source_counts", {}).get("T_JA")
        == {"ja": 400, "en": 0, "ru": 0}
        and audit.get("teacher_source_counts", {}).get("T_EN")
        == {"ja": 200, "en": 200, "ru": 0}
        and audit.get("teacher_source_counts", {}).get("T_RU")
        == {"ja": 200, "en": 0, "ru": 200}
    )
    if not required:
        raise ValueError(f"completed seed {seed} fails resume audit")
    return True


def ensure_no_partial_seed_files(outdir, seed):
    expected = {
        outdir / f"natural_s2_seed{seed}_summary.csv",
        outdir / f"natural_s2_seed{seed}_audit.json",
        outdir / f"natural_s2_seed{seed}_curves.csv",
    }
    existing = {p for p in expected if p.exists()}
    if existing and existing != expected:
        raise SystemExit(
            f"seed {seed} has partial output files; remove/repair before resume: "
            + ", ".join(str(p) for p in sorted(existing))
        )


def run_seed(repo_root, data_dir, outdir, seed):
    cmd = [
        sys.executable,
        str(repo_root / "experiments" / "natural_ja_en_s2_hidden_teacher.py"),
        "--seed",
        str(seed),
        "--data-dir",
        str(data_dir),
        "--outdir",
        str(outdir),
    ]
    env = os.environ.copy()
    env["THREADS"] = "1"
    print("launch", seed, flush=True)
    subprocess.run(cmd, cwd=repo_root, env=env, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Reserved for future deterministic cohort launchers; currently must be 1.",
    )
    args = ap.parse_args()
    if args.max_parallel != 1:
        raise SystemExit("current locked runner requires --max-parallel 1")

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = args.data_dir.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    hashes = verify_corpus(data_dir)

    # Important: this runner does not adjudicate after each seed. It only prints
    # execution state. Scientific adjudication is run once after all fixed seeds exist.
    completed = []
    pending = []
    for seed in SEEDS:
        ensure_no_partial_seed_files(outdir, seed)
        if verify_seed_output(outdir, seed, hashes):
            completed.append(seed)
        else:
            pending.append(seed)

    state = {
        "fixed_seeds": SEEDS,
        "completed_before_run": completed,
        "pending_before_run": pending,
        "corpus_sha256": hashes,
        "policy": "no intermediate adjudication",
    }
    (outdir / "R1_COHORT_STATE_BEFORE_RUN.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(state, indent=2), flush=True)

    for seed in pending:
        run_seed(repo_root, data_dir, outdir, seed)
        if not verify_seed_output(outdir, seed, hashes):
            raise SystemExit(f"seed {seed} did not produce a valid complete output")

    if not all(verify_seed_output(outdir, seed, hashes) for seed in SEEDS):
        raise SystemExit("fixed cohort is still incomplete; adjudication not run")

    adjudication_path = outdir / "natural_r1_confirmatory_adjudication.json"
    cmd = [
        sys.executable,
        str(repo_root / "experiments" / "adjudicate_natural_ja_en_s2_hidden_teacher.py"),
        str(outdir),
        "--json-out",
        str(adjudication_path),
    ]
    subprocess.run(cmd, cwd=repo_root, check=True)
    print(f"final adjudication saved: {adjudication_path}", flush=True)


if __name__ == "__main__":
    main()
