#!/usr/bin/env python3
"""Verify public aggregate records, NOT independently rerun model training.

Full curve/checkpoint verification is recorded in the execution artifact. A
matching compact CSV is not, by itself, proof of what a model was trained on.
"""
import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

SEEDS = set(range(32000, 32010))
CONDS = {"JA_nat", "JA_TJA", "JA_TEN", "JA_TRU"}
SOURCE = "efbcc3f2fe5e809f539a95eff3d120c21ec98826"
CORE = "aab161824538e29f71f2a5f00fd44364f2de9bd4"
WRAPPER = "9dcc9f5a4c4c76ed62e173ffb99b9f911de84996"
PREFIX = "natural_r2_halfweight_"
SUFFIX = "_2026-09-08"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def positive(value):
    x = float(value)
    if not math.isfinite(x) or x <= 0:
        raise ValueError("expected a positive finite recorded value")
    return x


def recompute(rows, audits):
    expected = {(s, c) for s in SEEDS for c in CONDS}
    keys = [(int(r["seed"]), r["condition"]) for r in rows]
    if len(keys) != 40 or set(keys) != expected:
        raise ValueError("missing, duplicate, or unknown seed/condition")
    by = dict(zip(keys, rows))
    for row in rows:
        for flag in ["smoke_only", "validation_only", "censored"]:
            if row[flag] != "False":
                raise ValueError("this public cohort must be nonvalidation and uncensored")
        for field in ["JA_post_nll", "EN_step0_nll", "span_updates", "reference_high", "reference_low"]:
            positive(row[field])
        if positive(row["span_updates"]) > 600:
            raise ValueError("span exceeds the locked budget")
    audit_keys = [int(a["seed"]) for a in audits]
    if len(audit_keys) != 10 or set(audit_keys) != SEEDS:
        raise ValueError("incomplete or duplicated audits")
    for audit in audits:
        if audit["original_audit_pass"] != "True" or audit["observed_actual_batch_audit_pass"] != "True":
            raise ValueError("recorded execution audit did not pass")
        if int(audit["updates"]) != 5400:
            raise ValueError("incorrect update count")
        if (audit["source_commit"], audit["core_git_blob"], audit["wrapper_git_blob"]) != (SOURCE, CORE, WRAPPER):
            raise ValueError("incorrect code provenance")
        for key in ["phase1_batch_sha256", "phase2_batch_sha256"]:
            if not re.fullmatch(r"[0-9a-f]{64}", audit[key]):
                raise ValueError("malformed actual-input digest")
    savings, differences, ja_ten, ja_tja = [], [], [], []
    for seed in sorted(SEEDS):
        pairs = {(float(by[(seed, c)]["reference_high"]), float(by[(seed, c)]["reference_low"])) for c in CONDS}
        if len(pairs) != 1:
            raise ValueError("conditions do not share reference thresholds")
        hi, lo = next(iter(pairs))
        if not hi > lo:
            raise ValueError("nonpositive reference interval")
        ten, tja = by[(seed, "JA_TEN")], by[(seed, "JA_TJA")]
        a, b = positive(ten["span_updates"]), positive(tja["span_updates"])
        differences.append(a - b)
        savings.append(1 - a / b)
        ja_ten.append(positive(ten["JA_post_nll"]))
        ja_tja.append(positive(tja["JA_post_nll"]))
    k = sum(x < 0 for x in differences)
    p = sum(math.comb(10, j) for j in range(k, 11)) / 1024
    mean_saving = statistics.mean(savings)
    jt, jj = statistics.mean(ja_ten), statistics.mean(ja_tja)
    retention = jt <= 1.01 * jj
    passed = k >= 9 and p <= 0.05 and mean_saving >= 0.05 and retention
    return {"verdict": "PASS" if passed else "FAIL", "n_complete": 10,
            "n_favorable": k, "exact_one_sided_sign_p": p,
            "mean_saving": mean_saving, "median_saving": statistics.median(savings),
            "min_saving": min(savings), "max_saving": max(savings),
            "mean_primary_difference_updates": statistics.mean(differences),
            "mean_JA_TEN_nll": jt, "mean_JA_TJA_nll": jj,
            "relative_JA_penalty": jt / jj - 1, "retention_gate_pass": retention,
            "optimizer_updates": sum(int(a["updates"]) for a in audits)}


def verify(root):
    protocol = json.loads((root / "protocols/natural_ja_en_s2_halfweight_tradeoff_2026-09-06.json").read_text())
    if protocol["fresh_seeds"] != sorted(SEEDS) or protocol["hidden_distillation_weight"] != 0.5 or protocol["status"] != "LOCKED_BEFORE_FRESH_OUTCOMES":
        raise ValueError("locked protocol disagrees")
    results = root / "results"
    got = recompute(read_csv(results / (PREFIX + "seed_summary" + SUFFIX + ".csv")),
                    read_csv(results / (PREFIX + "execution_audits" + SUFFIX + ".csv")))
    expected = json.loads((results / (PREFIX + "verification" + SUFFIX + ".json")).read_text())
    for key, value in got.items():
        target = expected[key]
        equal = (value == target) if isinstance(value, (str, bool, int)) else abs(value - float(target)) <= 1e-12
        if not equal:
            raise ValueError(f"published result differs: {key}")
    return {"aggregate_verification": "PASS", "recomputed_result": got,
            "scope": "Compact record consistency, not raw-curve or training reproduction."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, allow_nan=False))
