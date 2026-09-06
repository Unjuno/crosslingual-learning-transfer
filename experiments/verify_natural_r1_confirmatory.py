#!/usr/bin/env python3
"""Recompute the locked R1 S2 adjudication from committed compact seed summaries."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PROTOCOL = ROOT / "protocols" / "natural_ja_en_s2_hidden_teacher_r1_2026-09-06.json"
SEEDS = list(range(31000, 31010))
CONDS = {"JA_nat", "JA_TJA", "JA_TEN", "JA_TRU"}
TOL = 1e-12


def rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def b(x):
    if x == "True":
        return True
    if x == "False":
        return False
    raise ValueError(x)


def close(a, c):
    return math.isclose(float(a), float(c), rel_tol=0.0, abs_tol=TOL)


def sign_p(k, n=10):
    return sum(math.comb(n, j) for j in range(k, n + 1)) / (2 ** n)


def main():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["status"] == "DESIGN_LOCKED_BEFORE_NEW_OUTCOMES"
    assert protocol["confirmatory_plan"]["fresh_seeds"] == SEEDS

    summary = defaultdict(dict)
    for r in rows(RESULTS / "natural_r1_confirmatory_seed_summary.csv"):
        seed = int(r["seed"])
        summary[seed][r["condition"]] = r
    assert sorted(summary) == SEEDS
    assert all(set(summary[s]) == CONDS for s in SEEDS)

    audit = {int(r["seed"]): r for r in rows(RESULTS / "natural_r1_confirmatory_audit_summary.csv")}
    assert sorted(audit) == SEEDS
    audit_pass = True
    for seed in SEEDS:
        a = audit[seed]
        ok = (
            b(a["scientific_evidence"])
            and b(a["corpus_hashes_match"])
            and int(a["prior_result_seed_hits_count"]) == 0
            and int(a["student_phase0_ja_updates"]) == 400
            and int(a["student_phase1_ja_updates"]) == 200
            and int(a["T_JA_ja"]) == 400
            and int(a["T_EN_ja"]) == 200
            and int(a["T_EN_en"]) == 200
            and int(a["T_RU_ja"]) == 200
            and int(a["T_RU_ru"]) == 200
            and int(a["fixed_byte_vocab"]) == 256
            and b(a["audit_pass"])
        )
        audit_pass = audit_pass and ok
    assert audit_pass

    seed_out = []
    for seed in SEEDS:
        d = summary[seed]
        ten, tja, nat, tru = d["JA_TEN"], d["JA_TJA"], d["JA_nat"], d["JA_TRU"]
        censored = b(ten["censored"]) or b(tja["censored"])
        st = float(ten["span_updates"])
        sj = float(tja["span_updates"])
        sn = float(nat["span_updates"])
        sr = float(tru["span_updates"])
        primary = None if censored else st - sj
        saving = None if censored or sj <= 0 else 1.0 - st / sj
        seed_out.append(
            {
                "seed": seed,
                "primary_effect": primary,
                "saving_vs_TJA": saving,
                "favorable_primary_sign": primary is not None and primary < 0,
                "censored_primary": censored,
                "JA_TEN_nll": float(ten["JA_post_nll"]),
                "JA_TJA_nll": float(tja["JA_post_nll"]),
                "practical_effect": st - sn,
                "specificity_effect": st - sr,
                "EN_step0_TEN_minus_TJA": float(ten["EN_step0_nll"]) - float(tja["EN_step0_nll"]),
                "audit_pass": True,
            }
        )

    k = sum(r["favorable_primary_sign"] for r in seed_out)
    p = sign_p(k)
    mean_saving = sum(r["saving_vs_TJA"] for r in seed_out) / 10
    mean_ja_ten = sum(r["JA_TEN_nll"] for r in seed_out) / 10
    mean_ja_tja = sum(r["JA_TJA_nll"] for r in seed_out) / 10
    safety = mean_ja_ten <= 1.01 * mean_ja_tja
    no_censoring = not any(r["censored_primary"] for r in seed_out)
    passed = k >= 9 and p <= 0.05 and mean_saving >= 0.05 and audit_pass and safety and no_censoring
    observed = {
        "verdict": "PASS" if passed else "FAIL",
        "n_favorable_primary_sign": k,
        "exact_one_sided_sign_p": p,
        "mean_saving_vs_TJA": mean_saving,
        "mean_JA_TEN_nll": mean_ja_ten,
        "mean_JA_TJA_nll": mean_ja_tja,
        "japanese_safety_pass": safety,
        "all_audits_pass": audit_pass,
        "no_primary_censoring": no_censoring,
        "mean_practical_effect_TEN_minus_nat": sum(r["practical_effect"] for r in seed_out) / 10,
        "mean_specificity_effect_TEN_minus_TRU": sum(r["specificity_effect"] for r in seed_out) / 10,
        "mean_EN_step0_TEN_minus_TJA": sum(r["EN_step0_TEN_minus_TJA"] for r in seed_out) / 10,
    }

    recorded = json.loads((RESULTS / "natural_r1_confirmatory_adjudication.json").read_text(encoding="utf-8"))
    for key, value in observed.items():
        if isinstance(value, float):
            assert close(value, recorded[key]), (key, value, recorded[key])
        else:
            assert value == recorded[key], (key, value, recorded[key])

    assert observed["verdict"] == "FAIL"
    assert observed["n_favorable_primary_sign"] == 10
    assert observed["mean_saving_vs_TJA"] >= 0.05
    assert observed["japanese_safety_pass"] is False
    print(json.dumps(observed, indent=2, sort_keys=True))
    print("OK: R1 efficacy gates pass, Japanese safety gate fails, locked overall verdict is FAIL.")


if __name__ == "__main__":
    main()
