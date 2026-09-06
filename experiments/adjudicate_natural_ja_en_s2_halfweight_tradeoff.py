#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

SEEDS = list(range(32000, 32010))
CONDS = ["JA_nat", "JA_TJA", "JA_TEN", "JA_TRU"]
PROTOCOL = "protocols/natural_ja_en_s2_halfweight_tradeoff_2026-09-06.json"
EXPECTED_CORPUS_HASHES = {
    "en_train": "74f59db34b5b8334cf463cf0bcf5f355e54cc7de98cc4eb35ef379ecb42e8258",
    "en_eval": "15221408b2c4c2fffee8a5f16feb651f2de5947e85906867f119fdf7210797da",
    "ja_train": "5a0d86cebeb5e9f71c86ea1bcf619c8a42bb21d04ed2d37ce35f5aee58bffd58",
    "ja_eval": "5a44582a08d6e3c48c610d663b22642c8b8612dc19f25a326b2ce906a504e7e4",
    "ru_train": "12a53e456480b56977c548021231c9adf9b4cee97902f6d1acb67e6cbb0d46ed",
    "ru_eval": "52a1cd50a20645577e988f6023dcf1853e856f5babd180a0e1d620029c689800",
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_bool(x):
    if x == "True":
        return True
    if x == "False":
        return False
    raise ValueError(x)


def finite_float(x):
    if x is None or x == "":
        return None
    v = float(x)
    return v if math.isfinite(v) else None


def sign_p(k, n=10):
    return sum(math.comb(n, j) for j in range(k, n + 1)) / (2 ** n)


def expected_audit(a):
    return (
        a["scientific_evidence"] is True
        and a["smoke_only"] is False
        and a.get("validation_only") is False
        and a["prior_result_seed_hits"] == []
        and a.get("corpus_hashes_match_frozen_manifest") is True
        and a["corpus_sha256"] == EXPECTED_CORPUS_HASHES
        and a["student_phase0_sources"] == {"ja": 400, "en": 0, "ru": 0}
        and a["student_phase1_sources"] == {"ja": 200, "en": 0, "ru": 0}
        and a["teacher_source_counts"]["T_JA"] == {"ja": 400, "en": 0, "ru": 0}
        and a["teacher_source_counts"]["T_EN"] == {"ja": 200, "en": 200, "ru": 0}
        and a["teacher_source_counts"]["T_RU"] == {"ja": 200, "en": 0, "ru": 200}
        and a["teacher_target_slot_positions_identical"] is True
        and a["fixed_byte_vocab"] == 256
        and a.get("hidden_distillation_weight") == 0.5
        and a.get("tradeoff_followup") is True
        and a.get("parent_locked_verdict") == "FAIL"
        and a.get("protocol") == PROTOCOL
        and a["schedule"] == {
            "base_steps": 400,
            "teacher_steps": 400,
            "phase1_steps": 200,
            "phase2_steps": 600,
            "eval_every": 10,
        }
    )


def mean_present(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    return sum(vals) / len(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    missing, seed_rows = [], []
    all_audits = True
    hash_ref = None
    for seed in SEEDS:
        sp = args.input_dir / f"natural_s2_seed{seed}_summary.csv"
        apath = args.input_dir / f"natural_s2_seed{seed}_audit.json"
        if not sp.exists() or not apath.exists():
            missing.append(seed)
            continue
        rows = read_csv(sp)
        by = {r["condition"]: r for r in rows}
        if len(rows) != 4 or sorted(by) != sorted(CONDS):
            raise ValueError(f"seed {seed}: malformed conditions")
        if any(as_bool(r["smoke_only"]) for r in rows):
            raise ValueError(f"seed {seed}: smoke result")
        if "validation_only" in rows[0] and any(as_bool(r["validation_only"]) for r in rows):
            raise ValueError(f"seed {seed}: validation-only result")

        audit = json.loads(apath.read_text(encoding="utf-8"))
        audit_ok = expected_audit(audit)
        all_audits = all_audits and audit_ok
        if hash_ref is None:
            hash_ref = audit["corpus_sha256"]
        else:
            all_audits = all_audits and audit["corpus_sha256"] == hash_ref

        ten, tja, nat, tru = by["JA_TEN"], by["JA_TJA"], by["JA_nat"], by["JA_TRU"]
        st, sj = finite_float(ten["span_updates"]), finite_float(tja["span_updates"])
        sn, sr = finite_float(nat["span_updates"]), finite_float(tru["span_updates"])
        censored = as_bool(ten["censored"]) or as_bool(tja["censored"]) or st is None or sj is None
        primary = None if censored else st - sj
        saving = None if censored or sj <= 0 else 1.0 - st / sj
        seed_rows.append({
            "seed": seed,
            "primary_effect": primary,
            "saving_vs_TJA": saving,
            "favorable_primary_sign": primary is not None and primary < 0,
            "censored_primary": censored,
            "JA_TEN_nll": float(ten["JA_post_nll"]),
            "JA_TJA_nll": float(tja["JA_post_nll"]),
            "practical_effect": None if st is None or sn is None else st - sn,
            "specificity_effect": None if st is None or sr is None else st - sr,
            "EN_step0_TEN_minus_TJA": float(ten["EN_step0_nll"]) - float(tja["EN_step0_nll"]),
            "audit_pass": audit_ok,
        })

    if missing:
        result = {"verdict": "UNCERTAIN", "reason": "missing fixed tradeoff seeds", "missing_seeds": missing, "n_complete": len(seed_rows)}
    else:
        k = sum(r["favorable_primary_sign"] for r in seed_rows)
        p = sign_p(k)
        savings = [r["saving_vs_TJA"] for r in seed_rows]
        all_savings = all(v is not None for v in savings)
        mean_saving = sum(savings) / 10 if all_savings else None
        mean_ja_ten = sum(r["JA_TEN_nll"] for r in seed_rows) / 10
        mean_ja_tja = sum(r["JA_TJA_nll"] for r in seed_rows) / 10
        safety = mean_ja_ten <= 1.01 * mean_ja_tja
        no_censoring = not any(r["censored_primary"] for r in seed_rows)
        passed = k >= 9 and p <= 0.05 and mean_saving is not None and mean_saving >= 0.05 and safety and all_audits and no_censoring
        result = {
            "verdict": "PASS" if passed else "FAIL",
            "hidden_distillation_weight": 0.5,
            "n_favorable_primary_sign": k,
            "exact_one_sided_sign_p": p,
            "mean_saving_vs_TJA": mean_saving,
            "mean_JA_TEN_nll": mean_ja_ten,
            "mean_JA_TJA_nll": mean_ja_tja,
            "japanese_safety_pass": safety,
            "all_audits_pass": all_audits,
            "no_primary_censoring": no_censoring,
            "mean_practical_effect_TEN_minus_nat": mean_present(seed_rows, "practical_effect"),
            "mean_specificity_effect_TEN_minus_TRU": mean_present(seed_rows, "specificity_effect"),
            "mean_EN_step0_TEN_minus_TJA": sum(r["EN_step0_TEN_minus_TJA"] for r in seed_rows) / 10,
            "seeds": seed_rows,
            "corpus_sha256": hash_ref,
        }

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
