#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

SEEDS = list(range(31000, 31010))
CONDS = ["JA_nat", "JA_TJA", "JA_TEN", "JA_TRU"]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_bool(x):
    if x == "True":
        return True
    if x == "False":
        return False
    raise ValueError(f"not a canonical bool: {x!r}")


def finite_float(x):
    if x is None or x == "":
        return None
    value = float(x)
    return value if math.isfinite(value) else None


def exact_one_sided_sign_p(k, n=10):
    return sum(math.comb(n, j) for j in range(k, n + 1)) / (2 ** n)


def expected_audit(audit):
    return (
        audit["scientific_evidence"] is True
        and audit["smoke_only"] is False
        and audit["prior_result_seed_hits"] == []
        and audit["student_phase0_sources"] == {"ja": 400, "en": 0, "ru": 0}
        and audit["student_phase1_sources"] == {"ja": 200, "en": 0, "ru": 0}
        and audit["teacher_source_counts"]["T_JA"]
        == {"ja": 400, "en": 0, "ru": 0}
        and audit["teacher_source_counts"]["T_EN"]
        == {"ja": 200, "en": 200, "ru": 0}
        and audit["teacher_source_counts"]["T_RU"]
        == {"ja": 200, "en": 0, "ru": 200}
        and audit["teacher_target_slot_positions_identical"] is True
        and audit["fixed_byte_vocab"] == 256
        and audit["schedule"]
        == {
            "base_steps": 400,
            "teacher_steps": 400,
            "phase1_steps": 200,
            "phase2_steps": 600,
            "eval_every": 10,
        }
    )


def mean_present(rows, key):
    vals = [row[key] for row in rows if row[key] is not None]
    return sum(vals) / len(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    missing = []
    seed_rows = []
    audit_pass = True
    corpus_hash_reference = None

    for seed in SEEDS:
        summary_path = args.input_dir / f"natural_s2_seed{seed}_summary.csv"
        audit_path = args.input_dir / f"natural_s2_seed{seed}_audit.json"
        if not summary_path.exists() or not audit_path.exists():
            missing.append(seed)
            continue

        rows = read_csv(summary_path)
        by_cond = {row["condition"]: row for row in rows}
        if sorted(by_cond) != sorted(CONDS) or len(rows) != 4:
            raise ValueError(f"seed {seed}: malformed summary conditions")
        if any(as_bool(row["smoke_only"]) for row in rows):
            raise ValueError(f"seed {seed}: smoke result cannot enter adjudication")

        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        seed_audit_pass = expected_audit(audit)
        audit_pass = audit_pass and seed_audit_pass
        hashes = audit["corpus_sha256"]
        if corpus_hash_reference is None:
            corpus_hash_reference = hashes
        else:
            audit_pass = audit_pass and hashes == corpus_hash_reference

        ten = by_cond["JA_TEN"]
        tja = by_cond["JA_TJA"]
        nat = by_cond["JA_nat"]
        tru = by_cond["JA_TRU"]

        span_ten = finite_float(ten["span_updates"])
        span_tja = finite_float(tja["span_updates"])
        span_nat = finite_float(nat["span_updates"])
        span_tru = finite_float(tru["span_updates"])
        censored = (
            as_bool(ten["censored"])
            or as_bool(tja["censored"])
            or span_ten is None
            or span_tja is None
        )

        primary = None if censored else span_ten - span_tja
        saving = (
            None
            if censored or span_tja is None or span_tja <= 0
            else 1.0 - span_ten / span_tja
        )
        practical = (
            None if span_ten is None or span_nat is None else span_ten - span_nat
        )
        specificity = (
            None if span_ten is None or span_tru is None else span_ten - span_tru
        )

        seed_rows.append(
            {
                "seed": seed,
                "primary_effect": primary,
                "saving_vs_TJA": saving,
                "favorable_primary_sign": primary is not None and primary < 0,
                "censored_primary": censored,
                "JA_TEN_nll": float(ten["JA_post_nll"]),
                "JA_TJA_nll": float(tja["JA_post_nll"]),
                "practical_effect": practical,
                "specificity_effect": specificity,
                "EN_step0_TEN_minus_TJA": float(ten["EN_step0_nll"])
                - float(tja["EN_step0_nll"]),
                "audit_pass": seed_audit_pass,
            }
        )

    if missing:
        result = {
            "verdict": "UNCERTAIN",
            "reason": "missing fixed confirmatory seeds",
            "missing_seeds": missing,
            "n_complete": len(seed_rows),
        }
    else:
        n_favorable = sum(row["favorable_primary_sign"] for row in seed_rows)
        sign_p = exact_one_sided_sign_p(n_favorable, 10)
        savings = [row["saving_vs_TJA"] for row in seed_rows]
        all_savings = all(value is not None for value in savings)
        mean_saving = sum(savings) / 10 if all_savings else None
        mean_ja_ten = sum(row["JA_TEN_nll"] for row in seed_rows) / 10
        mean_ja_tja = sum(row["JA_TJA_nll"] for row in seed_rows) / 10
        safety = mean_ja_ten <= 1.01 * mean_ja_tja
        no_censoring = not any(row["censored_primary"] for row in seed_rows)

        passed = (
            n_favorable >= 9
            and sign_p <= 0.05
            and mean_saving is not None
            and mean_saving >= 0.05
            and audit_pass
            and safety
            and no_censoring
        )
        result = {
            "verdict": "PASS" if passed else "FAIL",
            "n_favorable_primary_sign": n_favorable,
            "exact_one_sided_sign_p": sign_p,
            "mean_saving_vs_TJA": mean_saving,
            "mean_JA_TEN_nll": mean_ja_ten,
            "mean_JA_TJA_nll": mean_ja_tja,
            "japanese_safety_pass": safety,
            "all_audits_pass": audit_pass,
            "no_primary_censoring": no_censoring,
            "mean_practical_effect_TEN_minus_nat": mean_present(
                seed_rows, "practical_effect"
            ),
            "mean_specificity_effect_TEN_minus_TRU": mean_present(
                seed_rows, "specificity_effect"
            ),
            "mean_EN_step0_TEN_minus_TJA": sum(
                row["EN_step0_TEN_minus_TJA"] for row in seed_rows
            )
            / 10,
            "seeds": seed_rows,
            "corpus_sha256": corpus_hash_reference,
        }

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
