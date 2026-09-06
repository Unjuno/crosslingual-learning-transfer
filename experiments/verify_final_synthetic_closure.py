#!/usr/bin/env python3
"""Recompute the final synthetic-phase closure verdicts from committed family-level CSVs.

This verifier intentionally uses only the Python standard library. It does not rerun
training. Its purpose is to make the public repository self-checking at the level of
committed family-level adjudication data and to detect drift between raw-enough result
CSVs, summary CSVs, locked protocols, and current_status.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PROTOCOLS = ROOT / "protocols"
TOL = 1e-12


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def as_bool(x: str) -> bool:
    if x == "True":
        return True
    if x == "False":
        return False
    raise ValueError(f"not a canonical bool: {x!r}")


def close(a: float, b: float, tol: float = TOL) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tol)


def average_ranks(values):
    indexed = sorted(enumerate(values), key=lambda t: t[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg = ((i + 1) + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg
        i = j
    return ranks


def pearson(x, y):
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    den = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    if den == 0:
        raise ValueError("undefined correlation: zero variance")
    return sum(a * b for a, b in zip(dx, dy)) / den


def spearman(x, y):
    return pearson(average_ranks(x), average_ranks(y))


def exact_one_sided_sign_p(k: int, n: int) -> float:
    return sum(math.comb(n, j) for j in range(k, n + 1)) / (2 ** n)


def verify_locked_protocol(path: Path, expected_seeds):
    p = read_json(path)
    assert p["status"] == "LOCKED_BEFORE_FRESH_OUTCOMES", path
    assert p["fresh_seeds"] == expected_seeds, (path, p["fresh_seeds"])
    return p


def verify_width_boundary():
    expected_seeds = [20000, 20100, 20200, 20300, 20400]
    verify_locked_protocol(
        PROTOCOLS / "random_rank_hidden_width_confirmatory_2026-09-02.json",
        expected_seeds,
    )
    family = read_csv(RESULTS / "random_rank_hidden_width_confirmatory_family_slopes.csv")
    summary = {int(r["seed"]): r for r in read_csv(RESULTS / "random_rank_hidden_width_confirmatory_summary.csv")}

    grouped = defaultdict(list)
    for r in family:
        grouped[(int(r["seed"]), int(r["hidden_width"]))].append(float(r["rank_slope"]))

    d_positive = 0
    h48_all_negative = True
    rows = []
    for seed in expected_seeds:
        s32 = grouped[(seed, 32)]
        s48 = grouped[(seed, 48)]
        assert len(s32) == len(s48) == 5
        m32 = max(s32)
        m48 = max(s48)
        d = m32 - m48
        n32 = sum(v < 0 for v in s32)
        n48 = sum(v < 0 for v in s48)
        dpos = d > 0
        h48neg = m48 < 0
        d_positive += int(dpos)
        h48_all_negative = h48_all_negative and h48neg

        ref = summary[seed]
        assert close(m32, float(ref["M_H32"]))
        assert close(m48, float(ref["M_H48"]))
        assert close(d, float(ref["D"]))
        assert n32 == int(ref["n_negative_H32"])
        assert n48 == int(ref["n_negative_H48"])
        assert dpos == as_bool(ref["primary_D_positive"])
        assert h48neg == as_bool(ref["H48_all_negative"])
        rows.append({"seed": seed, "D": d, "n_negative_H32": n32, "n_negative_H48": n48})

    verdict = "PASS" if d_positive == 5 and h48_all_negative else "FAIL"
    assert verdict == "FAIL"
    return {"verdict": verdict, "D_positive_seeds": d_positive, "rows": rows}


def verify_prestate_predictor():
    expected_seeds = [21000, 21100, 21200, 21300, 21400]
    verify_locked_protocol(
        PROTOCOLS / "prestate_state_rank_predictor_confirmatory_2026-09-03.json",
        expected_seeds,
    )
    family = read_csv(RESULTS / "prestate_state_rank_predictor_confirmatory_family.csv")
    summary = {int(r["seed"]): r for r in read_csv(RESULTS / "prestate_state_rank_predictor_confirmatory_summary.csv")}

    by_seed = defaultdict(list)
    pooled_x, pooled_y = [], []
    for r in family:
        seed = int(r["seed"])
        x = float(r["state_rank_centered"])
        y = float(r["rank_slope_centered"])
        by_seed[seed].append((x, y, float(r["rank_slope"])))
        pooled_x.append(x)
        pooled_y.append(y)

    negative_seed_rhos = 0
    seed_rows = []
    for seed in expected_seeds:
        cells = by_seed[seed]
        assert len(cells) == 10
        rho = spearman([c[0] for c in cells], [c[1] for c in cells])
        n_negative_slopes = sum(c[2] < 0 for c in cells)
        ref = summary[seed]
        assert close(rho, float(ref["rho_seed_centered"]))
        assert (rho < 0) == as_bool(ref["rho_negative"])
        assert n_negative_slopes == int(ref["n_negative_rank_slopes"])
        negative_seed_rhos += int(rho < 0)
        seed_rows.append({"seed": seed, "rho": rho, "n_negative_rank_slopes": n_negative_slopes})

    pooled_rho = spearman(pooled_x, pooled_y)
    reference_pooled = float(next(iter(summary.values()))["pooled_rho"])
    assert close(pooled_rho, reference_pooled)
    assert all(close(float(r["pooled_rho"]), reference_pooled) for r in summary.values())

    verdict = "PASS" if negative_seed_rhos == 5 and pooled_rho <= -0.20 else "FAIL"
    assert verdict == "FAIL"
    return {
        "verdict": verdict,
        "negative_seed_rhos": negative_seed_rhos,
        "pooled_rho": pooled_rho,
        "rows": seed_rows,
    }


def verify_transformer_replication():
    expected_seeds = [22000, 22100]
    verify_locked_protocol(
        PROTOCOLS / "random_rank_transformer_replication_2026-09-03.json",
        expected_seeds,
    )
    family = read_csv(RESULTS / "random_rank_transformer_replication_family.csv")
    summary = {int(r["seed"]): r for r in read_csv(RESULTS / "random_rank_transformer_replication_summary.csv")}

    by_seed = defaultdict(list)
    for r in family:
        by_seed[int(r["seed"])].append(float(r["rank_slope"]))

    rows = []
    all_pass = True
    for seed in expected_seeds:
        slopes = by_seed[seed]
        assert len(slopes) == 5
        mean_slope = sum(slopes) / len(slopes)
        n_negative = sum(v < 0 for v in slopes)
        p = exact_one_sided_sign_p(n_negative, len(slopes))
        passed = n_negative == 5
        all_pass = all_pass and passed

        ref = summary[seed]
        assert close(mean_slope, float(ref["mean_rank_slope"]))
        assert n_negative == int(ref["n_negative"])
        assert len(slopes) == int(ref["n_families"])
        assert close(p, float(ref["one_sided_sign_p"]))
        assert passed == as_bool(ref["pass_all5"])
        rows.append({"seed": seed, "mean_rank_slope": mean_slope, "n_negative": n_negative, "p": p})

    verdict = "PASS" if all_pass else "FAIL"
    assert verdict == "PASS"
    return {"verdict": verdict, "rows": rows}


def main():
    observed = {
        "hidden_width_boundary": verify_width_boundary(),
        "prestate_state_effective_rank_predictor": verify_prestate_predictor(),
        "transformer_random_rank_replication": verify_transformer_replication(),
    }

    status = read_json(RESULTS / "current_status_2026-09-03.json")
    expected = {
        "hidden_width_boundary": observed["hidden_width_boundary"]["verdict"],
        "prestate_state_effective_rank_predictor": observed["prestate_state_effective_rank_predictor"]["verdict"],
        "transformer_random_rank_replication": observed["transformer_random_rank_replication"]["verdict"],
    }
    assert status["synthetic_phase"] == "FROZEN"
    assert status["final_closure_tests"] == expected

    print(json.dumps({"synthetic_phase": "FROZEN", "final_closure_tests": observed}, indent=2, sort_keys=True))
    print("OK: committed family-level data, summaries, locked protocols, and current_status agree.")
    print("NOTE: this verifies adjudication data; it does not rerun training or reconstruct omitted raw audit logs.")


if __name__ == "__main__":
    main()
