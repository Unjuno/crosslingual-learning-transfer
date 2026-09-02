#!/usr/bin/env python3
"""Adjudicate the locked seed18500 state↔teacher-signal confirmatory protocol.

This module intentionally uses only the Python standard library. It consumes the
per-source CSV files emitted by ``state_signal_cycle_adjudication.py`` and
recomputes every preregistered gate without fitting or tuning anything.

Verdicts:
- PASS: complete dataset and all locked scientific/audit criteria pass.
- FAIL: complete dataset but at least one locked criterion fails.
- UNCERTAIN: dataset is incomplete, malformed, duplicated, or non-finite.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


DEFAULT_SEED = 18500
FAMILIES = tuple(range(5))
VARIANTS = tuple(range(6))
MODES = ("Bsub_aligned", "Bsub_cycleavg")

# Locked audit thresholds from protocols/state_signal_cycle_confirm_seed18500.md.
A_DISTANCE = 0.5
A_DISTANCE_TOL = 1e-8
ENTROPY_GAP_TOL = 1e-7
B_HEAD_CHANGE_TOL = 1e-10
COMMON_A_RANGE_TOL = 0.01
NORM_RELERR_TOL = 1e-5
DEGENERACY_TOL = 1e-6

REQUIRED_COLUMNS = {
    "family",
    "seed",
    "source_idx",
    "target_idx",
    "mode",
    "effect",
    "tv_A_source",
    "tv_A_target",
    "entropy_gap_source",
    "entropy_gap_target",
    "common_A_range",
    "b_head_change",
    "max_delta_norm_relerr",
    "degenerate_fraction",
}


@dataclass(frozen=True)
class FamilySummary:
    family: int
    M_aligned: float
    M_cycleavg: float
    D: float
    ratio: float | None
    aligned_nontrivial: bool
    ratio_le_0_60: bool


class DataError(ValueError):
    """Raised when the input cannot support a locked-protocol verdict."""


def _finite_float(row: Mapping[str, str], key: str, source: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataError(f"{source}: invalid numeric field {key!r}: {row.get(key)!r}") from exc
    if not math.isfinite(value):
        raise DataError(f"{source}: non-finite field {key!r}: {value!r}")
    return value


def _int_field(row: Mapping[str, str], key: str, source: str) -> int:
    value = _finite_float(row, key, source)
    ivalue = int(value)
    if float(ivalue) != value:
        raise DataError(f"{source}: expected integer {key!r}, got {value!r}")
    return ivalue


def discover_csvs(inputs: Sequence[str]) -> List[Path]:
    """Expand files, directories, and glob patterns deterministically."""
    found: List[Path] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            found.extend(sorted(p.rglob("*.csv")))
        elif p.is_file():
            found.append(p)
        else:
            found.extend(Path(x) for x in sorted(glob.glob(raw, recursive=True)))
    unique = sorted({p.resolve() for p in found})
    if not unique:
        raise DataError("no CSV inputs found")
    return unique


def load_rows(paths: Sequence[Path], seed: int) -> List[dict]:
    rows: List[dict] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            fields = set(reader.fieldnames or [])
            missing = REQUIRED_COLUMNS - fields
            if missing:
                raise DataError(f"{path}: missing columns: {sorted(missing)}")
            for line_no, raw in enumerate(reader, start=2):
                source = f"{path}:{line_no}"
                row_seed = _int_field(raw, "seed", source)
                if row_seed != seed:
                    # A directory may contain other experiments; ignore them rather than
                    # silently mixing confirmatory seeds.
                    continue
                mode = raw.get("mode", "")
                if mode not in MODES:
                    continue
                parsed = dict(raw)
                parsed["_source"] = source
                rows.append(parsed)
    if not rows:
        raise DataError(f"no rows found for seed={seed} and modes={MODES}")
    return rows


def validate_complete(rows: Sequence[Mapping[str, str]], seed: int) -> None:
    expected = {
        (f, s, t, mode)
        for f in FAMILIES
        for s in VARIANTS
        for t in VARIANTS
        for mode in MODES
    }
    seen: Dict[Tuple[int, int, int, str], str] = {}
    for row in rows:
        source = row.get("_source", "<row>")
        f = _int_field(row, "family", source)
        s = _int_field(row, "source_idx", source)
        t = _int_field(row, "target_idx", source)
        row_seed = _int_field(row, "seed", source)
        mode = row["mode"]
        if row_seed != seed:
            raise DataError(f"{source}: mixed seed {row_seed}, expected {seed}")
        key = (f, s, t, mode)
        if key in seen:
            raise DataError(f"duplicate cell {key}: {seen[key]} and {source}")
        seen[key] = source
    actual = set(seen)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        msg = []
        if missing:
            msg.append(f"missing {len(missing)} cells (first 8: {missing[:8]})")
        if extra:
            msg.append(f"unexpected {len(extra)} cells (first 8: {extra[:8]})")
        raise DataError("; ".join(msg))


def build_effects(rows: Sequence[Mapping[str, str]]) -> Dict[Tuple[int, str, int, int], float]:
    effects: Dict[Tuple[int, str, int, int], float] = {}
    for row in rows:
        source = row.get("_source", "<row>")
        key = (
            _int_field(row, "family", source),
            row["mode"],
            _int_field(row, "source_idx", source),
            _int_field(row, "target_idx", source),
        )
        effects[key] = _finite_float(row, "effect", source)
    return effects


def matching_penalties(
    effects: Mapping[Tuple[int, str, int, int], float], family: int, mode: str
) -> List[float]:
    """Return locked P_j=-C_j values for j=1..5."""
    e00 = effects[(family, mode, 0, 0)]
    penalties: List[float] = []
    for j in range(1, 6):
        ejj = effects[(family, mode, j, j)]
        e0j = effects[(family, mode, 0, j)]
        ej0 = effects[(family, mode, j, 0)]
        c_j = 0.5 * (e00 + ejj - e0j - ej0)
        penalties.append(-c_j)
    return penalties


def summarize_families(effects: Mapping[Tuple[int, str, int, int], float]) -> List[FamilySummary]:
    out: List[FamilySummary] = []
    for family in FAMILIES:
        m_aligned = mean(matching_penalties(effects, family, "Bsub_aligned"))
        m_cycle = mean(matching_penalties(effects, family, "Bsub_cycleavg"))
        if m_aligned <= 0.0:
            ratio = None
        else:
            ratio = m_cycle / m_aligned
        out.append(
            FamilySummary(
                family=family,
                M_aligned=m_aligned,
                M_cycleavg=m_cycle,
                D=m_cycle - m_aligned,
                ratio=ratio,
                aligned_nontrivial=m_aligned >= 1.0,
                ratio_le_0_60=ratio is not None and ratio <= 0.60,
            )
        )
    return out


def audit(rows: Sequence[Mapping[str, str]]) -> dict:
    max_a_distance_dev = 0.0
    max_entropy_gap = 0.0
    max_b_head_change = 0.0
    max_common_a_range = 0.0
    max_norm_relerr = 0.0
    max_degeneracy = 0.0

    for row in rows:
        source = row.get("_source", "<row>")
        max_a_distance_dev = max(
            max_a_distance_dev,
            abs(_finite_float(row, "tv_A_source", source) - A_DISTANCE),
            abs(_finite_float(row, "tv_A_target", source) - A_DISTANCE),
        )
        max_entropy_gap = max(
            max_entropy_gap,
            abs(_finite_float(row, "entropy_gap_source", source)),
            abs(_finite_float(row, "entropy_gap_target", source)),
        )
        max_b_head_change = max(max_b_head_change, abs(_finite_float(row, "b_head_change", source)))
        max_common_a_range = max(max_common_a_range, abs(_finite_float(row, "common_A_range", source)))
        max_norm_relerr = max(max_norm_relerr, abs(_finite_float(row, "max_delta_norm_relerr", source)))
        max_degeneracy = max(max_degeneracy, abs(_finite_float(row, "degenerate_fraction", source)))

    checks = {
        "A_distance_deviation_le_1e-8": max_a_distance_dev <= A_DISTANCE_TOL,
        "entropy_gap_le_1e-7": max_entropy_gap <= ENTROPY_GAP_TOL,
        "B_head_change_le_1e-10": max_b_head_change <= B_HEAD_CHANGE_TOL,
        "common_A_range_le_0_01": max_common_a_range <= COMMON_A_RANGE_TOL,
        "norm_relerr_le_1e-5": max_norm_relerr <= NORM_RELERR_TOL,
        "degeneracy_le_1e-6": max_degeneracy <= DEGENERACY_TOL,
    }
    return {
        "max_A_distance_deviation": max_a_distance_dev,
        "max_entropy_gap": max_entropy_gap,
        "max_B_head_change": max_b_head_change,
        "max_common_A_range": max_common_a_range,
        "max_norm_relerr": max_norm_relerr,
        "max_degeneracy": max_degeneracy,
        "checks": checks,
        "pass": all(checks.values()),
    }


def adjudicate_rows(rows: Sequence[Mapping[str, str]], seed: int = DEFAULT_SEED) -> dict:
    try:
        validate_complete(rows, seed)
        effects = build_effects(rows)
        families = summarize_families(effects)
        audit_result = audit(rows)

        mean_aligned = mean(f.M_aligned for f in families)
        mean_cycle = mean(f.M_cycleavg for f in families)
        pooled_ratio = None if mean_aligned <= 0.0 else mean_cycle / mean_aligned
        n_negative = sum(f.D < 0.0 for f in families)
        n_ratio = sum(f.ratio_le_0_60 for f in families)
        n_nontrivial = sum(f.aligned_nontrivial for f in families)

        criteria = {
            "1_D_negative_5of5": n_negative == 5,
            "2_pooled_ratio_le_0_50": pooled_ratio is not None and pooled_ratio <= 0.50,
            "3_family_ratio_le_0_60_at_least_4of5": n_ratio >= 4,
            "4_M_aligned_ge_1_at_least_4of5": n_nontrivial >= 4,
            "5_audits_pass": audit_result["pass"],
        }
        verdict = "PASS" if all(criteria.values()) else "FAIL"
        return {
            "seed": seed,
            "verdict": verdict,
            "complete": True,
            "criteria": criteria,
            "family_sign_p_one_sided": 2.0 ** -5 if n_negative == 5 else None,
            "n_D_negative": n_negative,
            "pooled_mean_M_aligned": mean_aligned,
            "pooled_mean_M_cycleavg": mean_cycle,
            "pooled_ratio": pooled_ratio,
            "n_family_ratio_le_0_60": n_ratio,
            "n_M_aligned_ge_1": n_nontrivial,
            "families": [asdict(f) for f in families],
            "audits": audit_result,
        }
    except DataError as exc:
        return {
            "seed": seed,
            "verdict": "UNCERTAIN",
            "complete": False,
            "error": str(exc),
        }


def adjudicate_inputs(inputs: Sequence[str], seed: int = DEFAULT_SEED) -> dict:
    try:
        paths = discover_csvs(inputs)
        rows = load_rows(paths, seed)
    except DataError as exc:
        return {"seed": seed, "verdict": "UNCERTAIN", "complete": False, "error": str(exc)}
    result = adjudicate_rows(rows, seed)
    result["input_files"] = [str(p) for p in paths]
    result["n_rows_used"] = len(rows)
    return result


def write_family_csv(path: Path, result: Mapping[str, object]) -> None:
    families = result.get("families")
    if not isinstance(families, list):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "family",
        "M_aligned",
        "M_cycleavg",
        "D",
        "ratio",
        "aligned_nontrivial",
        "ratio_le_0_60",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(families)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="CSV file(s), directory/directories, or glob pattern(s) containing per-source outputs",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--family-csv-out", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = adjudicate_inputs(args.inputs, args.seed)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if args.family_csv_out is not None:
        write_family_csv(args.family_csv_out, result)
    # FAIL is a valid completed scientific outcome, so return 1. UNCERTAIN is a
    # data/infrastructure error and returns 2 for automation pipelines.
    return 0 if result["verdict"] == "PASS" else (1 if result["verdict"] == "FAIL" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
