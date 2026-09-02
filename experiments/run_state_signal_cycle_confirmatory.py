#!/usr/bin/env python3
"""Run the locked seed18500 state↔teacher-signal confirmatory experiment.

The runner launches one subprocess per (family, source) cell and then invokes the
locked adjudicator. It refuses to start real jobs if the archived helper modules
required by the representative experiment snapshot are absent.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import state_signal_cycle_confirmatory_adjudicator as adjudicator


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_SCRIPT = HERE / "state_signal_cycle_adjudication.py"
DEFAULT_OUTDIR = REPO / "results" / "raw" / "state_signal_cycle_seed18500"
REQUIRED_HELPERS = (
    "teacher_hidden_entropy_matched_distance_transformer.py",
    "teacher_hidden_geometry_intervention.py",
    "support_stationary_matrix.py",
)


def parse_int_list(raw: str, allowed: Iterable[int], label: str) -> Tuple[int, ...]:
    allowed_set = set(allowed)
    values = tuple(int(x.strip()) for x in raw.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError(f"{label} must not be empty")
    bad = [x for x in values if x not in allowed_set]
    if bad:
        raise argparse.ArgumentTypeError(f"invalid {label}: {bad}; allowed={sorted(allowed_set)}")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"duplicate {label}: {values}")
    return values


def missing_helpers() -> list[Path]:
    return [HERE / name for name in REQUIRED_HELPERS if not (HERE / name).is_file()]


def expected_csv(outdir: Path, family: int, source: int, seed: int) -> Path:
    return outdir / f"family{family}_seed{seed}_source{source}.csv"


def output_complete(path: Path, family: int, source: int, seed: int) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except (OSError, csv.Error):
        return False
    if len(rows) != 12:
        return False
    cells = set()
    for row in rows:
        try:
            if int(row["family"]) != family or int(row["seed"]) != seed or int(row["source_idx"]) != source:
                return False
            target = int(row["target_idx"])
            mode = row["mode"]
        except (KeyError, ValueError):
            return False
        if target not in range(6) or mode not in adjudicator.MODES:
            return False
        cells.add((target, mode))
    return cells == {(t, m) for t in range(6) for m in adjudicator.MODES}


def command_for(
    python: str,
    script: Path,
    outdir: Path,
    family: int,
    source: int,
    seed: int,
) -> tuple[list[str], dict[str, str]]:
    env = os.environ.copy()
    env.update({
        "FAMILY": str(family),
        "SEED": str(seed),
        "SOURCE": str(source),
        "OUTDIR": str(outdir),
        # Locked confirmatory requires all six targets. Override any inherited
        # partial TARGETS setting to prevent silently incomplete matrices.
        "TARGETS": "0,1,2,3,4,5",
    })
    return [python, str(script)], env


def run_one(
    python: str,
    script: Path,
    outdir: Path,
    family: int,
    source: int,
    seed: int,
    resume: bool,
) -> tuple[int, int, int, str]:
    target = expected_csv(outdir, family, source, seed)
    if resume and output_complete(target, family, source, seed):
        return family, source, 0, "SKIP complete"
    cmd, env = command_for(python, script, outdir, family, source, seed)
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
    detail = (proc.stdout + "\n" + proc.stderr).strip()
    if proc.returncode == 0 and not output_complete(target, family, source, seed):
        return family, source, 3, f"process returned 0 but output is incomplete: {target}\n{detail}"
    return family, source, proc.returncode, detail


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=18500)
    parser.add_argument("--families", default="0,1,2,3,4")
    parser.add_argument("--sources", default="0,1,2,3,4,5")
    parser.add_argument("--jobs", type=int, default=1, help="maximum concurrent subprocesses")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--resume", action="store_true", help="skip already-complete 12-row source outputs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--family-csv-out", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        families = parse_int_list(args.families, range(5), "families")
        sources = parse_int_list(args.sources, range(6), "sources")
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.seed != 18500:
        parser.error("locked confirmatory runner only permits --seed 18500")

    jobs = [(f, s) for f in families for s in sources]
    args.outdir = args.outdir.resolve()
    args.script = args.script.resolve()

    if args.dry_run:
        for family, source in jobs:
            cmd, env = command_for(args.python, args.script, args.outdir, family, source, args.seed)
            print(
                f"FAMILY={env['FAMILY']} SEED={env['SEED']} SOURCE={env['SOURCE']} "
                f"TARGETS={env['TARGETS']} OUTDIR={env['OUTDIR']} " + " ".join(cmd)
            )
        return 0

    missing = missing_helpers()
    if missing:
        print("ERROR: representative experiment snapshot is not runnable from this checkout.", file=sys.stderr)
        print("Missing archived helper modules:", file=sys.stderr)
        for path in missing:
            print(f"  - {path.relative_to(REPO)}", file=sys.stderr)
        print(
            "Restore/publish these exact helper snapshots before running seed18500; "
            "do not substitute reconstructed implementations for a confirmatory rerun.",
            file=sys.stderr,
        )
        return 2
    if not args.script.is_file():
        print(f"ERROR: experiment script not found: {args.script}", file=sys.stderr)
        return 2

    args.outdir.mkdir(parents=True, exist_ok=True)
    failures = []
    with ThreadPoolExecutor(max_workers=min(args.jobs, len(jobs))) as pool:
        futures = {
            pool.submit(
                run_one,
                args.python,
                args.script,
                args.outdir,
                family,
                source,
                args.seed,
                args.resume,
            ): (family, source)
            for family, source in jobs
        }
        for future in as_completed(futures):
            family, source, code, detail = future.result()
            status = "OK" if code == 0 else f"FAIL({code})"
            print(f"family={family} source={source}: {status}", flush=True)
            if detail and (code != 0 or detail.startswith("SKIP")):
                print(detail, flush=True)
            if code != 0:
                failures.append((family, source, code, detail))

    if failures:
        print(f"ERROR: {len(failures)}/{len(jobs)} jobs failed; adjudication not run.", file=sys.stderr)
        return 2

    result = adjudicator.adjudicate_inputs([str(args.outdir)], args.seed)
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if args.family_csv_out is not None:
        adjudicator.write_family_csv(args.family_csv_out, result)
    return 0 if result["verdict"] == "PASS" else (1 if result["verdict"] == "FAIL" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
