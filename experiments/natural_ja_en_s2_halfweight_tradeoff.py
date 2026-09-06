#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import natural_ja_en_s2_hidden_teacher as core

SEEDS = list(range(32000, 32010))
WEIGHT = 0.5
PROTOCOL = "protocols/natural_ja_en_s2_halfweight_tradeoff_2026-09-06.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    if args.seed not in SEEDS:
        raise SystemExit(f"locked half-weight seeds are {SEEDS}")

    core.HIDDEN_WEIGHT = WEIGHT
    core.CONFIRMATORY_SEEDS = SEEDS
    core.run(
        seed=args.seed,
        data_dir=args.data_dir,
        outdir=args.outdir,
        smoke=False,
        validation_natural=False,
    )

    audit_path = args.outdir / f"natural_s2_seed{args.seed}_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["protocol"] = PROTOCOL
    audit["parent_locked_verdict"] = "FAIL"
    audit["hidden_distillation_weight"] = WEIGHT
    audit["tradeoff_followup"] = True
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"half-weight audit stamped seed={args.seed}", flush=True)


if __name__ == "__main__":
    main()
