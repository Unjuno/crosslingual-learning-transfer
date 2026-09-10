# Repository map

## Current public entry points

| Path | Purpose |
|---|---|
| `README.md`, `README.ja.md` | English and Japanese public summaries |
| `docs/FINAL_STATUS_2026-09-10.md`, `results/FINAL_STATUS.json` | One final status / frozen scope |
| `docs/CLAIM_EVIDENCE_MATRIX.md` | Claim → protocol/result/checker, with availability levels |
| `docs/REPRODUCIBILITY.md` | What can actually be verified or rerun |
| `scripts/verify_publication.py` | Standard-library, no-training public verification |
| `publication/` | Version, release notes, file inventory, scientific-file hashes, archive availability |
| `CITATION.cff`, `CONTRIBUTING.md`, `SECURITY.md` | Citation and maintenance boundaries |

## Historical evidence (kept in place)

[Final synthetic status](FINAL_SYNTHETIC_STATUS_2026-09-03.md), [R1 registered FAIL](NATURAL_R1_STATUS_2026-09-06.md), [R2 qualified registered PASS](NATURAL_R2_HALFWEIGHT_2026-09-08.md), [R2 diagnostic limitations](R2_DIAGNOSTICS_2026-09-08.md), and [archived body/head diagnostic](../research/archived_adaptation_locus_2026_09_09/README_JA.md).

`docs/EXPERIMENT_LOG.md`, date-stamped reports and `results/current_status_*.json` reflect the time they were written. They are not the current authority. `results/evidence_summary.csv` is a historical cross-phase ledger, not a complete latest index. No old result/seed/protocol has been removed, renamed or silently reclassified for publication.

## Direct compact checks

```bash
python experiments/verify_final_synthetic_closure.py
python experiments/verify_natural_r1_confirmatory.py
python experiments/verify_halfweight_public_2026_09_08.py
python experiments/verify_halfweight_dense_2026_09_08.py
python research/archived_adaptation_locus_2026_09_09/experiments/verify_records.py
```

The unified command also runs tests and publication checks. None of these commands launches model training. Training snapshots and corpus preparation scripts remain under `experiments/`; they have separate runtime prerequisites and should not be mass-executed.
