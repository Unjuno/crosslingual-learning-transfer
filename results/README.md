# Public result records

**Current authority: [FINAL_STATUS.json](FINAL_STATUS.json)** and [the final report](../docs/FINAL_STATUS_2026-09-10.md). Date-stamped `current_status_*` files and `evidence_summary.csv` are historical snapshots, not a complete latest index.

## Recompute all public checks

```bash
python scripts/verify_publication.py
```

Run from the repository root. Verification PASS means internal record consistency, not success of every scientific hypothesis.

## Current evidence groups

| Group | Records | Scientific interpretation |
|---|---|---|
| Synthetic closure | `random_rank_hidden_width_confirmatory_*`, `prestate_state_rank_predictor_confirmatory_*`, `random_rank_transformer_replication_*` | Two specified hypotheses failed; small Transformer rank result supported in tested cohorts |
| Natural R1 | `natural_r1_confirmatory_*` | Registered FAIL: Japanese-retention constraint exceeded |
| Natural R2 | `natural_r2_halfweight_seed_summary_2026-09-08.csv`, `natural_r2_halfweight_execution_audits_2026-09-08.csv`, `natural_r2_halfweight_verification_2026-09-08.json` | Registered-grid PASS, not a robust population saving of at least five percent |
| Dense and other R2 diagnostics | `natural_r2_halfweight_dense_spans_2026-09-08.csv`, `natural_r2_measurement_diagnostics_2026-09-08.json` | Every-update mean 4.5692%; archival/diagnostic, not new independent trials |
| Historical mechanisms | `evidence_summary.csv` and earlier compact files | Partial evidence snapshots; not all raw learning is reproducible here |

Archived body/head results are in [research/archived_adaptation_locus_2026_09_09](../research/archived_adaptation_locus_2026_09_09/README_JA.md), not duplicated in this directory. They add zero independent seeds.

See [claim-to-evidence mapping](../docs/CLAIM_EVIDENCE_MATRIX.md) for file-level boundaries. Raw traces, checkpoints and third-party corpora are not included. Scientific CSV/JSON files and locked protocols were not changed by publication cleanup.
