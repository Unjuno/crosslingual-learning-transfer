# Claim-to-evidence matrix — final public map

**As of 2026-09-10.** [Final authority](FINAL_STATUS_2026-09-10.md). Verification level describes available evidence, not effect strength, peer review or complete training reproducibility.

**A:** compact adjudication/record arithmetic is recomputable from committed cells and a verifier. **B:** historical compact evidence only; relevant raw artifacts or complete execution variants are absent. **C:** blocked / not counted. All levels exclude claims of automatically reproduced historical learning.

| Claim | Final status | Level | Protocol / committed evidence | Public verification boundary |
|---|---|---|---|---|
| A-side interventions can help or harm later B learning | Supported in controlled systems | B | `protocols/gru_aligned_transfer_confirmatory.json`; `results/cross_architecture_confirmatory_summary.csv`; `results/signed_transfer_harness_cross_cohort_summary.csv`; `results/evidence_summary.csv` | Representative code and numerical summaries, not every original trace |
| Shared representation/decoder coordinates affect performance-aligned transfer | Supported in tested settings | B | `protocols/performance_aligned_learning_minimal_gru.json`; `results/performance_aligned_cross_cohort_summary.csv`; `results/decoder_alignment_summary.csv`; `results/coordinate_alignment_cross_cohort_summary.csv` | Historical mechanism snapshots; fresh-head nonreplication is retained |
| Specified H32/H48 width boundary | Not supported: FAIL | A | `protocols/random_rank_hidden_width_confirmatory_2026-09-02.json`; `results/random_rank_hidden_width_confirmatory_family_slopes.csv`; `results/random_rank_hidden_width_confirmatory_summary.csv` | Final synthetic verifier recomputes paired family/seed adjudication |
| Selected state effective-rank predictor generalizes | Not supported: FAIL | A | `protocols/prestate_state_rank_predictor_confirmatory_2026-09-03.json`; `results/prestate_state_rank_predictor_confirmatory_family.csv`; corresponding summary | Final synthetic verifier recomputes centered Spearman statistics |
| Random-rank dependence appears in the small causal Transformer | Supported in tested cohorts | A | `protocols/random_rank_transformer_replication_2026-09-03.json`; `results/random_rank_transformer_replication_family.csv`; corresponding summary | Final synthetic verifier checks signs/slopes; a negative slope is not strict monotonicity |
| Teacher-mediated hidden transfer and target-decodable subspace interventions | Supported in documented synthetic settings | B | `docs/EXPERIMENT_LOG.md`; `results/evidence_summary.csv` | Not every hidden-teacher helper, raw trace or checkpoint is published |
| Behavioral dissimilarity recovers some held-out synthetic structure | Supported within tested shifts | B | Distance-recovery, topology/shell/operator rows of `results/evidence_summary.csv` | Summary evidence, not all calibration matrices |
| Support changes matter more than fixed-support probability changes | Supported in tested comparisons | B | `results/support_vs_weight_seed18000.json`; `results/evidence_summary.csv` | Not a generic total-variation metric or universal language distance |
| Stable state–signal binding confirmed at seed18500 | BLOCKED / NOT COUNTED | C | `protocols/state_signal_cycle_confirm_seed18500.md`; strict runner/adjudicator | Three exact helper snapshots absent; pilot not promoted to confirmation |
| Earlier target-aware Japanese-window pilot beats ordinary Japanese | Not supported: FAIL | B | `protocols/natural_ja_en_blended5.json`; `results/natural_ja_en_blended5_confirmatory_tests.csv` | Corpus not redistributed; primary failure preserved |
| Natural R1 English-side benefit | Constrained sub-result; overall FAIL | A | `protocols/natural_ja_en_s2_hidden_teacher_r1_2026-09-06.json`; `results/natural_r1_confirmatory_seed_summary.csv`; audit summary and adjudication JSON | `experiments/verify_natural_r1_confirmatory.py`; 10/10 favorable, but Japanese-retention gate fails |
| R2 weight0.5 satisfies registered utility criterion | PASS on registered grid | A | `protocols/natural_ja_en_s2_halfweight_tradeoff_2026-09-06.json`; `results/natural_r2_halfweight_seed_summary_2026-09-08.csv`; execution audits and verification JSON | `experiments/verify_halfweight_public_2026_09_08.py`; point estimate, not ≥5% population guarantee |
| R2 five-percent margin survives dense timing | Not established: mean 4.5692% | A | `results/natural_r2_halfweight_dense_spans_2026-09-08.csv`; `results/natural_r2_measurement_diagnostics_2026-09-08.json` | `experiments/verify_halfweight_dense_2026_09_08.py`; reused seeds, diagnostic only |
| Single 50:50 Student midpoint meets its exploratory criterion | FAIL_EXPLORATORY | B | `protocols/r2_measurement_midpoint_exploratory_2026-09-08.json`; measurement diagnostics JSON | No full midpoint-curve reconstruction in compact checkout |
| Fixed body retains endpoint advantage under common initial head | EXPLORATORY_ARCHIVED_DIAGNOSTIC | A | `research/archived_adaptation_locus_2026_09_09/protocols/HEAD_SWAP_PLAN.json`; same directory `results/factorial_cells.csv`, `results/head_swap_analysis.json` | Its `experiments/verify_records.py` checks 2×2 arithmetic; local-plan chronology, zero new independent seeds |
| Independent UD-corpus confirmation | NOT_RUN | — | Pinned preparation script / source references only | No UD outcome |
| Target-unaware Japanese-text-only, human or large-LLM generality | NOT_ESTABLISHED | — | No passing experiment for these claims | S2 target-aware teaching is not S0 or S1 |

## One verification entry point

From the repository root:

```bash
python scripts/verify_publication.py
```

It runs five compact verification families, record tests and publication guards. It does not rerun training or supply absent raw traces/checkpoints. [Reproduction boundaries](REPRODUCIBILITY.md) and [repository map](REPOSITORY_MAP.md) provide direct commands and historical reports. Dated reports and the older cross-phase ledger remain unchanged as history.
