# Result summaries

Compact CSV summaries used by the repository claim ledger. Positive, negative, null, failed, and replication-failure outcomes are retained together.

## Cross-phase index

- `evidence_summary.csv` — compact PASS/FAIL table spanning the main research phases.

## Existence and signed-transfer controls

- `cross_architecture_confirmatory_summary.csv` — Transformer/GRU confirmatory comparison.
- `signed_transfer_harness_cross_cohort_summary.csv` — positive, negative, and neutral harness calibration plus replication.
- `matched_wrong_exact_tests.csv` — source-distance/entropy/margin-matched target-vs-wrong map.

## Performance-aligned and representation mechanism

- `performance_aligned_cross_cohort_summary.csv` — base-head and fresh-head cohorts, including failed fresh-head replication.
- `decoder_alignment_summary.csv` — transfer effect versus retained decoder cosine alignment.
- `coordinate_alignment_cross_cohort_summary.csv` — identity versus geometry-preserving coordinate rotation.
- `shared_coordinate_mechanism_summary.csv` — subspace, partial-rank, top/bottom, and random-rank mechanism tests.
- `random_rank_primary_tests.csv` — preregistered GRU random-rank primary slope and per-rank descriptive tests.

### Final learner-conditioning / architecture closure

- `random_rank_hidden_width_summary.csv` — exploratory cohort-level rank slopes across GRU hidden widths and seeds.
- `random_rank_hidden_width_family_slopes.csv` — family-level exploratory slopes and per-rank effects for the width sweep.
- `random_rank_hidden_width_audit_summary.csv` — orthogonality and retained-state-energy audits for the exploratory width sweep.
- `random_rank_hidden_width_confirmatory_summary.csv` — locked H32-vs-H48 boundary test; **FAIL**.
- `random_rank_hidden_width_confirmatory_family_slopes.csv` — family-level data for that fresh confirmation.
- `prestate_predictor_exploratory_selection.csv` — constrained four-feature pre-state scalar screen; state effective rank selected for one fresh test.
- `prestate_state_rank_predictor_confirmatory_summary.csv` — locked effective-rank predictor confirmation; **FAIL**.
- `prestate_state_rank_predictor_confirmatory_family.csv` — family-level predictor/outcome cells.
- `random_rank_transformer_replication_summary.csv` — fresh causal-Transformer rank replication; **PASS** in seeds 22000 and 22100 (5/5 each).
- `random_rank_transformer_replication_family.csv` — family-level Transformer effects and slopes.

## Structural surrogate phase

- `surrogate_all_realization_topology.csv` — natural/BG/UNI topology correlations.
- `surrogate_exact_seed_resampling_256_summary.csv` — exact 4^4 seed-resampling summaries.
- `surrogate_unigram_degree_margin_checks.csv` — proof-by-audit that UNI controls preserve unigram counts and directed in/out bigram margins.

## Natural-language pilot

- `natural_ja_en_blended5_confirmatory_tests.csv` — Japanese -> English 95/5 pilot; the primary comparison against ordinary Japanese failed.

These are summary artifacts, not raw data dumps. Large checkpoints, raw per-step logs, duplicated intermediate CSVs, and third-party corpora are intentionally excluded.
