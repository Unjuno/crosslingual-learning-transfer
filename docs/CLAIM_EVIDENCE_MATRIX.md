# Claim-to-evidence matrix

This file maps the public claims to the strongest evidence that is actually committed in the repository. It is deliberately stricter than a prose summary: a claim is not labeled fully reproducible when the training curves, checkpoints, corpus, or exact helper snapshots needed to rerun it are absent.

## Evidence levels

- **A — committed adjudication is recomputable:** family/cell/seed-level adjudication data, locked protocol, analyzer or verifier, and compact summary are all present. Training itself may still be expensive and raw per-step logs may be omitted.
- **B — compact evidence snapshot:** representative code and/or compact summary are present, but the repository does not contain every raw artifact required to reproduce the full historical training run exactly.
- **C — blocked / not counted:** a confirmatory claim cannot be faithfully rerun from the public snapshot and is excluded from supported claims.

## Matrix

| Claim | Status | Evidence level | Protocol / design | Committed result evidence | Public verification boundary |
|---|---|---:|---|---|---|
| A-only interventions can improve or worsen later-B learning in the controlled systems | Supported | B | `protocols/gru_aligned_transfer_confirmatory.json`; signed-transfer controls summarized in the evidence ledger | `results/evidence_summary.csv`; representative scripts in `experiments/` | Historical training outputs are compacted; the repository verifies the recorded ledger, not every original minibatch trace. |
| Performance-aligned transfer depends on shared representation / decoder coordinates | Supported | B | `protocols/performance_aligned_learning_minimal_gru.json`, decoder/rotation protocols | `results/performance_aligned_cross_cohort_summary.csv`, `results/decoder_alignment_summary.csv`, `results/coordinate_alignment_cross_cohort_summary.csv`, `results/evidence_summary.csv` | Representative mechanisms are public; not every historical raw curve/checkpoint is committed. |
| Preserving more shared state-coordinate rank shifts transfer favorably in the original GRU random-rank test | Supported in documented GRU cohort | B | `protocols/random_rank_state_alignment_gru.json` | `results/random_rank_primary_tests.csv`, `results/evidence_summary.csv` | Compact primary result is committed; final closure verifier does not reconstruct the original GRU training curve from omitted raw logs. |
| A sharp H=48 hidden-width stability boundary exists | **Not supported** | **A** | `protocols/random_rank_hidden_width_confirmatory_2026-09-02.json` | `results/random_rank_hidden_width_confirmatory_family_slopes.csv`, `results/random_rank_hidden_width_confirmatory_summary.csv` | `python experiments/verify_final_synthetic_closure.py` recomputes the family-level adjudication. |
| State-embedding effective rank robustly predicts learner-conditioned rank-transfer variation | **Not supported** | **A** | `protocols/prestate_state_rank_predictor_confirmatory_2026-09-03.json` | `results/prestate_state_rank_predictor_confirmatory_family.csv`, `results/prestate_state_rank_predictor_confirmatory_summary.csv` | The verifier recomputes seed-wise and pooled Spearman statistics from committed cells. |
| Shared-coordinate random-rank dependence generalizes from GRU to the documented one-layer causal Transformer | Supported | **A** | `protocols/random_rank_transformer_replication_2026-09-03.json` | `results/random_rank_transformer_replication_family.csv`, `results/random_rank_transformer_replication_summary.csv` | The verifier recomputes mean slopes, signs, and exact one-sided sign probabilities from committed family data. |
| Teacher-mediated A-side hidden signals can improve later B learning when the teacher knows A+B | Supported in documented GRU/Transformer systems | B | Hidden-teacher protocols summarized in `docs/EXPERIMENT_LOG.md` and claim ledger | `results/evidence_summary.csv`; representative hidden-teacher/state-signal scripts | Several historical helper variants/raw outputs are not all published; treat the evidence ledger as a compact executed-study snapshot. |
| Transfer behavior recovers held-out synthetic B-B structural dissimilarity across the documented zero-shot shifts | Supported in tested synthetic family | B | Distance-recovery sequence documented in `docs/EXPERIMENT_LOG.md` | `results/evidence_summary.csv` | Compact outcome summaries are public; the repository is not a complete archive of every intermediate calibration/output matrix. |
| Transfer-derived dissimilarity is more sensitive to support changes than to fixed-support probability-weight rearrangements | Supported in documented synthetic tests | B | SUPPORT-vs-WEIGHT adjudications documented in the experiment log | `results/evidence_summary.csv` | Confirmatory outcome is in the ledger; not all raw matrices are committed. |
| Persistent state-to-teacher-signal binding is confirmed by seed18500 | **Blocked / not counted** | **C** | locked seed18500 protocol and runner/adjudicator safeguards | no complete confirmatory result | Three exact archived helper snapshots are missing. The runner refuses a reconstructed substitute. |
| Earlier target-aware Japanese-window phase-1 curriculum has been shown to accelerate later English over ordinary Japanese | **Not supported** | B | `protocols/natural_ja_en_blended5.json` | `results/natural_ja_en_blended5_confirmatory_tests.csv`, `results/evidence_summary.csv`; `experiments/natural_ja_en_blended5_curriculum.py` | The corpus is not redistributed; the preregistered primary comparison failed. |
| In the domain-limited natural S2 cohort, English-aware hidden teaching on Japanese-sourced Student inputs improves TEN-vs-TJA performance-aligned English span | **Supported as a constrained English-side sub-result** | **A** | `protocols/natural_ja_en_s2_hidden_teacher_r1_2026-09-06.json`; frozen corpus hashes in `protocols/natural_ja_en_s2_corpus_manifest_2026-09-06.json` | `results/natural_r1_confirmatory_seed_summary.csv`, `results/natural_r1_confirmatory_audit_summary.csv`, `results/natural_r1_confirmatory_adjudication.json` | `python experiments/verify_natural_r1_confirmatory.py` recomputes 10/10 favorable signs, p=.0009765625 and mean 6.3521% span saving. Raw checkpoint curves are omitted from the compact public snapshot. |
| The natural S2 cohort passes its full preregistered utility/safety criterion | **Not supported — locked FAIL** | **A** | same locked R1 protocol | same seed/audit/adjudication files | The verifier recomputes the safety failure: mean JA_TEN NLL is 1.1355% worse than JA_TJA, above the locked 1% maximum. The threshold is not relaxed. |
| Target-unaware Japanese text alone has been shown to accelerate later English | **Not established** | — | no passing S0/S1 protocol | none | The completed natural cohort is S2: the English-aware teacher explicitly contains English-derived information. |

## Public adjudication self-checks

Synthetic closure:

```bash
python experiments/verify_final_synthetic_closure.py
```

Natural R1/R2 S2 cohort:

```bash
python experiments/verify_natural_r1_confirmatory.py
```

The first command checks the three final synthetic closure experiments directly from committed family/cell-level CSVs and asserts consistency with their summary CSVs, locked fresh-seed protocols, and `results/current_status_2026-09-03.json`.

The second command recomputes the locked natural S2 efficacy/safety adjudication from committed 10-seed summaries and audits and checks it against `results/natural_r1_confirmatory_adjudication.json`.

These commands **do not rerun training** and **do not recreate omitted checkpoints or raw per-step logs**. That distinction is intentional.
