# Crosslingual Learning Transfer

**Can learning A prepare a model to learn B more efficiently later?**

This is a public experimental research record, not a finished curriculum product, a universal language-distance law, or evidence about human learning. Positive results, failed criteria, and unresolved reproducibility limits remain visible together.

## Latest result — 2026-09-08

The previously locked **natural S2 halfweight follow-up passed its registered criteria** on fresh seeds **32000–32009**:

| Criterion | Observed |
|---|---:|
| English performance-interval improvement versus Japanese-teacher control | **10/10 seeds** |
| One-sided sign-test probability | **0.0009765625** |
| Mean per-seed English interval saving | **5.1881%** |
| Median saving; seed range | **5.8849%; 0.7798–10.5482%** |
| Japanese phase1 mean-NLL penalty versus Japanese-teacher control | **0.6293%**, below the locked 1% limit |
| Frozen-corpus and execution audits | **10/10 pass** |
| Overall registered decision | **PASS** |

The original weight-1.0 R1 study **remains FAIL**: its English-side saving was 6.3521%, but Japanese phase1 loss worsened by 1.1355%, beyond its locked 1% limit. The new trial used the **already published weight-0.5 protocol**, not a retrospective relaxation of R1. The two trials used different seed cohorts, so their difference is not a paired causal dose-response estimate.

A descriptive paired-seed 95% bootstrap interval for the new mean saving is **3.08–7.37%**. The registered 5% condition is a point-estimate criterion; the result does not guarantee at least 5% saving in the underlying population.

See [the new trial report](docs/NATURAL_R2_HALFWEIGHT_2026-09-08.md), [the registered protocol](protocols/natural_ja_en_s2_halfweight_tradeoff_2026-09-06.json), and [the original R1 report](docs/NATURAL_R1_STATUS_2026-09-06.md).

## What “Japanese-only” does and does not mean

Before English adaptation, the Student is updated on **Japanese-sourced samples only**. However, the English-aware teacher is allowed to transmit English-derived information through hidden-state targets. Japanese software text can itself contain ASCII and Latin fragments.

This is **S2: target-aware hidden teaching through Japanese inputs**, not target-unaware Japanese-text-only learning. The experiment uses small byte-level models and frozen Vim/TeX software translation strings. It does not establish broad natural-language, human-learning, or production-LLM generality.

The retention gate is measured **after Japanese phase1 and before English adaptation**. It is not a guarantee against forgetting during subsequent English-only training. English interval savings exclude teacher preparation, earlier Japanese training and wall-clock overhead.

Read [information boundaries](docs/INFORMATION_BOUNDARIES.md) before interpreting the results.

## Synthetic phase: frozen, not discarded

The synthetic mechanism-search phase was frozen on 2026-09-03. Within the documented systems, the evidence supports signed, directed, learner-conditioned and support-sensitive transfer behavior:

- A-side interventions can improve or harm subsequent B learning.
- Shared/target-decodable representation geometry and coordinate interventions affect transfer.
- Shared-coordinate rank dependence was observed in GRU and in two fresh causal-Transformer cohorts. A negative fitted slope does not mean every rank increment is monotonically beneficial.
- Transfer behavior recovers some held-out synthetic structural differences, but does not uniformly recover probability-weight changes on fixed supports.

Several tested reductions failed, including a simple monotonic distance law, a proposed hidden-width stability boundary, and a selected effective-rank predictor. These failures constrain those particular hypotheses; they do not prove that every possible scalar description is impossible or that a specific alternative mechanism has been identified.

The locked state–signal binding test at seed18500 remains **BLOCKED / NOT COUNTED** because exact helper snapshots are missing. A reconstructed implementation would be a separate study, not a recovery of that locked run.

The [final synthetic report](docs/FINAL_SYNTHETIC_STATUS_2026-09-03.md), [historical evidence ledger](results/evidence_summary.csv), and [claim/evidence matrix](docs/CLAIM_EVIDENCE_MATRIX.md) preserve the earlier scope and failures. Historical claim documents describe their stated snapshot date; the Sep-8 halfweight report adds the new natural-study result without rewriting the old adjudications.

## Bounded measurement diagnostics

Separate Sep-8 protocols investigate archived Student-weight averaging, evaluation resolution, exact-line-filtered documents, output-head restoration and byte-class loss decomposition. These are explicitly exploratory or post-adjudication diagnostics, **not additional independent cohorts**, and do not replace either the original R1 FAIL or the registered halfweight decision.

The single 50:50 Student-parameter midpoint is not the same operation as a hidden-loss weight of 0.5. Its original exploratory gate failed because the mean interval saving was below 5%, despite satisfying the phase1 Japanese-retention criterion. No midpoint coefficient sweep is authorized by that protocol.

## Public verification

Recompute the new compact result and run its malformed/negative record guards:

```bash
python experiments/verify_halfweight_public_2026_09_08.py
python -m unittest discover -s experiments -p 'test_halfweight_public_2026_09_08.py' -v
```

Earlier committed adjudications remain independently available:

```bash
python experiments/verify_final_synthetic_closure.py
python experiments/verify_natural_r1_confirmatory.py
```

These standard-library checks verify **committed record arithmetic**, not end-to-end model training. During the Sep-8 halfweight execution, a separate verifier recomputed all spans from the raw 3,050 curve records and reevaluated all 40 phase1 Student checkpoints, with zero discrepancy. Full execution traces/checkpoints are separate artifacts; compact verification must not be confused with those stronger checks.

Fresh-seed guards intentionally reject reuse of registered outcomes as new evidence. Future same-seed replays must be labeled reproduction. No replay enlarges the independent seed count.

## Reproduction and repository map

- `protocols/` — locked study definitions and separately labeled exploratory plans.
- `experiments/` — execution snapshots, runners, analyzers and record verifiers.
- `results/` — compact seed-level summaries, audits and PASS/FAIL/BLOCKED decisions.
- `docs/NATURAL_R2_HALFWEIGHT_2026-09-08.md` — latest registered trial, uncertainty and limitations.
- `docs/NATURAL_R1_STATUS_2026-09-06.md` — original natural S2 FAIL.
- `docs/FINAL_SYNTHETIC_STATUS_2026-09-03.md` — frozen synthetic phase.
- `docs/REPRODUCIBILITY.md` — environment, omitted artifacts and data boundaries.
- `docs/INFORMATION_BOUNDARIES.md` — target-information paths.

The tested Sep-8 environment was Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5 and pandas 2.2.3, float32 with deterministic algorithms and one PyTorch thread/job. The model has 141,056 parameters, two width-64 causal Transformer layers, four heads, context 128 bytes and batch 16. Hardware clocks were not fixed. Cross-platform bitwise identity is not claimed.

Some historical scripts retain original execution paths. Consult the reproduction documentation before rerunning. Source corpus versions must reproduce the frozen SHA256 manifest; substituting another corpus is a new experiment.

## Scope and licensing

**Supported within tested settings:** controlled transfer effects, selected representation interventions, support-sensitive synthetic structural recovery, and a target-aware natural-text hidden-teaching configuration that passed the stated English-efficiency/Japanese-phase1-retention criterion on one fresh cohort.

**Not established:** target-unaware Japanese-text-only acceleration, universal distance or mechanism laws, savings including all preparation costs, retention throughout later English training, unseen-corpus generality, human effects, or large-production-LLM generality.

Code is Apache-2.0. Third-party natural-language corpus contents are not redistributed here and retain their original licenses.
