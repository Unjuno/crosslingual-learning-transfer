# Final synthetic-phase status — 2026-09-03

This note freezes the current synthetic mechanism-search phase. It is intentionally a research record rather than a paper: positive results, failed confirmatory hypotheses, and unresolved reproducibility blockers are kept together.

## Bottom line

The controlled experiments support a **signed, directed, learner-conditioned structural transfer geometry**. They do not support a universal symmetric scalar language distance.

The most durable findings are:

- A-only learning can make later B learning faster or slower.
- Performance-aligned transfer depends on shared representation / decoder coordinates.
- B-decodable hidden geometry causally affects later-B sample efficiency.
- Source-target matching in the hidden-transfer channel predicts synthetic B-B structural dissimilarity.
- The recovered dissimilarity is strongly support-sensitive and is not generic total variation.
- Shared-coordinate rank dependence is not GRU-specific: a fresh causal-Transformer replication passed in two independent cohorts.

At the same time, learner-conditioned variability cannot currently be reduced to a simple hidden-width threshold or a simple pre-curriculum state-embedding effective-rank scalar.

## Final three closure experiments

### 1. Hidden-width boundary confirmation — FAIL

Locked protocol: `protocols/random_rank_hidden_width_confirmatory_2026-09-02.json`.

Fresh seeds 20000, 20100, 20200, 20300, 20400; H=32 vs H=48; five families each.

The prespecified seed statistic was `D_s = max_f slope(H32) - max_f slope(H48)` and required `D_s > 0` in all five fresh seeds.

Observed D values:

| seed | D | H32 negative families | H48 negative families |
|---:|---:|---:|---:|
| 20000 | +0.085181 | 5/5 | 5/5 |
| 20100 | -0.000143 | 5/5 | 5/5 |
| 20200 | -0.007747 | 5/5 | 5/5 |
| 20300 | -0.050168 | 4/5 | 3/5 |
| 20400 | -0.003254 | 5/5 | 5/5 |

Primary result: **FAIL**. Only 1/5 seeds had D>0. A sharp H=48 stability boundary must not be claimed.

Audits passed across all 50 runs: maximum orthogonal-rotation error 1.073e-6; keep0 retained-energy absolute error 0; keep12 retained-energy absolute error 1.03e-7.

### 2. Simple pre-state predictor confirmation — FAIL

Exactly four pre-phase1 scalar candidates were screened on the completed width-confirmatory cohort. The strongest exploratory candidate was state-embedding effective rank (overall Spearman approximately -0.256). It was then frozen as the only candidate for a fresh test.

Locked protocol: `protocols/prestate_state_rank_predictor_confirmatory_2026-09-03.json`.

Fresh seeds 21000–21400; H=32/48; five families each. Within each seed and width, predictor and rank-slope outcome were centered across families before the prespecified Spearman test.

Observed seed-level rho:

| seed | rho |
|---:|---:|
| 21000 | +0.6485 |
| 21100 | -0.2727 |
| 21200 | -0.3818 |
| 21300 | +0.0424 |
| 21400 | -0.0788 |

Pooled centered rho = **+0.0364**.

Primary result: **FAIL**. The simple effective-rank predictor did not generalize. No multivariate rescue, nonlinear fit, extra feature search, or rescue seed was performed.

Interpretation: learner-conditioned variability is real, but the constrained low-dimensional pre-state predictors tested here do not robustly explain it.

### 3. Transformer shared-coordinate rank replication — PASS

The GRU random-rank experiment was ported to a one-layer causal Transformer while retaining the same synthetic families, A-only target curriculum, state-coordinate rotation intervention, keep dimensions {0,3,6,12}, decoder alignment rho=.25, and 80–20 performance-aligned rank-slope metric.

A previously used seed5400/family0 run was used only for implementation sanity. Fresh outcomes were then locked in `protocols/random_rank_transformer_replication_2026-09-03.json`.

| fresh seed | mean rank slope | negative families | exact one-sided sign p |
|---:|---:|---:|---:|
| 22000 | -0.382130 | 5/5 | .03125 |
| 22100 | -0.252719 | 5/5 | .03125 |

Primary result: **PASS in both independent cohorts**.

Rotation audits passed across all ten Transformer runs: maximum orthogonal error 8.35e-7; keep0 retained-energy error 0; keep12 retained-energy error 7.83e-8.

This supports architecture generality of the *shared-coordinate rank dependence itself*. It does not make the transfer geometry learner-independent: the failed width-boundary and predictor tests show substantial learner/initialization conditioning remains.

## State-to-signal binding status

The seed18400 pilot remains mechanistically interesting: cycle-averaging state-to-Bsub-signal directions nearly removed the matching benefit. A separate seed18500 confirmatory protocol was locked.

However, the public repository snapshot is missing three exact archived helper modules required to rerun that locked experiment:

- `teacher_hidden_entropy_matched_distance_transformer.py`
- `teacher_hidden_geometry_intervention.py`
- `support_stationary_matrix.py`

The repository therefore treats seed18500 as **BLOCKED / NOT COUNTED**, rather than reconstructing a nominally equivalent implementation after the fact. PR #2 contains the adjudicator/runner and explicit blocker checks.

## Synthetic-phase stopping decision

The synthetic mechanism search is now frozen. Additional width grids, scalar predictors, family-specific rescues, or post-hoc mechanism fishing are not planned.

The current synthetic claim is deliberately narrower than the original motivating idea:

> Later-learning behavior defines a useful directed, learner-conditioned, support-sensitive structural transfer geometry in the tested neural systems; it is not a universal scalar language distance.

If the project continues, the next phase should be a separate **natural-language / pretrained-model transfer** program. The existing Japanese→English 95/5 pilot did not beat ordinary Japanese on its preregistered primary comparison, so natural-language acceleration remains unestablished.

## Reproducibility boundary

Final closure runs were executed CPU-only with PyTorch 2.10.0+cpu, NumPy 2.3.5, pandas 2.2.3, deterministic PyTorch algorithms, and one PyTorch thread per job. Compact family-level and adjudication CSVs are committed; large raw per-step curves and checkpoints are not.
