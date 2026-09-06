# Crosslingual Learning Transfer

**Can learning system A make system B easier — or harder — to learn later, even when B is not shown during the A-only intervention?**

This repository collects controlled neural-network experiments on that question. It is an **experimental evidence snapshot**, not a finished curriculum method and not evidence about human language learning.

## Current headline

The synthetic mechanism-search phase is **frozen as of 2026-09-03**. The strongest supported synthetic interpretation is a **signed, directed, learner-conditioned, support-sensitive structural transfer geometry**, not a universal symmetric language-distance scalar.

In the documented synthetic Transformer/GRU systems:

- A-only learning can create **positive, neutral, or negative** later-B transfer.
- Stable sample-efficiency transfer depends on **shared / B-decodable representation geometry**, not only on starting loss.
- A teacher-mediated A-side hidden signal can carry target-specific information without exposing B examples to the Student during phase 1.
- Transfer-derived dissimilarity can recover held-out B-B structural distances across fresh seeds, unseen overlap topologies, changed A-distance shells, a changed transition operator, and unseen A-family templates.
- That behavioral distance is **not generic total variation**: it is much more sensitive to transition-support changes than to probability-weight rearrangements on a fixed support.
- Shared-coordinate rank dependence replicated in two fresh **causal Transformer** cohorts, reducing concern that this mechanism is GRU-specific.
- A proposed H=32 vs H=48 stability boundary **failed** fresh confirmation, and a simple pre-curriculum state-embedding effective-rank predictor also **failed** fresh confirmation.

A first prospective natural-text S2 hidden-teacher cohort has also been completed. An English-aware teacher improved the Student's later English performance-aligned span against a matched Japanese-only teacher in **10/10 fixed seeds**, with **6.35% mean span saving**, but the preregistered Japanese-safety gate narrowly failed. The locked natural study verdict is therefore **FAIL**, not PASS.

For the final synthetic-phase adjudication, see [`docs/FINAL_SYNTHETIC_STATUS_2026-09-03.md`](docs/FINAL_SYNTHETIC_STATUS_2026-09-03.md). For the natural S2 result, see [`docs/NATURAL_R1_STATUS_2026-09-06.md`](docs/NATURAL_R1_STATUS_2026-09-06.md).

For public verification scope, see [`docs/CLAIM_EVIDENCE_MATRIX.md`](docs/CLAIM_EVIDENCE_MATRIX.md). For the distinction between “Student sees only A” and “the whole system is target-unaware”, see [`docs/INFORMATION_BOUNDARIES.md`](docs/INFORMATION_BOUNDARIES.md).

## Selected current evidence

| Result | Status | Main evidence |
|---|---|---|
| A-only target-compatible curricula can improve later B learning | **Supported** | Transformer 7/8 fresh seeds at d=.25; GRU 8/8 |
| Target-incompatible A curricula can harm later B learning | **Supported** | signed-transfer calibration + independent replication |
| Performance-aligned transfer depends on shared coordinates | **Supported** | base-coordinate cohorts replicated; completely fresh B-head result did not replicate |
| Random-rank shared-coordinate dependence generalizes beyond GRU | **Supported** | causal Transformer seeds 22000 and 22100: 5/5 families each, exact p=.03125 each |
| A sharp H=48 hidden-width stability boundary exists | **Not supported** | fresh seeds 20000–20400: prespecified D>0 only 1/5 seeds |
| State-embedding effective rank predicts rank-transfer variation | **Not supported** | fresh seeds 21000–21400; pooled centered Spearman = +.036 |
| Teacher hidden(A) from an A+B teacher improves later B learning vs A-only teacher | **Supported in synthetic systems** | GRU 5/5; Transformer confirmatory + independent replication |
| Very low-bandwidth teacher hidden signals retain transfer | **Supported conditionally** | 2D/quantized hidden-signal studies; target-specific q4 later-B contrasts replicated |
| Entropy-matched distance has an interior transfer window | **Supported in two confirmatory cohorts** | seeds 10900 and 11000: 5/5 families, p=.03125 each; later opportunistic cohort 4/5 |
| B-decodable hidden subspace causally amplifies transfer | **Supported** | Bsub vs Borth, equal-rank random, and equal-rank A-head controls; confirmatory + replication |
| A fixed transfer->distance calibration recovers unseen B-B distances | **Supported in the tested synthetic family** | zero-shot cohorts across seeds/topology/shell/operator/unseen A families; MAE roughly .06-.11 TV |
| Transfer distance is equally sensitive to all TV components | **Not supported** | within-row geometry failed; SUPPORT > WEIGHT replicated 5/5, p=.03125 |
| Stationary-distribution shift explains fixed-support-count residuals | **Not supported** | fresh seed18200 mean family Spearman=.12, 3/5 positive |
| Long sequence context is the main source of identity residuals | **Not supported** | L=6 -> L=1 did not reduce residual dispersion |
| Stable state-to-teacher-signal binding is confirmed | **Blocked / not counted** | seed18400 pilot is promising; locked seed18500 rerun requires three missing exact archived helper snapshots |
| Earlier target-aware Japanese-window pilot accelerates English over ordinary Japanese | **Not supported** | preregistered 95/5 gain comparison failed |
| Natural S2 English-aware hidden teacher improves TEN-vs-TJA English span | **Strong constrained sub-result** | seeds 31000–31009: 10/10 favorable, exact p=.0009765625, mean span saving 6.3521% |
| Natural S2 study passes its full utility/safety criterion | **Not supported / locked FAIL** | mean JA_TEN NLL was 1.1355% worse than JA_TJA; locked safety maximum was 1% |

Lower AUC / fewer B steps means better later-B learning in the reported sample-efficiency experiments.

The compact cross-phase ledger is in [`results/evidence_summary.csv`](results/evidence_summary.csv). Positive findings and failures are intentionally kept together.

## Current mechanism picture

```text
A-only input
   |
   v
representation written during phase 1
   |
   +--> decoder/shared-coordinate accessibility matters
   +--> B-decodable subspace placement matters strongly
   +--> learner architecture / initialization modulates the realized effect
   v
Student representation after phase 1
   |
   v
later B learning speed
```

A second line treats later-learning behavior as a probe of task structure:

```text
source teacher B_s
      |
      v
A-side teacher signal -> Student
      |
      v
later target B_t learning
      |
      v
matching synergy / transfer penalty
      |
      +--> predicts support-sensitive B_s <-> B_t dissimilarity
```

This behavioral dissimilarity generalizes surprisingly well within the synthetic setup, but it is not an exact metric and does not uniformly recover fine probability-weight differences.

## Important negative results

The project deliberately retains failed hypotheses, including:

- a simple monotonic scalar distance law,
- surface-only target-specific teaching as a robust mechanism,
- boundary-cut / special-row scalar explanations,
- first-order gradient-alignment prediction,
- a prospective hidden-MSE + teacher-B-NLL mechanism predictor,
- full distance-window explanation by Bsub geometry alone,
- strict monotonic 5x5 TV recovery,
- general TV recovery when distance comes mostly from within-support probability rearrangement,
- stationary-distribution shift as the fixed-support-count mechanism,
- fixed additive row weights,
- multi-token context as the sole identity-residual mechanism,
- a sharp H=48 hidden-width stability boundary,
- state-embedding effective rank as a robust scalar predictor of rank-transfer variation,
- the first natural S2 study's full preregistered utility/safety PASS criterion.

These failures constrain the synthetic mechanism toward **learner-conditioned, state/successor-specific representation and optimization interactions** and constrain the natural result to a **target-aware transfer channel with a small Japanese-specialist tradeoff**, not a finished safe curriculum.

## Natural-language status

The earlier Japanese -> English pilot selected Japanese-sourced windows using an offline English transition motif. A 95/5 blend passed safety checks and differed from a Russian-target control on one contrast, but it **did not establish positive acceleration over ordinary Japanese**.

The prospective R1/R2 S2 cohort is now complete. It uses a fixed byte tokenizer and a domain-limited Vim/TeX software-translation corpus. Before English phase 2, the Student is updated only on Japanese-sourced samples; however, the T_EN teacher is explicitly trained with English and may transmit English-derived information through hidden targets.

Locked result:

- TEN vs TJA span favorable in **10/10** fixed seeds;
- exact one-sided sign p = **.0009765625**;
- mean span saving = **6.3521%**;
- all audits pass; no censoring;
- Japanese safety vs TJA teacher: **FAIL** (TEN mean NLL 1.1355% worse; allowed maximum 1%).

Overall locked verdict: **FAIL**.

This is evidence for an S2 target-aware hidden-teaching channel, not for target-unaware Japanese-text-only acceleration. The English-aware Student also starts phase 2 with better English NLL, so knowledge prepositioning and subsequent learning dynamics both contribute.

Human learning, broad natural-language generality, target-unaware S0/S1 curriculum effects, and large production LLM generality remain unestablished.

## Repository map

- `docs/FINAL_SYNTHETIC_STATUS_2026-09-03.md` — final synthetic-phase adjudication and stopping decision.
- `docs/CLAIMS_AND_LIMITATIONS.md` — public claim ledger including the natural S2 result and its failed safety gate.
- `docs/CLAIM_EVIDENCE_MATRIX.md` — claim-by-claim public evidence/reproducibility level.
- `docs/INFORMATION_BOUNDARIES.md` — target-information paths and A-only/S0–S3 distinctions.
- `docs/NATURAL_R1_STATUS_2026-09-06.md` — locked natural S2 design, execution, verdict, and interpretation boundary.
- `docs/EXPERIMENT_LOG.md` — chronological experiment phases and decision points.
- `docs/RECENT_RESULTS_2026-09-01.md` — earlier hidden-transfer/distance/mechanism snapshot.
- `docs/REPRODUCIBILITY.md` — environment assumptions, exact final-closure package record, and data boundaries.
- `docs/TEST_HARNESS.md` — controls and harness audits.
- `experiments/` — representative execution snapshots, public closure verifiers, and natural R1 preparation/runner/adjudicator scripts.
- `protocols/` — locked protocols and the frozen R1 corpus hash manifest.
- `results/` — compact PASS/FAIL summaries and natural R1 seed/audit/adjudication evidence; large raw logs/checkpoints are excluded.

## Public verification

Recompute the final three synthetic closure adjudications directly from the committed family/cell-level CSVs:

```bash
python experiments/verify_final_synthetic_closure.py
```

Recompute the compact locked natural R1/R2 adjudication from the committed seed summaries/audits:

```bash
python experiments/verify_natural_r1_confirmatory.py
```

The verifiers use only the Python standard library and check the committed evidence against locked protocols/adjudications. These are **adjudication reproductions**, not full training reruns: omitted checkpoints, many historical raw curves, and third-party corpus contents are not reconstructed.

For the closest recorded package versions of the final synthetic closure runs:

```bash
python -m pip install -r requirements-final-synthetic.txt
```

For broader historical scripts and the R1 implementation:

```bash
python -m pip install -r requirements.txt
```

The scripts are execution snapshots rather than a polished library. Several older files retain paths from the original execution environment; see [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) before rerunning them.

## Scope and claim discipline

**Supported:** controlled synthetic neural-network existence results, several representation-geometry mechanisms, cross-architecture shared-coordinate rank dependence, support-sensitive behavioral distance recovery under the documented synthetic setups, and a strong target-aware S2 English-side hidden-teacher effect in one domain-limited natural-text cohort.

**Not supported:** a universal language-distance law, exact metric structure, a simple hidden-width threshold, a robust scalar pre-state mechanism predictor, decoder-independent universal meta-learning, a target-unaware Japanese-only curriculum, a fully passed natural utility/safety criterion, broad natural-language or large-production-LLM generality, or human-learning effects.

## Data and licensing

Code in this repository is released under Apache-2.0. External natural-language corpora are **not redistributed** here; they remain subject to their original licenses.
