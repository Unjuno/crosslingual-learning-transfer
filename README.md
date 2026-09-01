# Crosslingual Learning Transfer

**Can learning system A make system B easier — or harder — to learn later, even when B is not shown during the A-only intervention?**

This repository collects controlled neural-network experiments on that question. It is an **experimental evidence snapshot**, not a finished curriculum method and not evidence about human language learning.

## Current headline

The project has moved beyond an existence test. In the documented synthetic Transformer/GRU systems:

- A-only learning can create **positive, neutral, or negative** later-B transfer.
- Stable sample-efficiency transfer depends on **shared / B-decodable representation geometry**, not only on starting loss.
- A teacher-mediated A-side hidden signal can carry target-specific information without exposing B examples to the Student during phase 1.
- Transfer-derived dissimilarity can recover held-out B-B structural distances across fresh seeds, unseen overlap topologies, changed A-distance shells, a changed transition operator, and unseen A-family templates.
- That behavioral distance is **not a generic total-variation metric**: it is much more sensitive to transition-support changes than to probability-weight rearrangements on a fixed support.
- Several simple mechanistic reductions have failed, including scalar distance laws, stationary-distribution shift, fixed row weights, and long-sequence context as the sole explanation.

The strongest current interpretation is therefore a **signed, directed, learner-conditioned structural transfer geometry**, not a universal symmetric language-distance scalar.

## Selected current evidence

| Result | Status | Main evidence |
|---|---|---|
| A-only target-compatible curricula can improve later B learning | **Supported** | Transformer 7/8 fresh seeds at d=.25; GRU 8/8 |
| Target-incompatible A curricula can harm later B learning | **Supported** | signed-transfer calibration + independent replication |
| Performance-aligned transfer depends on shared coordinates | **Supported** | base-coordinate cohorts replicated; completely fresh B-head result did not replicate |
| Teacher hidden(A) from an A+B teacher improves later B learning vs A-only teacher | **Supported** | GRU 5/5; Transformer confirmatory + independent replication |
| Very low-bandwidth teacher hidden signals retain transfer | **Supported conditionally** | 2D/quantized hidden-signal studies; target-specific q4 later-B contrasts replicated |
| Entropy-matched distance has an interior transfer window | **Supported in two confirmatory cohorts** | seeds 10900 and 11000: 5/5 families, p=.03125 each; later opportunistic cohort 4/5 |
| B-decodable hidden subspace causally amplifies transfer | **Supported** | Bsub vs Borth, equal-rank random, and equal-rank A-head controls; confirmatory + replication |
| A fixed transfer->distance calibration recovers unseen B-B distances | **Supported in the tested synthetic family** | zero-shot cohorts across seeds/topology/shell/operator/unseen A families; MAE roughly .06-.11 TV |
| Transfer distance is equally sensitive to all TV components | **Not supported** | within-row geometry failed; SUPPORT > WEIGHT replicated 5/5, p=.03125 |
| Stationary-distribution shift explains fixed-support-count residuals | **Not supported** | fresh seed18200 mean family Spearman=.12, 3/5 positive |
| Long sequence context is the main source of identity residuals | **Not supported** | L=6 -> L=1 did not reduce residual dispersion |
| Stable state-to-teacher-signal binding is required for matching transfer | **Promising, not yet confirmed** | seed18400 pilot: cycle-averaging reduced mean matching penalty 2.369 -> .124 B-step; seed18500 confirmatory is in progress |
| Natural Japanese -> English acceleration over ordinary Japanese | **Not supported** | preregistered 95/5 gain comparison failed |

Lower AUC / fewer B steps means better later-B learning in the reported sample-efficiency experiments.

The compact cross-phase ledger is in [`results/evidence_summary.csv`](results/evidence_summary.csv). Positive findings and failures are intentionally kept together.

## Current mechanism picture

```text
A-only input
   |
   v
teacher-conditioned hidden signal on A
   |
   +--> amount/bandwidth matters somewhat
   |
   +--> B-decodable subspace placement matters strongly
   |
   +--> persistent state <-> signal-direction binding may matter
   v
Student representation after phase 1
   |
   v
later B learning speed
```

A second, related line treats later-learning behavior as a probe of task structure:

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

This behavioral dissimilarity generalizes surprisingly well within the synthetic setup, but it is not yet an exact metric and does not uniformly recover fine probability-weight differences.

## Important negative results

The project deliberately retains failed hypotheses. Among the more informative failures:

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
- multi-token context as the sole identity-residual mechanism.

These failures narrow the mechanism toward **state/successor-specific representation and optimization interactions**.

## Natural-language status

A Japanese -> English pilot selected Japanese-only windows using an offline English transition motif. A 95/5 blend passed safety checks and differed from a Russian-target control on one contrast, but it **did not establish positive acceleration over ordinary Japanese**.

Human learning, large production LLMs, and natural-language generality remain untested.

## Repository map

- `docs/CLAIMS_AND_LIMITATIONS.md` — current claim ledger.
- `docs/EXPERIMENT_LOG.md` — chronological experiment phases and decision points.
- `docs/RECENT_RESULTS_2026-09-01.md` — compact update for the hidden-transfer/distance/mechanism sequence.
- `docs/REPRODUCIBILITY.md` — environment assumptions and data boundaries.
- `docs/TEST_HARNESS.md` — controls and harness audits.
- `experiments/` — representative execution snapshots.
- `protocols/` — selected locked protocols, including current state-signal binding tests.
- `results/` — compact PASS/FAIL summaries; large raw logs/checkpoints are excluded.

## Reproduction

```bash
python -m pip install -r requirements.txt
```

The scripts are execution snapshots rather than a polished library. Several retain `/mnt/data/...` paths from the original execution environment; see [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) before rerunning them.

## Scope and claim discipline

**Supported:** controlled synthetic neural-network existence results, several representation-geometry mechanisms, and support-sensitive behavioral distance recovery under the documented setups.

**Not supported:** a universal language-distance law, exact metric structure, decoder-independent universal meta-learning, a proven natural-language curriculum, human-learning effects, or generality to large production LLMs.

## Data and licensing

Code in this repository is released under Apache-2.0. External natural-language corpora are **not redistributed** here; they remain subject to their original licenses.
