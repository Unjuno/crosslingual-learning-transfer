# Recent results snapshot — 2026-09-01

This note summarizes the hidden-transfer / distance-recovery / mechanism sequence that occurred after the repository's initial public snapshot. It is intentionally concise and preserves negative results.

## Strong replicated findings

### Teacher-mediated hidden transfer

A Student receives only A inputs during phase 1. Teacher hidden targets differ according to whether the Teacher knows A+B or only A/control W. Later-B learning is evaluated independently.

- GRU Teacher_AB vs Teacher_A: 5/5 families, p=.03125.
- Transformer: original and independent replication both 5/5.
- Transformer Teacher_AB vs Teacher_AW target-specific control: 5/5.

### Entropy-matched interior distance window

Rows were modified while preserving the row probability multiset and entropy exactly.

- seed10900: interior-minus-endpoints mean -1.644 B-step, 5/5, p=.03125.
- seed11000 independent replication: mean -1.990, 5/5, p=.03125.
- a later opportunistic cohort was 4/5; therefore the effect is replicated but not deterministic across seeds.

### B-decodable subspace geometry

At d=.5, norm-preserving hidden-delta placement in the B-head row space improved later-B learning relative to:

- the B-head orthogonal complement,
- an equal-rank random subspace,
- an equal-rank A-head subspace.

The main contrasts passed fresh confirmatory tests and independent replications. The B-head subspace generally retained performance close to the original teacher signal.

### Source-target matching and zero-shot distance recovery

Teacher signals constructed for B_s transfer best to a matching later target B_t. This remained true when all B variants were exactly equidistant from A.

A transfer->distance calibration learned once generalized without refit across:

- fresh seeds,
- unseen subset-overlap topology,
- TV(A,B) shells 1/3, 1/2, and 2/3,
- a changed transition-destination operator,
- unseen A-family templates.

Typical pooled zero-shot MAE was roughly .06-.11 TV with high rank accuracy in these synthetic families.

### SUPPORT > WEIGHT

The behavioral distance is not generic TV.

- seed17900: SUPPORT Spearman .920 vs WEIGHT .414; family difference 5/5, p=.03125.
- seed18000 replication: SUPPORT .885 vs WEIGHT .506; family difference 5/5, p=.03125.

The current interpretation is a **support-sensitive structural dissimilarity**, not total variation itself.

## Important failures

The following hypotheses were explicitly tested and not supported:

- simple monotonic scalar distance law,
- target-specific surface-only teaching as a robust mechanism,
- boundary-cut and special-row scalar explanations,
- first-order gradient alignment,
- prospective hidden-MSE + teacher-B-NLL predictor,
- Bsub geometry as a complete explanation of the distance-window shape,
- strict monotonic 5x5 TV recovery,
- general TV recovery for within-support probability rearrangement,
- stationary-distribution shift as the fixed-support-count residual mechanism,
- fixed additive row weights,
- long-sequence context as the sole residual mechanism.

## Current active test: state-to-signal binding

At L=1, the teacher hidden signal is kept in the B-decodable subspace and its norm is preserved, but the mapping from current state to signal direction is cycle-averaged.

seed18400 family0 pilot:

- ALIGNED mean matching penalty: 2.369 B-step.
- CYCLEAVG mean matching penalty: .124 B-step.
- range and SD of identity residuals both shrank by ~58%.
- audits passed.

The original pilot was formally FAIL because the protocol required the matching penalty to remain >=.5 B-step; instead it nearly vanished. This motivated a **new**, separately locked claim that persistent state-to-signal binding is necessary for target-specific matching transfer.

seed18500 is the fresh five-family confirmatory test for that new claim. It is not complete as of this snapshot and is not counted as evidence.

## Current interpretation

The evidence increasingly points to a transfer geometry that is:

- directed,
- learner-conditioned,
- support-sensitive,
- target-decodable,
- and likely dependent on local state/successor-to-representation binding rather than one global scalar graph statistic.

Natural-language and large-LLM generalization remain open.
