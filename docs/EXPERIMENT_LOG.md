# Experiment log

## Phase 0 — directed transfer and structural surrogates

The project started by measuring directed sequential transfer across six character-level language corpora (`de`, `en`, `ja`, `ru`, `es`, `it`). Exact directed-bigram-preserving surrogates reproduced transfer topology much more consistently than unigram/degree-margin controls, supporting a role for specific transition coupling but not bigram sufficiency.

## Phase 1 — A-only curriculum existence tests

Synthetic A/B systems with disjoint surface vocabularies and shared latent states established that A-only interventions can alter later B learning. Transformer and GRU cohorts supported target-compatible transfer, while the near-vs-far interaction was architecture-dependent.

## Phase 2 — signed-transfer harness

The harness was calibrated for positive, neutral, and negative transfer. Matched-wrong controls held source distance, entropy, and transition margins while changing target compatibility; target-compatible curricula beat matched-wrong controls at several distances.

## Phase 3 — performance alignment and coordinate dependence

Performance-aligned hitting time was introduced to compare the B steps needed to traverse the same B-NLL interval. Base-coordinate effects replicated; a completely fresh B decoder did not. Decoder alignment dose, geometry-preserving rotation, and preserved shared rank further localized the mechanism to shared coordinates.

## Phase 4 — natural Japanese -> English pilot

A Japanese-only 95/5 bridge selected by an offline English motif passed safety checks but did not beat ordinary Japanese on the preregistered English acceleration primary. Natural-language positive acceleration remains unestablished.

## Phase 5 — teacher-mediated hidden transfer

The research question was tightened: an A+B teacher and an A-only/control teacher teach the same-init Student using **A inputs only**. The Student then learns B independently.

- GRU hidden(A) transfer: Teacher_AB beat Teacher_A in 5/5 families, p=.03125.
- Transformer hidden(A) transfer: large beneficial effect in an original cohort and an independent replication, both 5/5.
- Target-specific Transformer control (Teacher_AB vs Teacher_AW) also passed 5/5.
- Hidden-signal dimension sweep showed transfer surviving into low-dimensional signals.
- 2-bit sign and q4 quantized channels retained measurable transfer, though not every stricter cross-target symmetry test passed.

This phase established that target-relevant information can be mediated through A-side hidden supervision without B examples being shown to the Student in phase 1.

## Phase 6 — entropy-matched distance window

A synthetic construction varied B relative to A while preserving the row probability multiset and row entropy exactly.

- Simple monotonic scalar distance slope: **FAIL**.
- Prespecified interior-vs-endpoint window: seed10900 **PASS**, 5/5, p=.03125.
- Independent seed11000 replication: **PASS**, 5/5, p=.03125.
- Later seed13800, whose primary purpose was a different predictor validation, showed the same window only 4/5; it is treated as secondary evidence rather than a third confirmatory PASS.

Several proposed explanations failed: boundary-cut mass, special swapped-row count, first-order gradient alignment, and a prospective hidden-MSE + teacher-B-NLL predictor.

## Phase 7 — causal representation geometry

Teacher hidden deltas were norm-preservingly moved between representation subspaces.

- B-head subspace vs B-head orthogonal complement at d=.5: prospective PASS and independent replication.
- B-head subspace vs equal-rank random subspace: confirmatory + independent replication, 5/5 each.
- B-head subspace vs equal-rank A-head subspace: confirmatory + independent replication, 5/5 each.

The full distance-window x subspace interaction did not meet the primary threshold. Subsequent prospective tests instead supported an approximately additive decomposition: large distance main effect + large subspace main effect + comparatively small interaction.

A norm-clamp intervention did not flatten the distance effect, ruling out hidden-delta magnitude as a sufficient explanation.

## Phase 8 — source-target matching and behavioral distance recovery

Teacher signal designed for source B_s was crossed with later target B_t.

- 3x3 source-target matching: confirmatory + independent replication PASS.
- Holding TV(A,B)=.5 for every variant while changing only B-B overlap: both matching and the B-B distance gradient replicated.
- Six-variant held-out recovery: transfer synergy predicted unseen B-B TV pairs in confirmatory and replication cohorts.
- A single calibration learned from seed16400 generalized without refit to fresh seeds, unseen overlap topology, A-distance shells 1/3 and 2/3, a changed transition-destination operator, and unseen A-family templates.

This established useful **behavioral distance recovery inside the synthetic family**, but not a universal scalar metric.

## Phase 9 — support vs probability-weight geometry

A stricter within-row design held the changed-row set fixed and generated B-B TV through within-row support/weight differences. Fine-grained rank recovery failed even when MAE remained deceptively low.

A preregistered SUPPORT-vs-WEIGHT adjudication then separated the two components:

- seed17900: SUPPORT pooled Spearman=.920 vs WEIGHT=.414; family difference 5/5, p=.03125.
- seed18000 independent replication: SUPPORT=.885 vs WEIGHT=.506; family difference 5/5, p=.03125.

Conclusion: transfer-derived dissimilarity is much more sensitive to transition-support structure than to probability rearrangement on a fixed support.

## Phase 10 — failed simple residual mechanisms

With support mismatch count held fixed, several proposed explanations were tested.

- Stationary-distribution TV: exploratory family0 signal did not replicate in fresh seed18200; family Spearman `[+.8,-.5,-.6,+.8,+.1]`, mean=.12 -> **FAIL**.
- A-exposure-weighted support mismatch: inconsistent across families.
- Teacher-signal scalar distances: inconsistent.
- Fixed additive row weights: strong in-fit R2 but failed out-of-fit on new codewords -> **FAIL**.
- Multi-token sequence context: reducing L=6 to L=1 increased rather than reduced identity dispersion -> **FAIL**.

These results move the mechanism away from simple graph scalars and toward local state/successor-specific representation interactions.

## Phase 11 — state-to-signal binding intervention

At L=1, the Bsub teacher hidden signal was modified while preserving per-state norm and information budget. Signal directions were cycled across the six changed states so that each state received all directions equally over time, removing stable state-to-direction binding.

Family0 seed18400 pilot:

- aligned mean matching penalty: **2.369 B-step**,
- cycle-averaged mean matching penalty: **0.124 B-step**,
- identity range/SD shrank by about 58%,
- all leakage/norm audits passed.

The original pilot gate was **FAIL** because it required the matching effect to remain at least .5 B-step; instead the intervention nearly erased the entire matching effect. This was treated as a new hypothesis rather than as a rescued PASS.

A fresh five-family confirmatory protocol (seed18500) was locked before outcomes. At the repository update time it is **in progress** and must not be counted as confirmed evidence.
