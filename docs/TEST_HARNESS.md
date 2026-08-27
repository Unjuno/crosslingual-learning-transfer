# Test harness and statistical checks

The project treats the experiment harness itself as something that must be tested.

## Functional controls

The signed-transfer calibration contains three reference behaviors:

1. **Positive control:** target-compatible A curriculum should improve later B under the calibrated near-distance condition.
2. **Neutral metamorphic control:** rerunning exactly the same ordinary-A condition with identical inputs/seeds should produce exactly the same recorded metrics.
3. **Negative control:** target-incompatible/anti-target A curriculum should worsen later B under the calibrated condition.

The neutral repeat produced a maximum recorded difference of zero in the calibration audit.

## Matched-wrong invariants

The stricter wrong-target construction matches the target-compatible kernel on source displacement and simple distributional properties, including directed row/column margins and row entropy, while changing source->target transition correspondence. These invariants are used to reduce the "wrong is merely farther/harder" alternative explanation.

## Geometry-preserving rotation audit

The coordinate-rotation experiments use orthogonal transformations of B state embeddings. Across the adjudication cohort, orthogonality and pairwise-distance errors were on the order of 1e-6 or smaller. This preserves within-B geometry while changing orientation relative to A/shared coordinates.

## Statistical-harness bug found and corrected

One newly written family-level exact sign-flip analyzer initially counted sign-flipped means relative to zero instead of comparing the permutation statistic to the **observed mean**. This could return p=.5 even for a 5/5 same-direction result.

- Training outputs were unaffected.
- The analyzer was corrected.
- A repository-wide pattern audit found no other analyzer using the same erroneous comparison pattern.

This incident is retained in the research record because harness verification is part of the methodology, not an implementation detail to hide.

## Claim policy

A result is not promoted from exploratory to confirmatory simply because it looks favorable. Several apparent effects were rerun on fresh cohorts and explicitly downgraded when replication failed; the fully fresh B-head performance-aligned effect is the main example.
