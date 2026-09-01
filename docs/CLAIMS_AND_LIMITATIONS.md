# Claims and limitations

This file is the claim ledger for the project. It separates direct experimental facts, mechanistic inference, active tests, and explicit non-claims.

## Supported experimental facts

1. **Directed transfer exists in the controlled systems.** A-only interventions can improve or worsen later B learning.
2. **Positive, neutral, and negative controls are distinguishable by the same harness.** Exact repeated-neutral controls can produce zero paired difference while target-compatible and anti-target curricula move in opposite directions.
3. **Performance-aligned later-B sample efficiency is a useful primary outcome.** Comparing the number of B steps required to traverse the same B-performance interval removes a major starting-performance confound.
4. **Shared representation/decoder coordinates matter.** Base-coordinate transfer replicated; completely fresh B-head transfer did not replicate stably. Decoder alignment dose, coordinate rotation, and preserved rank all modulate transfer in the synthetic GRU system.
5. **Teacher-mediated hidden representation transfer exists.** A Student trained only on A inputs can later learn B faster when its A-side hidden target comes from a teacher that knows A+B rather than A alone. This was confirmed in GRU and Transformer settings with no B examples shown to the Student during phase 1.
6. **The hidden channel can be compressed substantially.** Low-dimensional and quantized hidden signals retain measurable transfer; however, the strongest target-specific claims depend on the exact compression/control setting.
7. **An entropy-matched interior distance window is reproducible in the tested Transformer system.** Two locked cohorts (seeds 10900 and 11000) showed the prespecified interior-vs-endpoint contrast in 5/5 families with p=.03125 each. A later cohort not preregistered for this claim was 4/5, so the window should not be described as deterministic across seeds.
8. **B-decodable hidden geometry has a causal effect.** At fixed interior distance, norm-preserving placement of the teacher hidden delta in the B-head row space outperformed its orthogonal complement, equal-rank random subspaces, and an equal-rank A-head subspace. The key contrasts replicated independently.
9. **Distance and B-decodable geometry are approximately additive in the tested factorization.** Both main effects are large while the distance x subspace interaction is small in two prospective cohorts. Bsub geometry does not by itself explain the entire distance-window shape.
10. **Source-target matching matters.** Teacher signals constructed for B_s transfer best when the later target is matching B_t; mismatch grows as B_s and B_t diverge, including designs where every B has the same distance from A.
11. **Transfer behavior can recover held-out synthetic B-B distances.** Family-specific calibration, a fixed global calibration, unseen overlap topology, changed A-distance shells, a changed transition operator, and unseen A-family templates all produced useful zero-shot distance recovery in the tested synthetic family.
12. **That recovered distance is support-sensitive rather than generic TV.** When B-B variation came from transition-support changes, recovery was strong. When support was fixed and only probability weights were rearranged, rank recovery deteriorated. SUPPORT > WEIGHT was confirmed and independently replicated (5/5 family difference, p=.03125 in each adjudication cohort).
13. **Several simple residual mechanisms are not supported.** Fresh tests rejected stationary-distribution shift, fixed additive row weights, and long-sequence context as sufficient explanations for fixed-support-count identity residuals.
14. **Natural Japanese -> English acceleration over ordinary Japanese is not established.** The 95/5 Japanese-only bridge passed safety checks and beat a Russian-target control on one contrast, but failed the preregistered primary against ordinary Japanese.

## Active but not yet confirmed

- **Persistent state-to-teacher-signal binding.** In a family0 pilot, cycling Bsub teacher-signal directions across changed states while preserving per-state norm and information budget reduced the mean target-specific matching penalty from 2.369 to 0.124 B-step. The original pilot gate failed because the intervention removed the whole matching effect rather than only the residual dispersion. A separate fresh confirmatory claim (seed18500) was locked before outcomes; it is currently incomplete and must not be counted as PASS.

## Mechanistic inference

The evidence is most consistent with transfer being encoded in a distributed, target-decodable representation whose usefulness depends on **where** information is written and on a **state/successor-specific binding** between structural coordinates and learner representation. This is stronger than a pure starting-loss artifact and stronger than a simple support-Hamming count, but weaker than a universal task-independent meta-learning mechanism.

The behavioral distance recovered from transfer is best described, at present, as a **support-sensitive, learner-conditioned structural dissimilarity**. It is not established as an exact metric or as total variation itself.

## Explicitly not established

- A single scalar language distance uniquely determines transfer.
- Transfer is symmetric; A->B need not equal B->A.
- Transfer-derived dissimilarity is an exact metric; triangle-inequality violations remain.
- Total variation is the intrinsic quantity measured by transfer.
- Support mismatch count alone is sufficient.
- Stationary-distribution shift, gradient alignment, hidden MSE, teacher B-NLL, or fixed row weights provide a general mechanism predictor.
- Any nonzero shared-coordinate alignment is universally sufficient.
- The observed boundaries and effect sizes are architecture universal.
- Japanese curriculum modifications have been shown to accelerate later English over an ordinary Japanese curriculum.
- The result generalizes to human education or large production LLMs.

## Main open questions

1. Does the state-to-teacher-signal binding effect replicate across all five families in seed18500?
2. Which local representation quantity explains fixed-support-count identity residuals: successor embedding geometry, decoder-state alignment, local curvature, or another non-additive interaction?
3. Why does the entropy-matched distance window peak in the interior if Bsub geometry is approximately additive with distance?
4. How learner-dependent is the recovered structural distance across architecture, initialization, optimizer, capacity, and pretraining state?
5. Can the synthetic support-sensitive transfer geometry be mapped to natural linguistic structure without leaking target-language data into the training intervention?
