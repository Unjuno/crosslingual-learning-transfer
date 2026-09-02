# Claims and limitations

This is the frozen claim ledger for the synthetic phase as of 2026-09-03. It separates direct experimental facts, mechanistic inference, blocked tests, and explicit non-claims.

## Supported experimental facts

1. **Directed transfer exists in the controlled systems.** A-only interventions can improve or worsen later B learning.
2. **Positive, neutral, and negative controls are distinguishable by the same harness.** Exact repeated-neutral controls can produce zero paired difference while target-compatible and anti-target curricula move in opposite directions.
3. **Performance-aligned later-B sample efficiency is a useful primary outcome.** Comparing the B steps required to traverse the same B-performance interval removes a major starting-performance confound.
4. **Shared representation/decoder coordinates matter.** Base-coordinate transfer replicated; completely fresh B-head transfer did not replicate stably. Decoder alignment dose, coordinate rotation, and preserved rank all modulate transfer.
5. **Shared-coordinate rank dependence is not GRU-specific in this synthetic setup.** A one-layer causal Transformer replication passed in two locked fresh cohorts: seeds 22000 and 22100 each had negative rank slope in 5/5 families (exact one-sided p=.03125 each).
6. **Teacher-mediated hidden representation transfer exists.** A Student trained only on A inputs can later learn B faster when its A-side hidden target comes from a teacher that knows A+B rather than A alone. This was confirmed in GRU and Transformer settings with no B examples shown to the Student during phase 1.
7. **The hidden channel can be compressed substantially.** Low-dimensional and quantized hidden signals retain measurable transfer; the strongest target-specific claims depend on the exact compression/control setting.
8. **An entropy-matched interior distance window is reproducible in the tested Transformer system.** Two locked cohorts (seeds 10900 and 11000) showed the prespecified interior-vs-endpoint contrast in 5/5 families with p=.03125 each. A later opportunistic cohort was 4/5, so the effect is not deterministic across seeds.
9. **B-decodable hidden geometry has a causal effect.** At fixed interior distance, norm-preserving placement of the teacher hidden delta in the B-head row space outperformed its orthogonal complement, equal-rank random subspaces, and an equal-rank A-head subspace. The key contrasts replicated independently.
10. **Distance and B-decodable geometry are approximately additive in the tested factorization.** Both main effects are large while the distance × subspace interaction is small in two prospective cohorts. Bsub geometry does not fully explain the distance-window shape.
11. **Source-target matching matters.** Teacher signals constructed for B_s transfer best when the later target is matching B_t; mismatch grows as B_s and B_t diverge, including designs where every B has the same distance from A.
12. **Transfer behavior can recover held-out synthetic B-B distances.** Family-specific calibration, a fixed global calibration, unseen overlap topology, changed A-distance shells, a changed transition operator, and unseen A-family templates all produced useful zero-shot distance recovery in the tested synthetic family.
13. **That recovered distance is support-sensitive rather than generic TV.** When B-B variation came from transition-support changes, recovery was strong. When support was fixed and only probability weights were rearranged, rank recovery deteriorated. SUPPORT > WEIGHT was confirmed and independently replicated (5/5 family difference, p=.03125 in each adjudication cohort).
14. **Natural Japanese -> English acceleration over ordinary Japanese is not established.** The 95/5 Japanese-only bridge passed safety checks and beat a Russian-target control on one contrast, but failed the preregistered primary against ordinary Japanese.

## Confirmatory failures that constrain the claim

1. **A sharp H=48 learner-width stability boundary is not supported.** In the locked fresh H32-vs-H48 test, the prespecified D_s>0 condition held in only 1/5 seeds. Seed20300 also produced nonnegative family slopes at both widths.
2. **State-embedding effective rank is not a robust scalar pre-state predictor.** It was the best of exactly four constrained exploratory candidates, but the fresh confirmatory seed-level centered Spearman values were +.648, -.273, -.382, +.042, -.079 and pooled centered rho was +.036.
3. **Several earlier simple residual mechanisms are not supported.** Fresh tests rejected stationary-distribution shift, fixed additive row weights, and long-sequence context as sufficient explanations for fixed-support-count identity residuals. Other failed reductions include simple scalar distance laws, first-order gradient alignment, hidden-MSE + teacher-B-NLL prediction, and generic TV recovery for fixed-support probability rearrangements.

## Blocked / not counted

- **Persistent state-to-teacher-signal binding.** The seed18400 family0 pilot remains promising: cycle-averaging Bsub teacher-signal directions while preserving per-state norm nearly removed the target-specific matching benefit. A separate seed18500 confirmatory claim was locked before outcomes. However, the public snapshot lacks three exact archived helper modules required for a faithful rerun. The project does not substitute a reconstructed implementation into a locked confirmatory test. Seed18500 is therefore **BLOCKED / NOT COUNTED**.

## Mechanistic inference

The evidence is most consistent with transfer being encoded in a distributed, target-decodable representation whose usefulness depends on **where** information is written and on learner-specific representation/optimization state. Shared-coordinate rank dependence survives a GRU-to-Transformer architecture change, but its realized family/seed strength is not captured by a simple hidden-width threshold or by state-embedding effective rank.

The behavioral distance recovered from transfer is best described as a **support-sensitive, directed, learner-conditioned structural dissimilarity**. It is not established as an exact metric or as total variation itself.

## Explicitly not established

- A single scalar language distance uniquely determines transfer.
- Transfer is symmetric; A->B need not equal B->A.
- Transfer-derived dissimilarity is an exact metric; triangle-inequality violations remain.
- Total variation is the intrinsic quantity measured by transfer.
- Support mismatch count alone is sufficient.
- A simple hidden-width threshold determines whether transfer geometry is stable.
- State-embedding effective rank, stationary-distribution shift, gradient alignment, hidden MSE, teacher B-NLL, or fixed row weights provide a general mechanism predictor.
- Any nonzero shared-coordinate alignment is universally sufficient.
- Effect sizes are architecture-, optimizer-, initialization-, or pretraining-state invariant.
- Japanese curriculum modifications have been shown to accelerate later English over an ordinary Japanese curriculum.
- The result generalizes to human education or large production LLMs.

## Synthetic-phase stopping decision

The synthetic mechanism-search phase is frozen. No additional width grids, rescue seeds, scalar-predictor fishing, or family-specific mechanism searches are planned for the current project sequence.

If the project continues, the highest-value questions are external-validity questions:

1. Can a pretrained learner receive **only Japanese inputs** during phase 1 yet be placed into a state that later learns English faster than after an ordinary Japanese phase-1 curriculum?
2. How much target-language knowledge may the teacher/curriculum designer use while keeping target examples completely hidden from the Student during phase 1?
3. Does any transfer geometry survive changes in tokenizer, model scale, pretraining state, and natural linguistic structure?

See `docs/FINAL_SYNTHETIC_STATUS_2026-09-03.md` for the final closure experiments and exact PASS/FAIL summaries.
