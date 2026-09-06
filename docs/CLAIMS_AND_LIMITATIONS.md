# Claims and limitations

This file is the public claim ledger. The synthetic phase was frozen on 2026-09-03; the first prospective natural-text S2 cohort was completed on 2026-09-06. Direct facts, failed locked gates, mechanistic inference, blocked tests, and explicit non-claims are kept separate.

## Supported experimental facts

1. **Directed transfer exists in the controlled synthetic systems.** A-only interventions can improve or worsen later B learning.
2. **Positive, neutral, and negative controls are distinguishable by the same harness.** Exact repeated-neutral controls can produce zero paired difference while target-compatible and anti-target curricula move in opposite directions.
3. **Performance-aligned later-B sample efficiency is a useful primary outcome.** Comparing the B steps required to traverse the same B-performance interval removes a major starting-performance confound.
4. **Shared representation/decoder coordinates matter.** Base-coordinate transfer replicated; completely fresh B-head transfer did not replicate stably. Decoder alignment dose, coordinate rotation, and preserved rank all modulate transfer.
5. **Shared-coordinate rank dependence is not GRU-specific in this synthetic setup.** A one-layer causal Transformer replication passed in two locked fresh cohorts: seeds 22000 and 22100 each had negative rank slope in 5/5 families (exact one-sided p=.03125 each).
6. **Teacher-mediated hidden representation transfer exists in the controlled synthetic systems.** A Student trained only on A inputs can later learn B faster when its A-side hidden target comes from a teacher that knows A+B rather than A alone. This was confirmed in GRU and Transformer settings with no B examples shown to the Student during phase 1.
7. **The hidden channel can be compressed substantially.** Low-dimensional and quantized hidden signals retain measurable transfer; the strongest target-specific claims depend on the exact compression/control setting.
8. **An entropy-matched interior distance window is reproducible in the tested Transformer system.** Two locked cohorts (seeds 10900 and 11000) showed the prespecified interior-vs-endpoint contrast in 5/5 families with p=.03125 each. A later opportunistic cohort was 4/5, so the effect is not deterministic across seeds.
9. **B-decodable hidden geometry has a causal effect.** At fixed interior distance, norm-preserving placement of the teacher hidden delta in the B-head row space outperformed its orthogonal complement, equal-rank random subspaces, and an equal-rank A-head subspace. The key contrasts replicated independently.
10. **Distance and B-decodable geometry are approximately additive in the tested factorization.** Both main effects are large while the distance × subspace interaction is small in two prospective cohorts. Bsub geometry does not fully explain the distance-window shape.
11. **Source-target matching matters.** Teacher signals constructed for B_s transfer best when the later target is matching B_t; mismatch grows as B_s and B_t diverge, including designs where every B has the same distance from A.
12. **Transfer behavior can recover held-out synthetic B-B distances.** Family-specific calibration, a fixed global calibration, unseen overlap topology, changed A-distance shells, a changed transition operator, and unseen A-family templates all produced useful zero-shot distance recovery in the tested synthetic family.
13. **That recovered synthetic distance is support-sensitive rather than generic TV.** When B-B variation came from transition-support changes, recovery was strong. When support was fixed and only probability weights were rearranged, rank recovery deteriorated. SUPPORT > WEIGHT was confirmed and independently replicated (5/5 family difference, p=.03125 in each adjudication cohort).
14. **The earlier target-aware Japanese-window pilot did not establish Japanese -> English acceleration over ordinary Japanese.** The 95/5 bridge passed safety checks and beat a Russian-target control on one contrast, but failed its preregistered primary against ordinary Japanese.
15. **A prospective natural-text S2 hidden-teacher cohort showed a strong English-side efficacy pattern, but the locked study still failed overall.** In a domain-limited Vim/TeX software-translation corpus, an English-aware teacher supplied hidden targets while the Student's pre-English update source remained Japanese only. JA_TEN beat the matched Japanese-only teacher control JA_TJA on the common performance-aligned English span in 10/10 fixed fresh seeds (exact one-sided p=.0009765625), with mean span saving 6.3521%. All audits passed and no seed was censored. However, the preregistered Japanese-safety gate failed, so the locked overall verdict is **FAIL**, not PASS.

## Confirmatory failures that constrain the claim

1. **A sharp H=48 learner-width stability boundary is not supported.** In the locked fresh H32-vs-H48 test, the prespecified D_s>0 condition held in only 1/5 seeds. Seed20300 also produced nonnegative family slopes at both widths.
2. **State-embedding effective rank is not a robust scalar pre-state predictor.** It was the best of exactly four constrained exploratory candidates, but the fresh confirmatory seed-level centered Spearman values were +.648, -.273, -.382, +.042, -.079 and pooled centered rho was +.036.
3. **Several earlier simple residual mechanisms are not supported.** Fresh tests rejected stationary-distribution shift, fixed additive row weights, and long-sequence context as sufficient explanations for fixed-support-count identity residuals. Other failed reductions include simple scalar distance laws, first-order gradient alignment, hidden-MSE + teacher-B-NLL prediction, and generic TV recovery for fixed-support probability rearrangements.
4. **The first natural-text S2 hidden-teacher study failed its locked overall gate because of Japanese retention relative to the Japanese-specialist teacher.** Mean post-phase1 Japanese eval NLL was 1.950955 for JA_TEN versus 1.929050 for JA_TJA, a 1.1355% degradation. The locked maximum was 1.0%. The excess over the permitted mean ceiling was 0.002614 NLL. This threshold is not relaxed after seeing the result.

## Natural S2 interpretation boundary

The R1/R2 natural-text study is **S2**, not target-unaware S0:

- the Student's phase-0 and phase-1 update source was Japanese only;
- the tokenizer was fixed byte-level and not fit on English;
- the English-aware teacher was explicitly trained with English-source updates and its hidden states were used as phase-1 targets on Japanese batches.

Therefore a positive English effect is evidence that **English target knowledge can be transmitted through an A-side/Japanese hidden-teaching channel**. It is not evidence that ordinary Japanese data, or a target-unaware Japanese curriculum, spontaneously creates the same benefit.

The English-aware condition also started phase 2 with lower English NLL than JA_TJA by about 0.0983 NLL on average. Knowledge prepositioning and subsequent learning dynamics are therefore both part of the observed S2 effect. The common-interval span reduces, but does not conceptually erase, the distinction between those components.

Post-adjudication diagnostics show that JA_TEN was already below the reference-high threshold at phase2 step 0 in only 1/10 seeds; mean low-threshold crossing was still earlier for JA_TEN (134.171 updates) than JA_TJA (145.489). These are descriptive diagnostics, not additional confirmatory gates.

## Blocked / not counted

- **Persistent state-to-teacher-signal binding.** The seed18400 family0 pilot remains promising: cycle-averaging Bsub teacher-signal directions while preserving per-state norm nearly removed the target-specific matching benefit. A separate seed18500 confirmatory claim was locked before outcomes. However, the public snapshot lacks three exact archived helper modules required for a faithful rerun. The project does not substitute a reconstructed implementation into a locked confirmatory test. Seed18500 is therefore **BLOCKED / NOT COUNTED**.

## Mechanistic inference

The synthetic evidence is most consistent with transfer being encoded in a distributed, target-decodable representation whose usefulness depends on **where** information is written and on learner-specific representation/optimization state. Shared-coordinate rank dependence survives a GRU-to-Transformer architecture change, but its realized family/seed strength is not captured by a simple hidden-width threshold or by state-embedding effective rank.

The natural S2 result is directionally consistent with that picture: target-aware hidden supervision on Japanese-sourced inputs changes both pre-English performance and the number of English updates needed to traverse a common performance interval. Because the locked safety gate fails and the corpus is domain-limited, it should be treated as evidence for a **target-aware transfer channel with a tradeoff**, not as a finished curriculum method.

The behavioral distance recovered from synthetic transfer is best described as a **support-sensitive, directed, learner-conditioned structural dissimilarity**. It is not established as an exact metric or as total variation itself.

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
- A target-unaware Japanese-text-only curriculum has been shown to accelerate later English.
- The natural S2 result has passed its full preregistered utility/safety criterion; it has not.
- The domain-limited Vim/TeX result generalizes to broad natural-language corpora, larger pretrained language models, or other language pairs.
- The result generalizes to human education.

## Stopping and next-decision rule

The synthetic mechanism-search phase remains frozen. No additional synthetic width grids, rescue seeds, scalar-predictor fishing, or family-specific mechanism searches are planned.

The completed natural S2 cohort is not retroactively rescued. A further experiment is justified only as a **new prospective tradeoff question**, for example whether a lower hidden-teaching strength can retain the English benefit while satisfying the same Japanese-safety criterion. Such a study must use new seeds and a newly locked protocol and must preserve the original R1/R2 FAIL verdict.

A move to target-unaware Japanese-text-only S0/S1 claims is premature until the S2 benefit/safety tradeoff is resolved or explicitly accepted as a negative result.

See:

- `docs/FINAL_SYNTHETIC_STATUS_2026-09-03.md`
- `docs/NATURAL_R1_STATUS_2026-09-06.md`
- `docs/INFORMATION_BOUNDARIES.md`
