# Claims and limitations

This file is the claim ledger for the project. It separates direct experimental facts from mechanistic inference and open uncertainty.

## Supported experimental facts

1. **Directed transfer exists in the controlled systems.** A-only curriculum interventions can improve or worsen later B learning.
2. **Positive and negative controls are distinguishable by the same harness.** In GRU calibration cohorts, target-compatible A curricula improved B AUC while anti/incorrect curricula worsened it; an exact repeated-neutral control produced zero difference.
3. **The effect is target-specific beyond simple source displacement.** Source-distance-, entropy-, and margin-matched wrong controls are worse than target-compatible controls at d=.125, .25, and .50 in 8/8 fresh seeds.
4. **A/B coordinate sharing modulates performance-aligned transfer.** With the base coordinate system retained, target-compatible curricula cross the same B-performance interval in fewer B steps across five kernel families and an independent replication.
5. **A completely fresh B decoder removes the stable cross-family result.** Initial fresh-head target-vs-wrong evidence failed independent replication.
6. **Decoder alignment dose matters.** In synthetic GRU experiments, transfer weakens monotonically as cosine alignment rho is reduced from 1 to .5 to .25 to 0; rho=.25 replicated independently.
7. **Within-B geometry alone is insufficient.** Orthogonally rotating B state embeddings preserves norms and pairwise distances but largely removes target advantage relative to the A/shared coordinate system.
8. **Shared latent rank matters.** Randomly preserving more dimensions of the B state-coordinate span shifts the transfer effect in a favorable direction in all five tested families.
9. **No privileged simple subspace was identified.** Gradient-salient decoder directions and high-energy state principal components were not uniquely necessary.
10. **Natural Japanese -> English acceleration over ordinary Japanese is not established.** The 95/5 Japanese-only bridge passed safety checks and beat a Russian-target control on one learning-gain contrast, but did not beat ordinary Japanese on the preregistered primary gain test.

## Structural-surrogate facts

- Exact directed-bigram-preserving realizations were highly reproducible: BG0 vs BG1 raw Cspec rho = 0.928.
- Unigram/degree-margin-preserving realizations were less reproducible: UNI0 vs UNI1 raw Cspec rho = 0.454.
- Natural vs exact-bigram surrogates correlated strongly (raw rho ~.82-.84); unigram controls were weaker and more variable.
- Unigram controls preserve exact unigram counts and directed bigram in/out degree margins, so the comparison is more accurately **specific joint source->target coupling vs margins**, not "bigram vs unigram only".

## Mechanistic inference

The evidence is most consistent with transferable information being stored in a distributed representation that is accessible when A and B share enough coordinate structure. This is stronger than a pure starting-loss artifact, but weaker than universal decoder-independent meta-learning.

## Explicitly not established

- Bigram structure is fully sufficient for crosslingual transfer.
- A single scalar distance uniquely determines transfer.
- Transfer is symmetric: A->B need not equal B->A.
- Any nonzero shared-coordinate alignment is universally sufficient.
- The observed synthetic boundary/functional form is architecture universal.
- Japanese curriculum modifications have been shown to accelerate later English over an ordinary Japanese curriculum.
- The result generalizes to human education or large production LLMs.

## Main open questions

1. Does shared-coordinate-rank dependence reproduce in the Transformer with a minimal 0/6/12 intervention?
2. Which natural linguistic representations play the role of the synthetic shared coordinates?
3. Can a natural Japanese-only curriculum beat a difficulty/frequency-matched ordinary-Japanese control on later English sample efficiency without degrading either language?
4. What predicts the sign of transfer across distances, doses, architectures, and tasks?
