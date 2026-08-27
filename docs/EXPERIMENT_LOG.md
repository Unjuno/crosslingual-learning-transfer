# Experiment log

## Phase 0 — directed transfer and structural surrogates

The project started by measuring directed sequential transfer across six character-level language corpora (`de`, `en`, `ja`, `ru`, `es`, `it`). The key methodological shift was to treat transfer as A->B rather than as a symmetric distance.

Controlled surrogate families were then introduced:

- **BG**: exact directed character-bigram multiset preserved by randomized Euler trails.
- **UNI**: sequence interior shuffled while token counts/endpoints are preserved. Because endpoints and counts are fixed, this also preserves directed bigram row/column degree margins.

Main outcome: BG realizations reproduced transfer topology much more consistently than UNI realizations. This supported a causal role for specific transition coupling, but simple scalar bigram distances did not cleanly explain pair-specific residual transfer.

## Phase 1 — A-only curriculum existence tests

A synthetic A/B system was constructed with disjoint surface vocabularies and shared latent states. During the curriculum intervention, the model receives A-surface tokens only. Some A steps can be generated from B-compatible latent transitions.

- Transformer fresh seeds 100-107: target-compatible vs ordinary A at d=.25, 7/8, p=.0078125.
- GRU fresh seeds 900-907: 8/8, p=.00390625.
- Target-compatible vs wrong control passed in both architectures.
- The near-vs-far interaction passed in the Transformer but not the GRU, so a universal distance law was not claimed.

## Phase 2 — signed-transfer harness

The harness was explicitly calibrated to detect positive, neutral, and negative transfer.

- target-compatible 25%: positive transfer.
- exact repeated ordinary-A control: zero difference.
- anti/incorrect target structure: negative transfer.

A stricter matched-wrong control then held source distance, row/column margins, and row entropy equal to the target-compatible kernel. At d=.125/.25/.50, target beat matched-wrong in 8/8 fresh seeds.

## Phase 3 — mechanism localization

Restoring the pre-curriculum shared output head showed that target/wrong information was also stored in the curriculum-updated backbone/state representation. The target condition began B substantially better. A simple "more loss reduction from its own start" metric did not support faster learning.

To remove the start-performance confound, performance-aligned hitting time was introduced: compare the number of B steps needed to traverse the same B-NLL interval.

- Base-coordinate retained: target vs ordinary was ~5 steps faster across 5/5 families and replicated.
- Fully fresh B decoder: the effect did not replicate.

This narrowed the mechanism to cross-task transfer that depends on shared representation/decoder coordinates rather than generic task-independent meta-learning.

## Phase 4 — coordinate dose and orientation

Decoder cosine alignment was controlled directly while head norm was held fixed.

- rho=1.0: target vs ordinary ~-5.89 steps.
- rho=.5: ~-3.36.
- rho=.25: ~-1.86; independently replicated at ~-2.07.
- rho=0: approximately zero/inconsistent.

A geometry-preserving orthogonal rotation of B state embeddings then changed only orientation relative to the A/shared representation. At rho=.25, identity coordinates retained target advantage while rotation largely removed it; a prespecified three-seed family-mean adjudication passed 5/5 families.

## Phase 5 — shared-coordinate rank

The state-embedding span was partially preserved while the complement was rotated.

- Keeping six of twelve principal state directions restored transfer relative to keeping none.
- Top-six vs bottom-six directions did not support a "high-energy directions are uniquely important" hypothesis.
- Random nested rank 0/3/6/12 produced a favorable slope in 5/5 families (mean -0.1704 B-step per preserved dimension, p=.03125).

Current synthetic mechanism model: shared representational capacity/rank matters more robustly than one privileged direction.

## Phase 6 — natural Japanese -> English pilot

Japanese-only windows were selected using offline rank-bigram motifs derived from English or Russian. The model updates during the curriculum remained Japanese-only.

A 95% ordinary-Japanese + 5% English-motif Japanese curriculum:

- passed safety checks for Japanese and English starting loss,
- differed from the Russian-target control on one learning-gain test (7/8, p=.02344),
- but did **not** improve the preregistered English learning-gain primary over ordinary Japanese (3/8, p=.6328).

Natural-language positive acceleration therefore remains an open target rather than a claimed result.
