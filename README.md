# Crosslingual Learning Transfer

Controlled neural-network experiments on how prior learning in one language-like task changes later learning in another. The repository is an **experimental evidence snapshot**, not a finished curriculum method and not evidence of human-language transfer.

## Research question

Can learning task/language **A** change the later learnability of **B**, even when the curriculum intervention itself contains only A-surface inputs? If so, when is the effect positive, neutral, or negative, and what shared representation structure is required?

The project began from directed language-transfer measurements and structural surrogate experiments, then moved to controlled curriculum interventions and mechanism ablations.

## Current evidence

| Result | Status | Main evidence |
|---|---|---|
| A-only target-compatible curricula can improve later B learning in a controlled NN system | **Supported** | Transformer: 7/8 fresh seeds at d=.25, p=.0078125; GRU: 8/8, p=.00390625 |
| A-only target-incompatible curricula can harm later B learning | **Supported** | GRU signed-transfer calibration and independent replication |
| Target-compatible structure beats a source-distance/entropy/margin-matched wrong structure at moderate distance | **Supported** | d=.125/.25/.50: 8/8 fresh seeds for target vs matched-wrong |
| Later-B sample efficiency differs after aligning comparisons to the same B-performance interval | **Supported conditionally** | Base-head coordinate retained: 5/5 kernel families in two cohorts |
| The effect survives a fully fresh/random B decoder | **Not supported** | Fresh-head replication failed |
| Transfer strength depends on retained A/B decoder alignment | **Supported in synthetic GRU** | rho 1.0 -> .5 -> .25 -> 0 produces monotone weakening in 5/5 families; rho=.25 replicated |
| Cross-task latent-coordinate orientation matters | **Supported in synthetic GRU** | Geometry-preserving B-state rotation largely removes the effect; 3-seed family adjudication 5/5 |
| Increasing preserved shared latent rank shifts transfer in a favorable direction | **Supported in synthetic GRU** | random nested rank 0/3/6/12: slope negative in 5/5 families, p=.03125 |
| A 95% ordinary Japanese + 5% English-motif-selected Japanese curriculum accelerates English over ordinary Japanese | **Not supported** | gain comparison 3/8, p=.6328125 |

Lower AUC / fewer B steps means better later-B learning in the reported experiments.

## Interpretation

The strongest current interpretation is not that prior learning creates a universal "learn faster" ability. Instead, A-only learning can place the network in a **target-helpful or target-harmful state**, and performance-aligned sample-efficiency effects are reliable when A and B retain enough shared representational/decoder coordinates. Destroying those coordinates can remove the stable effect.

This makes the research object a **signed, directed transfer relation**, not a symmetric language-distance scalar.

## Earlier structural-surrogate result

Character-level six-language experiments showed that exact directed-bigram-preserving surrogates reproduce the natural transfer topology much more strongly and consistently than unigram/degree-margin-preserving controls. For example, BG0 vs BG1 raw Cspec topology rho was 0.928, whereas UNI0 vs UNI1 was 0.454. This supports a role for directed transition coupling, but **does not establish bigram sufficiency** or a complete scalar distance mechanism.

## Natural-language status

A Japanese -> English pilot selected Japanese-only windows using an offline English transition motif. A 95/5 blend was safe for Japanese and did not damage English starting loss on average, and it differed from a Russian-target control, but it **did not establish positive acceleration over ordinary Japanese**. Human learning and large-LLM generality remain untested.

## Repository map

- `docs/CLAIMS_AND_LIMITATIONS.md` — claim ledger: fact / inference / uncertainty.
- `docs/EXPERIMENT_LOG.md` — chronological experimental phases and decisions.
- `docs/REPRODUCIBILITY.md` — execution assumptions and what is/is not included.
- `docs/TEST_HARNESS.md` — controls, exact tests, and a statistical-harness bug that was found and corrected.
- `docs/milestones/` — contemporaneous milestone notes.
- `experiments/` — exact snapshots of representative executed experiment scripts.
- `protocols/` — locked protocol JSON files for selected confirmatory experiments.
- `results/` — compact summary CSVs only; raw checkpoints/corpora are intentionally excluded.

## Scope and claim discipline

**Supported:** controlled neural-network existence and mechanism results under the documented synthetic setups.

**Not supported:** a universal language-distance law, a proven Japanese curriculum for faster English acquisition, human-learning effects, or generality to large production LLMs.

## Data and licensing

Code in this repository is released under Apache-2.0. External natural-language corpora are **not redistributed** here; they remain subject to their original licenses. The natural-language experiment script is included as an execution snapshot and requires the original corpus/handoff environment.

See `docs/REPRODUCIBILITY.md` for details.
