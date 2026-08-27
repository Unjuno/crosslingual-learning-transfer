# Crosslingual Learning Transfer

**Can learning system A make system B easier — or harder — to learn later, even when B is not shown during the A-only curriculum intervention?**

This repository collects controlled neural-network experiments on that question. It is an **experimental evidence snapshot**, not a finished curriculum method and not evidence about human language learning.

## What this repository shows

In the documented synthetic systems, prior A-only learning can move later B learning in either direction:

- **positive transfer:** a target-compatible A curriculum can improve later B learning,
- **negative transfer:** a target-incompatible A curriculum can make later B learning worse,
- **neutral control:** repeating the same A curriculum produces exactly zero paired difference in the calibrated harness.

The strongest current mechanism result is that stable sample-efficiency transfer depends on **shared A/B representation coordinates**. The effect weakens as decoder alignment is removed, is strongly reduced by geometry-preserving coordinate rotation, and shifts in a favorable direction as more shared latent rank is preserved.

This does **not** imply a universal scalar language distance or a proven language-teaching recipe.

## Why this matters

The research object is better described as a **signed, directed learning-transfer relation** than as a symmetric language-similarity score. If the phenomenon survives future natural-language and larger-model tests, it would motivate curricula designed not only to teach the current task, but also to avoid harmful internal states and potentially prepare reusable structure for later tasks.

That application remains a hypothesis. The current contribution is the controlled experimental evidence and mechanism constraints.

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

The compact cross-phase table is in [`results/evidence_summary.csv`](results/evidence_summary.csv). PASS and FAIL outcomes are kept together rather than filtering to positive findings.

## Current interpretation

The evidence does not support a universal "learn faster" ability. A more precise interpretation is:

```text
A-only curriculum
      |
      v
target-helpful / neutral / target-harmful state
      |
      v
later B learning
      |
      +-- effect depends on shared A/B representation coordinates
```

With a retained base coordinate system, target-compatible curricula cross the same B-performance interval in fewer B updates across five kernel families and an independent cohort. When the B decoder is completely fresh/random, that stable cross-family advantage does not replicate.

## Earlier structural-surrogate result

Character-level six-language experiments showed that exact directed-bigram-preserving surrogates reproduce the natural transfer topology much more strongly and consistently than unigram/degree-margin-preserving controls. For example, BG0 vs BG1 raw Cspec topology rho was 0.928, whereas UNI0 vs UNI1 was 0.454.

The UNI controls preserve exact unigram counts and directed bigram in/out degree margins, so the relevant contrast is more accurately **specific joint source->target transition coupling versus marginal/degree structure**. This supports a role for directed transition structure, but **does not establish bigram sufficiency** or a complete scalar distance mechanism.

## Natural-language status

A Japanese -> English pilot selected Japanese-only windows using an offline English transition motif. A 95/5 blend was safe for Japanese and did not damage English starting loss on average, and it differed from a Russian-target control, but it **did not establish positive acceleration over ordinary Japanese**.

Human learning and large-production-LLM generality remain untested.

## Repository map

- `docs/CLAIMS_AND_LIMITATIONS.md` — claim ledger: supported facts, inference, and uncertainty.
- `docs/EXPERIMENT_LOG.md` — chronological experiment sequence and decision points.
- `docs/REPRODUCIBILITY.md` — environment assumptions, determinism, and data boundaries.
- `docs/TEST_HARNESS.md` — positive/neutral/negative controls and the statistical-harness bug that was found and corrected.
- `experiments/` — representative executed experiment snapshots across surrogate, curriculum, natural-language, and mechanism phases.
- `protocols/` — locked confirmatory/mechanism protocol JSON files.
- `results/` — compact summary CSVs, including positive findings, failures, and replication failures.

## Reproduction

```bash
python -m pip install -r requirements.txt
```

The scripts are execution snapshots rather than a polished library. Several retain `/mnt/data/...` paths from the original container; see [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) before rerunning them. Natural-language experiments require the original external corpus environment, which is not redistributed here.

## Scope and claim discipline

**Supported:** controlled neural-network existence and mechanism results under the documented synthetic setups.

**Not supported:** a universal language-distance law, bigram sufficiency, decoder-independent universal meta-learning, a proven Japanese curriculum for faster English acquisition, human-learning effects, or generality to large production LLMs.

## Data and licensing

Code in this repository is released under Apache-2.0. External natural-language corpora are **not redistributed** here; they remain subject to their original licenses.
