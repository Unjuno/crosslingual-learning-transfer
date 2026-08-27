# Reproducibility

## Scope

This repository contains a compact research snapshot: representative executed scripts, locked protocol JSON files, and summary CSVs. It intentionally excludes large checkpoints, raw per-step outputs, and third-party corpora.

## Environment

Representative synthetic runs used:

- Python 3
- PyTorch
- NumPy
- pandas
- CPU execution
- `torch.use_deterministic_algorithms(True)`
- typically `THREADS=1`

Install the minimal Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Determinism

Experiment scripts explicitly seed Python, NumPy, and PyTorch generators. Confirmatory experiments use fixed fresh seed ranges recorded in their protocol JSON files. Statistical tests are paired exact sign-flip tests when the unit count is small enough for exact enumeration.

## Important path note

The files under `experiments/` are **execution snapshots**, preserved close to the versions used in the original run environment. Some scripts contain `/mnt/data/...` output paths. Set supported environment variables such as `OUTDIR`, or adapt only the output root before local execution. Do not change model/data/protocol parameters when attempting an exact replication.

## Natural-language corpus

The Japanese/English experiment depends on a previously prepared six-language corpus environment. The corpus itself is not redistributed here because third-party data retain their original licenses. The natural-language script is included for methodological transparency, not as a standalone data bundle.

## Surrogate experiment conventions

The structural-surrogate phase used six languages and a character-level toy Transformer. Exact-bigram surrogates preserve the directed character-bigram multiset; UNI controls preserve unigram counts/endpoints and, consequently, directed bigram in/out degree margins.

## Interpreting p-values

Many confirmatory mechanism tests use five kernel families. If all 5/5 family means have the preregistered sign, a one-sided exact sign-flip test yields p=.03125. This is a small-sample exact test, not evidence that the synthetic family sample represents all possible languages/tasks.

## Raw artifacts omitted

Not committed:

- model checkpoints,
- raw minibatch/per-step logs,
- large corpus files,
- duplicated intermediate CSVs,
- local handoff ZIP files.

The committed `results/` files are the compact values used in the claim ledger.
