# Reproducibility

## Scope

This repository is a compact research snapshot: representative executed scripts, locked protocol JSON files, family/cell-level data for the final closure tests, and compact summary CSVs. It intentionally excludes large checkpoints, many raw per-step outputs, and third-party corpora.

The repository therefore distinguishes **training reproduction** from **adjudication reproduction**. The final three synthetic closure verdicts can be recomputed from committed family/cell-level CSVs, but not every historical training run can be recreated bit-for-bit from the public snapshot.

See also:

- `docs/CLAIM_EVIDENCE_MATRIX.md` — claim-by-claim evidence level.
- `docs/INFORMATION_BOUNDARIES.md` — target-information pathways and the meaning of “A-only”.

## Final synthetic closure environment

The executed final closure runs were recorded as:

- PyTorch `2.10.0+cpu`
- NumPy `2.3.5`
- pandas `2.2.3`
- CPU execution
- `torch.use_deterministic_algorithms(True)`
- one PyTorch thread per job (`THREADS=1`)

The exact Python minor version of the executed closure host was not archived in the public record, so this repository does not invent one retroactively. GitHub Actions currently uses Python 3.11 for syntax/protocol/adjudication checks; that is a CI environment, not a claim about the historical training host.

For the closest installable package match to the final closure sequence:

```bash
python -m pip install -r requirements-final-synthetic.txt
```

`requirements-final-synthetic.txt` records `torch==2.10.0`; the executed build reported the CPU local tag `2.10.0+cpu`, so select the appropriate CPU wheel/index for the platform when an exact backend match matters.

The broader `requirements.txt` remains intentionally loose because the repository spans many historical snapshots and not all old dependency versions were preserved.

## One-command public adjudication check

Run:

```bash
python experiments/verify_final_synthetic_closure.py
```

This standard-library verifier recomputes the three final closure adjudications from committed family/cell-level data:

1. H32-vs-H48 hidden-width boundary — expected **FAIL**;
2. pre-state state-effective-rank predictor — expected **FAIL**;
3. causal-Transformer random-rank replication — expected **PASS**.

It also checks that:

- fresh-seed lists and `LOCKED_BEFORE_FRESH_OUTCOMES` status match the committed protocols;
- recomputed statistics match the committed summary CSVs;
- the final verdicts match `results/current_status_2026-09-03.json`.

This command **does not rerun training**, **does not regenerate omitted checkpoints**, and **does not reconstruct raw audit logs that were not committed**.

## Determinism

Experiment scripts explicitly seed Python, NumPy, and PyTorch generators. Confirmatory experiments use fixed fresh seed ranges recorded in their protocol JSON files. Statistical tests are paired exact sign/sign-flip style tests where specified by the protocol and small family counts permit exact enumeration.

For historical reruns, do not change model/data/protocol parameters while still calling the run an exact replication. A changed implementation, restored approximation, or alternative dependency stack should be labeled a new replication.

## Important path note

The files under `experiments/` are **execution snapshots**, preserved close to the versions used in the original run environment. Some older scripts contain `/mnt/data/...` output paths. Set supported environment variables such as `OUTDIR`, or adapt only the output root before local execution.

## Natural-language corpus

The Japanese/English experiment depends on a previously prepared six-language corpus environment. The corpus itself is not redistributed here because third-party data retain their original licenses. The natural-language script is included for methodological transparency, not as a standalone data bundle.

The next natural-language phase must also report target-information boundaries explicitly; “Japanese-only” is insufficient by itself because a curriculum designer or teacher can still use English-derived information. See `docs/INFORMATION_BOUNDARIES.md`.

## Surrogate experiment conventions

The structural-surrogate phase used six languages and a character-level toy Transformer. Exact-bigram surrogates preserve the directed character-bigram multiset; UNI controls preserve unigram counts/endpoints and, consequently, directed bigram in/out degree margins.

## Interpreting p-values

Many confirmatory mechanism tests use five kernel families. If all 5/5 family means have the preregistered sign, a one-sided exact sign test yields p=.03125. This is a small-sample exact test, not evidence that the synthetic family sample represents all possible languages/tasks.

## Raw artifacts omitted

Not committed in general:

- model checkpoints,
- many raw minibatch/per-step logs,
- large corpus files,
- duplicated intermediate CSVs,
- local handoff ZIP files.

For the **final closure tests**, family/cell-level adjudication data are committed and machine-verifiable. For several earlier research phases, only compact summaries and representative scripts are available. The exact boundary for each public claim is recorded in `docs/CLAIM_EVIDENCE_MATRIX.md`.
