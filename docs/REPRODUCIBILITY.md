# Reproducibility and public availability

[Final scope](FINAL_STATUS_2026-09-10.md). Publication is separated into **record verification**, **training rerun**, and **historical provenance**. None implies the others.

## 1. Fully available: compact-record verification

Unpack the research release or check out its tag. From the repository root:

```bash
python scripts/verify_publication.py
```

Use Python **3.11–3.13**. No pip packages, GPU, corpus, model download or network access is required. The command runs five existing verifier families: final synthetic closure, natural R1, registered R2 halfweight, dense R2 timing and archived body/head records. It also runs standard-library record tests plus publication guards. Detailed subprocess results are stored in `dist/publication-verification.json` (ignored by git).

A successful exit means the committed quantities/coverage agree with their registered or descriptive definitions, final status is consistent, scientific snapshot hashes are unchanged, relative Markdown file links resolve, and the bounded exposure scan passes. Historical experiment FAIL/BLOCKED statuses **must remain** in a successful package. It does not validate all historical claims or raw learning.

To run one family directly, use the commands indexed in [the repository map](REPOSITORY_MAP.md). `--static-only` on the unified checker skips verifier/test execution and is **not** the full release gate. Invalid/missing records fail closed; no successful scientific result is synthesized from absent data.

## 2. Available code, conditional training reproduction

`experiments/` contains original execution snapshots. Do not import or execute every script indiscriminately: some train on import, some depend on system corpus files, and some retain historical `/mnt/data/` paths. Compilation is not a runtime test. Read the individual protocol and script before launching an expensive run.

The final synthetic package record is `requirements-final-synthetic.txt`. The natural reference environment reported Python3.13.5, PyTorch2.10.0+cpu, NumPy2.3.5, pandas2.2.3, float32, deterministic algorithms and one thread/job. `requirements.txt` is an unpinned convenience list, **not a lockfile**. The recorded versions are not a complete transitive wheel/OS/ABI lock. CPU wheel selection and BLAS kernels are platform dependent; no cross-platform bitwise guarantee is made. No quantization or zero-point is used in the natural float32 trials; older quantized-signal studies have their own definitions.

Natural R1/R2 scripts require the exact corpus outputs in `protocols/natural_ja_en_s2_corpus_manifest_2026-09-06.json`. The preparation script expects particular Vim/TeX source files. Another package version can change the resulting hashes. Do not relax a hash check to call changed data an exact rerun. Fresh-seed guards reject seeds already in committed results; a replay must be explicitly labeled reproduction, not counted as fresh evidence.

The archived Sep-9 public subdirectory supplies compact record verification, not every training/replay implementation. Complete execution code and weights existed in separately delivered execution archives; their public download locations are **not provided by this release**. Do not describe this git checkout as a full self-contained training archive.

## 3. Explicitly blocked or unavailable

Seed18500 depends on the exact historical files `teacher_hidden_entropy_matched_distance_transformer.py`, `teacher_hidden_geometry_intervention.py`, and `support_stationary_matrix.py`, which are absent. Its runner must fail rather than substitute reconstructed code. Status: **BLOCKED / NOT COUNTED**.

Third-party corpus text, raw minibatch/per-checkpoint traces, full calibration matrices, large checkpoints and local handoff archives are not shipped here. [Artifact catalog](../publication/artifact_catalog.json) records hashes/sizes of available conversation archives for provenance, with public availability set to false. A hash identifies a file but does not make it downloadable or prove its creation history.

Independent UD-corpus training was not run. A pinned preparation script is a future feasibility asset, not a result.

## 4. Units, statistics and interpretation

Natural byte NLL uses natural logarithms (nats per predicted byte; SI dimensionless). Update count is a count. Equal context and batch make update count proportional to sampled target-byte count within a run; different tokenizers and corpora are not directly comparable. Timing interpolation estimates fractional crossings; it is not an executed fractional update.

Keep seed as the paired unit for seed uncertainty; documents add a separate evaluation-sampling dimension. Do not count checkpoints/tokens/replays as independent training seeds. Registered point-estimate gates differ from confidence bounds and from later diagnostics. Retention timing (before versus after English) must be stated.

## 5. Package and CI scope

`publication/files.sha256.json` is the release inventory; `publication/scientific-snapshot.json` fixes existing protocols, records and experiment sources from the evidence-base commit. New metadata does not retroactively preregister a study. Release bundles contain only manifest-listed tracked text files and checksums. They exclude git internals, credentials, corpus data and weights.

The exposure check is a heuristic on this snapshot, **not a comprehensive security audit, secret-history scan or branch-protection policy**. Existing public history is not rewritten. CI checks do not cryptographically prove that historical training was run. Release tags are never deliberately moved by the publishing workflow; repository-wide immutable-release settings are not configured here.
