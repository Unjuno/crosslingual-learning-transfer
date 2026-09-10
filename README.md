# Crosslingual Learning Transfer

[日本語](README.ja.md) · [Final status](docs/FINAL_STATUS_2026-09-10.md) · [Evidence map](docs/CLAIM_EVIDENCE_MATRIX.md) · [Reproduction](docs/REPRODUCIBILITY.md)

**Frozen research snapshot — 10 September 2026.** No further outcome-seeking experiments are planned in this sequence. Corrections remain welcome. This is an experimental record, **not a peer-reviewed paper, production curriculum, or human-learning recommendation**.

## The question and the answer

Can the way a model learns task A change its subsequent learning of task B?

In the controlled systems studied here, it can help or harm. Shared representation and decoder interventions affect the outcome. A small natural-text experiment also found a consistent English-side benefit when an **English-aware teacher supplies hidden targets on Japanese-sourced Student inputs**.

**This is not “learning Japanese text alone makes English 5% easier.”** Target information is allowed in the teacher. The data are domain-limited Vim/TeX translation strings, and the numerical five-percent boundary is sensitive to measurement resolution.

## Final evidence at a glance

| Evidence | Result | Boundary |
|---|---|---|
| Synthetic intervention sequence | Helpful, harmful and neutral transfer; representation interventions matter | Tested task families and small GRU/Transformer learners only |
| Natural R1, hidden-loss weight 1.0 | **Registered FAIL**: mean English interval saving 6.3521%, Japanese phase1 loss penalty 1.1355% | Japanese-retention ceiling was 1%; failure is retained |
| Natural R2, weight 0.5, fresh seeds 32000–32009 | **Registered PASS**: 10/10 favorable; mean saving 5.1881%; Japanese penalty 0.6293% | Original ten-update evaluation grid; point-estimate criterion |
| Same R2 trajectories, every-update timing | Mean saving **4.5692%** interpolated / **4.7066%** first-observed; 10/10 favorable | Diagnostic, not a replacement verdict; robust ≥5% is **not established** |
| Common-head / frozen-body diagnostic | English-teacher body favorable in 10/10 archived seeds; mean endpoint difference −0.020383 nats/byte | **Zero new independent seeds**; not pure learning-rate or semantic evidence |
| State–signal confirmation, seed18500 | **BLOCKED / NOT COUNTED** | Three exact helper snapshots are missing |

For R2, the registered saving median is **5.8849%**, range **0.7798–10.5482%**. Its descriptive paired-seed 95% bootstrap interval is **3.08–7.37%**. These are English performance-interval updates, **not end-to-end compute or wall-clock savings**. The retention gate is evaluated **before** English adaptation, not after it. Results from different metrics or seed cohorts are not interchangeable.

Natural-text model: 141,056 parameters, 2 causal Transformer layers, width 64, 4 heads, fixed 256-byte vocabulary, context 128 bytes, batch 16. Reported natural R2 execution: CPU AMD EPYC 9V74, float32, PyTorch 2.10.0+cpu, one thread/job, no fixed CPU clock. The archived Sep-9 diagnostic used Intel Xeon Platinum 8573C. See [methods and environment](docs/REPRODUCIBILITY.md).

## Verify the public record — no training or package installation

From the repository root, with **Python 3.11–3.13**:

```bash
python scripts/verify_publication.py
```

This one command runs all five compact result verifiers and record tests, checks frozen scientific-file hashes, final-status consistency, local Markdown links, publication file inventory and common secret/file-exposure patterns. It **does not** rerun learning, authenticate reported execution provenance, inspect all git history, or recreate missing data/checkpoints. A checker PASS means the records are internally consistent; it does not turn a scientific FAIL into a PASS.

The [release notes](publication/RELEASE_NOTES.md) describe the `v1.0.0-research` snapshot. Release assets include a checksummed compact source-and-record ZIP and a validation report. Large raw traces, model weights and third-party corpus text are **not included or publicly hosted by this release**. Source hashes are not substitutes for missing files.

## Read in this order

1. [Final status and stopping decision](docs/FINAL_STATUS_2026-09-10.md): supported, failed, blocked and untested questions.
2. [Claim-to-evidence map](docs/CLAIM_EVIDENCE_MATRIX.md): exactly which files support which claims.
3. [Reproduction boundaries](docs/REPRODUCIBILITY.md): record arithmetic versus training, dependencies and unavailable artifacts.

The [repository map](docs/REPOSITORY_MAP.md) indexes historical reports, protocols, results and the archived body/head diagnostic. Existing scientific scripts and results keep their paths and bytes. Dated status files are historical snapshots; **[results/FINAL_STATUS.json](results/FINAL_STATUS.json) is the current authority**.

## What is not established

Target-unaware Japanese-text-only acceleration; a universal language-distance metric; pure learning-rate improvement separated from prepositioned knowledge; independent-corpus generalization; post-English Japanese retention; total-compute savings; human-learning or large-LLM generality. Details: [claim boundaries](docs/CLAIMS_AND_LIMITATIONS.md) and [information paths](docs/INFORMATION_BOUNDARIES.md).

## Use, corrections and citation

Code retains **Apache-2.0**; third-party data retain their own terms and are not relicensed. Read [third-party provenance](docs/THIRD_PARTY_DATA.md). Use [CITATION.cff](CITATION.cff) and the exact tag/commit when referencing this software/research record; no DOI or paper is claimed. [CONTRIBUTING.md](CONTRIBUTING.md) explains corrections, and [SECURITY.md](SECURITY.md) explains reporting and scan limits. No social-media post is made by the release workflow.
