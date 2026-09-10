# Final status — 10 September 2026

**FROZEN RESEARCH SNAPSHOT.** The present experimental sequence is closed. Publication cleanup adds no training runs and changes no historical scientific decisions. [Machine-readable authority](../results/FINAL_STATUS.json) · [日本語の入口](../README.ja.md).

## Final claim ledger

| Category | What is retained | What it does not establish |
|---|---|---|
| **SUPPORTED, controlled synthetic evidence** | Prior A-side interventions can help or harm later B learning; shared/target-decodable representation interventions matter; random-rank dependence was observed with GRU and small Transformer learners | Universal language distance; architecture/optimizer invariance; a theorem of monotonic transfer |
| **REGISTERED FAIL** | R1 weight1.0: 10/10 English-side favorable signs, mean span saving 6.3521%; Japanese loss penalty 1.1355% exceeded the registered 1% ceiling | Failed retention is not repaired by the favorable English sub-result |
| **REGISTERED PASS, qualified** | R2 weight0.5: new seeds 32000–32009, 10/10 favorable; mean registered saving 5.1881%; Japanese phase1 penalty 0.6293% | The point-estimate gate is not a population guarantee; a different seed cohort is not a paired weight comparison |
| **ADVERSE MEASUREMENT DIAGNOSTIC** | Every-update R2 timing gives 4.5692% interpolated / 4.7066% first-observed saving, with 10/10 favorable directions | No robust ≥5% saving claim; no post-hoc replacement of the historical verdict |
| **EXPLORATORY / ARCHIVED** | Body/head substitution: common-head frozen English-teacher bodies outperform Japanese-teacher bodies in 10/10 already-used seeds at the fixed endpoint | Zero new independent seeds; initial knowledge and learning efficiency remain distinct; no semantics/grammar measurement |
| **FAILED HYPOTHESES** | Specified H32/H48 stability boundary, selected effective-rank predictor, natural S1 window pilot, single 50:50 Student midpoint criterion | Failure of selected hypotheses is not proof that all scalar models or all curricula are impossible |
| **BLOCKED / NOT COUNTED** | seed18500 state–signal confirmation requires three exact helper snapshots | A reconstructed implementation would be a new study, not faithful historical reproduction |
| **NOT TESTED / NOT ESTABLISHED** | Independent-corpus validation, large LLMs, humans, target-unaware Japanese text, total-compute saving, post-English retention guarantee | None may be inferred from the tested toy/domain-limited systems |

The current [claim/evidence map](CLAIM_EVIDENCE_MATRIX.md) supplies exact files and verification levels. Dated reports preserve the chronology rather than overwrite earlier failures.

## Quantities and interpretation

The natural R1/R2 outcome is the English update count needed to traverse a common seed-specific reference-defined performance interval. Fixed byte vocabulary, context and batch size make updates proportional to sampled target bytes **within that experiment**. Comparing different architectures, corpora, byte/token units or end-to-end costs requires new accounting.

R2 uses a 141,056-parameter, two-layer width-64 causal Transformer (4 heads, FF256, context128 bytes, batch16). Historical environment: Python3.13.5, PyTorch2.10.0+cpu, NumPy2.3.5, pandas2.2.3, float32, deterministic algorithms, one thread/job; AMD EPYC 9V74, clock uncontrolled. Archived body/head diagnostics used Intel Xeon Platinum 8573C. These are not speed benchmarks.

The R2 median registered saving is 5.8849% (range 0.7798–10.5482%); descriptive paired-seed 95% interval for its mean is 3.08–7.37%. The archived endpoint body-origin effect is −0.020383 nats/byte, **not a percent saving**. Dense replays and output-head diagnostics reuse seeds and must not be pooled as fresh trials.

## Information and data boundaries

Student updates before English adaptation are sourced from the Japanese corpus, while the teacher is allowed to encode English-derived information (S2). Japanese source text can contain ASCII, Latin fragments and loanwords. This is neither zero target information in the system nor target-unaware Japanese-text-only learning. S1 target-aware text selection and S0 target-unaware teaching are distinct; see [information boundaries](INFORMATION_BOUNDARIES.md).

Vim/TeX translation strings are domain-limited. Exact-line filtering does not eliminate semantic/template overlap. Japanese retention was measured before English adaptation; subsequent English training can still degrade Japanese prediction. Lower byte NLL and loss decompositions are not direct measurements of semantic competence or human skill.

## Why stop here

The sequence has existence results, bounded mechanism interventions, confirmatory successes and failures, timing sensitivity and archived body/head controls. More tuning on the same corpus would not resolve the largest open question: external validity. No further hidden-weight grids, midpoint-coefficient sweeps or rescue seeds are planned in this sequence.

Corrections, clearer documentation and reproducibility fixes remain permitted. A new corpus/model or a stricter information boundary requires a separately scoped prospective study. Freezing research is not archiving the GitHub repository or preventing corrections.

## Publication boundary

The compact release is independently checkable for arithmetic and file integrity. It is **not** a complete raw-data/weight archive. Historical descriptions of execution checks are preserved as reported records; the release checker does not rerun those evaluations. [Reproduction](REPRODUCIBILITY.md), [artifact availability](../publication/artifact_catalog.json), and [publication controls](PUBLICATION_CHECKLIST.md) state the exact boundaries. This release adds no paper or DOI and makes no peer-review claim.
