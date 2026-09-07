# Natural R2 halfweight confirmation — 2026-09-08

## Registered result: PASS, within the stated scope

The previously published one-shot protocol `protocols/natural_ja_en_s2_halfweight_tradeoff_2026-09-06.json` was executed on exactly seeds 32000–32009. No model, corpus, teacher budget, metric, threshold, seed or loss-weight change was made during the cohort. The sole configuration change relative to R1 is hidden-distillation weight 0.5 rather than 1.0, as specified in that earlier protocol.

| Registered gate | Observed | Decision |
|---|---:|---|
| Favorable TEN-vs-TJA interval difference in at least 9/10 seeds | 10/10 | PASS |
| One-sided sign p at most .05 | .0009765625 | PASS |
| Mean per-seed English interval saving at least 5% | 5.1880787% | PASS |
| Japanese phase1 mean NLL penalty against TJA at most 1% | 0.6292635% | PASS |
| All execution and frozen-corpus audits | 10/10 pass | PASS |
| No primary censoring by 600 English updates | None | PASS |

The original weight-1.0 R1 result remains **FAIL**. Passing a new configuration does not rewrite an earlier failure.

## What was actually run and verified

Exactly 54,000 optimizer updates: ten separate seed-specific initializations, Japanese common-base training, all three teacher specializations, all four phase1 Student conditions, and all five English phase2 curves including the reference. This produced 50 English curves and 3,050 recorded checkpoints.

The original public core and halfweight wrapper were executed unchanged (Git blob IDs are in the audit CSV). A separate observation layer recorded actual input-batch digests and saved checkpoints. Independent analysis recomputed reference thresholds and interval spans from raw curves, then reevaluated the Japanese losses from all 40 saved phase1 Student states. Maximum span discrepancy and checkpoint Japanese-NLL discrepancy were both zero. The original and independent adjudicators agree.

This is a new 10-seed cohort. Replaying older 31000-series checkpoints in separate measurement/midpoint diagnostics does not enlarge this sample.

## Effect size and uncertainty

Per-seed English saving median: **5.8848583%**; range **0.7797882–10.5481937%**. Mean span difference: **−7.7560020 updates**.

Mean Japanese phase1 NLL was **1.94491204** for TEN and **1.93274995** for TJA. Both ordinary Japanese and Russian-target-teacher secondary English comparisons favored TEN in 10/10 seeds, with mean differences −16.001727 and −13.803485 updates respectively. Those are secondary contrasts, not additional preregistered primary claims.

A descriptive paired-seed percentile bootstrap (10,000 draws; fixed analysis seed 98208) gives a 95% interval of **3.08350–7.36704%** for mean saving, and **0.59327–0.67384%** for the ratio-of-mean Japanese penalty. The fixed efficacy criterion uses a point estimate, not a lower confidence limit: this result does **not** establish a population saving of at least 5%. Uncertainty is conditional on this small, fixed corpus. No token-level pseudoreplication is used.

The weight-1.0 and weight-0.5 studies use different seed cohorts. Their difference is not a paired causal dose-response estimate.

## Information, data and cost boundaries

- Student updates before phase2 use Japanese-sourced corpus samples, but an English-aware teacher supplies hidden targets. This is not target-unaware Japanese-text-only learning.
- Japanese software text may include ASCII, Latin fragments and loanwords. Corpus provenance is not a guarantee of zero English-derived information.
- Data are the exact frozen Vim/TeX software translation strings, not broad natural-language data. Previously audited exact train/eval line overlaps remain in the registered data. Filtered-document evaluations are separate diagnostics, never a retroactive replacement.
- The English interval is defined by the original seed-specific reference curve; evaluation was every ten updates with linear interpolation. Measurement-resolution diagnostics are labeled separately.
- English step0 NLL is lower for TEN by 0.05570775 on average. Knowledge prepositioning can coexist with the interval-efficiency result; this is not pure optimizer learning-rate improvement.
- Japanese retention is tested immediately after Japanese phase1, **before** English adaptation. It is not a guarantee against forgetting during later English-only learning.
- Reported savings concern English interval updates/tokens, excluding teacher preparation, prior Japanese training and wall-clock cost. No total-compute saving is established.
- No human-learning, large-LLM, language-universal or unseen-corpus generalization claim is made.

## Execution environment

141,056 parameters; two causal Transformer layers; width 64; four attention heads; FF width 256; fixed byte vocabulary 256; context 128 bytes; batch 16. Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5, pandas 2.2.3, float32, deterministic algorithms, one PyTorch thread/job, at most three cohort jobs. CPU: AMD EPYC 9V74; clock not fixed. These are sample-efficiency results, not hardware-speed benchmarks.

## Public verification and artifact boundary

```
python experiments/verify_halfweight_public_2026_09_08.py
python -m unittest discover -s experiments -p 'test_halfweight_public_2026_09_08.py' -v
```

These commands verify the committed 40-condition summaries, ten execution-audit records and published arithmetic. They do not rerun training or independently prove data provenance. Full raw-curve, observed-input and checkpoint validation was performed in the execution environment and is retained in separate execution/checkpoint artifacts. Compact public verification and end-to-end training reproduction are deliberately distinguished.

The original fresh-seed runner is a guard against reusing registered outcomes as new evidence. Any future same-seed replay must be explicitly labeled reproduction, not counted as another fresh cohort.
