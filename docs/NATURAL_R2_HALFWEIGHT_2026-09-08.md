# Natural R2 halfweight confirmation — 2026-09-08

## Registered PASS, with an important measurement limitation

The previously published one-shot protocol `protocols/natural_ja_en_s2_halfweight_tradeoff_2026-09-06.json` was executed on exactly seeds 32000–32009, without model, corpus, teacher-budget, metric, threshold, seed or loss-weight changes during the cohort.

**The registered 10-update-grid decision is PASS. However, a subsequently fixed measurement diagnostic reduced the mean saving from 5.1881% to 4.5692% when both crossings were measured every update. The direction remained favorable in 10/10 seeds, but a robust “at least 5%” effect is not established.** This qualification is part of the result, not an optional footnote. The original weight-1.0 R1 remains FAIL, and neither historical decision is rewritten using a post-hoc metric.

## Registered gates

| Gate | Observed | Decision |
|---|---:|---|
| Favorable TEN-vs-TJA interval difference in at least 9/10 seeds | 10/10 | PASS |
| One-sided sign probability at most .05 | .0009765625 | PASS |
| Mean per-seed English interval saving at least 5% | 5.1880787% | PASS on the registered grid |
| Japanese phase1 mean-NLL penalty against TJA at most 1% | 0.6292635% | PASS |
| All execution and frozen-corpus audits | 10/10 pass | PASS |
| No primary censoring by 600 English updates | None | PASS |

Only the previously specified hidden-distillation weight changed from 1.0 in R1 to 0.5 in this trial. Both trials use different seed cohorts; their difference is not a paired causal dose-response estimate.

## Execution and verification

Exactly 54,000 optimizer updates were completed: ten initializations, Japanese common-base training, all three teacher specializations, all four phase1 Student conditions and all five English curves including the reference. This produced 50 English curves and 3,050 recorded checkpoints.

The public core and halfweight wrapper were executed unchanged. An observation layer recorded actual input-batch digests and saved weights without changing training. Independent analysis recomputed thresholds/spans from raw curves and reevaluated Japanese NLL from all 40 saved phase1 Student states. Both maximum discrepancies were zero, and the original and independent adjudicators agreed.

Per-seed saving median: **5.8848583%**; range **0.7797882–10.5481937%**. Mean span difference: **−7.7560020 updates**. Mean Japanese phase1 NLL: TEN **1.94491204**, TJA **1.93274995**. Secondary English comparisons against ordinary Japanese and the Russian-target teacher favored TEN in 10/10 seeds, with mean differences −16.001727 and −13.803485 updates. These remain secondary contrasts.

## Resolution and evaluation sensitivity

After the registered decision, a separate public plan fixed 250-update replays of the same two phase1 Students in every seed. Original English batches/optimizer settings were unchanged; every original-grid NLL value matched the completed primary run exactly. These replays are not additional independent evidence.

| Descriptive measurement | Mean saving | Favorable seeds |
|---|---:|---:|
| Registered 10-update-grid interpolation | 5.1881% | 10/10 |
| Every-update measurement, interpolated crossings | **4.5692%** | 10/10 |
| Every-update measurement, integer first-observed crossings | **4.7066%** | 10/10 |
| Exact-line-filtered document evaluation every 5 updates, interpolated | 6.7664% | 10/10 |
| Same filtered-document evaluation, first-observed crossings | 6.8798% | 8/10; two ties |

The filtered-document evaluation changes both document boundaries and the evaluated target population. It is a separate diagnostic, not a better-looking replacement for the registered test. Its numerical thresholds are the original fixed reference thresholds, applied identically to both arms.

A descriptive paired-seed percentile bootstrap (10,000 draws; analysis seed 98208) on the **registered** statistic gives a 95% interval of **3.08350–7.36704%** for mean saving and **0.59327–0.67384%** for the Japanese mean-loss ratio penalty. The fixed gate uses a point estimate, not a lower confidence limit. Seed uncertainty and evaluation-resolution sensitivity are different limitations; no unsupported combined uncertainty or population guarantee is claimed.

## Information, data, retention and cost boundaries

Student updates before English adaptation use Japanese-sourced samples, but an English-aware teacher supplies hidden targets. Japanese software text can include ASCII/Latin fragments. This is not target-unaware Japanese-text-only learning or zero English-derived information.

The frozen corpus contains domain-limited Vim/TeX software translation strings. Previously audited exact train/eval line overlaps remain in the registered data. Exact-line-filtered document diagnostics do not remove every template or semantic overlap. No broad language or unseen-corpus generality is established.

English step0 NLL was lower for TEN by 0.05570775 on average. Knowledge prepositioning can coexist with interval-efficiency improvement; the result is not pure optimizer learning-rate improvement.

**The Japanese gate is measured before English adaptation.** In the separate filtered-document diagnostic, halfweight TEN Japanese NLL rose from 1.89675 after phase1 to 2.64420 after 250 English updates. Passing phase1 retention is therefore not a guarantee against later forgetting.

Savings concern English interval updates/tokens and exclude teacher preparation, earlier Japanese training and wall-clock overhead. No total-compute saving, human-learning effect or large-LLM result is claimed.

## Environment

141,056 parameters; two width-64 causal Transformer layers; four heads; FF width 256; fixed byte vocabulary 256; context 128 bytes; batch 16. Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5, pandas 2.2.3, float32, deterministic algorithms, one PyTorch thread/job, at most three primary-cohort jobs. CPU: AMD EPYC 9V74; clock not fixed. These are sample-efficiency measurements, not hardware-speed benchmarks.

## Public verification boundary

```bash
python experiments/verify_halfweight_public_2026_09_08.py
python -m unittest discover -s experiments -p 'test_halfweight_public_2026_09_08.py' -v
```

These commands verify committed summaries/audit records and their arithmetic, not training or raw-curve reconstruction. Full execution/checkpoint verification was performed locally and is retained in separate trace and checkpoint artifacts. Any future replay of these completed seeds is reproduction and must not enlarge the independent sample size.

See also `docs/R2_DIAGNOSTICS_2026-09-08.md` and `results/natural_r2_measurement_diagnostics_2026-09-08.json`. No further hidden-loss-weight or midpoint-coefficient search is part of this sequence.
