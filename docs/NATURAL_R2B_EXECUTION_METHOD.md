# R2b half-weight execution and verification

This record concerns the already preregistered half-weight S2 experiment, not an additional parameter search. The parent weight-1.0 cohort remains FAIL. The half-weight cohort is fixed to seeds 32000–32009, weight 0.5, and all other parent settings. It must not acquire rescue seeds, alternative weights, or relaxed retention/efficacy thresholds.

## Provenance

Scientific core and wrapper are byte-identical to main at `efbcc3f2fe5e809f539a95eff3d120c21ec98826`. Their blob IDs are in `protocols/natural_r2b_execution_seal_2026-09-08.json`. GitHub Actions run 34142957353 checked seed-valued fields in all 25 historical result CSVs at that revision and found no half-weight seed. Its source archive and SHA256 report are retained as artifact 10026583444.

`observe_natural_r2b_execution.py` calls the original half-weight entry point. Its reusable R1 observer measures actual update counts, input-source labels, and per-arm batch-stream hashes. It also records runtime and stage-end checkpoints. It does not change the model, loss, minibatches, update order, or scientific metric. Stage checkpoints contain the last model of each stage only; they are not a full multi-arm mid-stage resumption format.

Instrumentation was checked on reduced-schedule validation seed30999: the uninstrumented and instrumented runs produced byte-identical curve, summary, and original-audit files. This is a technical validation, not scientific evidence.

## Counting units

| Field | Meaning | Unit and scope |
|---|---|---|
| seed | Independent initialized training repetition | Integer identifier, not an observation count for individual tokens |
| B_step | English optimizer update index | Updates, from 0 to 600, sampled every 10 |
| span_updates | Interpolated updates to traverse the prescribed reference-defined interval | Updates, not wall time; interval is fixed within a seed across arms |
| EN_nll / JA_post_nll | Mean negative log probability of the next byte | Natural-log units per predicted byte; mathematically dimensionless |
| saving_vs_TJA | Paired proportional English-span reduction relative to Japanese-teacher control | Dimensionless fraction; 0.05 denotes 5% |
| hidden_distillation_weight | Weight of cosine hidden loss added to Japanese LM loss | Dimensionless scalar, fixed at 0.5 |
| runtime_seconds | Observed wall time including evaluation/tracing/checkpoint work | Seconds; not an isolated training-throughput benchmark |

Batch size 16 and context 128 give 2048 predicted byte targets per optimizer update. Update spans and byte-target spans therefore have the same fractional saving. There are 5400 optimizer updates per seed: 400 base, 1200 across three teachers, 800 across four phase1 arms, and 3000 across the five English curves. The ten-seed total is 54000 updates. Teacher preparation cost is not subtracted from the reported English-only saving.

Hardware: CPU-only AMD EPYC 7763, four-core cgroup quota, 4 GiB RAM limit; one PyTorch thread per seed and up to four concurrent seed jobs. CPU frequency was not pinned, and no fixed-clock throughput claim is made. Software: Python3.13.5, PyTorch2.10.0+cpu, NumPy2.3.5, pandas2.2.3. CPU float32, deterministic algorithms enabled, no quantization.

## Verification layers

1. Original locked adjudicator: `adjudicate_natural_ja_en_s2_halfweight_tradeoff.py`.
2. Independent verifier: `verify_natural_r2b_raw.py` recomputes every interval, crossing, and span from all raw curves; checks exact seed/condition/grid coverage, scientific flags, corpus hashes, source counts, actual batch streams, executed code IDs, and agreement with stored summaries.
3. Archive verifier: `verify_archived_natural_r2b.py` verifies archive/member SHA256, extracts only forty allowlisted regular files, runs layer2 and checks agreement with layer1. It requires only the standard library.

A valid scientific FAIL is successful verification (exit0). Missing, corrupted, validation-only, mislabelled, or inconsistent evidence is UNCERTAIN (exit2), not an efficacy finding. Test fixtures are generated only in temporary directories and never placed among scientific raw results.

## Scope and limitations

S2 means Japanese-sourced Student updates before English phase2, with English-aware hidden teacher targets explicitly allowed. It does not mean absence of English-derived information in the system, nor Japanese-text-only curriculum success. The Japanese software messages contain shared ASCII and Latin-script fragments. No tokenizer is fit on English: all 256 byte values are fixed a priori.

The frozen corpus has exact post-normalization line overlap: 5/296 English, 8/296 Japanese, and 9/296 Russian evaluation lines also occur in training. No complete129-byte English or Japanese evaluation window is identical to a training window, but that does not establish semantic or near-duplicate independence. The original corpus remains unchanged and these limitations must accompany reported outcomes. Generated source text is not redistributed.

The reference-defined interval reduces but does not prove removal of initialization/starting-performance differences. If a condition already starts below the high threshold, the first crossing is zero as in the locked code; such cases should be reported descriptively, not silently removed or redefined.

## Execution correction

At the start of this continuation an old R1 branch was mistaken for current main, and repeat jobs31000–31002 were started. Live PR history exposed that PR7 had already completed them and PR8 had locked R2b. The repeat jobs were interrupted and quarantined before use as any fresh evidence. The old seal was corrected in commit710659c96cf767172b2c39f84820d4b8b619ed4a. All R2b freshness checks were then performed against current pinned main before fresh32000 outcomes. This version-control error is disclosed rather than hidden.
