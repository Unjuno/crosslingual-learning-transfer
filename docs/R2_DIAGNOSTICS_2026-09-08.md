# R2 measurement and checkpoint diagnostics — 2026-09-08

## Status and provenance

All fixed diagnostic jobs completed. They use already completed checkpoints, so they add **zero independent seeds**. Five separate public plans distinguish the original archival midpoint probe from subsequent timing, output-head, byte-decomposition and halfweight-resolution diagnostics. Later plans are post-hoc relative to their motivating observations, not retrospective preregistrations of confirmatory hypotheses.

The only new independent cohort in this execution sequence is the separately registered halfweight trial on seeds 32000–32009. Its fixed decision is PASS, the weight-1.0 R1 remains FAIL, and the midpoint's original exploratory decision remains FAIL.

## 1. Measurement resolution matters to the five-percent margin

The halfweight trial's registered 10-update interpolation gives mean saving **5.1881%**. Replaying unchanged training with every-update evaluation gives **4.5692%** under interpolation and **4.7066%** with integer first-observed crossings. All ten paired directions remain favorable, and all original-grid NLL values match exactly.

Consequently, the passing numerical margin is not robust to resolution. This does not alter what the predeclared statistic returned, but it prevents a confident general claim of at least 5% savings. The integer timing result also prevents interpreting fractional interpolation as physically executed fractional updates.

On the changed exact-line-filtered document evaluation, the halfweight saving is 6.7664% with interpolation and 6.8798% using five-update first-observed crossings; the latter has eight favorable seeds and two ties. These are different evaluation targets, not a more favorable replacement primary.

For the archived weight-1.0 cohort, fully dense timing gives 8.1469% mean saving versus the historical 6.3521%. Thus coarse timing need not bias the estimate in one direction for every configuration.

## 2. A single parameter midpoint did not meet its criterion

The sole candidate averages the same-seed Japanese-teacher and English-teacher **Student parameters**, 50:50, after phase1. It is not the hidden-distillation weight of 0.5 used in the fresh trial. Preparing both endpoint Students has additional cost.

The original exploratory measurement gave **3.6862%** mean English saving and **0.5226%** Japanese mean-NLL penalty. It failed the predeclared 5% saving requirement. Subsequent fully dense measurements gave 4.3141% interpolated saving and 4.4985% first-observed saving, still below 5%. On filtered documents, the mean is 4.2892% interpolated; first-observed measurement is favorable in only 6/10 seeds and includes an adverse seed. No further coefficient was searched.

## 3. Duplicate lines are not the entire phase1 tradeoff

Original normalized eval/train exact overlaps were eight Japanese lines and five English lines out of 296. After removing same-language exact overlaps and duplicate eval lines, the old weight-1.0 Japanese mean-loss penalty was **1.0408%** rather than the historical **1.1355%**.

A descriptive two-way seed-by-document bootstrap interval is **0.8706–1.2002%**. It crosses 1%, so this changed evaluation does not establish that the true penalty exceeds that boundary with high confidence. It also does not repair the historical registered failure. For halfweight, the filtered phase1 mean-loss penalty is **0.5547%**.

Filtered evaluation scores 288 Japanese documents/14,061 target bytes and 291 English documents/9,409 target bytes, with per-document 128-position blocks, masked padding, no cross-document context, and no score for each document's first byte. All evaluated English bytes are ASCII. This is exact-line filtering, not semantic/template decontamination.

## 4. Phase1 retention is not post-English retention

On archived weight-1.0 English-teacher Students, filtered Japanese NLL rises from **1.89786** after phase1 to **4.38012** after 600 English-only updates. Restoring **head.weight only** from phase1 recovers a mean **70.09%** of that Japanese loss increase (seed range 51.74–78.15%), but English NLL worsens by **0.31483 nats/byte** on average.

An exact log-probability decomposition divides byte NLL into probability assigned to the correct byte class and prediction within that class. Classes are ASCII, UTF8-continuation, and leading-or-other bytes. The Japanese increase of **2.482258 nats/byte** comprises **2.147817** from class probability and **0.334441** from within-class prediction. The mean per-seed class-probability share is **85.67%**, range 78.61–91.08%.

The restored head recovers **1.773025** class-probability loss and only **0.002345** within-class loss on average. This points to a substantial output-distribution component of the byte-NLL forgetting. It does not mean that 86% of Japanese semantics was erased, or that all memory lives in the output head.

Conversely, in the halfweight phase1 English advantage, the mean clean-English NLL difference is −0.053005, decomposed into −0.010902 class-probability loss and −0.042104 within-class loss. The starting advantage is therefore not solely an ASCII-probability shift, but within-ASCII prediction is not itself a proof of semantic or grammatical transfer.

Halfweight also exhibits later forgetting: filtered Japanese NLL rises from **1.89675** after phase1 to **2.64420** after 250 English updates. The original retention gate was not designed to control this.

## 5. What to do with these results

Do not tune more hidden-loss weights or midpoint coefficients on this corpus. Preserve the registered PASS/FAIL decisions together with their measurement limitations. The next independent validity question requires data fixed before outcomes, near-duplicate/template safeguards, densely measured or fixed absolute targets, and explicit retention measurement after English learning as well as before it. A Japanese-text-only curriculum remains a separate unestablished objective.

## Audit and publication scope

Across the new trial and replays: **83,000 optimizer updates**, 80 full 600-update English curves, 50 prefix-only diagnostic replays, and 18,450 English evaluation records including the changed metrics. The head and class-decomposition diagnostics use no optimizer updates. The fresh cohort's 40 saved phase1 states reproduced the recorded Japanese NLL exactly; all replay original-grid NLL discrepancies were zero.

The per-target float64 decomposition identity error was at most 2.665e-15; the maximum difference from the original float32 aggregate evaluation was 1.804e-8 nats/byte. These are arithmetic checks, not effect-size confidence bounds.

Compact numerical results are in `results/natural_r2_measurement_diagnostics_2026-09-08.json`. Full raw traces, executable diagnostic code, tests and checkpoints are separate execution artifacts, not all committed here. This public diagnostic report is therefore descriptive documentation, not a claim of fully self-contained raw-data reproduction from the compact repository alone.
