# Crosslingual Learning Transfer

**Can learning A prepare a model to learn B more efficiently later?**

This repository is an experimental research record, not a finished curriculum product, a universal language-distance law, or evidence about human learning. Successes, failed criteria, measurement limitations and unresolved reproducibility problems are retained together.

## Latest independent result — 2026-09-08

**A previously locked lower-strength hidden-teaching configuration passed its registered criterion, but its “at least 5%” margin is sensitive to measurement resolution.**

| Study or measurement | English interval saving | Japanese phase1 penalty vs Japanese-teacher control | Status |
|---|---:|---:|---|
| Original R1, hidden-loss weight 1.0, seeds 31000–31009 | Mean 6.3521% | 1.1355%, above the 1% limit | **Registered FAIL, unchanged** |
| Registered halfweight follow-up, seeds 32000–32009 | Mean **5.1881%**, favorable in **10/10** | **0.6293%**, within the 1% limit | **Registered PASS** |
| Same halfweight checkpoints, every-update timing | Mean **4.5692%** interpolated; **4.7066%** first-observed | Same already-measured phase1 states | **Diagnostic: favorable in 10/10, but below 5%** |
| Single 50:50 Student-parameter midpoint, archival seeds | Original exploratory metric 3.6862% | 0.5226% | **Exploratory FAIL; no coefficient sweep** |

The registered halfweight mean is computed from the original **10-update evaluation grid**. Its median is 5.8849%, range 0.7798–10.5482%, and one-sided sign probability .0009765625. A descriptive paired-seed 95% bootstrap interval for that mean is **3.08–7.37%**. Neither this interval nor the denser timing establishes a robust population saving of at least 5%.

The new configuration used the **already published weight-0.5 protocol**, not a retrospective relaxation of R1. Different weight studies use different seed cohorts, so their difference is not a paired causal dose-response estimate. Replayed checkpoints do not enlarge the independent seed count.

Read [the registered halfweight report](docs/NATURAL_R2_HALFWEIGHT_2026-09-08.md), [all measurement diagnostics](docs/R2_DIAGNOSTICS_2026-09-08.md), [the locked halfweight protocol](protocols/natural_ja_en_s2_halfweight_tradeoff_2026-09-06.json), and [the original R1 FAIL](docs/NATURAL_R1_STATUS_2026-09-06.md).

## Latest mechanism diagnostic — 2026-09-09

A bounded archived-checkpoint diagnostic used the **already completed 32000–32009 seeds**, so it adds **zero new independent seeds**. It trained 80 adaptation branches for 20,000 optimizer updates to separate where the later-English advantage remains.

With a **common initial output head**, an English-aware-teacher-derived frozen model body still had lower English NLL after 250 head-only English updates in **10/10 archived seeds**. The mean endpoint body-origin effect was **−0.020383 nats/byte**. The corresponding mean endpoint effect of initial output-head origin was approximately **−0.000018 nats/byte**, with its descriptive seed-bootstrap interval crossing zero.

This makes an explanation based only on the initially more English-favorable output head insufficient for this endpoint effect. It does **not** establish a fresh replication, pure learning-rate improvement, semantic/grammatical transfer, independent-corpus generality, target-unaware Japanese-text-only transfer, or a human/large-LLM result. The frozen “body” includes embeddings and LayerNorm as well as Transformer blocks, and the English-aware Student was already advantaged before English adaptation.

See the [Sep-9 archived diagnostic snapshot](research/archived_adaptation_locus_2026_09_09/README_JA.md) and its [execution report](research/archived_adaptation_locus_2026_09_09/docs/EXECUTION_REPORT_JA.md). The R1 registered FAIL, R2 registered-grid PASS and R2 measurement-resolution caveat remain unchanged.

## What “Japanese-only” means here

Before English adaptation, Student updates use **Japanese-sourced corpus samples**. However, an English-aware teacher may transmit English-derived information through hidden-state targets. Japanese software text can include ASCII and Latin fragments.

This is **S2: target-aware hidden teaching through Japanese inputs**, not target-unaware Japanese-text-only learning. Models are small fixed-byte causal Transformers trained on frozen Vim/TeX software translation strings. Broad natural language, human learning and production LLMs are not established.

The Japanese-retention gate is tested **after Japanese phase1, before English adaptation**. It is not a guarantee against later forgetting. In the halfweight diagnostic, filtered-document Japanese NLL rose from 1.89675 after phase1 to 2.64420 after 250 English-only updates. Savings exclude teacher preparation, earlier Japanese training and wall-clock costs.

[Information boundaries](docs/INFORMATION_BOUNDARIES.md) distinguish source-language input restrictions from restrictions on target-derived information.

## What the additional diagnostics found

Original-grid NLL values matched exactly when trajectories were replayed with denser evaluation. The changed timing estimates are therefore a measurement issue, not different training.

Exact-line-filtered document evaluation still showed a favorable transfer pattern, but it changes document boundaries and the evaluated population. It does not remove every template/semantic overlap and is not substituted for the registered metric.

On archival weight-1.0 checkpoints, restoring **only the phase1 output head** recovered about 70% of the later Japanese byte-NLL increase while worsening English NLL. An exact loss decomposition attributed about 86% of the Japanese increase to changes in probability assigned to byte classes. These are not percentages of semantic knowledge forgotten or recovered. Full scope and numerical details are in the diagnostic report.

The Sep-9 locus diagnostic further shows a retention/performance tradeoff across update locations. With the English-aware starting Student and 250 English updates, full-model adaptation reached the best English endpoint of the three tested update rules, while head-only adaptation can exactly recover the original Japanese function if the saved Japanese head is explicitly routed back in. That identity is by construction and requires task/language routing plus an extra saved head; it is not evidence of free retention.

## Synthetic phase — frozen

The synthetic mechanism-search phase was frozen on 2026-09-03. Within the documented setups:

- A-side interventions can help or harm subsequent B learning.
- Shared/target-decodable representation geometry and coordinate interventions affect transfer.
- Shared-coordinate rank dependence was observed in GRU and two fresh causal-Transformer cohorts. A negative fitted slope does not imply strictly monotonic benefit at every rank increment.
- Transfer behavior recovers some held-out synthetic structural differences but does not uniformly recover fixed-support probability-weight changes.

Tested simple distance, width-boundary and effective-rank-predictor hypotheses failed. This does not prove that every scalar description is impossible or that a particular alternative mechanism has been identified. The state–signal binding confirmation at seed18500 remains **BLOCKED / NOT COUNTED**, pending exact helper snapshots; a reconstructed implementation would be a separate study.

See the [synthetic closure report](docs/FINAL_SYNTHETIC_STATUS_2026-09-03.md), [historical evidence ledger](results/evidence_summary.csv), and [claim/evidence matrix](docs/CLAIM_EVIDENCE_MATRIX.md). Historical documents retain their stated snapshot dates; later reports add results without rewriting prior adjudications.

## Public verification

```bash
python experiments/verify_halfweight_public_2026_09_08.py
python -m unittest discover -s experiments -p 'test_halfweight_public_2026_09_08.py' -v
python experiments/verify_final_synthetic_closure.py
python experiments/verify_natural_r1_confirmatory.py
python research/archived_adaptation_locus_2026_09_09/experiments/verify_records.py
python -m unittest discover -s research/archived_adaptation_locus_2026_09_09/experiments -p 'test_public_records.py' -v
```

These standard-library commands verify **committed record arithmetic**, not end-to-end training. In the Sep-8 halfweight execution, independent raw-curve recomputation and reevaluation of all 40 phase1 checkpoints had zero discrepancy. In the Sep-9 archived diagnostic, 80 saved endpoint states were separately reevaluated 240 times with maximum NLL discrepancy about 1.22e-8 nats/byte in the execution archive. Full execution traces/checkpoints are separate artifacts; compact verification must not be confused with those stronger checks.

Fresh-seed guards intentionally reject reuse of registered outcomes as new trials. Future same-seed replays must be labeled reproduction and do not add independent evidence.

## Repository and implementation boundaries

`protocols/` contains locked definitions and labeled exploratory plans. `experiments/` contains execution snapshots, runners and verifiers. `results/` contains compact seed summaries, audits and decisions. `docs/REPRODUCIBILITY.md` explains omitted artifacts and data boundaries. Later archived diagnostics with zero new independent seeds live under `research/` so they are not confused with registered primary cohorts.

The Sep-8 primary environment was Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5 and pandas 2.2.3; float32, deterministic algorithms, one PyTorch thread/job. Model: 141,056 parameters, two width-64 causal Transformer layers, four heads, context 128 bytes, batch 16. CPU clocks were not fixed. Cross-platform bitwise identity and hardware-speed gains are not claimed. The Sep-9 archived replay used the same software/model specification on an Intel Xeon Platinum 8573C and explicitly does not claim bitwise identity to the earlier CPU execution.

Source corpus versions must match the frozen SHA256 manifest. Another corpus is a new experiment. The Sep-9 snapshot contains a pinned Universal Dependencies preparation utility for a possible independent-corpus phase, but **no UD training result has been run or claimed**. Some older scripts retain original execution paths. No additional loss-weight or midpoint-coefficient search is part of this completed sequence; independent data and preregistered dense/fixed-threshold measurements remain the next validity questions.

## Licensing

Code is Apache-2.0. Third-party corpus text is not redistributed here and retains its original licenses.
