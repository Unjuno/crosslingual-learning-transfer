# Hidden-width sensitivity of shared-coordinate rank transfer — 2026-09-02

## Question

Does the previously observed negative slope between preserved shared latent rank and later-B performance-aligned span survive changes in GRU hidden width?

This is an exploratory extension of the preregistered `random_rank_state_alignment_gru` experiment. The width sequence was partly adaptive and is therefore not presented as a new confirmatory claim.

## Method held fixed

The original synthetic family construction, A-only target-compatible curriculum, state-embedding dimension D=32, keep dimensions {0,3,6,12}, decoder cosine alignment rho=.25, B-learning horizon, and 80-20 performance-aligned span metric were retained. Only GRU hidden width H and seed were varied.

For each family, the primary descriptive quantity is the OLS slope of `target25 - A_nat` performance-aligned span versus `keep_dim`. A negative slope means that retaining more shared state-coordinate rank makes the prior target-compatible curriculum increasingly beneficial for later B learning.

## Results

| Seed | H | Mean slope | Negative families | Descriptive one-sided sign p |
|---:|---:|---:|---:|---:|
| 5400 | 32 | -0.2771 | 5/5 | .03125 |
| 5400 | 64 | -0.1704 | 5/5 | .03125 |
| 5400 | 128 | -0.2010 | 5/5 | .03125 |
| 5500 | 32 | -0.1153 | 4/5 | .1875 |
| 5500 | 48 | -0.2045 | 5/5 | .03125 |
| 5500 | 64 | -0.2331 | 5/5 | .03125 |
| 5500 | 96 | -0.1331 | 5/5 | .03125 |
| 5500 | 128 | -0.1418 | 5/5 | .03125 |
| 5600 | 32 | -0.2397 | 4/5 | .1875 |
| 5600 | 48 | -0.2301 | 5/5 | .03125 |

The original H=64/seed5400 analysis is reproduced to numerical rounding (published mean slope approximately -0.170399). Orthogonal-rotation audit error stayed at or below 1.0133e-6 across the new runs, and keep12 retained essentially all state-embedding energy.

## Interpretation

The broad mechanism is robust: every tested seed/width cohort has a negative mean rank slope. However, strict familywise sign consistency is not width-invariant. H=32 produced only 4/5 negative families in both fresh seeds 5500 and 5600, with the sign reversal occurring in different families. H=48 produced 5/5 negative families in both of those seeds, while H=64, H=96, and H=128 also showed 5/5 in the cohorts tested.

This weakens any claim that shared-coordinate transfer geometry is independent of learner capacity. The better-supported interpretation is that the rank mechanism persists on average but becomes less reliable across task families in the narrow H=32 learner.

It would be incorrect to infer a sharp universal threshold at H=48 from these data. Widths were sampled sparsely, later widths were selected adaptively, and only three seeds were used. A future confirmatory capacity study should lock widths and fresh seeds before outcomes, ideally using a normalized capacity axis or parameter-count-matched architecture controls.

## Execution environment

Runs were CPU-only with PyTorch 2.10.0+cpu, NumPy 2.3.5, pandas 2.2.3, deterministic algorithms enabled, and one PyTorch thread per job. Execution host exposed 5 Intel Xeon Platinum 8370C cores at a nominal 2.80 GHz. Runtime is not treated as a scientific outcome.

Compact cohort summaries are in `results/random_rank_hidden_width_summary.csv`; family-level slopes are in `results/random_rank_hidden_width_family_slopes.csv`; intervention audits are in `results/random_rank_hidden_width_audit_summary.csv`.
