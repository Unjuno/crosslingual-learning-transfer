# Hidden-width exploratory follow-up — 2026-09-02

> **Superseded for final adjudication:** see `docs/FINAL_SYNTHETIC_STATUS_2026-09-03.md`.

This note records the adaptive exploratory width sequence that motivated the later locked H32-vs-H48 confirmation. It must not be read as confirmatory evidence.

## Exploratory observation

The original GRU random-rank shared-coordinate intervention was rerun across hidden widths while retaining state embedding dimension 32, keep dimensions {0,3,6,12}, decoder alignment rho=.25, and the 80–20 performance-aligned span metric.

Exploratory cohorts suggested that strict familywise sign consistency could weaken at H=32 while appearing stable at H>=48 in the tested seeds. This motivated a separately locked fresh test.

## Final disposition

The subsequent locked confirmation on seeds 20000–20400 **FAILED** the proposed H32-vs-H48 stability boundary:

- prespecified D_s>0 occurred in only 1/5 fresh seeds;
- seed20300 produced nonnegative family slopes at both H32 and H48;
- all rotation audits passed.

Therefore no sharp H=48 threshold is claimed. The exploratory width pattern is retained only as hypothesis-generation history.

See:

- `protocols/random_rank_hidden_width_confirmatory_2026-09-02.json`
- `results/random_rank_hidden_width_confirmatory_summary.csv`
- `results/random_rank_hidden_width_confirmatory_family_slopes.csv`
- `docs/FINAL_SYNTHETIC_STATUS_2026-09-03.md`
