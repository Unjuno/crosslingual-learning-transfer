# Natural R1 status — 2026-09-06

## Stage

R1/R2 S2 confirmatory is **COMPLETE — LOCKED OVERALL VERDICT: FAIL**.

The failure is specifically the preregistered Japanese-safety gate. The English-learning efficacy gates passed strongly; the project does not relax the safety threshold after observing outcomes.

## Question

Can an English-aware teacher, acting only through hidden targets on Japanese-sourced Student inputs, prepare an English-naive Student to learn English more efficiently than an otherwise matched Japanese-only teacher?

This is an **S2 information-boundary** experiment. The Student receives Japanese-sourced samples only before phase 2, but the English-aware teacher is explicitly allowed to contain English-derived information. The result therefore does not test target-unaware Japanese text alone.

## Locked design

Protocol:

`protocols/natural_ja_en_s2_hidden_teacher_r1_2026-09-06.json`

- fixed byte-level vocabulary 256;
- 2-layer causal Transformer, d_model=64, 4 heads;
- 400 Japanese-only common Student updates;
- matched 400-update T_JA / T_EN / T_RU teachers;
- 200 Japanese-only Student phase-1 updates;
- hidden cosine distillation weight 1.0;
- English-only phase 2, maximum 600 updates;
- fixed fresh seeds 31000–31009;
- primary PASS required all of:
  - >=9/10 favorable `span(JA_TEN) - span(JA_TJA)` signs;
  - exact one-sided sign p<=.05;
  - mean `1 - span(JA_TEN)/span(JA_TJA)` >=5%;
  - all audits and frozen-corpus checks;
  - no primary censoring;
  - mean Japanese eval NLL for JA_TEN <= 1.01 × mean Japanese eval NLL for JA_TJA.

No rescue seed, model change, loss-weight change, threshold change, teacher-budget change, or corpus substitution was made after confirmatory outcomes.

## Frozen corpus and execution audit

The cohort used the frozen Vim/TeX software translation-string corpus recorded in:

`protocols/natural_ja_en_s2_corpus_manifest_2026-09-06.json`

All ten seeds matched the exact six frozen corpus SHA256 values. Every seed reported:

- Student phase 0: 400 Japanese-source updates, no English/Russian-source updates;
- Student phase 1: 200 Japanese-source updates, no English/Russian-source updates;
- T_JA: 400 Japanese updates;
- T_EN: 200 Japanese + 200 English updates;
- T_RU: 200 Japanese + 200 Russian updates;
- fixed byte vocabulary 256;
- no prior committed seed-valued result hit;
- no primary censoring.

All 10/10 audits passed.

## Locked confirmatory result

Primary comparison: `JA_TEN` versus `JA_TJA` on the common reference-defined 80–20 English performance interval.

| Gate | Locked requirement | Observed | Result |
|---|---:|---:|---|
| Favorable seed sign | >=9/10 | **10/10** | PASS |
| Exact one-sided sign p | <=.05 | **0.0009765625** | PASS |
| Mean span saving | >=5% | **6.3521%** | PASS |
| Audit / frozen corpus | all pass | **10/10 pass** | PASS |
| Primary censoring | none | **none** | PASS |
| Japanese safety vs T_JA teacher | JA_TEN <= 1.01 × JA_TJA | **1.1355% worse** | **FAIL** |

Overall locked verdict: **FAIL**.

Mean Japanese eval NLL after phase 1:

- JA_TJA: **1.9290505**
- JA_TEN: **1.9509552**
- allowed safety ceiling: **1.9483410**
- excess over the locked ceiling: **0.0026142 NLL**

The safety failure is numerically narrow, but it is still a preregistered failure and is not reclassified.

## Efficacy pattern retained as a constrained finding

Although the overall study fails, the English-side pattern is unusually consistent and should be retained as a **sub-result, not as a PASS of the study**:

- JA_TEN beats JA_TJA in performance-aligned span in **10/10 seeds**;
- mean span: JA_TEN **132.995** updates vs JA_TJA **142.336** updates;
- mean primary difference: **-9.341 updates**;
- mean saving: **6.352%**;
- JA_TEN also beats ordinary JA_nat in span in **10/10 seeds**;
- JA_TEN beats Russian-target JA_TRU in span in **10/10 seeds**.

The locked adjudicator records mean secondary differences:

- TEN − JA_nat: **-25.235 updates**;
- TEN − JA_TRU: **-21.228 updates**.

These secondary comparisons do not repair the failed safety gate.

## Starting-performance versus learning-efficiency interpretation

JA_TEN begins phase 2 with lower English NLL than JA_TJA by **0.0983 NLL on average**. That is expected to be possible in an S2 design: English information can be transmitted through the teacher's hidden targets even though the Student's phase-1 samples are Japanese-sourced.

Therefore the result contains a **knowledge-prepositioning component** and must not be described as pure learning-rate improvement.

Post-adjudication diagnostics nevertheless show that the common performance-interval result is not explained only by starting below the high threshold:

- JA_TEN was already below the reference-high threshold at phase2 step 0 in only **1/10** seeds;
- JA_TJA was below it in **0/10**;
- mean high-threshold crossing: TEN **1.176** vs TJA **3.154** updates;
- mean low-threshold crossing: TEN **134.171** vs TJA **145.489** updates.

Thus TEN reaches both the beginning and the end of the common performance interval earlier, with the larger difference occurring at the low threshold. This is descriptive post-adjudication evidence, not a new confirmatory gate.

## Japanese tradeoff interpretation

The failed safety gate compares JA_TEN against **JA_TJA**, which is a Japanese-specialist hidden-teacher control. Relative to ordinary JA_nat, JA_TEN's mean Japanese NLL is actually about **0.91% lower** (better). This is post-adjudication context only; the locked safety comparator remains JA_TJA and the overall verdict remains FAIL.

The result therefore suggests a real target-specific English benefit paired with a small tradeoff against the strongest Japanese-specialist teacher condition. Whether that tradeoff can be reduced without losing the English effect is a new question and requires a new prospective protocol; it cannot be answered by changing the present threshold.

## Public result files

- `results/natural_r1_confirmatory_seed_summary.csv`
- `results/natural_r1_confirmatory_audit_summary.csv`
- `results/natural_r1_confirmatory_adjudication.json`
- `results/natural_r1_confirmatory_posthoc_diagnostics.json`

Recompute the compact locked adjudication with:

```bash
python experiments/verify_natural_r1_confirmatory.py
```

The compact verifier recomputes the locked gates from committed seed summaries/audits. Raw per-checkpoint curves are intentionally not committed in the compact repository snapshot.

## Current scientific status

- Synthetic phase: frozen and publicly auditable.
- Natural S2 target-aware hidden-teacher effect: **English efficacy strongly supported in this domain-limited cohort, but locked overall study FAIL due Japanese safety gate**.
- Target-unaware Japanese-text-only acceleration: **not established**.
- Broad natural-language generality: **not established**.
- Human-learning implication: **not established**.
