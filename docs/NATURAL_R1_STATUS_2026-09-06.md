# Natural R1 status — 2026-09-06

## Stage

R1 is **DESIGN LOCKED / IMPLEMENTATION VALIDATED / CONFIRMATORY NOT STARTED**.

No scientific outcome from seeds 31000–31009 has been observed or committed.

## Question

Can an English-aware teacher, acting only through hidden targets on Japanese-sourced Student inputs, prepare an English-naive Student to learn English more efficiently than an otherwise matched Japanese-only teacher?

This is an **S2 information-boundary** experiment. The Student receives Japanese-sourced samples only before phase 2, but the English-aware teacher is explicitly allowed to contain English-derived information. A PASS would therefore not establish a target-unaware Japanese-text-only effect.

See `docs/INFORMATION_BOUNDARIES.md`.

## Locked design

Protocol:

`protocols/natural_ja_en_s2_hidden_teacher_r1_2026-09-06.json`

Key fixed elements:

- byte-level vocabulary: 256, not fit on any language corpus;
- 2-layer causal Transformer, d_model=64, 4 heads;
- common Student base: 400 Japanese-only updates;
- teachers: T_JA, T_EN, T_RU, 400 updates each;
- Student phase 1: 200 Japanese-only updates under JA_nat / JA_TJA / JA_TEN / JA_TRU;
- hidden teaching loss: final-LayerNorm tokenwise `1 - cosine_similarity`, weight 1.0;
- phase 2: English LM learning only, maximum 600 updates;
- fixed fresh seeds: 31000–31009;
- primary gate: at least 9/10 favorable TEN-vs-TJA signs, exact one-sided sign p<=.05, mean >=5% span saving, all audits pass, and Japanese safety gate passes;
- no rescue seeds, model changes, loss-weight changes, threshold changes, teacher-budget changes, or corpus substitution after confirmatory outcomes.

## Frozen corpus

The first R1 test uses a domain-limited natural-text corpus made from locally installed Vim and TeX Live Japanese/Russian translation resources, with English msgids as the aligned English text.

This is **software UI/help translation text**, not a broad web/book/news corpus.

Preparation script:

```bash
python experiments/prepare_natural_r1_system_translations.py --outdir /path/to/natural_r1
```

Frozen source/output hashes:

`protocols/natural_ja_en_s2_corpus_manifest_2026-09-06.json`

Scientific runs refuse input files whose SHA256 values differ from that manifest.

Generated corpus text is not committed; source-package licenses remain controlling.

## Validation already performed

Seed 30999 is reserved for non-scientific implementation validation.

Completed checks:

- deterministic corpus preparation;
- exact frozen corpus hashes;
- finite losses on a reduced schedule;
- Student phase-0/phase-1 source is Japanese only;
- teacher language-source counts follow the intended reduced validation schedule;
- fixed byte vocabulary accepts all three corpora;
- output schema and audit record are produced.

Validation record:

`results/natural_r1_implementation_validation_2026-09-06.json`

No seed30999 transfer effect or sign is retained as scientific evidence or used to change the locked design.

## Confirmatory execution policy

The confirmatory cohort is treated as one fixed experiment. Do not run an individual confirmatory seed interactively and then revise the study.

Use the cohort runner:

```bash
python experiments/run_natural_ja_en_s2_confirmatory.py \
  --data-dir /path/to/natural_r1 \
  --outdir /path/to/natural_r1_results
```

The runner:

1. verifies the frozen corpus hashes before launching anything;
2. checks that no fixed confirmatory seed already occurs as a seed-valued entry in committed `results/*.csv`;
3. launches/resumes exactly seeds 31000–31009;
4. validates each completed seed's summary/audit pair before treating it as resumable;
5. after all ten are complete, runs `adjudicate_natural_ja_en_s2_hidden_teacher.py` exactly once and saves the final adjudication.

If the cohort is incomplete, the adjudicator returns **UNCERTAIN** rather than turning partial data into a scientific verdict.

## Current scientific status

- Synthetic phase: frozen and publicly auditable.
- R1 natural design: locked.
- R1 implementation: validated.
- R1 confirmatory outcomes: **none yet**.
- Natural Japanese -> English acceleration: **still not established**.
