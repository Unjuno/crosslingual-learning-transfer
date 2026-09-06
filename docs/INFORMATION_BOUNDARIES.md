# Information boundaries

The phrase “B is not shown during phase 1” can mean several different things. This repository uses the term only for the **Student input stream** unless a stricter condition is explicitly stated.

That distinction matters because a Student can receive only A-labeled inputs while the curriculum designer, teacher, initialization, or auxiliary target construction still contains information derived from B.

## Four useful levels

| Level | Student phase-1 input | Curriculum / teacher may use B knowledge? | What the result can establish |
|---|---|---|---|
| **S0 — target-unaware system** | A only | No | A-only experience itself produces a target-general preparation effect. |
| **S1 — target-aware curriculum, A-only Student input** | A only | Yes, to choose/order/generate A-side training examples | B-aware curriculum design can write a later-useful state without exposing B examples to the Student. |
| **S2 — target-aware teacher signal, A-only Student input** | A only | Yes, including hidden/logit/representation targets | Target knowledge can be transmitted through an A-side teaching channel. |
| **S3 — target-exposed Student** | A and/or B | Yes | Ordinary transfer/adaptation; this does not test the phase-1 input restriction. |

The synthetic hidden-teacher and target-compatible curriculum studies are primarily **S1/S2**, not S0.

## Concrete information paths in the public experiments

### Random-rank GRU / Transformer sequence

The random-rank scripts train a common base model on uniform A and B streams before the A-only curriculum phase. During the A-only phase, `target25` still feeds A-labeled sequences to the Student, but a subset of those sequences is sampled from the target-derived transition matrix `pbt`.

Therefore:

- the **phase-1 curriculum input IDs are A-side**;
- the **curriculum construction is target-aware**;
- the base learner has already participated in a common A/B preparation stage;
- the installed B decoder is explicitly constructed at cosine alignment `rho=0.25` to the A decoder in the documented random-rank mechanism tests.

This setup supports a causal statement about **where target-relevant structure is written and later accessed**. It does not support the stronger claim that a learner or curriculum designer with zero B information spontaneously discovers an optimal B-preparing A curriculum.

Representative implementations:

- `experiments/random_rank_state_alignment_gru.py`
- `experiments/random_rank_state_alignment_transformer.py`

### Hidden-teacher sequence

When a Student trained on A-side inputs receives a hidden target from a teacher that knows A+B, the Student does not receive B examples directly during the intervention, but **B knowledge is present in the teacher signal**.

The correct interpretation is therefore S2: target-aware representation teaching through an A-side channel.

### Natural Japanese -> English pilot

The Japanese-only pilot used an offline English-target transition motif to select/construct part of the Japanese curriculum. The Student phase-1 examples were Japanese, but the curriculum designer was target-aware.

The preregistered primary comparison against ordinary Japanese failed, so the repository does not claim that this S1-style natural-language bridge accelerates English learning.

Representative files:

- `protocols/natural_ja_en_blended5.json`
- `experiments/natural_ja_en_blended5_curriculum.py`
- `results/natural_ja_en_blended5_confirmatory_tests.csv`

## Claims that must not be conflated

These are separate empirical questions:

1. **Input restriction:** Can the Student see only A during phase 1?
2. **System information restriction:** Can the teacher/curriculum designer also avoid all B-derived information?
3. **Prior-history restriction:** Has the Student ever been pretrained on B or a multilingual corpus containing B?
4. **Transfer outcome:** Is later B starting performance better?
5. **Learning-efficiency outcome:** After controlling for starting performance, does the Student require fewer B tokens/updates to reach the same B performance?

A positive answer to one does not imply the others.

## Required reporting for any next natural-language phase

Every experiment should declare these fields before outcomes are inspected:

- Student checkpoint and documented prior language exposure;
- tokenizer construction data;
- exact phase-1 Student-visible inputs;
- any B-derived information used by curriculum selection;
- teacher checkpoint and target-language exposure;
- whether hidden/logit/embedding teacher signals are allowed;
- phase-1 token/update budget;
- phase-2 B token/update budget;
- starting-B performance and fixed target-performance thresholds;
- target-language development data versus untouched final test data.

The next project phase should not use the phrase “Japanese-only makes English easier” without also stating which information-boundary level is being tested.
