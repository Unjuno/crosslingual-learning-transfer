# Representative experiment snapshots

These files are representative snapshots from the executed research sequence. They are intentionally kept close to the executed implementations rather than rewritten as a polished library.

## Structural surrogate phase

- `study2_direction_fixed6_bigram_surrogate.py` — six-language character-level Transformer with original, exact-bigram-preserving, and unigram/degree-margin control corpora.

## Controlled curriculum phase

- `gru_aligned_transfer_confirmatory.py` — independent-architecture A-only curriculum existence test.
- `matched_wrong_map25_gru.py` — signed transfer with source-distance/entropy/margin-matched wrong controls.

## Performance and representation mechanism

- `performance_aligned_learning_minimal_gru.py` — same-B-performance-interval sample-efficiency test.
- `decoder_alignment_threshold_gru.py` — decoder cosine-alignment dose response.
- `embedding_rotation_ablation_rho025_gru.py` — geometry-preserving B-state coordinate rotation.
- `random_rank_state_alignment_gru.py` — shared latent-coordinate rank intervention.

### Final learner-conditioning closure

- `run_random_rank_hidden_width.py` — hidden-width wrapper that reuses the GRU random-rank implementation while varying learner width and seed.
- `analyze_random_rank_hidden_width.py` — recomputes the locked 80–20 performance-aligned span and family rank slopes from width-run curves.
- `prestate_random_rank_features.py` — constrained pre-phase1 scalar feature extractor used for the four-candidate screen and effective-rank confirmation.
- `analyze_prestate_state_rank_predictor.py` — applies the locked within-seed×width centering and Spearman adjudication for the effective-rank predictor.
- `random_rank_state_alignment_transformer.py` — one-layer causal Transformer port of the random-rank shared-coordinate experiment.
- `analyze_random_rank_transformer.py` — computes Transformer family rank slopes and exact one-sided sign summaries.

Final protocols/results:

- `protocols/random_rank_hidden_width_confirmatory_2026-09-02.json` — H32 vs H48 boundary confirmation; **FAIL**.
- `protocols/prestate_state_rank_predictor_confirmatory_2026-09-03.json` — simple pre-state effective-rank predictor; **FAIL**.
- `protocols/random_rank_transformer_replication_2026-09-03.json` — independent causal-Transformer replication; **PASS** in two fresh cohorts.
- `docs/FINAL_SYNTHETIC_STATUS_2026-09-03.md` — final synthetic-phase adjudication and stopping decision.

Example width run:

```bash
HIDDEN_WIDTH=48 SEED=20000 FAMILY_ONLY=0 OUTDIR=/tmp/random_rank_width \
python experiments/run_random_rank_hidden_width.py
```

Example Transformer run:

```bash
SEED=22000 FAMILY_ONLY=0 OUTDIR=/tmp/random_rank_transformer \
python experiments/random_rank_state_alignment_transformer.py
```

## Hidden-teacher / state-binding phase

- `state_signal_cycle_adjudication.py` — L=1 Transformer intervention that preserves Bsub signal norm while cycle-averaging state-to-signal-direction assignment. The default execution is the locked seed18500 follow-up; use explicit environment variables to reproduce the earlier seed18400 pilot.
- `run_state_signal_cycle_confirmatory.py` — locked 5-family × 6-source seed18500 launcher. It forces all six targets, validates resumable 12-row source outputs, and runs the confirmatory adjudicator after all jobs finish.
- `state_signal_cycle_confirmatory_adjudicator.py` — standard-library-only implementation of every locked seed18500 PASS/audit gate. Complete negative outcomes return `FAIL`; incomplete or malformed data return `UNCERTAIN` rather than being interpreted scientifically.

### Reproducibility blocker for seed18500

The public snapshot does not contain three exact archived helper modules required for a faithful rerun:

- `teacher_hidden_entropy_matched_distance_transformer.py`
- `teacher_hidden_geometry_intervention.py`
- `support_stationary_matrix.py`

The confirmatory runner fails before launching jobs when these files are absent. This is deliberate: reconstructed or behaviorally similar replacements must not be substituted into a locked confirmatory rerun. Restore the exact executed helper snapshots first; until then seed18500 is **BLOCKED / NOT COUNTED**.

The broader hidden-teacher and distance-recovery sequence contains many execution variants; rather than publish every near-duplicate script, the repository keeps representative implementations plus compact protocols/results and a chronological log in `docs/EXPERIMENT_LOG.md`.

## Natural-language pilot

- `natural_ja_en_blended5_curriculum.py` — Japanese-only 95/5 curriculum pilot with English-target and Russian-target offline motif selectors.

Some older snapshots retain `/mnt/data/...` roots from the original execution environment. See `docs/REPRODUCIBILITY.md` before rerunning them. Large checkpoints, raw logs, and third-party corpora are intentionally excluded.
