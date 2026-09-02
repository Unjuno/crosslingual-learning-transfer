# Representative experiment snapshots

These files are representative snapshots from the executed research sequence. They are intentionally preserved close to the original container versions rather than rewritten as a polished package.

## Structural surrogate phase

- `study2_direction_fixed6_bigram_surrogate.py` — six-language character-level Transformer with original, exact-bigram-preserving, and unigram/degree-margin control corpora.

## Controlled curriculum phase

- `gru_aligned_transfer_confirmatory.py` — independent-architecture A-only curriculum existence test.
- `matched_wrong_map25_gru.py` — signed transfer with source-distance/entropy/margin-matched wrong controls.

## Performance and early mechanism phase

- `performance_aligned_learning_minimal_gru.py` — same-B-performance-interval sample-efficiency test.
- `decoder_alignment_threshold_gru.py` — decoder cosine-alignment dose response.
- `embedding_rotation_ablation_rho025_gru.py` — geometry-preserving B-state coordinate rotation.
- `random_rank_state_alignment_gru.py` — shared latent-coordinate rank intervention.

## Hidden-teacher / state-binding phase

- `state_signal_cycle_adjudication.py` — L=1 Transformer intervention that preserves Bsub signal norm while cycle-averaging state-to-signal-direction assignment. The default execution is the locked seed18500 follow-up; use explicit environment variables to reproduce the earlier seed18400 pilot.
- `run_state_signal_cycle_confirmatory.py` — locked 5-family × 6-source seed18500 launcher. It forces all six targets, validates resumable 12-row source outputs, and runs the confirmatory adjudicator after all jobs finish.
- `state_signal_cycle_confirmatory_adjudicator.py` — standard-library-only implementation of every locked seed18500 PASS/audit gate. Complete negative outcomes return `FAIL`; incomplete or malformed data return `UNCERTAIN` rather than being interpreted scientifically.

### Reproducibility blocker for the representative hidden-teacher snapshot

`state_signal_cycle_adjudication.py` depends on three archived helper snapshots that are not currently published in this repository:

- `teacher_hidden_entropy_matched_distance_transformer.py`
- `teacher_hidden_geometry_intervention.py`
- `support_stationary_matrix.py`

The confirmatory runner fails before launching jobs when these files are absent. This is deliberate: reconstructed or behaviorally similar replacements must not be substituted into a locked confirmatory rerun. Restore the exact executed helper snapshots first; only then rerun seed18500.

The broader hidden-teacher and distance-recovery sequence contains many execution variants; rather than publish every near-duplicate script, the repository keeps the current representative implementation plus compact protocols/results and a chronological log in `docs/EXPERIMENT_LOG.md`.

## Natural-language pilot

- `natural_ja_en_blended5_curriculum.py` — Japanese-only 95/5 curriculum pilot with English-target and Russian-target offline motif selectors.

Some snapshots retain `/mnt/data/...` roots from the execution environment. See `docs/REPRODUCIBILITY.md` before rerunning them. Large checkpoints, raw logs, and third-party corpora are intentionally excluded.
