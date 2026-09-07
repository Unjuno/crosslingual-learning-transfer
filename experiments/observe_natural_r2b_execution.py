#!/usr/bin/env python3
"""Observational instrumentation of the already locked half-weight experiment."""
from pathlib import Path
import argparse
import sys
import observe_natural_r1_execution as obs
import natural_ja_en_s2_halfweight_tradeoff as half

PINNED_COMMIT = 'efbcc3f2fe5e809f539a95eff3d120c21ec98826'
HALF_WRAPPER_BLOB = '9dcc9f5a4c4c76ed62e173ffb99b9f911de84996'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', required=True, type=int)
    ap.add_argument('--data-dir', required=True, type=Path)
    ap.add_argument('--outdir', required=True, type=Path)
    args = ap.parse_args()
    if args.seed not in half.SEEDS:
        raise ValueError('Seed is not in the locked R2b cohort')
    if obs.file_blob(Path(half.__file__)) != HALF_WRAPPER_BLOB:
        raise RuntimeError('Half-weight wrapper is not the pinned source')
    if obs.file_blob(Path(half.core.__file__)) != obs.PINNED_CORE_BLOB:
        raise RuntimeError('Core is not the pinned source')
    obs.PINNED_COMMIT = PINNED_COMMIT
    observer = obs.Observer(args.seed, args.outdir, checkpoints=True)
    observer.install()
    done = False
    try:
        half.main()  # Original argv and original scientific entry point.
        done = True
    finally:
        trace = observer.save_trace(done, validation=False)
        trace['experiment'] = half.PROTOCOL
        trace['halfweight_wrapper_blob'] = obs.file_blob(Path(half.__file__))
        trace['hidden_distillation_weight'] = half.core.HIDDEN_WEIGHT
        obs.atomic_json(args.outdir / f'seed{args.seed}_execution_trace.json', trace)
    required = ['phase1_same_batches','phase2_same_batches',
                'pre_phase2_student_only_ja','teacher_target_slots_match']
    if not all(trace[x] for x in required) or trace['steps_observed'] != 5400:
        raise RuntimeError('Measured execution trace failed')
    observer.event('complete', trace_audits_pass=True)


if __name__ == '__main__':
    main()
