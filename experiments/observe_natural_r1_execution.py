#!/usr/bin/env python3
"""Observe the pinned R1 implementation without changing training computations.

Adds source/batch tracing, progress, stage checkpoints and runtime provenance.
The original core file remains byte-for-byte identical to Git blob aab1618.
"""
from __future__ import annotations
import argparse
import collections
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from datetime import datetime, timezone

import natural_ja_en_s2_hidden_teacher as core

PINNED_CORE_BLOB = 'aab161824538e29f71f2a5f00fd44364f2de9bd4'
PINNED_COMMIT = '21775d9529c4a65cf4283ff03df2f489482e91ba'


def file_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Observer:
    def __init__(self, seed: int, outdir: Path, checkpoints: bool):
        self.seed = seed
        self.outdir = outdir
        self.checkpoints = checkpoints
        self.start = time.monotonic()
        self.phase = 'setup'
        self.calls = collections.Counter()
        self.ids = {}
        self.sources = {}
        self.digests = {}
        self.schedules = {}
        self.phases_finished = []
        self.last_batch = None
        self.corpus_ids = {}
        self.phase2_index = 0
        self.steps = 0
        self.last_model = None
        self.last_optimizer = None
        self.log = outdir / f'seed{seed}_events.jsonl'
        self.log.parent.mkdir(parents=True, exist_ok=True)
        if self.log.exists():
            raise FileExistsError(f'Never overwrite an execution log: {self.log}')
        self.event('start', pinned_commit=PINNED_COMMIT, core_blob=PINNED_CORE_BLOB)

    def event(self, event: str, **fields):
        record = dict(event=event, seed=self.seed, phase=self.phase,
                      utc=utc_now(), elapsed_seconds=time.monotonic()-self.start,
                      optimizer_steps=self.steps, **fields)
        with self.log.open('a') as f:
            f.write(json.dumps(record, allow_nan=False) + '\n')
        atomic_json(self.outdir / f'seed{self.seed}_progress.json', record)
        print(json.dumps(record), flush=True)

    def slot(self, model) -> str:
        if id(model) not in self.ids:
            self.ids[id(model)] = len(self.ids)
        ix = self.ids[id(model)]
        if self.phase == 'teachers':
            return ['T_JA', 'T_EN', 'T_RU'][ix]
        if self.phase == 'phase1':
            return ['JA_nat', 'JA_TJA', 'JA_TEN', 'JA_TRU'][ix]
        return self.phase

    def enter(self, name):
        self.phase = name
        self.ids = {}
        self.event('phase_start')

    def finish(self):
        self.event('phase_done')
        if self.checkpoints and self.last_model is not None:
            root = self.outdir / 'checkpoints' / f'seed{self.seed}'
            root.mkdir(parents=True, exist_ok=True)
            path = root / f'{self.phase}.pt'
            # Stage-end state of the last model only; observational checkpoint,
            # not claimed to be a complete multi-arm mid-stage resume image.
            core.torch.save({'model': self.last_model.state_dict(),
                             'optimizer': self.last_optimizer.state_dict(),
                             'torch_rng': core.torch.random.get_rng_state(),
                             'seed': self.seed, 'phase': self.phase}, path)
        self.phases_finished.append(self.phase)
        self.last_model = self.last_optimizer = None

    def install(self):
        orig_load = core.load_corpora
        @functools.wraps(orig_load)
        def load(*args, **kwargs):
            result = orig_load(*args, **kwargs)
            (corp, _), _ = result
            self.corpus_ids = {v.data_ptr(): k for k, v in corp.items()}
            return result
        core.load_corpora = load

        orig_batch = core.batch_from
        @functools.wraps(orig_batch)
        def batch(corpus, batch_size, seed):
            z = orig_batch(corpus, batch_size, seed)
            self.last_batch = (id(z), self.corpus_ids.get(corpus.data_ptr()), seed)
            return z
        core.batch_from = batch

        orig_step = core.optimize_step
        @functools.wraps(orig_step)
        def step(model, optimizer, z, teacher=None):
            if self.last_batch is None or self.last_batch[0] != id(z):
                raise RuntimeError('untraceable training batch')
            source, batch_seed = self.last_batch[1:]
            slot = self.slot(model)
            key = self.phase + '/' + slot
            self.calls[key] += 1
            self.sources.setdefault(key, collections.Counter())[source] += 1
            self.digests.setdefault(key, hashlib.sha256()).update(z.numpy().tobytes())
            self.schedules.setdefault(key, []).append(source)
            result = orig_step(model, optimizer, z, teacher)
            if not all(math.isfinite(v) for v in result):
                raise FloatingPointError(f'nonfinite loss in {key}')
            self.last_model, self.last_optimizer = model, optimizer
            self.steps += 1
            if self.steps % 200 == 0:
                self.event('progress')
            return result
        core.optimize_step = step

        for name, phase in [('train_common_base', 'base'),
                            ('train_teachers', 'teachers'),
                            ('train_phase1', 'phase1')]:
            original = getattr(core, name)
            def make_wrapper(fn, label):
                @functools.wraps(fn)
                def wrapped(*args, **kwargs):
                    self.enter(label)
                    result = fn(*args, **kwargs)
                    self.finish()
                    return result
                return wrapped
            setattr(core, name, make_wrapper(original, phase))

        original_curve = core.train_english_curve
        @functools.wraps(original_curve)
        def curve(*args, **kwargs):
            label = ['REF_EN', 'JA_nat', 'JA_TJA', 'JA_TEN', 'JA_TRU'][self.phase2_index]
            self.phase2_index += 1
            self.enter('phase2_' + label)
            result = original_curve(*args, **kwargs)
            if not all(math.isfinite(v) for _, v in result):
                raise FloatingPointError('nonfinite eval NLL')
            atomic_json(self.outdir / f'seed{self.seed}_{label}_curve.partial.json', result)
            self.finish()
            return result
        core.train_english_curve = curve

    def save_trace(self, completed: bool, validation: bool):
        digests = {k: v.hexdigest() for k, v in self.digests.items()}
        p1 = [v for k, v in digests.items() if k.startswith('phase1/')]
        p2 = [v for k, v in digests.items() if k.startswith('phase2_')]
        base_sources = [v for k,v in self.sources.items() if k.startswith(('base/', 'phase1/'))]
        ten = self.schedules.get('teachers/T_EN', [])
        tru = self.schedules.get('teachers/T_RU', [])
        trace = dict(seed=self.seed, complete=completed, validation_only=validation,
                     steps_observed=self.steps, phases_finished=self.phases_finished,
                     model_update_counts=dict(self.calls),
                     actual_source_counts={k:dict(v) for k,v in self.sources.items()},
                     actual_batch_sha256=digests,
                     phase1_same_batches=(len(p1)==4 and len(set(p1))==1),
                     phase2_same_batches=(len(p2)==5 and len(set(p2))==1),
                     pre_phase2_student_only_ja=all(set(v)=={'ja_train'} for v in base_sources),
                     teacher_target_slots_match=(bool(ten) and len(ten)==len(tru) and
                         [i for i,x in enumerate(ten) if x=='en_train']==
                         [i for i,x in enumerate(tru) if x=='ru_train']),
                     runtime_seconds=time.monotonic()-self.start,
                     pinned_commit=PINNED_COMMIT, core_blob=file_blob(Path(core.__file__)),
                     python=sys.version, torch=core.torch.__version__, numpy=core.np.__version__,
                     pandas=core.pd.__version__, platform=platform.platform(),
                     torch_num_threads=core.torch.get_num_threads(),
                     deterministic=core.torch.are_deterministic_algorithms_enabled())
        atomic_json(self.outdir / f'seed{self.seed}_execution_trace.json', trace)
        return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', required=True, type=int)
    ap.add_argument('--data-dir', required=True, type=Path)
    ap.add_argument('--outdir', required=True, type=Path)
    ap.add_argument('--validation-natural', action='store_true')
    ap.add_argument('--checkpoints', action='store_true')
    args = ap.parse_args()
    if file_blob(Path(core.__file__)) != PINNED_CORE_BLOB:
        raise RuntimeError('Pinned training code differs; refusing execution')
    obs = Observer(args.seed, args.outdir, args.checkpoints)
    obs.install()
    done = False
    try:
        core.run(args.seed, args.data_dir, args.outdir, smoke=False,
                 validation_natural=args.validation_natural)
        done = True
    finally:
        trace = obs.save_trace(done, args.validation_natural)
    required = ['phase1_same_batches','phase2_same_batches',
                'pre_phase2_student_only_ja','teacher_target_slots_match']
    if not all(trace[x] for x in required):
        raise RuntimeError('Measured execution trace failed audit')
    obs.event('complete', trace_audits_pass=True)


if __name__ == '__main__':
    main()
