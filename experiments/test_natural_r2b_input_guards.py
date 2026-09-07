"""Synthetic in-memory fixtures for verifier guards; never scientific outputs."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import verify_natural_r2b_raw as v


def csv_write(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture(root, seed, ten_speed=1.1, ten_ja=2.005):
    # These linear curves are test vectors, not sampled/trained model results.
    speeds = dict(REF_EN=1., JA_nat=1., JA_TJA=1., JA_TEN=ten_speed, JA_TRU=1.)
    curves = {arm: [(step, max(.2, 5 - .005 * speed * step)) for step in v.GRID]
              for arm, speed in speeds.items()}
    curve_rows = [dict(seed=seed, condition=arm, B_step=step, EN_nll=loss,
                      smoke_only=False, validation_only=False)
                  for arm, points in curves.items() for step, loss in points]
    summary = []
    for arm in v.ARMS:
        sp = v.span(curves[arm], 4.4, 2.6)
        summary.append(dict(seed=seed, condition=arm,
                            JA_post_nll=ten_ja if arm == 'JA_TEN' else 2.,
                            EN_step0_nll=5., span_updates='' if sp is None else sp,
                            censored=sp is None, reference_high=4.4, reference_low=2.6,
                            smoke_only=False, validation_only=False))
    csv_write(root / f'natural_s2_seed{seed}_curves.csv', curve_rows)
    csv_write(root / f'natural_s2_seed{seed}_summary.csv', summary)
    audit = dict(seed=seed, scientific_evidence=True, smoke_only=False, validation_only=False,
                 corpus_sha256=v.CORPUS, corpus_hashes_match_frozen_manifest=True,
                 protocol=v.PROTOCOL, hidden_distillation_weight=.5,
                 tradeoff_followup=True, parent_locked_verdict='FAIL', prior_result_seed_hits=[],
                 student_phase0_sources=dict(ja=400, en=0, ru=0),
                 student_phase1_sources=dict(ja=200, en=0, ru=0),
                 teacher_source_counts=dict(T_JA=dict(ja=400, en=0, ru=0),
                                            T_EN=dict(ja=200, en=200, ru=0),
                                            T_RU=dict(ja=200, en=0, ru=200)),
                 schedule=dict(base_steps=400, teacher_steps=400, phase1_steps=200,
                               phase2_steps=600, eval_every=10),
                 fixed_byte_vocab=256, teacher_target_slot_positions_identical=True)
    counts = v.expected_sources()
    trace = dict(seed=seed, complete=True, validation_only=False,
                 core_blob=v.CORE_BLOB, halfweight_wrapper_blob=v.WRAPPER_BLOB,
                 hidden_distillation_weight=.5, steps_observed=5400,
                 actual_source_counts=counts,
                 model_update_counts={k: sum(x.values()) for k,x in counts.items()},
                 phase1_same_batches=True, phase2_same_batches=True,
                 pre_phase2_student_only_ja=True, teacher_target_slots_match=True,
                 deterministic=True, actual_batch_sha256={k: 'a'*64 for k in counts},
                 torch_num_threads=1, runtime_seconds=1.)
    (root / f'natural_s2_seed{seed}_audit.json').write_text(json.dumps(audit))
    (root / f'seed{seed}_execution_trace.json').write_text(json.dumps(trace))


class InputGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='TEST_ONLY_r2b_')
        self.root = Path(self.tmp.name)
        fixture(self.root, 32000)
    def tearDown(self):
        self.tmp.cleanup()
    def mutate_csv(self, suffix, mutate):
        p = self.root / f'natural_s2_seed32000_{suffix}.csv'
        rows = v.read_csv(p); mutate(rows); csv_write(p, rows)
    def mutate_json(self, name, mutate):
        p = self.root / name; obj = json.loads(p.read_text()); mutate(obj); p.write_text(json.dumps(obj))
    def rejects(self):
        with self.assertRaises((ValueError, KeyError)):
            v.verify_seed(self.root, 32000)
    def test_valid_linear_fixture(self):
        self.assertTrue(v.verify_seed(self.root, 32000)['all_audits_pass'])
    def test_false_summary_span(self):
        self.mutate_csv('summary', lambda rows: rows[2].update(span_updates='1'))
        self.rejects()
    def test_missing_curve_point(self):
        self.mutate_csv('curves', lambda rows: rows.pop())
        self.rejects()
    def test_duplicate_curve_point(self):
        self.mutate_csv('curves', lambda rows: rows[1].update(B_step='0'))
        self.rejects()
    def test_mislabelled_seed(self):
        self.mutate_csv('summary', lambda rows: rows[0].update(seed='32001'))
        self.rejects()
    def test_smoke_not_science(self):
        self.mutate_csv('summary', lambda rows: rows[0].update(smoke_only='True'))
        self.rejects()
    def test_validation_not_science(self):
        self.mutate_csv('curves', lambda rows: rows[0].update(validation_only='True'))
        self.rejects()
    def test_hidden_weight_mismatch(self):
        self.mutate_json('natural_s2_seed32000_audit.json',lambda a:a.update(hidden_distillation_weight=1.))
        self.rejects()
    def test_incorrect_execution_sources(self):
        self.mutate_json('seed32000_execution_trace.json',lambda a:a['actual_source_counts']['phase1/JA_TEN'].update(en_train=1))
        self.rejects()
    def test_nonidentical_actual_batches(self):
        self.mutate_json('seed32000_execution_trace.json',lambda a:a['actual_batch_sha256'].update({'phase1/JA_TEN':'b'*64}))
        self.rejects()
    def test_full_cohort_pass(self):
        for seed in v.SEEDS:fixture(self.root,seed)
        self.assertEqual(v.verify(self.root)['verdict'],'PASS')
    def test_full_cohort_scientific_efficacy_failure(self):
        for seed in v.SEEDS:fixture(self.root,seed,ten_speed=.95)
        result=v.verify(self.root)
        self.assertEqual(result['verdict'],'FAIL')
        self.assertFalse(result['gates']['direction'])
    def test_full_cohort_scientific_retention_failure(self):
        for seed in v.SEEDS:fixture(self.root,seed,ten_ja=2.04)
        result=v.verify(self.root)
        self.assertEqual(result['verdict'],'FAIL')
        self.assertFalse(result['gates']['Japanese_retention'])
    def test_primary_censoring_not_dropped(self):
        for seed in v.SEEDS:fixture(self.root,seed,ten_speed=.1)
        result=v.verify(self.root)
        self.assertEqual(result['verdict'],'FAIL')
        self.assertIsNone(result['mean_saving_vs_TJA'])
        self.assertFalse(result['gates']['no_censoring'])


if __name__ == '__main__':
    unittest.main()
