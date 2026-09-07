import unittest
from pathlib import Path
import tempfile
from verify_natural_r2b_raw import (crossing, span, best_curve, sign_p, canonical_bool,
                                  number, verify, expected_sources)

class VerificationTests(unittest.TestCase):
    def test_linear_crossing(self):
        self.assertEqual(crossing([(0,5.),(10,3.)],4.),5.)
    def test_first_crossing_uses_cumulative_best(self):
        self.assertEqual(crossing([(0,5.),(10,3.),(20,4.)],3.5),7.5)
    def test_shared_interval_span(self):
        self.assertAlmostEqual(span([(0,5.),(10,3.)],4.6,3.4),6.)
    def test_censoring_never_zero_filled(self):
        self.assertIsNone(span([(0,5.),(10,4.)],4.6,3.4))
    def test_already_below_interval(self):
        self.assertEqual(span([(0,2.),(10,1.)],4.,3.),0.)
    def test_duplicate_steps_rejected(self):
        with self.assertRaises(ValueError):best_curve([(0,5.),(0,4.)])
    def test_nonpositive_interval_rejected(self):
        with self.assertRaises(ValueError):span([(0,5.),(10,3.)],3.,4.)
    def test_nan_rejected(self):
        with self.assertRaises(ValueError):number('nan','x')
    def test_noncanonical_bool_rejected(self):
        with self.assertRaises(ValueError):canonical_bool('false')
    def test_exact_sign_gate_boundaries(self):
        self.assertEqual(sign_p(10),1/1024)
        self.assertEqual(sign_p(9),11/1024)
        self.assertGreater(sign_p(8),0.05)
    def test_input_missing_not_scientific_fail(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):verify(Path(d))
    def test_update_budget(self):
        self.assertEqual(sum(sum(x.values()) for x in expected_sources().values()),5400)

if __name__=='__main__':unittest.main()
