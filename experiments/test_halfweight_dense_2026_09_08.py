import csv
import unittest
from pathlib import Path
import verify_halfweight_dense_2026_09_08 as v

class Tests(unittest.TestCase):
    def setUp(self):
        p=Path(__file__).resolve().parents[1]/'results/natural_r2_halfweight_dense_spans_2026-09-08.csv'
        with p.open(newline='') as f:self.rows=list(csv.DictReader(f))
    def test_observed_resolution(self):
        r=v.recompute(self.rows)
        self.assertLess(r['dense_interpolated_mean_saving'],.05)
        self.assertEqual(r['dense_interpolated_favorable'],10)
    def test_missing(self):
        with self.assertRaises(ValueError):v.recompute(self.rows[:-1])
    def test_nonfinite(self):
        self.rows[0]['low_crossing']='nan'
        with self.assertRaises(ValueError):v.recompute(self.rows)
    def test_mismatch(self):
        self.rows[0]['interpolated_span']='80'
        with self.assertRaises(ValueError):v.recompute(self.rows)
    def test_noninteger_observation(self):
        self.rows[0]['discrete_span']='110.5'
        with self.assertRaises(ValueError):v.recompute(self.rows)
if __name__=='__main__':unittest.main()
