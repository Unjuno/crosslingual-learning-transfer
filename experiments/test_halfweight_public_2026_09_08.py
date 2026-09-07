import copy
import unittest
from pathlib import Path
import verify_halfweight_public_2026_09_08 as v

class Tests(unittest.TestCase):
    def setUp(self):
        r=Path(__file__).resolve().parents[1]/'results'
        self.rows=v.read_csv(r/'natural_r2_halfweight_seed_summary_2026-09-08.csv')
        self.audits=v.read_csv(r/'natural_r2_halfweight_execution_audits_2026-09-08.csv')
    def test_observed_pass(self):
        self.assertEqual(v.recompute(self.rows,self.audits)['verdict'],'PASS')
    def test_missing_cell(self):
        with self.assertRaises(ValueError):v.recompute(self.rows[:-1],self.audits)
    def test_duplicate_cell(self):
        self.rows[-1]=copy.deepcopy(self.rows[0])
        with self.assertRaises(ValueError):v.recompute(self.rows,self.audits)
    def test_nonfinite(self):
        self.rows[0]['span_updates']='nan'
        with self.assertRaises(ValueError):v.recompute(self.rows,self.audits)
    def test_smoke(self):
        self.rows[0]['smoke_only']='True'
        with self.assertRaises(ValueError):v.recompute(self.rows,self.audits)
    def test_wrong_digest(self):
        self.audits[0]['phase1_batch_sha256']='bad'
        with self.assertRaises(ValueError):v.recompute(self.rows,self.audits)
    def test_duplicate_audit(self):
        self.audits[-1]=copy.deepcopy(self.audits[0])
        with self.assertRaises(ValueError):v.recompute(self.rows,self.audits)
    def test_retention_failure(self):
        for row in self.rows:
            if row['condition']=='JA_TEN':row['JA_post_nll']=str(float(row['JA_post_nll'])*1.02)
        self.assertEqual(v.recompute(self.rows,self.audits)['verdict'],'FAIL')
    def test_no_speedup(self):
        by={(r['seed'],r['condition']):r for r in self.rows}
        for row in self.rows:
            if row['condition']=='JA_TEN':row['span_updates']=by[(row['seed'],'JA_TJA')]['span_updates']
        got=v.recompute(self.rows,self.audits)
        self.assertEqual(got['n_favorable'],0);self.assertEqual(got['verdict'],'FAIL')
    def test_threshold_mismatch(self):
        self.rows[0]['reference_high']='3.9'
        with self.assertRaises(ValueError):v.recompute(self.rows,self.audits)
if __name__=='__main__':unittest.main()
