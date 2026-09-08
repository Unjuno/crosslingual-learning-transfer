import copy,unittest
from verify_pud_r3_public import check,SEEDS,ARMS
class PublicTests(unittest.TestCase):
    def fixture(self):
        rows=[{'seed':s,'arm':a,'english_step0_nll':4.,'target_update':90 if a=='JA_TEN' else 100,'high_update':50,
               'japanese_phase1_nll':2.,'japanese_at_target_nll':2.005,'english_end600_nll':2.5,'japanese_end600_nll':3.}
              for s in SEEDS for a in ARMS]
        audits=[{'seed':s,'optimizer_updates':5400,'dense_curve_rows':3005,'all_execution_checks':True,
                 'checkpoint_record_max_error':0.,'data_manifest_sha256':'a70abd42d4ca57a9f27a6a784388559dc7ebb4366ced28d4aef7285a08e41236',
                 'phase1_digests':{a:'0'*64 for a in ARMS-{'REF_EN'}},'phase2_digests':{a:'1'*64 for a in ARMS}} for s in SEEDS]
        return rows,audits
    def test_fixture_pass(self):self.assertEqual(check(*self.fixture())['joint_utility_verdict'],'PASS')
    def test_retention_fail(self):
        rows,a=self.fixture()
        for r in rows:
            if r['arm']=='JA_TEN':r['japanese_at_target_nll']=3.
        self.assertEqual(check(rows,a)['joint_utility_verdict'],'FAIL')
    def test_missing(self):
        r,a=self.fixture()
        with self.assertRaises(ValueError):check(r[:-1],a)
    def test_duplicate(self):
        r,a=self.fixture()
        with self.assertRaises(ValueError):check(r+[r[0]],a)
    def test_nonfinite(self):
        r,a=self.fixture();r[0]['japanese_phase1_nll']=float('nan')
        with self.assertRaises(ValueError):check(r,a)
    def test_false_audit(self):
        r,a=self.fixture();a[0]['all_execution_checks']=False
        with self.assertRaises(ValueError):check(r,a)
    def test_actual_batches_disagree(self):
        r,a=self.fixture();a[0]['phase2_digests']['JA_TEN']='2'*64
        with self.assertRaises(ValueError):check(r,a)
    def test_fractional_crossing(self):
        r,a=self.fixture();r[0]['target_update']=20.5
        with self.assertRaises(ValueError):check(r,a)
    def test_data_substitution(self):
        r,a=self.fixture();a[0]['data_manifest_sha256']='0'*64
        with self.assertRaises(ValueError):check(r,a)
    def test_checkpoint_metric_discrepancy(self):
        r,a=self.fixture();a[0]['checkpoint_record_max_error']=1e-3
        with self.assertRaises(ValueError):check(r,a)
if __name__=='__main__':unittest.main()
