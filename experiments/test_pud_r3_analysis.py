import unittest, tempfile
from pathlib import Path
import analyze_pud_r3 as an
class AnalysisTests(unittest.TestCase):
    def fixture(self):
        curves={};summaries={}
        for s in an.SEEDS:
            rows=[]
            for a in an.ARMS:
                t=90 if a=='JA_TEN' else 100
                curve=[3.5-.5*min(i/t,1.) for i in range(601)]
                curves[s,a]=curve
                rows.append({'seed':s,'arm':a,'target_update':t,'high_update':an.hit(curve,3.25),
                             'english_step0_nll':curve[0],'english_end600_nll':curve[-1],
                             'japanese_phase1_nll':2.,'japanese_at_target_nll':2.005,
                             'japanese_end600_nll':2.1})
            summaries[s]=rows
        return curves,summaries
    def test_known_pass(self):
        c,s=self.fixture();a=an.summarize(c,s)
        self.assertEqual(a['joint_utility_verdict'],'PASS');self.assertAlmostEqual(a['mean_saving'],.1)
    def test_retention_failure_does_not_erase_efficacy(self):
        c,s=self.fixture()
        for rows in s.values():
            for r in rows:
                if r['arm']=='JA_TEN':r['japanese_at_target_nll']=3.
        a=an.summarize(c,s);self.assertEqual(a['efficacy_verdict'],'PASS');self.assertEqual(a['joint_utility_verdict'],'FAIL')
    def test_constant_no_benefit(self):
        c,s=self.fixture()
        for seed in an.SEEDS:
            c[seed,'JA_TEN']=c[seed,'JA_TJA'][:]
            for r in s[seed]:
                if r['arm']=='JA_TEN':r.update(target_update=100,high_update=50)
        a=an.summarize(c,s);self.assertEqual(a['efficacy_verdict'],'FAIL');self.assertAlmostEqual(a['mean_saving'],0.)
    def test_missing_seed(self):
        with tempfile.TemporaryDirectory() as d:
            a=an.analyze(Path(d));self.assertEqual(a['joint_utility_verdict'],'UNCERTAIN');self.assertEqual(len(a['missing_seeds']),10)
    def test_false_summary_crossing(self):
        c,s=self.fixture();s[33000][0]['target_update']=99
        with self.assertRaises(ValueError):an.summarize(c,s)
    def test_sign_probability(self):
        self.assertEqual(an.sign_p(10,10),1/1024);self.assertEqual(an.sign_p(9,10),11/1024)
    def test_no_interpolation(self):
        self.assertEqual(an.hit([3.5,3.1,2.9],3.0),2)
    def test_first_observed_crossing(self):
        self.assertEqual(an.hit([4,3,3.1,2.9],3.0),1)
    def test_censoring_is_not_favorable(self):
        c,s=self.fixture();c[33000,'JA_TEN']=[4.]*601
        for r in s[33000]:
            if r['arm']=='JA_TEN':r.update(target_update=None,high_update=None,english_step0_nll=4.,english_end600_nll=4.,japanese_at_target_nll=None)
        a=an.summarize(c,s);self.assertEqual(a['joint_utility_verdict'],'UNCERTAIN');self.assertEqual(a['favorable_seeds'],9)
    def test_duplicate_arm_rejected(self):
        c,s=self.fixture();s[33000].append(s[33000][0].copy())
        with self.assertRaises(ValueError):an.summarize(c,s)
