import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('publication',ROOT/'scripts/verify_publication.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


class PublicationTests(unittest.TestCase):
    def test_safe_path(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(v.safe_path(Path(d),'docs/note.md'),Path(d)/'docs/note.md')
    def test_parent_path_rejected(self):
        with self.assertRaises(ValueError):v.safe_path(ROOT,'../secret.txt')
    def test_absolute_path_rejected(self):
        with self.assertRaises(ValueError):v.safe_path(ROOT,'/tmp/private.txt')
    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'x').write_text('test');(p/'link').symlink_to(p/'x')
            with self.assertRaises(ValueError):v.safe_path(p,'link')
    def test_weights_rejected(self):
        with self.assertRaises(ValueError):v.check_file_policy('model.pt')
    def test_corpus_rejected(self):
        with self.assertRaises(ValueError):v.check_file_policy('ja_train.txt')
    def test_env_rejected(self):
        with self.assertRaises(ValueError):v.check_file_policy('.env')
    def test_compact_csv_allowed(self):v.check_file_policy('results/numbers.csv')
    def test_github_token_rejected(self):
        with self.assertRaises(ValueError):v.check_text('x','gh'+'p_'+'a'*36)
    def test_private_key_rejected(self):
        with self.assertRaises(ValueError):v.check_text('x','-----BEGIN '+'PRIVATE KEY-----')
    def test_runner_placeholder_allowed(self):v.check_text('workflow','${{ secrets.GITHUB_TOKEN }}')
    def test_nonpublic_link_rejected(self):
        with self.assertRaises(ValueError):v.check_text('x','sand'+'box:/mnt/data/result.zip')
    def test_missing_local_link(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):v.local_links(Path(d),'README.md','[x](missing.md)')
    def test_external_link_not_fetched(self):
        self.assertEqual(v.local_links(ROOT,'README.md','[x](https://example.org/x)'),0)
    def test_link_inside_code_ignored(self):
        self.assertEqual(v.local_links(ROOT,'README.md','```\n[x](missing.md)\n```'),0)
    def test_nan_json_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json';p.write_text('{"x": NaN}')
            with self.assertRaises(ValueError):v.read_json(p)
    def test_mutable_action_rejected(self):
        with self.assertRaises(ValueError):v.check_workflow('test.yml','permissions:\n  contents: read\n- uses: actions/checkout@v4')
    def test_unexpected_write_rejected(self):
        with self.assertRaises(ValueError):v.check_workflow('test.yml','contents: read\ncontents: write')
    def test_snapshot_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'publication').mkdir();(p/'record.csv').write_text('changed')
            (p/v.SNAPSHOT).write_text(json.dumps({'source_commit':'63a6a056f9575d027dde09d3d19be172e821cd95','files':{'record.csv':'0'*64}}))
            with self.assertRaises(ValueError):v.check_snapshot(p)
    def test_historical_verdict_drift_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for name in ['results/FINAL_STATUS.json','publication/release.json','results/current_status_2026-09-09.json','results/natural_r1_confirmatory_adjudication.json','results/natural_r2_halfweight_verification_2026-09-08.json','CITATION.cff']:
                q=p/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/name).read_bytes())
            q=p/'results/FINAL_STATUS.json';obj=json.loads(q.read_text());obj['historical_decisions']['natural_R1_weight1']='PASS';q.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):v.check_final_status(p)
    def test_untrusted_release_branch_rejected(self):
        import os,subprocess,sys
        env=dict(os.environ);env['GITHUB_REPOSITORY']='not-the/canonical-repo';env['GITHUB_REF']='refs/heads/feature'
        r=subprocess.run([sys.executable,str(ROOT/'scripts/publish_release.py')],env=env,text=True,capture_output=True)
        self.assertNotEqual(r.returncode,0);self.assertIn('canonical main branch',r.stderr)
    def test_current_final_status(self):v.check_final_status(ROOT)
    def test_historical_snapshot(self):self.assertGreater(v.check_snapshot(ROOT),50)


if __name__=='__main__':unittest.main()
