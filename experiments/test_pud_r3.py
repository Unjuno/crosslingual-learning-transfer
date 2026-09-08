import copy, hashlib, importlib, json, tempfile, unittest
from pathlib import Path
import numpy as np
import torch
import run_pud_r3 as r
from prepare_pud_r3 import parse, grams, normalize

class Tests(unittest.TestCase):
    def test_eval_exact_byte_count(self):
        rows=[{'text':'abc'},{'text':'日本語ですよ。'*30},{'text':'z'*129}]
        batches,n=r.make_eval(rows)
        self.assertEqual(n,sum(len(x['text'].encode())-1 for x in rows))
        targets=[]
        for _,y,_,_ in batches:targets.extend(y[y!=-100].tolist())
        self.assertEqual(targets,[v for x in rows for v in x['text'].encode()[1:]])
    def test_no_sentence_crossing(self):
        b,n=r.make_eval([{'text':'abc'},{'text':'XYZ'}])
        x,y,ids,ct=b[0]
        self.assertEqual(x[1,:2].tolist(),[88,89]);self.assertEqual(y[0,:2].tolist(),[98,99])
    def test_uniform_nll(self):
        class Flat(torch.nn.Module):
            def forward(self,x):return torch.zeros((*x.shape,256))
        e,_=r.evaluate(Flat(),r.make_eval([{'text':'日ab'*90}]))
        self.assertAlmostEqual(e,float(np.log(256)),places=6)
    def test_evaluation_rng_unchanged(self):
        torch.manual_seed(99999);m=r.core.Model();state=torch.get_rng_state().clone()
        r.evaluate(m,r.make_eval([{'text':'sample 日本語'}]))
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
    def test_causal_prefix(self):
        torch.manual_seed(99998);m=r.core.Model();a=torch.zeros((1,128),dtype=torch.long);b=a.clone();b[:,90:]=255
        with torch.no_grad():pa=m(a);pb=m(b)
        self.assertTrue(torch.equal(pa[:,:90],pb[:,:90]))
    def test_observed_digest(self):
        obs=r.Observer();z=torch.zeros((16,129),dtype=torch.long)
        obs.update('a','ja',z);obs.update('b','ja',z)
        self.assertEqual(obs.export()['digests']['a'],obs.export()['digests']['b'])
        z[0,0]=1;obs.update('a','ja',z)
        self.assertNotEqual(obs.export()['digests']['a'],obs.export()['digests']['b'])
    def test_original_core_unchanged(self):
        b=Path(r.core.__file__).read_bytes()
        self.assertEqual(hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),r.CORE_BLOB)
    def test_parser_excludes_companion_english(self):
        txt=''.join(f'# newdoc id = d{i}\n# sent_id = {i}\n# text = 日本語\n# text_en = DO NOT USE\n# english_text = EXCLUDE\n\n' for i in range(1000))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input';p.write_text(txt)
            rows=parse(p);self.assertTrue(all(x['text']=='日本語' for x in rows.values()))
    def test_json_rejects_nan(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):r.write_json(Path(d)/'x',{'a':float('nan')})
    def test_manifest_rejects_change(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'MANIFEST.json').write_text('{}')
            with self.assertRaises(ValueError):r.read_data(Path(d))
    def test_normalized_duplicate_rule(self):
        self.assertEqual(grams('ABC def!'),grams('ａｂｃ　def'))
    def test_parameter_count(self):self.assertEqual(sum(p.numel() for p in r.core.Model().parameters()),141056)

if __name__=='__main__':unittest.main()
