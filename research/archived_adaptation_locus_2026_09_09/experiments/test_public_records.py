import csv,json,shutil,tempfile,unittest
from pathlib import Path
from verify_records import verify,ROOT
class PublicTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);(self.root/'results').mkdir()
  for n in ['all_summaries.csv','analysis.json','factorial_cells.csv','head_swap_analysis.json']:shutil.copy2(ROOT/'results'/n,self.root/'results'/n)
 def tearDown(self):self.tmp.cleanup()
 def csv_edit(self,name,func):
  p=self.root/'results'/name
  with p.open(newline='') as f:r=list(csv.DictReader(f));fields=list(r[0])
  func(r)
  with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(r)
 def test_valid(self):self.assertEqual(verify(self.root)['record_verification'],'PASS')
 def test_missing(self):
  self.csv_edit('all_summaries.csv',lambda r:r.pop())
  with self.assertRaises(ValueError):verify(self.root)
 def test_duplicate(self):
  self.csv_edit('all_summaries.csv',lambda r:r.append(r[0]))
  with self.assertRaises(ValueError):verify(self.root)
 def test_nonfinite(self):
  self.csv_edit('all_summaries.csv',lambda r:r[0].update(EN_final250='nan'))
  with self.assertRaises(ValueError):verify(self.root)
 def test_factorial_missing(self):
  self.csv_edit('factorial_cells.csv',lambda r:r.pop())
  with self.assertRaises(ValueError):verify(self.root)
 def test_factorial_tampered(self):
  self.csv_edit('factorial_cells.csv',lambda r:r[0].update(EN_final250='7.'))
  with self.assertRaises(ValueError):verify(self.root)
 def test_report_tampered(self):
  p=self.root/'results/head_swap_analysis.json';r=json.loads(p.read_text());r['verdict']='FAIL_EXPLORATORY';p.write_text(json.dumps(r))
  with self.assertRaises(ValueError):verify(self.root)
 def test_seed_tampered(self):
  self.csv_edit('factorial_cells.csv',lambda r:r[0].update(seed='9999'))
  with self.assertRaises(ValueError):verify(self.root)
if __name__=='__main__':unittest.main()
