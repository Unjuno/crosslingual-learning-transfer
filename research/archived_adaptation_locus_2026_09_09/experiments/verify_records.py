#!/usr/bin/env python3
"""Standard-library arithmetic/coverage audit, NOT an independent training run."""
import csv,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SEEDS=set(range(32000,32010));CONDS={'JA_TJA','JA_TEN'};MODES={'FULL','HEAD_ONLY','BODY_ONLY'}

def read(path):
 with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def finite(x):
 v=float(x)
 if not math.isfinite(v):raise ValueError('nonfinite record')
 return v

def avg(x):return sum(x)/len(x)

def near(a,b):
 if not math.isclose(finite(a),finite(b),rel_tol=1e-10,abs_tol=1e-10):raise ValueError(f'published arithmetic mismatch: {a} != {b}')

def verify(root=ROOT):
 rows=read(root/'results/all_summaries.csv');by={}
 expected={(s,c,m) for s in SEEDS for c in CONDS for m in MODES}
 for r in rows:
  k=(int(r['seed']),r['condition'],r['mode'])
  if k in by or k not in expected:raise ValueError('duplicate or unexpected cell')
  for n in ['EN_initial','EN_final250','EN_gain','JA_initial','JA_final_shared','JA_final_routed','JA_shared_increase','JA_routed_increase']:r[n]=finite(r[n])
  near(r['EN_gain'],r['EN_initial']-r['EN_final250']);near(r['JA_shared_increase'],r['JA_final_shared']-r['JA_initial']);near(r['JA_routed_increase'],r['JA_final_routed']-r['JA_initial'])
  if k[2]=='HEAD_ONLY':near(r['JA_routed_increase'],0)
  if k[2]=='BODY_ONLY':near(r['JA_final_shared'],r['JA_final_routed'])
  by[k]=r
 if set(by)!=expected:raise ValueError('incomplete locus matrix')
 published=json.loads((root/'results/analysis.json').read_text())
 for m in MODES:
  values=[by[s,'JA_TEN',m]['EN_final250']-by[s,'JA_TJA',m]['EN_final250'] for s in sorted(SEEDS)]
  st=published['modes'][m]
  near(avg(values),st['mean_TEN_minus_TJA_EN_final250'])
  if sum(v<0 for v in values)!=st['n_favorable_final250']:raise ValueError('direction count')
 h=published['modes']['HEAD_ONLY'];gate=h['n_favorable_final250']>=9 and h['mean_TEN_minus_TJA_EN_final250']<=-.01
 if published['verdict']!=('PASS_EXPLORATORY' if gate else 'FAIL_EXPLORATORY'):raise ValueError('locus verdict')
 cells=read(root/'results/factorial_cells.csv');f={}
 for r in cells:
  key=(int(r['seed']),r['body'],r['initial_head'])
  if key in f or key[0] not in SEEDS or key[1] not in CONDS or key[2] not in CONDS:raise ValueError('factorial key')
  r['EN_initial']=finite(r['EN_initial']);r['EN_final250']=finite(r['EN_final250']);f[key]=r
 if len(f)!=40:raise ValueError('factorial matrix incomplete')
 sp=json.loads((root/'results/head_swap_analysis.json').read_text());body=[]
 for s in sorted(SEEDS):
  jj=f[s,'JA_TJA','JA_TJA'];ej=f[s,'JA_TEN','JA_TJA'];je=f[s,'JA_TJA','JA_TEN'];ee=f[s,'JA_TEN','JA_TEN']
  near(jj['EN_final250'],by[s,'JA_TJA','HEAD_ONLY']['EN_final250']);near(ee['EN_final250'],by[s,'JA_TEN','HEAD_ONLY']['EN_final250'])
  body.append(((ej['EN_final250']-jj['EN_final250'])+(ee['EN_final250']-je['EN_final250']))/2)
 main=sp['statistics']['EN_final250_body_main'];near(avg(body),main['mean'])
 if sum(v<0 for v in body)!=main['n_negative']:raise ValueError('factorial sign count')
 sg=main['n_negative']>=9 and main['mean']<=-.01
 if sp['verdict']!=('PASS_EXPLORATORY' if sg else 'FAIL_EXPLORATORY'):raise ValueError('swap verdict')
 return {'record_verification':'PASS','locus_verdict':published['verdict'],'factorial_verdict':sp['verdict'],'locus_cells':60,'factorial_cells':40,'optimizer_updates_new_this_turn':20000,'independent_new_seeds':0,'scope':'Coverage and means/decisions only; raw/checkpoint validation is a separate script.'}

if __name__=='__main__':print(json.dumps(verify(),indent=2))
