#!/usr/bin/env python3
"""Recompute published R3 gates, not an end-to-end training replay.
Requires NumPy for the locked bootstrap generator; no PyTorch is needed.
"""
import csv,json,math,re
from pathlib import Path
import numpy as np
SEEDS=tuple(range(33000,33010))
ARMS={'REF_EN','JA_nat','JA_TJA','JA_TEN','JA_TRU'}
ROOT=Path(__file__).resolve().parents[1]

def require(ok,message):
    if not ok:raise ValueError(message)

def check(records,audits):
    cells={}
    for row in records:
        seed=int(row['seed']); arm=row['arm'];key=seed,arm
        require(seed in SEEDS and arm in ARMS and key not in cells,'invalid/duplicate identity')
        for field in ('english_step0_nll','japanese_phase1_nll','japanese_at_target_nll','english_end600_nll','japanese_end600_nll'):
            val=float(row[field]);require(math.isfinite(val) and val>0,'invalid loss')
        for field in ('target_update','high_update'):
            val=float(row[field]);require(math.isfinite(val) and val.is_integer() and 0<=val<=600,'invalid crossing')
        require(int(row['high_update'])<=int(row['target_update']),'reversed crossings')
        cells[key]=row
    require(set(cells)=={(s,a) for s in SEEDS for a in ARMS},'incomplete fixed cell grid')
    require(len(audits)==10 and {int(a['seed']) for a in audits}==set(SEEDS),'incomplete audit grid')
    for a in audits:
        require(a['optimizer_updates']==5400 and a['dense_curve_rows']==3005,'wrong budget or grid')
        require(a['all_execution_checks'] is True and a['checkpoint_record_max_error']==0.,'failed integrity')
        require(a['data_manifest_sha256']=='a70abd42d4ca57a9f27a6a784388559dc7ebb4366ced28d4aef7285a08e41236','wrong data manifest')
        for phase,arms in [('phase1',ARMS-{'REF_EN'}),('phase2',ARMS)]:
            d=a[phase+'_digests'];require(set(d)==arms,'digest identity mismatch')
            require(len(set(d.values()))==1,'unequal observed inputs')
            require(all(re.fullmatch('[0-9a-f]{64}',v) for v in d.values()),'malformed digest')
    ten=[cells[s,'JA_TEN'] for s in SEEDS];tja=[cells[s,'JA_TJA'] for s in SEEDS]
    require(all(int(r['target_update'])>0 for r in tja),'nonpositive control crossing')
    saving=np.array([1-int(x['target_update'])/int(y['target_update']) for x,y in zip(ten,tja)])
    k=int((saving>0).sum()); p=sum(math.comb(10,j) for j in range(k,11))/1024
    ix=np.random.default_rng(98309).integers(0,10,size=(10000,10))
    ci=np.quantile(saving[ix].mean(1),[.025,.975]).tolist()
    def m(rs,key):return sum(float(r[key]) for r in rs)/10
    pre=m(ten,'japanese_phase1_nll')/m(tja,'japanese_phase1_nll')-1
    post=m(ten,'japanese_at_target_nll')/m(tja,'japanese_at_target_nll')-1
    own=m(ten,'japanese_at_target_nll')/m(ten,'japanese_phase1_nll')-1
    efficacy=k>=9 and p<=.05 and float(saving.mean())>=.05 and ci[0]>=.05
    retention=pre<=.01 and post<=.01 and own<=.05
    return {'efficacy_verdict':'PASS' if efficacy else 'FAIL','retention_verdict':'PASS' if retention else 'FAIL',
            'joint_utility_verdict':'PASS' if efficacy and retention else 'FAIL','favorable_seeds':k,
            'exact_one_sided_sign_p':p,'mean_saving':float(saving.mean()),'seed_bootstrap_95CI':ci,
            'phase1_JA_penalty_ratio_of_means':pre,'target_JA_penalty_vs_TJA_ratio_of_means':post,
            'target_JA_increase_vs_own_phase1_ratio_of_means':own}

def main():
    with (ROOT/'results/natural_r3_pud_summary_2026-09-09.csv').open() as f:rows=list(csv.DictReader(f))
    audits=json.loads((ROOT/'results/natural_r3_pud_audit_2026-09-09.json').read_text())
    result=check(rows,audits)
    expected=json.loads((ROOT/'results/natural_r3_pud_verdict_2026-09-09.json').read_text())
    for k,v in result.items():
        if isinstance(v,str):require(v==expected[k],f'verdict mismatch: {k}')
        else:require(np.allclose(v,expected[k],rtol=0,atol=1e-12),f'numeric mismatch: {k}')
    print(json.dumps({'public_record_verification':'PASS','scope':'Published arithmetic/audit fields only, not fresh training or proof of provenance',**result},indent=2))
if __name__=='__main__':main()
