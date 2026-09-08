#!/usr/bin/env python3
"""Independent raw-record R3 analysis, with fixed censoring and bootstrap rules."""
import argparse, csv, hashlib, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
SEEDS=tuple(range(33000,33010)); ARMS=('REF_EN','JA_nat','JA_TJA','JA_TEN','JA_TRU')
TOL=1e-11

def require(ok,msg):
    if not ok:raise ValueError(msg)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def hit(values,target):
    return next((i for i,x in enumerate(values) if x<=target),None)
def mean(v):return sum(v)/len(v)
def sign_p(k,n):return sum(math.comb(n,j) for j in range(k,n+1))/(2**n)
def summarize(curves,summaries):
    records=[]
    for seed in SEEDS:
        rows={r['arm']:r for r in summaries[seed]}
        require(set(rows)==set(ARMS) and len(summaries[seed])==5,'arm grid')
        h={a:hit(curves[seed,a],3.) for a in ARMS}
        hi={a:hit(curves[seed,a],3.25) for a in ARMS}
        for a in ARMS:
            require(h[a]==rows[a]['target_update'],'target summary discrepancy')
            require(hi[a]==rows[a]['high_update'],'high summary discrepancy')
            require(abs(curves[seed,a][0]-rows[a]['english_step0_nll'])<TOL,'initial loss discrepancy')
            require(abs(curves[seed,a][-1]-rows[a]['english_end600_nll'])<TOL,'final loss discrepancy')
        valid=h['JA_TEN'] is not None and h['JA_TJA'] is not None and h['JA_TJA']>0
        saving=1-h['JA_TEN']/h['JA_TJA'] if valid else None
        iv=all(rows[a]['english_step0_nll']>3.25 and h[a] is not None and hi[a] is not None and h[a]>hi[a] for a in ('JA_TEN','JA_TJA'))
        ivsave=1-(h['JA_TEN']-hi['JA_TEN'])/(h['JA_TJA']-hi['JA_TJA']) if iv else None
        row={'seed':seed,'TEN_target_update':h['JA_TEN'],'TJA_target_update':h['JA_TJA'],
             'saving':saving,'interval_saving':ivsave,'interval_valid':iv,
             'TEN_JA_phase1':rows['JA_TEN']['japanese_phase1_nll'],
             'TJA_JA_phase1':rows['JA_TJA']['japanese_phase1_nll'],
             'TEN_JA_at_target':rows['JA_TEN']['japanese_at_target_nll'],
             'TJA_JA_at_target':rows['JA_TJA']['japanese_at_target_nll'],
             'TEN_JA_end600':rows['JA_TEN']['japanese_end600_nll'],
             'TJA_JA_end600':rows['JA_TJA']['japanese_end600_nll'],
             'TEN_minus_TJA_English_step0':rows['JA_TEN']['english_step0_nll']-rows['JA_TJA']['english_step0_nll']}
        for comp,label in [('JA_nat','nat'),('JA_TRU','RU')]:
            row['saving_vs_'+label]=1-h['JA_TEN']/h[comp] if h['JA_TEN'] is not None and h[comp] is not None and h[comp]>0 else None
        records.append(row)
    valid=all(r['saving'] is not None for r in records)
    rng=np.random.default_rng(98309)
    indices=rng.integers(0,10,size=(10000,10))
    gs=[r['saving'] for r in records]
    saving=mean(gs) if valid else None
    ci=np.quantile(np.asarray(gs)[indices].mean(1),[.025,.975]).tolist() if valid else None
    favorable=sum(g is not None and g>0 for g in gs)
    pre=mean([r['TEN_JA_phase1'] for r in records])/mean([r['TJA_JA_phase1'] for r in records])-1
    post_valid=all(r['TEN_JA_at_target'] is not None and r['TJA_JA_at_target'] is not None for r in records)
    post=mean([r['TEN_JA_at_target'] for r in records])/mean([r['TJA_JA_at_target'] for r in records])-1 if post_valid else None
    own=mean([r['TEN_JA_at_target'] for r in records])/mean([r['TEN_JA_phase1'] for r in records])-1 if post_valid else None
    gates={'primary_measurable':valid,'at_least_9_favorable':favorable>=9,'sign_p_at_most_05':sign_p(favorable,10)<=.05,
           'mean_saving_at_least_05':saving>=.05 if saving is not None else None,
           'bootstrap_lower_at_least_05':ci[0]>=.05 if ci else None,
           'phase1_JA_penalty_at_most_01':pre<=.01,
           'target_JA_penalty_vs_TJA_at_most_01':post<=.01 if post is not None else None,
           'target_JA_increase_vs_own_phase1_at_most_05':own<=.05 if own is not None else None}
    efficacy_keys=['at_least_9_favorable','sign_p_at_most_05','mean_saving_at_least_05','bootstrap_lower_at_least_05']
    efficacy='UNCERTAIN' if not valid else ('PASS' if all(gates[k] for k in efficacy_keys) else 'FAIL')
    # Missing measurement is never relabeled biological failure; known failed retention is separate.
    utility='UNCERTAIN' if not valid or not post_valid else ('PASS' if all(gates.values()) else 'FAIL')
    ret='UNCERTAIN' if not post_valid else ('PASS' if all(gates[k] for k in gates if 'JA_' in k) else 'FAIL')
    return {'experiment':'natural_r3_pud_fixed_target','completed_seeds':10,'efficacy_verdict':efficacy,
            'joint_utility_verdict':utility,'retention_verdict':ret,'gates':gates,
            'favorable_seeds':favorable,'exact_one_sided_sign_p':sign_p(favorable,10),
            'mean_saving':saving,'median_saving':float(np.median(gs)) if valid else None,
            'saving_range':[min(gs),max(gs)] if valid else None,'seed_bootstrap_95CI':ci,
            'phase1_JA_penalty_ratio_of_means':pre,'target_JA_penalty_vs_TJA_ratio_of_means':post,
            'target_JA_increase_vs_own_phase1_ratio_of_means':own,
            'TEN_mean_English_step0_difference':mean([r['TEN_minus_TJA_English_step0'] for r in records]),
            'TEN_JA_after600_mean':mean([r['TEN_JA_end600'] for r in records]),
            'TJA_JA_after600_mean':mean([r['TJA_JA_end600'] for r in records]),
            'secondary_interval_valid_seeds':sum(r['interval_valid'] for r in records),
            'secondary_interval_saving_available_cases':mean([r['interval_saving'] for r in records if r['interval_valid']]) if any(r['interval_valid'] for r in records) else None,
            'secondary_saving_vs_nat':mean([r['saving_vs_nat'] for r in records]) if all(r['saving_vs_nat'] is not None for r in records) else None,
            'secondary_saving_vs_RU':mean([r['saving_vs_RU'] for r in records]) if all(r['saving_vs_RU'] is not None for r in records) else None,
            'uncertainty_scope':'Optimization seeds conditional on one fixed 35-article holdout; not population-language uncertainty. Bootstrap interval is approximate.',
            'seed_records':records}

def analyze(root):
    curves={};summaries={};checkpoints=0;actualupdates=0;provenance={}
    missing=[s for s in SEEDS if not (root/f'seed{s}'/'audit.json').exists()]
    if missing:return {'joint_utility_verdict':'UNCERTAIN','reason':'incomplete fixed seed set','missing_seeds':missing}
    for s in SEEDS:
        p=root/f'seed{s}';audit=json.loads((p/'audit.json').read_text())
        require(audit['seed']==s and audit['status']=='COMPLETE','wrong or unfinished audit')
        require(all(audit['checks'].values()),'recorded audit failure')
        require(audit['optimizer_updates']==5400,'training budget mismatch')
        actualupdates+=audit['optimizer_updates']
        require(sha(p/'curves.csv')==audit['curve_sha256'],'raw curve hash mismatch')
        require(sha(p/'summary.json')==audit['summary_sha256'],'summary hash mismatch')
        require(sha(p/'endpoint_documents.csv')==audit['endpoint_document_sha256'],'endpoint hash mismatch')
        provenance[s]={k:audit[k] for k in ('protocol_commit','manifest_sha256','core_git_blob','runner_sha256')}
        if s!=SEEDS[0]:require(provenance[s]==provenance[SEEDS[0]],'implementation changed midcohort')
        for label,digest in audit['checkpoint_sha256'].items():
            require(sha(p/'checkpoints'/(label+'.pt'))==digest,'checkpoint hash mismatch');checkpoints+=1
        groups=defaultdict(dict)
        with (p/'curves.csv').open() as f:
            for r in csv.DictReader(f):
                require(int(r['seed'])==s and r['arm'] in ARMS,'raw identity mismatch')
                st=int(r['update']);val=float(r['english_nll'])
                require(math.isfinite(val) and val>=0 and st not in groups[r['arm']],'nonfinite, negative, or duplicate raw record')
                groups[r['arm']][st]=val
        require(set(groups)==set(ARMS),'missing arm')
        for a,g in groups.items():
            require(set(g)==set(range(601)),'incomplete dense curve')
            curves[s,a]=[g[i] for i in range(601)]
        summaries[s]=json.loads((p/'summary.json').read_text())
        doc=defaultdict(lambda:[0.,0])
        with (p/'endpoint_documents.csv').open() as f:
            for row in csv.DictReader(f):
                key=row['arm'],row['stage'],row['language']
                v=float(row['loss_sum']);ct=int(row['target_bytes'])
                require(int(row['seed'])==s and math.isfinite(v) and v>=0 and ct>0,'bad document loss')
                doc[key][0]+=v;doc[key][1]+=ct
        for row in summaries[s]:
            for field,stage,lang in [('japanese_phase1_nll','phase1','ja'),('japanese_at_target_nll','target','ja'),('japanese_end600_nll','end600','ja'),('english_end600_nll','end600','en')]:
                if row[field] is not None:
                    loss,n=doc[row['arm'],stage,lang];require(n==audit['eval_target_bytes'][lang],'document byte count')
                    require(abs(loss/n-row[field])<TOL,'document sum disagrees with mean')
    result=summarize(curves,summaries)
    result['integrity']={'passed':True,'optimizer_updates':actualupdates,'raw_curve_rows':len(curves)*601,'checkpoint_hashes_verified':checkpoints,'provenance':provenance[SEEDS[0]]}
    return result
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('input',type=Path);ap.add_argument('--out',required=True,type=Path);args=ap.parse_args()
    ans=analyze(args.input);text=json.dumps(ans,indent=2,allow_nan=False)+'\n';args.out.write_text(text);print(text)
