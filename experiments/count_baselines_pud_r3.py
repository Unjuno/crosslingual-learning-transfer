#!/usr/bin/env python3
"""Post-hoc fixed Laplace byte-count baselines; zero neural optimizer updates."""
import csv,json
from pathlib import Path
import numpy as np
import run_pud_r3 as r
ROOT=Path(__file__).resolve().parents[1]

def metrics(u,b,x,y):
    up=(u[y]+1.)/(u.sum()+256.)
    bp=(b[x,y]+1.)/(b.sum(1)[x]+256.)
    return float(-np.log(up).mean()),float(-np.log(bp).mean())

def main():
    streams,ev,_=r.read_data(ROOT/'data')
    parts=[np.frombuffer(row['text'].encode(),dtype=np.uint8).astype(np.int64) for row in ev['en']]
    x=np.concatenate([b[:-1] for b in parts]);y=np.concatenate([b[1:] for b in parts])
    train=streams['en'].numpy()
    u=np.bincount(train,minlength=256)
    b=np.bincount(train[:-1]*256+train[1:],minlength=65536).reshape(256,256)
    full=metrics(u,b,x,y)
    out=ROOT/'results/count_baselines';out.mkdir(exist_ok=True)
    records=[];summ=[]
    for seed in range(33000,33010):
        u=np.zeros(256,dtype=np.int64);b=np.zeros((256,256),dtype=np.int64);first=[None,None]
        for st in range(601):
            if st:
                z=r.core.batch_from(streams['en'],16,5140000+seed*1000+st-1).numpy()
                xx=z[:,:-1].reshape(-1);yy=z[:,1:].reshape(-1)
                u+=np.bincount(yy,minlength=256)
                b+=np.bincount(xx*256+yy,minlength=65536).reshape(256,256)
            values=metrics(u,b,x,y)
            for j,name in enumerate(('unigram','bigram')):
                if first[j] is None and values[j]<=3.:first[j]=st
                records.append({'seed':seed,'model':name,'update':st,'english_nll':values[j]})
        summ.append({'seed':seed,'unigram_target_update':first[0],'bigram_target_update':first[1],'unigram_nll600':values[0],'bigram_nll600':values[1]})
    with (out/'curves.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    ans={'plan_commit':'5c8a435aac3c75c6e0a158c050b4d0e37e314b1d','alpha':1.,'full_stream_unigram_nll':full[0],
         'full_stream_bigram_nll':full[1],'evaluation_target_bytes':len(y),'seeds':summ,
         'neural_optimizer_updates':0,'new_independent_seeds':0,'estimator_updates':12000,'dense_evaluation_records':len(records)}
    r.write_json(out/'summary.json',ans);print(json.dumps(ans,indent=2))
if __name__=='__main__':main()
