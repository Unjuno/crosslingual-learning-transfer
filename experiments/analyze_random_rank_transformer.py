import argparse, math
from pathlib import Path
import numpy as np, pandas as pd


def cumulative_best(frame):
    out=frame.sort_values('B_step').copy()
    out['best_nll']=np.minimum.accumulate(out['B_nll'].to_numpy(dtype=float))
    return out


def first_crossing(frame,threshold):
    g=cumulative_best(frame); x=g['B_step'].to_numpy(dtype=float); y=g['best_nll'].to_numpy(dtype=float)
    if y[0]<=threshold: return float(x[0])
    for i in range(1,len(x)):
        if y[i]<=threshold:
            x0,x1=x[i-1],x[i]; y0,y1=y[i-1],y[i]
            if y1==y0: return float(x1)
            return float(x0+(x1-x0)*(y0-threshold)/(y0-y1))
    return float('nan')


def effect(a_nat,target):
    a=cumulative_best(a_nat); t=cumulative_best(target)
    top=min(float(a['best_nll'].iloc[0]),float(t['best_nll'].iloc[0]))
    bottom=max(float(a['best_nll'].iloc[-1]),float(t['best_nll'].iloc[-1]))
    high=bottom+0.8*(top-bottom); low=bottom+0.2*(top-bottom)
    span_a=first_crossing(a,low)-first_crossing(a,high)
    span_t=first_crossing(t,low)-first_crossing(t,high)
    return span_t-span_a


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('input_dir',type=Path); ap.add_argument('--family-out',type=Path,required=True); ap.add_argument('--summary-out',type=Path,required=True); args=ap.parse_args()
    rows=[]
    for p in sorted(args.input_dir.glob('family*_seed*.csv')):
        if p.name.startswith('audit_'): continue
        d=pd.read_csv(p); effects=[]
        for keep in [0,3,6,12]:
            effects.append(effect(d[(d.keep_dim==keep)&(d.condition=='A_nat')],d[(d.keep_dim==keep)&(d.condition=='target25')]))
        slope=float(np.polyfit([0,3,6,12],effects,1)[0])
        rows.append({'seed':int(d.seed.iloc[0]),'family':int(d.family.iloc[0]),'rank_slope':slope,'effect_keep0':effects[0],'effect_keep3':effects[1],'effect_keep6':effects[2],'effect_keep12':effects[3]})
    family=pd.DataFrame(rows).sort_values(['seed','family'])
    summaries=[]
    for seed,g in family.groupby('seed'):
        n=int(len(g)); k=int((g.rank_slope<0).sum())
        p=sum(math.comb(n,j) for j in range(k,n+1))/(2**n)
        summaries.append({'seed':int(seed),'mean_rank_slope':float(g.rank_slope.mean()),'n_negative':k,'n_families':n,'one_sided_sign_p':p,'pass_all5':bool(n==5 and k==5)})
    summary=pd.DataFrame(summaries).sort_values('seed')
    args.family_out.parent.mkdir(parents=True,exist_ok=True);args.summary_out.parent.mkdir(parents=True,exist_ok=True)
    family.to_csv(args.family_out,index=False);summary.to_csv(args.summary_out,index=False)


if __name__=='__main__': main()
