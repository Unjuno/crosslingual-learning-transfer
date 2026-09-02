import argparse
from pathlib import Path
import pandas as pd


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('family_slopes',type=Path)
    ap.add_argument('feature_dir',type=Path)
    ap.add_argument('--family-out',type=Path,required=True)
    ap.add_argument('--summary-out',type=Path,required=True)
    args=ap.parse_args()

    slopes=pd.read_csv(args.family_slopes)
    features=pd.concat([pd.read_csv(p) for p in sorted(args.feature_dir.glob('feature_seed*_H*_f*.csv'))],ignore_index=True)
    d=slopes.merge(features[['seed','hidden_width','family','state_effective_rank']],on=['seed','hidden_width','family'],validate='one_to_one')
    d['state_rank_centered']=d['state_effective_rank']-d.groupby(['seed','hidden_width'])['state_effective_rank'].transform('mean')
    d['rank_slope_centered']=d['rank_slope']-d.groupby(['seed','hidden_width'])['rank_slope'].transform('mean')

    rows=[]
    for seed,g in d.groupby('seed'):
        if len(g)!=10:
            raise ValueError(f'seed {seed} has {len(g)} cells, expected 10')
        rho=float(g['state_rank_centered'].corr(g['rank_slope_centered'],method='spearman'))
        rows.append({'seed':int(seed),'rho_seed_centered':rho,'rho_negative':rho<0,'n_negative_rank_slopes':int((g['rank_slope']<0).sum())})
    summary=pd.DataFrame(rows).sort_values('seed')
    pooled=float(d['state_rank_centered'].corr(d['rank_slope_centered'],method='spearman'))
    summary['pooled_rho']=pooled

    args.family_out.parent.mkdir(parents=True,exist_ok=True)
    args.summary_out.parent.mkdir(parents=True,exist_ok=True)
    d.sort_values(['seed','hidden_width','family']).to_csv(args.family_out,index=False)
    summary.to_csv(args.summary_out,index=False)


if __name__=='__main__':
    main()
