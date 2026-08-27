import os, sys, copy, json, random
from pathlib import Path
import numpy as np, pandas as pd, torch
os.environ['SURR_MODE']='orig';os.environ['SURR_ID']='0';os.environ['THREADS']='1'
ROOT=Path('/mnt/data/ll_handoff_recovered/language_learning_distance_emergency_handoff_2026-08-26')
sys.path.insert(0,str(ROOT/'02_CODE'))
import study2_direction_fixed6_bigram_surrogate as s

torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
OUT=Path(os.environ.get('OUTDIR','/mnt/data/language_learning_distance_continuation_2026-08-27/natural_ja_en_blended5'))
OUT.mkdir(parents=True,exist_ok=True)
SEED=int(os.environ.get('SEED_ONLY','700'))
NBIN=8; CAND_N=1024; DOC_K=256; CURR_STEPS=100; BRIDGE_STEPS_N=5; CPS=[0,1,2,5,10,20,40]; BS=s.BS
A='ja';B='en';WRONG='ru'

def rank_map(lang):
    x=s.TR[lang].numpy();ids,cnt=np.unique(x,return_counts=True);order=ids[np.argsort(-cnt)]
    mp=np.full(s.VOC,NBIN-1,dtype=np.int64);n=len(order)
    for r,tok in enumerate(order):mp[tok]=min(NBIN-1,(r*NBIN)//max(1,n))
    return torch.tensor(mp,dtype=torch.long)
RM={l:rank_map(l) for l in s.LANGS}
def bin_kernel(lang,eps=.5):
    b=RM[lang][s.TR[lang]]; M=torch.full((NBIN,NBIN),eps,dtype=torch.float64)
    cnt=torch.bincount(b[:-1]*NBIN+b[1:],minlength=NBIN*NBIN).reshape(NBIN,NBIN).double();M+=cnt;M/=M.sum(1,keepdim=True);return torch.log(M)
LOGP={l:bin_kernel(l) for l in s.LANGS}
def score(cand,target):
    bins=RM[A][cand];x=bins[:,:-1];y=bins[:,1:]
    lt=LOGP[target][x,y].mean(1);others=[l for l in s.LANGS if l not in (A,target)]
    lo=torch.stack([LOGP[l][x,y].mean(1) for l in others]).mean(0);return lt-lo

def docs(seed):
    g=torch.Generator().manual_seed(71000+seed);cand=s.sample(s.TR[A],CAND_N,g);se=score(cand,B);sr=score(cand,WRONG)
    ie=torch.argsort(se,descending=True)[:DOC_K];ir=torch.argsort(sr,descending=True)[:DOC_K]
    gr=torch.Generator().manual_seed(71100+seed);inn=torch.randperm(CAND_N,generator=gr)[:DOC_K]
    D={'nat':cand[inn].clone(),'en':cand[ie].clone(),'ru':cand[ir].clone()}
    meta=pd.DataFrame([
      {'seed':seed,'doc':'nat','EN_score':float(se[inn].mean()),'RU_score':float(sr[inn].mean())},
      {'seed':seed,'doc':'en','EN_score':float(se[ie].mean()),'RU_score':float(sr[ie].mean())},
      {'seed':seed,'doc':'ru','EN_score':float(se[ir].mean()),'RU_score':float(sr[ir].mean())}])
    return D,meta

def curriculum(ref,D,kind,seed):
    m=copy.deepcopy(ref);op=torch.optim.AdamW(m.parameters(),lr=s.LR,weight_decay=0)
    r=random.Random(71200+seed); bridge=set(r.sample(range(CURR_STEPS),BRIDGE_STEPS_N))
    gidx=torch.Generator().manual_seed(71300+seed)
    idxs=[torch.randint(0,DOC_K,(BS,),generator=gidx) for _ in range(CURR_STEPS)]
    for st,ix in enumerate(idxs):
        key='nat' if kind=='nat' or st not in bridge else kind
        s.step(m,op,D[key][ix])
    return m,sorted(bridge)

@torch.no_grad()
def ev(m,lang,seed):
    g=torch.Generator().manual_seed(seed);z=s.sample(s.TE[lang],32*6,g);return float(s.loss(m,z))
def adapt(m0,seed):
    m=copy.deepcopy(m0);op=torch.optim.AdamW(m.parameters(),lr=s.LR,weight_decay=0);g=torch.Generator().manual_seed(72000+seed)
    L={};done=0
    for cp in CPS:
      for _ in range(cp-done):s.step(m,op,s.sample(s.TR[B],BS,g))
      done=cp;L[cp]=ev(m,B,73000+seed*100+cp)
    auc=sum(.5*(L[a]+L[b])*(b-a) for a,b in zip(CPS[:-1],CPS[1:]))/40;gain=L[0]-L[40]
    return L,auc,gain

def main():
    ref=s.reference(SEED);D,meta=docs(SEED);rows=[];positions=None
    for cond,kind in [('JA_nat','nat'),('JA_EN05','en'),('JA_RU05','ru')]:
      m,pos=curriculum(ref,D,kind,SEED); positions=pos
      ja=ev(m,A,74000+SEED);L,auc,gain=adapt(m,SEED)
      r={'seed':SEED,'condition':cond,'JA_post_nll':ja,'EN_auc_nll':auc,'EN_gain40':gain,'bridge_steps':BRIDGE_STEPS_N}
      for cp in CPS:r[f'EN_nll_{cp}']=L[cp]
      rows.append(r)
    pd.DataFrame(rows).to_csv(OUT/f'blended5_seed{SEED}.csv',index=False);meta.to_csv(OUT/f'blended5_selection_seed{SEED}.csv',index=False)
    (OUT/f'bridge_positions_seed{SEED}.json').write_text(json.dumps(positions),encoding='utf8')
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
if __name__=='__main__':main()
