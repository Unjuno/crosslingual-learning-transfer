import copy, os, random, json
from pathlib import Path
import numpy as np, pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(int(os.environ.get('THREADS','1')))
torch.use_deterministic_algorithms(True)
OUT=Path(os.environ.get('OUTDIR','/mnt/data/language_learning_distance_continuation_2026-08-27/matched_wrong_map25_gru'))
OUT.mkdir(parents=True,exist_ok=True)
K=12; CTX=24; D=32; H=64; BS=24; LR=2e-3
BASE_STEPS=120; CURR_STEPS=100; CPS=[0,1,2,5,10,20,40]
DISTANCES=[0.125,0.25,0.50,0.75,1.00]
DOSES=[0.25]
SEEDS=list(range(1600,1608))

def perm_matrix(p):
    m=np.zeros((K,K),dtype=np.float64); m[np.arange(K),np.asarray(p)]=1.; return m
def shift_perm(s): return [(i+s)%K for i in range(K)]
def make_struct(shifts,weights):
    m=sum(w*perm_matrix(shift_perm(s)) for s,w in zip(shifts,weights))
    return .92*m+.08*np.ones((K,K))/K
PA=make_struct([1,3,7],[.55,.30,.15]); PQ=make_struct([2,5,9],[.50,.35,.15]); PU=np.ones((K,K))/K

def make_target(distance): return (1-distance)*PA+distance*PQ
def tv(P,Q): return .5*np.mean(np.abs(P-Q).sum(1))
def row_entropy(P): return -(P*np.log(P)).sum(1)
WRONG_SHIFT=7
PWRONG_ENDPOINT=np.roll(PQ,WRONG_SHIFT,axis=1)
def anti_for(PB):
    d=tv(PA,PB)/tv(PA,PQ) if tv(PA,PQ)>0 else 0.0
    P=(1-d)*PA+d*PWRONG_ENDPOINT
    return WRONG_SHIFT,P

def sample_markov(P,lang,n,gen):
    starts=torch.arange(K).repeat(n//K); starts=starts[torch.randperm(n,generator=gen)]
    x=torch.empty((n,CTX+1),dtype=torch.long); cur=starts.clone(); off=0 if lang=='A' else K
    x[:,0]=cur+off; Pt=torch.tensor(P,dtype=torch.float32)
    for t in range(1,CTX+1):
        cur=torch.multinomial(Pt[cur],1,generator=gen).squeeze(1); x[:,t]=cur+off
    return x

class GRULM(nn.Module):
    def __init__(self):
        super().__init__(); self.state=nn.Embedding(K,D); self.lang=nn.Embedding(2,D)
        self.rnn=nn.GRU(D,H,batch_first=True); self.ln=nn.LayerNorm(H)
        self.hA=nn.Linear(H,K,bias=False); self.hB=nn.Linear(H,K,bias=False)
    def forward(self,x,lang):
        li=0 if lang=='A' else 1; st=x if li==0 else x-K
        z=self.state(st)+self.lang(torch.full_like(st,li)); h,_=self.rnn(z); h=self.ln(h)
        return self.hA(h) if li==0 else self.hB(h)

def infer_lang(z): return 'A' if int(z.max())<K else 'B'
def nll(m,z):
    lang=infer_lang(z); y=z[:,1:] if lang=='A' else z[:,1:]-K
    return F.cross_entropy(m(z[:,:-1],lang).reshape(-1,K),y.reshape(-1))
def step(m,opt,z):
    v=nll(m,z); opt.zero_grad(set_to_none=True); v.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step(); return float(v.detach())
@torch.no_grad()
def ev(m,z): return float(nll(m,z))
def seedall(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)

def train_base(seed):
    seedall(800000+seed); m=GRULM(); opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=0)
    for st in range(BASE_STEPS):
        lang='A' if st%2==0 else 'B'; g=torch.Generator().manual_seed(810000+seed*1000+st)
        step(m,opt,sample_markov(PU,lang,BS,g))
    return m

def bridge_positions(seed,distance,dose):
    n=int(round(dose*CURR_STEPS)); code=int(round(distance*1000))*100+int(round(dose*100))
    r=random.Random(820000+seed*100000+code)
    return set(r.sample(range(CURR_STEPS),n))

def run_condition(base,seed,distance,PB,PANTI,name,kind=None,dose=0.0,evalA=None,evalB=None):
    m=copy.deepcopy(base); opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=0)
    pos=bridge_positions(seed,distance,dose) if dose else set()
    for st in range(CURR_STEPS):
        P=PA
        if st in pos: P=PB if kind=='target' else PANTI
        g=torch.Generator().manual_seed(830000+seed*100000+int(round(distance*1000))*100+st)
        step(m,opt,sample_markov(P,'A',BS,g))
    a_post=ev(m,evalA)
    opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=0)
    losses={0:ev(m,evalB)}; done=0
    for cp in CPS[1:]:
        for st in range(done,cp):
            g=torch.Generator().manual_seed(840000+seed*100000+int(round(distance*1000))*100+st)
            step(m,opt,sample_markov(PB,'B',BS,g))
        done=cp; losses[cp]=ev(m,evalB)
    auc=sum(.5*(losses[a]+losses[b])*(b-a) for a,b in zip(CPS[:-1],CPS[1:]))/40
    gain40=losses[0]-losses[40]
    r={'seed':seed,'distance':distance,'condition':name,'kind':kind or 'natural','dose':dose,
       'A_post_nll':a_post,'B_auc_nll':auc,'B_gain40':gain40}
    for cp in CPS: r[f'B_nll_{cp}']=losses[cp]
    return r

def main():
    seed=int(os.environ.get('SEED_ONLY','1600')); assert seed in SEEDS
    base=train_base(seed); rows=[]; inv=[]
    for di,distance in enumerate(DISTANCES):
        PB=make_target(distance); sh,PANTI=anti_for(PB)
        evalA=sample_markov(PA,'A',BS*8,torch.Generator().manual_seed(850000+seed*100+di))
        evalB=sample_markov(PB,'B',BS*8,torch.Generator().manual_seed(860000+seed*100+di))
        rows.append(run_condition(base,seed,distance,PB,PANTI,'A_nat',evalA=evalA,evalB=evalB))
        for dose in DOSES:
            tag=f'{int(round(dose*100)):02d}'
            rows.append(run_condition(base,seed,distance,PB,PANTI,'target'+tag,'target',dose,evalA,evalB))
            rows.append(run_condition(base,seed,distance,PB,PANTI,'anti'+tag,'anti',dose,evalA,evalB))
        inv.extend([
            {'distance':distance,'kernel':'PA','anti_shift':sh,'row_sum_err':abs(PA.sum(1)-1).max(),'col_sum_err':abs(PA.sum(0)-1).max(),'mean_row_entropy':row_entropy(PA).mean(),'tv_to_target':tv(PA,PB)},
            {'distance':distance,'kernel':'PB_target','anti_shift':sh,'row_sum_err':abs(PB.sum(1)-1).max(),'col_sum_err':abs(PB.sum(0)-1).max(),'mean_row_entropy':row_entropy(PB).mean(),'tv_to_target':0.0},
            {'distance':distance,'kernel':'PANTI','anti_shift':sh,'row_sum_err':abs(PANTI.sum(1)-1).max(),'col_sum_err':abs(PANTI.sum(0)-1).max(),'mean_row_entropy':row_entropy(PANTI).mean(),'tv_to_target':tv(PANTI,PB)}])
    pd.DataFrame(rows).to_csv(OUT/f'map_seed{seed}.csv',index=False)
    if seed==SEEDS[0]: pd.DataFrame(inv).to_csv(OUT/'kernel_invariants.csv',index=False)
    print('saved seed',seed,'rows',len(rows),flush=True)
if __name__=='__main__': main()
