import copy, math, random, os, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(int(os.environ.get('THREADS','1')))
torch.use_deterministic_algorithms(True)
OUT=Path(os.environ.get('OUTDIR','/mnt/data/language_learning_distance_continuation_2026-08-27/gru_aligned_transfer_confirmatory'))
OUT.mkdir(parents=True,exist_ok=True)
K=12; CTX=24; D=32; H=64; BS=24; LR=2e-3
BASE_STEPS=120; CURR_STEPS=100; CPS=[0,1,2,5,10,20,40]
SEEDS=list(range(900,908)); DISTANCES=[0.25,0.5,0.75,1.0]
CONDS=[('A_nat',0.0,'target'),('A_B25',0.25,'target'),('A_wrong25',0.25,'wrong')]

def perm_matrix(perm):
    M=np.zeros((K,K),dtype=np.float64); M[np.arange(K),np.asarray(perm)]=1.0; return M
def shift_perm(s): return [(i+s)%K for i in range(K)]
def make_struct(shifts,weights):
    M=sum(w*perm_matrix(shift_perm(s)) for s,w in zip(shifts,weights))
    return 0.92*M+0.08*np.ones((K,K))/K
PA=make_struct([1,3,7],[0.55,0.30,0.15])
PQ=make_struct([2,5,9],[0.50,0.35,0.15])
PC=make_struct([4,6,11],[0.45,0.35,0.20])
PU=np.ones((K,K),dtype=np.float64)/K
assert np.allclose(PA.sum(0),1) and np.allclose(PA.sum(1),1)
assert np.allclose(PQ.sum(0),1) and np.allclose(PQ.sum(1),1)
assert np.allclose(PC.sum(0),1) and np.allclose(PC.sum(1),1)
def PB_of(d): return (1-d)*PA+d*PQ
def structural_tv(P,Q): return 0.5*np.mean(np.abs(P-Q).sum(axis=1))
def sample_markov(P,lang,n,gen):
    starts=torch.arange(K).repeat(n//K); starts=starts[torch.randperm(n,generator=gen)]
    x=torch.empty((n,CTX+1),dtype=torch.long); cur=starts.clone(); off=0 if lang=='A' else K; x[:,0]=cur+off
    Pt=torch.tensor(P,dtype=torch.float32)
    for t in range(1,CTX+1):
        cur=torch.multinomial(Pt[cur],1,generator=gen).squeeze(1); x[:,t]=cur+off
    return x
class GRULM(nn.Module):
    def __init__(self):
        super().__init__(); self.state=nn.Embedding(K,D); self.lang=nn.Embedding(2,D)
        self.rnn=nn.GRU(D,H,num_layers=1,batch_first=True); self.ln=nn.LayerNorm(H)
        self.hA=nn.Linear(H,K,bias=False); self.hB=nn.Linear(H,K,bias=False)
    def forward(self,x,lang):
        li=0 if lang=='A' else 1; state=x if li==0 else x-K
        z=self.state(state)+self.lang(torch.full_like(state,li)); h,_=self.rnn(z); h=self.ln(h)
        return self.hA(h) if li==0 else self.hB(h)
def infer_lang(z): return 'A' if int(z.max())<K else 'B'
def nll(m,z):
    lang=infer_lang(z); y=z[:,1:] if lang=='A' else z[:,1:]-K
    return F.cross_entropy(m(z[:,:-1],lang).reshape(-1,K),y.reshape(-1))
def step(m,opt,z):
    v=nll(m,z); opt.zero_grad(set_to_none=True); v.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); return float(v.detach())
@torch.no_grad()
def eval_fixed(m,z): return float(nll(m,z))
def seedall(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)
def train_base(seed):
    seedall(1000+seed); m=GRULM(); opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=0); g=torch.Generator().manual_seed(11000+seed)
    for st in range(BASE_STEPS):
        lang='A' if st%2==0 else 'B'; step(m,opt,sample_markov(PU,lang,BS,g))
    return m
def run_condition(base,seed,d,cond,alpha,kind,evalB,evalA):
    m=copy.deepcopy(base); opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=0)
    g_choose=random.Random(21000+seed*1000+int(round(d*100))); gA=torch.Generator().manual_seed(22000+seed*1000+int(round(d*100)))
    PB=PB_of(d); bridge=PB if kind=='target' else PC; n_bridge=int(round(alpha*CURR_STEPS)); steps=set(g_choose.sample(range(CURR_STEPS),n_bridge)) if n_bridge else set()
    for st in range(CURR_STEPS): step(m,opt,sample_markov(bridge if st in steps else PA,'A',BS,gA))
    a_post=eval_fixed(m,evalA); opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=0); gB=torch.Generator().manual_seed(31000+seed*1000+int(round(d*100)))
    losses={0:eval_fixed(m,evalB)}; done=0
    for cp in CPS[1:]:
        for _ in range(cp-done): step(m,opt,sample_markov(PB,'B',BS,gB))
        done=cp; losses[cp]=eval_fixed(m,evalB)
    auc=sum(.5*(losses[a]+losses[b])*(b-a) for a,b in zip(CPS[:-1],CPS[1:]))/40
    r={'seed':seed,'distance_mix':d,'kernel_tv_A_B':structural_tv(PA,PB),'condition':cond,'A_post_nll':a_post,'B_auc_nll':auc}
    for cp in CPS:r[f'B_nll_{cp}']=losses[cp]
    return r
def main():
    seed=int(os.environ.get('SEED_ONLY','900')); assert seed in SEEDS
    base=train_base(seed); rows=[]
    for d in DISTANCES:
        PB=PB_of(d); ge=torch.Generator().manual_seed(41000+seed*1000+int(round(d*100))); ga=torch.Generator().manual_seed(42000+seed*1000+int(round(d*100)))
        evalB=sample_markov(PB,'B',BS*8,ge); evalA=sample_markov(PA,'A',BS*8,ga)
        for cond,alpha,kind in CONDS: rows.append(run_condition(base,seed,d,cond,alpha,kind,evalB,evalA))
    pd.DataFrame(rows).to_csv(OUT/f'gru_transfer_seed{seed}.csv',index=False); print('saved',seed,len(rows),flush=True)
if __name__=='__main__': main()
