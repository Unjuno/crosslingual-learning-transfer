import copy, os, random, json
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(int(os.environ.get('THREADS','1')))
torch.use_deterministic_algorithms(True)
ROOT=Path('/mnt/data/language_learning_distance_continuation_2026-08-27/performance_aligned_learning_minimal_gru')
ROOT.mkdir(parents=True,exist_ok=True)
K=12; CTX=24; D=32; H=64; BS=24; LR=2e-3
BASE=120; CURR=100; BMAX=60; EVAL_EVERY=2; DOSE=.25
SEEDS=[3800,3801]; FAMS=list(range(5))

def pm(s):
 m=np.zeros((K,K)); m[np.arange(K),(np.arange(K)+s)%K]=1; return m
U=np.ones((K,K))/K
PA=.92*(.35*pm(1)+.35*pm(-1)+.15*pm(3)+.15*pm(-3))+.08*U
ORIENTS=[[2,4,5],[2,8,5],[10,4,5],[2,4,7],[10,8,5]]
def wts(fid):
 rng=np.random.default_rng(271828+fid*997)
 for _ in range(1000):
  w=rng.dirichlet([3,3,3])
  if w.min()>.12:return w
 raise RuntimeError
def endpoint(fid):
 w=wts(fid); sh=ORIENTS[fid]
 q=.92*sum(float(a)*pm(int(s)) for a,s in zip(w,sh))+.08*U
 qw=.92*sum(float(a)*pm(int(-s)) for a,s in zip(w,sh))+.08*U
 return q,qw
def blend(q):return .5*PA+.5*q

def sample(P,l,n,g):
 starts=torch.arange(K).repeat(n//K); starts=starts[torch.randperm(n,generator=g)]
 x=torch.empty((n,CTX+1),dtype=torch.long); cur=starts.clone(); off=0 if l=='A' else K; x[:,0]=cur+off
 Pt=torch.tensor(P,dtype=torch.float32)
 for t in range(1,CTX+1):
  cur=torch.multinomial(Pt[cur],1,generator=g).squeeze(1); x[:,t]=cur+off
 return x
class M(nn.Module):
 def __init__(self):
  super().__init__(); self.state=nn.Embedding(K,D); self.lang=nn.Embedding(2,D); self.rnn=nn.GRU(D,H,batch_first=True); self.ln=nn.LayerNorm(H); self.head=nn.Linear(H,K,bias=False)
 def forward(self,x,l):
  li=0 if l=='A' else 1; st=x if li==0 else x-K
  z=self.state(st)+self.lang(torch.full_like(st,li)); h,_=self.rnn(z); return self.head(self.ln(h))
def nll(m,z,l):
 y=z[:,1:] if l=='A' else z[:,1:]-K
 return F.cross_entropy(m(z[:,:-1],l).reshape(-1,K),y.reshape(-1))
def step(m,o,z,l):
 v=nll(m,z,l); o.zero_grad(set_to_none=True); v.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.); o.step()
@torch.no_grad()
def ev(m,z,l):return float(nll(m,z,l))
def seedall(s):random.seed(s); np.random.seed(s); torch.manual_seed(s)
def train_base(seed,fid):
 seedall(1810000+fid*10000+seed); m=M(); o=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=0)
 for st in range(BASE):
  l='A' if st%2==0 else 'B'; g=torch.Generator().manual_seed(1811000+fid*1000000+seed*1000+st); step(m,o,sample(U,l,BS,g),l)
 return m
def positions(seed,fid):return set(random.Random(1812000+fid*100000+seed).sample(range(CURR),int(DOSE*CURR)))
def curriculum(base,seed,fid,kind,pbt,pwt):
 m=copy.deepcopy(base); o=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=0); ps=positions(seed,fid) if kind else set()
 for st in range(CURR):
  P=PA if st not in ps else (pbt if kind=='target' else pwt)
  g=torch.Generator().manual_seed(1813000+fid*1000000+seed*1000+st); step(m,o,sample(P,'A',BS,g),'A')
 m.head.load_state_dict(copy.deepcopy(base.head.state_dict()))
 return m
def curve(m,seed,fid,pbt,eb):
 o=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=0); rows=[(0,ev(m,eb,'B'))]
 for st in range(1,BMAX+1):
  g=torch.Generator().manual_seed(1814000+fid*1000000+seed*1000+(st-1)); step(m,o,sample(pbt,'B',BS,g),'B')
  if st%EVAL_EVERY==0: rows.append((st,ev(m,eb,'B')))
 return rows
def main():
 fid=int(os.environ.get('FAMILY_ONLY','0')); seed=int(os.environ.get('SEED_ONLY','3800')); assert fid in FAMS and seed in SEEDS
 q,qw=endpoint(fid); pbt=blend(q); pwt=blend(qw); base=train_base(seed,fid)
 eb=sample(pbt,'B',BS*6,torch.Generator().manual_seed(1815000+fid*10000+seed)); out=[]
 for name,kind in [('A_nat',None),('target25','target'),('wrong25','wrong')]:
  m=curriculum(base,seed,fid,kind,pbt,pwt)
  for st,v in curve(m,seed,fid,pbt,eb): out.append({'family':fid,'seed':seed,'condition':name,'B_step':st,'B_nll':v})
 pd.DataFrame(out).to_csv(ROOT/f'family{fid}_seed{seed}.csv',index=False); print('saved',fid,seed,flush=True)
if __name__=='__main__':main()
