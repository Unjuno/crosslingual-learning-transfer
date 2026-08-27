import copy, os, random, json
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(int(os.environ.get('THREADS','1')))
torch.use_deterministic_algorithms(True)
ROOT=Path('/mnt/data/language_learning_distance_continuation_2026-08-27/decoder_alignment_threshold_gru')
ROOT.mkdir(parents=True,exist_ok=True)
K=12; CTX=24; D=32; H=64; BS=24; LR=2e-3
BASE=120; CURR=100; BMAX=60; EV=2; DOSE=.25
SEED=int(os.environ.get('FIXED_SEED','4200')); FAMS=list(range(5))
STAGE=os.environ.get('STAGE','stage1')
RHOS=[1.0,0.5,0.0] if STAGE=='stage1' else ([float(x) for x in os.environ['RHO_LIST'].split(',')] if STAGE=='custom' else [float(os.environ['RHO_ONLY'])])

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
def blend(q): return .5*PA+.5*q

def sample(P,l,n,g):
    starts=torch.arange(K).repeat(n//K); starts=starts[torch.randperm(n,generator=g)]
    x=torch.empty((n,CTX+1),dtype=torch.long); cur=starts.clone(); off=0 if l=='A' else K; x[:,0]=cur+off
    Pt=torch.tensor(P,dtype=torch.float32)
    for t in range(1,CTX+1):
        cur=torch.multinomial(Pt[cur],1,generator=g).squeeze(1); x[:,t]=cur+off
    return x

class M(nn.Module):
    def __init__(self):
        super().__init__(); self.state=nn.Embedding(K,D); self.lang=nn.Embedding(2,D); self.rnn=nn.GRU(D,H,batch_first=True); self.ln=nn.LayerNorm(H); self.hA=nn.Linear(H,K,bias=False); self.hB=None
    def forward(self,x,l):
        li=0 if l=='A' else 1; st=x if li==0 else x-K
        z=self.state(st)+self.lang(torch.full_like(st,li)); h,_=self.rnn(z); h=self.ln(h)
        return (self.hA if li==0 or self.hB is None else self.hB)(h)
def nll(m,z,l):
    y=z[:,1:] if l=='A' else z[:,1:]-K
    return F.cross_entropy(m(z[:,:-1],l).reshape(-1,K),y.reshape(-1))
def step(m,o,z,l):
    v=nll(m,z,l); o.zero_grad(set_to_none=True); v.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.); o.step()
@torch.no_grad()
def ev(m,z,l): return float(nll(m,z,l))
def seedall(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)

def train_base(seed,fid):
    seedall(2110000+fid*10000+seed); m=M(); o=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=0)
    for st in range(BASE):
        l='A' if st%2==0 else 'B'; g=torch.Generator().manual_seed(2111000+fid*1000000+seed*1000+st); step(m,o,sample(U,l,BS,g),l)
    return m

def positions(seed,fid): return set(random.Random(2112000+fid*100000+seed).sample(range(CURR),int(DOSE*CURR)))
def curriculum(base,seed,fid,kind,pbt,pwt):
    m=copy.deepcopy(base); o=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=0); ps=positions(seed,fid) if kind else set()
    for st in range(CURR):
        P=PA if st not in ps else (pbt if kind=='target' else pwt)
        g=torch.Generator().manual_seed(2113000+fid*1000000+seed*1000+st); step(m,o,sample(P,'A',BS,g),'A')
    return m

def aligned_head(base_weight, rho, seed, fid):
    state=torch.random.get_rng_state(); torch.manual_seed(2113500+fid*10000+seed)
    r=torch.randn_like(base_weight); torch.random.set_rng_state(state)
    b=base_weight.detach().clone(); bf=b.reshape(-1); rf=r.reshape(-1)
    rf=rf - (torch.dot(rf,bf)/torch.dot(bf,bf))*bf
    rf=rf / torch.linalg.vector_norm(rf) * torch.linalg.vector_norm(bf)
    w=rho*bf + (max(0.0,1-rho*rho)**0.5)*rf
    w=w.reshape_as(b)
    cos=float(torch.dot(w.reshape(-1),bf)/(torch.linalg.vector_norm(w)*torch.linalg.vector_norm(bf)))
    nr=float(torch.linalg.vector_norm(w)/torch.linalg.vector_norm(bf))
    return w,cos,nr

def install_head(m,w):
    m.hB=nn.Linear(H,K,bias=False)
    with torch.no_grad(): m.hB.weight.copy_(w)

def curve(m,seed,fid,pbt,eb,rho):
    o=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=0); rows=[(0,ev(m,eb,'B'))]
    for st in range(1,BMAX+1):
        g=torch.Generator().manual_seed(2114000+fid*1000000+seed*1000+(st-1)); step(m,o,sample(pbt,'B',BS,g),'B')
        if st%EV==0: rows.append((st,ev(m,eb,'B')))
    return rows

def main():
    fid=int(os.environ.get('FAMILY_ONLY','0')); seed=int(os.environ.get('SEED_ONLY',str(SEED))); assert fid in FAMS and seed==SEED
    q,qw=endpoint(fid); pbt=blend(q); pwt=blend(qw); base=train_base(seed,fid)
    bw=base.hA.weight.detach().clone()
    eb=sample(pbt,'B',BS*6,torch.Generator().manual_seed(2115000+fid*10000+seed)); out=[]; audits=[]
    cms={}
    for name,kind in [('A_nat',None),('target25','target'),('wrong25','wrong')]: cms[name]=curriculum(base,seed,fid,kind,pbt,pwt)
    for rho in RHOS:
        w,cos,nr=aligned_head(bw,float(rho),seed,fid); audits.append({'family':fid,'seed':seed,'rho':rho,'cos_to_base':cos,'norm_ratio':nr})
        for name in ['A_nat','target25','wrong25']:
            m=copy.deepcopy(cms[name]); install_head(m,w)
            for st,v in curve(m,seed,fid,pbt,eb,float(rho)):
                out.append({'family':fid,'seed':seed,'rho':rho,'condition':name,'B_step':st,'B_nll':v})
    tag='stage1' if STAGE=='stage1' else ('custom_'+'_'.join(f'{r:.3f}' for r in RHOS) if STAGE=='custom' else f'rho{RHOS[0]:.2f}')
    pd.DataFrame(out).to_csv(ROOT/f'{tag}_family{fid}_seed{seed}.csv',index=False)
    pd.DataFrame(audits).to_csv(ROOT/f'{tag}_audit_family{fid}_seed{seed}.csv',index=False)
    print('saved',tag,fid,seed,flush=True)
if __name__=='__main__': main()
