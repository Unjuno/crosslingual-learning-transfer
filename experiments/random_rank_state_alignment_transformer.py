import copy, os, random
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(int(os.environ.get('THREADS', '1')))
torch.use_deterministic_algorithms(True)
ROOT = Path(os.environ.get('OUTDIR', 'results/raw_transformer_random_rank'))
ROOT.mkdir(parents=True, exist_ok=True)
K=12; CTX=24; D=32; BS=24; LR=2e-3; BASE=120; CURR=100; BMAX=60; EV=2; DOSE=.25; RHO=.25
SEED=int(os.environ.get('SEED','5400')); FAMS=list(range(5)); KEEP=[0,3,6,12]

def pm(s): m=np.eye(K); return np.roll(m,s,axis=1)
U=np.ones((K,K))/K
PA=.92*(.35*pm(1)+.35*pm(-1)+.15*pm(3)+.15*pm(-3))+.08*U
ORIENTS=[[2,4,5],[2,8,5],[10,4,5],[2,4,7],[10,8,5]]
def wts(fid):
    rng=np.random.default_rng(271828+fid*997)
    for _ in range(1000):
        w=rng.dirichlet([3,3,3])
        if w.min()>.12: return w
    raise RuntimeError
def endpoint(fid):
    w=wts(fid); sh=ORIENTS[fid]
    return .92*sum(float(a)*pm(int(s)) for a,s in zip(w,sh))+.08*U
def blend(q): return .5*PA+.5*q

def sample(P,l,n,g):
    starts=torch.arange(K).repeat(n//K); starts=starts[torch.randperm(n,generator=g)]
    x=torch.empty((n,CTX+1),dtype=torch.long); cur=starts.clone(); off=0 if l=='A' else K
    x[:,0]=cur+off; Pt=torch.tensor(P,dtype=torch.float32)
    for t in range(1,CTX+1):
        cur=torch.multinomial(Pt[cur],1,generator=g).squeeze(1); x[:,t]=cur+off
    return x

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.state=nn.Embedding(K,D); self.lang=nn.Embedding(2,D); self.pos=nn.Embedding(CTX+1,D)
        layer=nn.TransformerEncoderLayer(d_model=D,nhead=4,dim_feedforward=64,dropout=0.0,batch_first=True,norm_first=True,activation='gelu')
        self.enc=nn.TransformerEncoder(layer,num_layers=1); self.ln=nn.LayerNorm(D)
        self.hA=nn.Linear(D,K,bias=False); self.hB=None; self.register_buffer('Brot',torch.eye(D))
    def forward(self,x,l):
        li=0 if l=='A' else 1; st=x if li==0 else x-K
        e=self.state(st); e=e if li==0 else e@self.Brot
        pos=torch.arange(x.shape[1],device=x.device).unsqueeze(0)
        z=e+self.lang(torch.full_like(st,li))+self.pos(pos)
        mask=torch.triu(torch.ones(x.shape[1],x.shape[1],dtype=torch.bool,device=x.device),diagonal=1)
        h=self.ln(self.enc(z,mask=mask))
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
    seedall(4810000+fid*10000+seed); m=M(); o=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=0)
    for st in range(BASE):
        l='A' if st%2==0 else 'B'; g=torch.Generator().manual_seed(4811000+fid*1000000+seed*1000+st)
        step(m,o,sample(U,l,BS,g),l)
    return m
def pos(seed,fid): return set(random.Random(4812000+fid*100000+seed).sample(range(CURR),int(DOSE*CURR)))
def curriculum(base,seed,fid,kind,pbt):
    m=copy.deepcopy(base); o=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=0); ps=pos(seed,fid) if kind else set()
    for st in range(CURR):
        P=PA if st not in ps else pbt; g=torch.Generator().manual_seed(4813000+fid*1000000+seed*1000+st)
        step(m,o,sample(P,'A',BS,g),'A')
    return m
def aligned_head(b,seed,fid):
    state=torch.random.get_rng_state(); torch.manual_seed(4813300+fid*10000+seed); r=torch.randn_like(b); torch.random.set_rng_state(state)
    bf=b.reshape(-1); rf=r.reshape(-1); rf-=torch.dot(rf,bf)/torch.dot(bf,bf)*bf; rf=rf/torch.linalg.vector_norm(rf)*torch.linalg.vector_norm(bf)
    return (RHO*bf+(1-RHO*RHO)**.5*rf).reshape_as(b)
def rotations(E,seed,fid):
    _,S,Vh=torch.linalg.svd(E,full_matrices=True); V=Vh.T; rng=np.random.default_rng(4813400+fid*10000+seed)
    perm=rng.permutation(12).tolist(); order=perm+list(range(12,D)); P=V[:,order]; outs=[]
    for keep in KEEP:
        n=D-keep
        if n==0: Rot=torch.eye(D)
        else:
            state=torch.random.get_rng_state(); torch.manual_seed(4813500+fid*10000+seed+keep*97); A=torch.randn(n,n); torch.random.set_rng_state(state)
            Q,R=torch.linalg.qr(A); sg=torch.sign(torch.diag(R)); sg[sg==0]=1; Q=Q*sg.unsqueeze(0)
            if torch.linalg.det(Q)<0: Q[:,0]*=-1
            B=torch.eye(D); B[keep:,keep:]=Q; Rot=P@B@P.T
        retained=float(sum(float(S[i]**2) for i in perm[:keep])/float(S@S)) if keep else 0.0
        outs.append((keep,Rot,retained,perm))
    return outs
def install(m,w,R):
    m.hB=nn.Linear(D,K,bias=False)
    with torch.no_grad(): m.hB.weight.copy_(w); m.Brot.copy_(R)
def curve(m,seed,fid,pbt,eb):
    o=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=0); r=[(0,ev(m,eb,'B'))]
    for st in range(1,BMAX+1):
        g=torch.Generator().manual_seed(4814000+fid*1000000+seed*1000+st-1); step(m,o,sample(pbt,'B',BS,g),'B')
        if st%EV==0: r.append((st,ev(m,eb,'B')))
    return r
def main():
    fid=int(os.environ.get('FAMILY_ONLY','0')); seed=int(os.environ.get('SEED',str(SEED)))
    pbt=blend(endpoint(fid)); base=train_base(seed,fid); w=aligned_head(base.hA.weight.detach(),seed,fid); E=base.state.weight.detach()
    eb=sample(pbt,'B',BS*6,torch.Generator().manual_seed(4815000+fid*10000+seed))
    cms={'A_nat':curriculum(base,seed,fid,None,pbt),'target25':curriculum(base,seed,fid,'target',pbt)}; rows=[]; aud=[]
    for keep,R,e,perm in rotations(E,seed,fid):
        aud.append({'family':fid,'seed':seed,'keep_dim':keep,'retained_state_energy':e,'orth_err':float(torch.max(torch.abs(R.T@R-torch.eye(D)))),'perm12':'-'.join(map(str,perm))})
        for c,m0 in cms.items():
            m=copy.deepcopy(m0); install(m,w,R)
            for st,v in curve(m,seed,fid,pbt,eb): rows.append({'family':fid,'seed':seed,'keep_dim':keep,'condition':c,'B_step':st,'B_nll':v})
    pd.DataFrame(rows).to_csv(ROOT/f'family{fid}_seed{seed}.csv',index=False)
    pd.DataFrame(aud).to_csv(ROOT/f'audit_family{fid}_seed{seed}.csv',index=False)
    print('saved',fid,seed,flush=True)
if __name__=='__main__': main()
