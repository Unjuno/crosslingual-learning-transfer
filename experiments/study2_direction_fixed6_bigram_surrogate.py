import ast, copy, gettext, hashlib, random, re, os, time, math
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(int(os.environ.get('THREADS','1')))
LANGS=['de','en','ja','ru','es','it']
CTX=32; LR=2e-3; BS=16; D=32; LAYERS=1; BASE=240; TEACH=160; DOCSTEPS=2
R=int(os.environ.get('TRAJ','2')); CPS=[0,5,10,20]; GEN_N=int(os.environ.get('GEN_N','128')); K=int(os.environ.get('K','16'))

def parse_po(path):
    entries=[];cur=None;mode=None
    def flush():
        nonlocal cur
        if cur and cur.get('msgid') and cur.get('msgstr'): entries.append((cur['msgid'],cur['msgstr']))
        cur=None
    for raw in Path(path).read_text('utf8').splitlines():
        s=raw.strip()
        if s.startswith('msgid '): flush();cur={'msgid':ast.literal_eval(s[6:]),'msgstr':''};mode='msgid'
        elif cur is not None and s.startswith('msgstr '): cur['msgstr']=ast.literal_eval(s[7:]);mode='msgstr'
        elif cur is not None and s.startswith('"'):
            try: cur[mode]+=ast.literal_eval(s)
            except: pass
        elif cur is not None and s=='' and cur.get('msgstr'): flush();mode=None
    flush(); return dict(entries)

def load_vim(l):
    if l=='en': return None
    p=f'/usr/share/vim/vim91/lang/{l}/LC_MESSAGES/vim.mo'
    with open(p,'rb') as f:g=gettext.GNUTranslations(f)
    return {k:v for k,v in g._catalog.items() if isinstance(k,str) and k and isinstance(v,str) and v}

def norm(s):
    s=s.replace('\x00',' ').replace('\n',' ');s=re.sub(r'\s+',' ',s).strip();return re.sub(r'\d+','0',s)

non=[l for l in LANGS if l!='en']; root=Path('/usr/share/texlive/tlpkg/translations')
po={l:parse_po(root/f'{l}.po') for l in non}; common0=sorted(set.intersection(*[set(po[l]) for l in non])); tex={'en':[norm(x) for x in common0]}
for l in non: tex[l]=[norm(po[l][x]) for x in common0]
vd={l:load_vim(l) for l in non}; common1=sorted(set.intersection(*[set(vd[l]) for l in non])); vim={'en':[norm(x) for x in common1]}
for l in non: vim[l]=[norm(vd[l][x]) for x in common1]
tridx=[i for i,x in enumerate(common1) if int(hashlib.md5(x.encode()).hexdigest()[:8],16)%10!=0]; teset=set(tridx); teidx=[i for i in range(len(common1)) if i not in teset]
chars=sorted(set('¶'+''.join(sum([tex[l]+vim[l] for l in LANGS],[])))); stoi={c:i for i,c in enumerate(chars)}; UNK=len(chars); VOC=len(chars)+1

def enc(msgs): s='¶'.join(x for x in msgs if x); return torch.tensor([stoi.get(c,UNK) for c in s],dtype=torch.long)
REF={l:enc(tex[l]) for l in LANGS}; TR={l:enc([vim[l][i] for i in tridx]) for l in LANGS}; TE={l:enc([vim[l][i] for i in teidx]) for l in LANGS}
# Counterfactual corpus intervention: preserve the exact directed character-bigram
# multiset of every language/split, but randomize the Euler trail (higher-order order).
SURR_MODE=os.environ.get('SURR_MODE','bg')  # bg | orig | uni
SURR_ID=int(os.environ.get('SURR_ID','0'))

def _random_euler(x, seed):
    a=x.tolist(); n=len(a)
    if n < 3: return x.clone()
    adj={}
    for u,v in zip(a[:-1],a[1:]): adj.setdefault(u,[]).append(v)
    rng=random.Random(seed)
    for vs in adj.values(): rng.shuffle(vs)
    stack=[a[0]]; trail=[]
    while stack:
        u=stack[-1]
        vs=adj.get(u)
        if vs:
            stack.append(vs.pop())
        else:
            trail.append(stack.pop())
    out=list(reversed(trail))
    assert len(out)==n and out[0]==a[0] and out[-1]==a[-1]
    return torch.tensor(out,dtype=torch.long)

def _unigram_shuffle(x, seed):
    a=x.tolist(); rng=random.Random(seed)
    mid=a[1:-1]; rng.shuffle(mid)
    return torch.tensor([a[0]]+mid+[a[-1]],dtype=torch.long)

def _bh(x):
    a=x.numpy().astype(np.int64,copy=False)
    return np.bincount(a[:-1]*VOC+a[1:], minlength=VOC*VOC)

def _apply_surrogates():
    global REF,TR,TE
    if SURR_MODE=='orig': return {}
    changed={}
    for si,(name,DCT) in enumerate([('REF',REF),('TR',TR),('TE',TE)]):
        for li,l in enumerate(LANGS):
            old=DCT[l]
            seed=930000 + SURR_ID*10000 + si*100 + li
            new=_random_euler(old,seed) if SURR_MODE=='bg' else _unigram_shuffle(old,seed)
            if SURR_MODE=='bg':
                assert np.array_equal(_bh(old),_bh(new)), (name,l,'bigram mismatch')
            else:
                assert torch.equal(torch.sort(old).values,torch.sort(new).values), (name,l,'unigram mismatch')
            changed[(name,l)]=float((old!=new).float().mean())
            DCT[l]=new
    return changed

SURR_CHANGED=_apply_surrogates()
C_IDS={c:torch.tensor(sorted(set(REF[c].tolist()+TR[c].tolist())),dtype=torch.long) for c in LANGS}

class M(nn.Module):
    def __init__(self):
        super().__init__();self.tok=nn.Embedding(VOC,D);self.pos=nn.Embedding(CTX,D)
        lay=nn.TransformerEncoderLayer(D,4,4*D,dropout=0,batch_first=True,activation='gelu',norm_first=True)
        self.bl=nn.TransformerEncoder(lay,LAYERS);self.ln=nn.LayerNorm(D);self.h=nn.Linear(D,VOC,bias=False)
        self.register_buffer('mask',torch.triu(torch.ones(CTX,CTX,dtype=torch.bool),1),persistent=False)
    def forward(self,x):
        z=self.tok(x)+self.pos(torch.arange(x.size(1)));return self.h(self.ln(self.bl(z,mask=self.mask[:x.size(1),:x.size(1)],is_causal=True)))

def seedall(s): random.seed(s);np.random.seed(s);torch.manual_seed(s)
def sample(st,n=BS,g=None): ix=torch.randint(0,len(st)-CTX-1,(n,),generator=g);return st[ix[:,None]+torch.arange(CTX+1)]
def loss(m,z): return F.cross_entropy(m(z[:,:-1]).reshape(-1,VOC),z[:,1:].reshape(-1))
def step(m,op,z):
    v=loss(m,z);op.zero_grad(set_to_none=True);v.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);op.step()
@torch.no_grad()
def ev(m,l,seed):
    g=torch.Generator().manual_seed(seed);return float(np.mean([loss(m,sample(TE[l],32,g)).item() for _ in range(6)]))

def reference(seed):
    seedall(seed);m=M();op=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=0);g=torch.Generator().manual_seed(81000+seed)
    sched=(LANGS*(BASE//len(LANGS)+1))[:BASE];random.Random(81100+seed).shuffle(sched)
    for l in sched: step(m,op,sample(REF[l],BS,g))
    return m

def teacher(ref,c,t,seed):
    # Same schedule positions and minibatch RNG across target labels within a carrier.
    ci=LANGS.index(c); m=copy.deepcopy(ref);op=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=0)
    g=torch.Generator().manual_seed(81200+seed*100+ci); n=int(.75*TEACH)
    sched=['T']*n+['C']*(TEACH-n); random.Random(81300+seed*100+ci).shuffle(sched)
    for which in sched: step(m,op,sample(TR[t if which=='T' else c],BS,g))
    return m

@torch.no_grad()
def gen_c(m,c,seed,n=GEN_N,pfx=8,temp=.9):
    ci=LANGS.index(c); ids=C_IDS[c];g=torch.Generator().manual_seed(81400+seed*100+ci);z=sample(TR[c],n,g)[:,:pfx].clone()
    # common random numbers across target teachers for same carrier
    torch.manual_seed(81500+seed*100+ci)
    while z.size(1)<CTX:
        logits=m(z)[:,-1,ids]/temp;p=torch.softmax(logits,-1);ix=torch.multinomial(p,1);z=torch.cat([z,ids[ix]],1)
    return z
@torch.no_grad()
def seq_lp(m,z):
    lg=m(z[:,:-1]);y=z[:,1:];return torch.log_softmax(lg,-1).gather(-1,y.unsqueeze(-1)).squeeze(-1).mean(1)

def train_doc_matched(ref,doc,idxs):
    m=copy.deepcopy(ref);op=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=0)
    for ix in idxs: step(m,op,doc[ix])
    return m

def prep(st,ref):
    m=copy.deepcopy(st)
    with torch.no_grad(): m.tok.weight.copy_(ref.tok.weight);m.pos.weight.copy_(ref.pos.weight);m.h.weight.zero_()
    return m

def auc(vals): return sum((vals[a]+vals[b])*.5*(b-a) for a,b in zip(CPS[:-1],CPS[1:]))/20

def adapt_carrier(states,ref,c,target,seed):
    # states: one trained document state per non-carrier language.
    names=[target]+[u for u in LANGS if u not in (c,target)]+['ref']; rec=[]
    for r in range(R):
        ms={u:prep(states[u],ref) for u in names if u!='ref'}; ms['ref']=prep(ref,ref)
        ops={u:torch.optim.AdamW(ms[u].parameters(),lr=LR,weight_decay=0) for u in names}
        # Same target adaptation/eval minibatches across all carriers for a given target.
        ti=LANGS.index(target);g=torch.Generator().manual_seed(81700+seed*1000+ti*50+r); vals={};done=0
        for cp in CPS:
            for _ in range(cp-done):
                z=sample(TR[target],BS,g)
                for u in names: step(ms[u],ops[u],z)
            done=cp; es=81800+seed*1000+ti*100+r*30+cp; vals[cp]={u:ev(ms[u],target,es) for u in names}
        rw={0:0.0}; rn={0:0.0}; out={'seed':seed,'traj':r,'src':c,'tgt':target,'L0':vals[0]['ref']}
        wrongs=[u for u in LANGS if u not in (c,target)]
        for cp in CPS[1:]:
            right=vals[cp][target]; wmean=float(np.mean([vals[cp][u] for u in wrongs])); refv=vals[cp]['ref']
            rw[cp]=wmean-right; rn[cp]=refv-right
            out[f'rwavg{cp}']=rw[cp];out[f'rn{cp}']=rn[cp]
        out['auc_rwavg']=auc(rw);out['auc_rn']=auc(rn);rec.append(out)
    return rec

def run_seed(seed):
    t0=time.time();ref=reference(seed);rows=[];sel=[]
    carr_only=os.environ.get('CARR_ONLY'); carrs=[carr_only] if carr_only else LANGS
    for c in carrs:
        ci=LANGS.index(c)
        targets=[t for t in LANGS if t!=c]
        teachers={t:teacher(ref,c,t,seed) for t in targets}
        candidates={t:gen_c(teachers[t],c,seed) for t in targets}
        docs={}; scoremeans={}
        for t in targets:
            p=candidates[t]; own=seq_lp(teachers[t],p); alt=torch.stack([seq_lp(teachers[u],p) for u in targets if u!=t]).mean(0); s=own-alt
            docs[t]=p[torch.argsort(s,descending=True)[:K]];scoremeans[t]=(float(torch.topk(s,K).values.mean()),float(s.mean()))
        # matched document minibatch indices across all target-aware docs for same carrier
        gdoc=torch.Generator().manual_seed(81600+seed*100+ci); idxs=[torch.randint(0,K,(BS,),generator=gdoc) for _ in range(DOCSTEPS)]
        states={t:train_doc_matched(ref,docs[t],idxs) for t in targets}
        for t in targets:
            rows.extend(adapt_carrier(states,ref,c,t,seed))
            top,allm=scoremeans[t];sel.append({'seed':seed,'src':c,'tgt':t,'top_score':top,'all_score':allm,'VOC':VOC,'carrier_vocab':len(C_IDS[c]),'tex_common':len(common0),'vim_common':len(common1)})
        print('seed',seed,'carrier',c,'elapsed',round(time.time()-t0,1),flush=True)
    return pd.DataFrame(rows),pd.DataFrame(sel)

if __name__=='__main__':
    seed=int(os.environ.get('SEED','0'));out=os.environ.get('OUTDIR','/mnt/data/research_handoff_2026-08-26/10_bigram_causal_surrogate/raw');Path(out).mkdir(parents=True,exist_ok=True)
    rows,sel=run_seed(seed);suffix=('_'+os.environ['CARR_ONLY']) if os.environ.get('CARR_ONLY') else '';rows.to_csv(Path(out)/f'fixed6_allpairs_seed{seed}{suffix}_rows.csv',index=False);sel.to_csv(Path(out)/f'fixed6_allpairs_seed{seed}{suffix}_selection.csv',index=False)
    sm=rows.groupby(['seed','src','tgt'],as_index=False).mean(numeric_only=True);sm.to_csv(Path(out)/f'fixed6_allpairs_seed{seed}{suffix}_summary.csv',index=False)
    print(sm[['src','tgt','auc_rwavg','auc_rn']].to_string(index=False));print('VOC',VOC,'common',len(common0),len(common1),'SURR_MODE',SURR_MODE,'SURR_ID',SURR_ID,'changed_mean',float(np.mean(list(SURR_CHANGED.values()))) if SURR_CHANGED else 0.0,flush=True)
