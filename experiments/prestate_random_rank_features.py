import os, sys
from pathlib import Path
import numpy as np, pandas as pd
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import random_rank_state_alignment_gru as core


def effective_rank_from_svals(s):
    e=s.double()**2
    p=e/e.sum()
    return float(torch.exp(-(p*torch.log(p.clamp_min(1e-30))).sum()))


def exact_b_nll(m, P):
    st=torch.arange(core.K,dtype=torch.long)
    e=m.state(st) @ m.Brot
    z=e + m.lang(torch.ones_like(st))
    h,_=m.rnn(z[:,None,:])
    h=m.ln(h[:,0,:])
    logits=m.hB(h)
    Pt=torch.tensor(P,dtype=logits.dtype)
    return -(Pt*torch.log_softmax(logits,dim=-1)).sum(dim=-1).mean()


def main():
    seed=int(os.environ['SEED'])
    width=int(os.environ['HIDDEN_WIDTH'])
    fid=int(os.environ['FAMILY_ONLY'])
    out=Path(os.environ.get('FEATURE_OUT','results/raw_prestate_features'))
    out.mkdir(parents=True,exist_ok=True)

    core.H=width
    core.SEED=seed
    pbt=core.blend(core.endpoint(fid))
    base=core.train_base(seed,fid)
    w=core.aligned_head(base.hA.weight.detach(),seed,fid)
    E=base.state.weight.detach()

    nlls=[]
    for keep,R,_,_ in core.rotations(E,seed,fid):
        core.install(base,w,R)
        with torch.no_grad():
            nlls.append((keep,float(exact_b_nll(base,pbt))))
    x=np.array([k for k,_ in nlls],dtype=float)
    y=np.array([v for _,v in nlls],dtype=float)
    rotation_nll_slope=float(np.polyfit(x,y,1)[0])

    state_effective_rank=effective_rank_from_svals(torch.linalg.svdvals(E))

    core.install(base,w,torch.eye(core.D))
    st=torch.arange(core.K,dtype=torch.long)
    e=base.state(st)
    z=e+base.lang(torch.ones_like(st))
    h,_=base.rnn(z[:,None,:])
    h=base.ln(h[:,0,:])
    b_hidden_effective_rank=effective_rank_from_svals(torch.linalg.svdvals(h.detach()))

    base.zero_grad(set_to_none=True)
    loss=exact_b_nll(base,pbt)
    loss.backward()
    b_state_grad_norm=float(torch.linalg.vector_norm(base.state.weight.grad).item())

    row={
        'seed':seed,
        'hidden_width':width,
        'family':fid,
        'pre_rotation_nll_slope':rotation_nll_slope,
        'state_effective_rank':state_effective_rank,
        'b_hidden_effective_rank':b_hidden_effective_rank,
        'b_state_grad_norm':b_state_grad_norm,
        'pre_identity_b_nll':float(loss.item()),
    }
    pd.DataFrame([row]).to_csv(out/f'feature_seed{seed}_H{width}_f{fid}.csv',index=False)
    print(row,flush=True)


if __name__=='__main__':
    main()
