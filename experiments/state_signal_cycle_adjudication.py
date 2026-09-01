import os, sys, copy, csv, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import teacher_hidden_entropy_matched_distance_transformer as core
import teacher_hidden_geometry_intervention as geo
import support_stationary_matrix as sm

CODES=sm.CODES


def signal_tables(tab,ta):
    # L must be 1. No stochastic layers; hidden tables are exact functions of state identity.
    x=torch.arange(core.K,dtype=torch.long).view(core.K,1)
    tab.eval(); ta.eval()
    with torch.no_grad():
        _,hab=tab(x,0,True); _,ha=ta(x,0,True)
        delta=hab-ha
        Q,rank,S=geo.b_head_basis(tab)
        R=geo.random_orthogonal(12345)  # unused by Bsub_norm
        td,rel,deg=geo.transform_delta(delta,Q,R,'Bsub_norm')
    return ha.detach(), td.detach(), rank, rel, deg


def cycle_map(changed, shift):
    m=torch.arange(core.K,dtype=torch.long)
    n=len(changed)
    for p,r in enumerate(changed):
        m[r]=changed[(p+shift)%n]
    return m


def remap_with_norm(td_table, x, changed, shift):
    # td_table: K x 1 x D. Preserve the aligned per-state signal norm exactly,
    # change only the Bsub direction by reassigning another changed-state direction.
    mp=cycle_map(changed,shift)
    idx=x[:,0]
    src=mp[idx]
    cand=td_table[src,0,:]
    ref=td_table[idx,0,:]
    rn=torch.linalg.vector_norm(ref,dim=-1,keepdim=True)
    cn=torch.linalg.vector_norm(cand,dim=-1,keepdim=True)
    deg=(cn<=1e-10)
    out=torch.zeros_like(cand)
    good=~deg.squeeze(-1)
    out[good]=cand[good]*(rn[good]/cn[good])
    rel=0.0
    valid=(rn.squeeze(-1)>1e-10)&good
    if valid.any():
        on=torch.linalg.vector_norm(out[valid],dim=-1)
        rel=float(((on-rn.squeeze(-1)[valid]).abs()/rn.squeeze(-1)[valid].clamp_min(1e-10)).max().item())
    return out[:,None,:], rel, float(deg.float().mean().item())


def student_phase_table(base,tab,ta,PA,seed,mode,changed):
    s=copy.deepcopy(base)
    params=[p for n,p in s.named_parameters() if not n.startswith('heads.1.')]
    opt=torch.optim.AdamW(params,lr=core.LR_S,weight_decay=0.0)
    g=torch.Generator().manual_seed(seed+300)
    ha_tab,td_tab,rank,base_rel,base_deg=signal_tables(tab,ta)
    max_rel=base_rel; deg_sum=base_deg; n=1
    for st in range(core.STUD_STEPS):
        x,y=core.sample_batch(PA,g)
        idx=x[:,0]
        with torch.no_grad():
            ha=ha_tab[idx,0,:][:,None,:]
            if mode=='A_baseline':
                ht=ha; rel=0.0; deg=0.0
            elif mode=='Bsub_aligned':
                ht=ha+td_tab[idx,0,:][:,None,:]; rel=base_rel; deg=base_deg
            elif mode=='Bsub_cycleavg':
                # shifts 1,2,...,5,0 repeat: every state receives every changed-state
                # Bsub direction exactly once per 6 training steps.
                shift=(st+1)%len(changed)
                td,rel,deg=remap_with_norm(td_tab,x,changed,shift)
                ht=ha+td
            else: raise ValueError(mode)
        logits,hs=s(x,0,True)
        ce=F.cross_entropy(logits.reshape(-1,core.K),y.reshape(-1))
        feat=F.mse_loss(hs,ht)
        opt.zero_grad(); (ce+core.LAM*feat).backward(); opt.step()
        max_rel=max(max_rel,rel); deg_sum+=deg; n+=1
    return s,dict(b_subspace_rank=rank,max_delta_norm_relerr=max_rel,degenerate_fraction=deg_sum/n)


def run_source(fam,seed,sidx,outdir):
    core.L=1
    core.seed_all(seed+fam*1000)
    PA=core.shift_kernel(fam,0)
    changed=sm.changed_rows(fam)
    PBs=[sm.B_from_code(PA,fam,c) for c in CODES]
    base_template=core.TinyT(); base_b_head=copy.deepcopy(base_template.heads[1].state_dict())
    base=copy.deepcopy(base_template)
    tab,ta,tabA,taA=core.matched_teachers(base,PA,PBs[sidx],seed+fam*100)
    modes=['A_baseline','Bsub_aligned','Bsub_cycleavg']
    models={}; audits={}
    for mode in modes:
        models[mode],audits[mode]=student_phase_table(base,tab,ta,PA,seed+313,mode,changed)
    bpre={m:max((s.heads[1].state_dict()[k]-base_b_head[k]).abs().max().item() for k in base_b_head) for m,s in models.items()}
    models,before,after,cal_target=geo.calibrate_common(models,PA,seed+529)
    bpost={m:max((s.heads[1].state_dict()[k]-base_b_head[k]).abs().max().item() for k in base_b_head) for m,s in models.items()}
    rows=[]
    target_env=os.environ.get('TARGETS','')
    target_ids=list(range(len(PBs))) if not target_env else [int(v) for v in target_env.split(',')]
    for tidx in target_ids:
        PBt=PBs[tidx]
        curves={m:core.b_curve(s,PBt,seed+701) for m,s in models.items()}
        for mode in ['Bsub_aligned','Bsub_cycleavg']:
            sp,spa,diff,th,tl=core.aligned_span(curves[mode],curves['A_baseline'])
            rows.append(dict(family=fam,seed=seed,source_variant=f'V{sidx}',target_variant=f'V{tidx}',source_idx=sidx,target_idx=tidx,mode=mode,
                tv_A_source=core.tv_distance(PA,PBs[sidx]),tv_A_target=core.tv_distance(PA,PBt),tv_source_target=core.tv_distance(PBs[sidx],PBt),
                entropy_gap_source=abs(core.kernel_entropy(PA)-core.kernel_entropy(PBs[sidx])),entropy_gap_target=abs(core.kernel_entropy(PA)-core.kernel_entropy(PBt)),
                teacher_A_gap=abs(tabA-taA),phase1_A_nll=before[mode],student_A_nll=after[mode],common_A_range=max(after.values())-min(after.values()),
                b_head_change=max(bpre[mode],bpost[mode]),b_subspace_rank=audits[mode]['b_subspace_rank'],max_delta_norm_relerr=audits[mode]['max_delta_norm_relerr'],degenerate_fraction=audits[mode]['degenerate_fraction'],
                span_mode=sp,span_A_baseline=spa,effect=diff,interval_high=th,interval_low=tl))
    Path(outdir).mkdir(parents=True,exist_ok=True)
    p=Path(outdir)/f'family{fam}_seed{seed}_source{sidx}.csv'
    with open(p,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(json.dumps({'family':fam,'source':sidx,'path':str(p),'rows':len(rows)}),flush=True)

if __name__=='__main__':
    fam=int(os.environ.get('FAMILY','0')); seed=int(os.environ.get('SEED','18400')); sidx=int(os.environ.get('SOURCE','0')); out=os.environ.get('OUTDIR','/mnt/data/teacher_hidden_state_signal_cycle_pilot_seed18400')
    run_source(fam,seed,sidx,out)
