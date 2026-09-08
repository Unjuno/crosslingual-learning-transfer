#!/usr/bin/env python3
"""One fixed readout-restoration diagnostic, separate from the registered trial."""
import argparse,copy,csv,json
from pathlib import Path
import torch
import run_pud_r3 as r
from audit_pud_r3_checkpoints import independent_metric
PLAN_COMMIT='34005b764e450448250ddc92a26821b457760225'

def restore_head(later,earlier):
    before=copy.deepcopy(later.state_dict()); after=copy.deepcopy(before)
    if set(before)!=set(earlier):raise ValueError('checkpoint key mismatch')
    if any(before[k].shape!=earlier[k].shape for k in before):raise ValueError('checkpoint shape mismatch')
    after['head.weight']=earlier['head.weight'].clone()
    if any(not torch.equal(after[k],before[k]) for k in before if k!='head.weight'):
        raise ValueError('nonhead tensor changed')
    out=copy.deepcopy(later);out.load_state_dict(after);return out

def run_one(seed,root,data,out):
    _,rows,_=r.read_data(data);prepared={l:r.make_eval(rows[l]) for l in ('ja','en')}
    p=root/f'seed{seed}';ck=p/'checkpoints';result=[]
    if not (p/'audit.json').exists():raise ValueError('unfinished checkpoint set')
    for arm in ('JA_TJA','JA_TEN'):
        early=torch.load(ck/f'phase1_{arm}.pt',map_location='cpu',weights_only=True)
        m0=r.core.Model();m0.load_state_dict(early)
        pre={l:independent_metric(m0,prepared[l]) for l in ('ja','en')}
        for stage in ('target','end600'):
            path=ck/f'{stage}_{arm}.pt'
            if not path.exists():continue
            state=torch.load(path,map_location='cpu',weights_only=True)
            later=r.core.Model();later.load_state_dict(state)
            same=restore_head(later,state)
            if any(not torch.equal(v,same.state_dict()[k]) for k,v in state.items()):raise ValueError('self restore changes state')
            x=prepared['en'][0][0][0]
            with torch.no_grad():
                if not torch.equal(later(x),same(x)):raise ValueError('negative-control predictions changed')
            restored=restore_head(later,early)
            before={l:independent_metric(later,prepared[l]) for l in ('ja','en')}
            after={l:independent_metric(restored,prepared[l]) for l in ('ja','en')}
            increase=before['ja']['nll_float64']-pre['ja']['nll_float64']
            recovered=before['ja']['nll_float64']-after['ja']['nll_float64']
            result.append({'seed':seed,'arm':arm,'stage':stage,'japanese_phase1':pre['ja']['nll_float64'],
                'japanese_before':before['ja']['nll_float64'],'japanese_after_head_restore':after['ja']['nll_float64'],
                'japanese_loss_increase':increase,'japanese_recovered_absolute':recovered,
                'fraction_loss_increase_recovered':recovered/increase if increase>0 else None,
                'english_before':before['en']['nll_float64'],'english_after_head_restore':after['en']['nll_float64'],
                'english_cost':after['en']['nll_float64']-before['en']['nll_float64'],
                'class_loss_increase':before['ja']['class_nll']-pre['ja']['class_nll'],
                'within_class_loss_increase':before['ja']['within_class_nll']-pre['ja']['within_class_nll'],
                'class_recovery':before['ja']['class_nll']-after['ja']['class_nll'],
                'within_class_recovery':before['ja']['within_class_nll']-after['ja']['within_class_nll'],
                'nonhead_tensors_unchanged':True,'negative_control_pass':True,'plan_commit':PLAN_COMMIT})
    out.mkdir(parents=True,exist_ok=True);r.write_json(out/'readout.json',result);return result
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--results',type=Path,required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();run_one(a.seed,a.results,a.data,a.out)
