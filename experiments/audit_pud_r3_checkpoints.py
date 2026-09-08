#!/usr/bin/env python3
"""Read-only checkpoint audit and exact byte-class loss decomposition.
No optimization, model selection, coefficient fit, or target changes.
"""
import argparse, csv, json, math
from pathlib import Path
import torch
import torch.nn.functional as F
import run_pud_r3 as r

@torch.no_grad()
def independent_metric(model, prepared):
    sums=[0.,0.,0.]; count=0; identity_error=0.; fp32=0.
    for x,y,_,_ in prepared[0]:
        logits=model(x); valid=y!=-100; labels=y[valid]
        active=logits[valid]
        logs=torch.log_softmax(active.double(),-1)
        nll=-logs.gather(-1,labels[:,None]).squeeze(1)
        cls=torch.where(labels<128,0,torch.where(labels<192,1,2))
        class_logs=torch.stack([torch.logsumexp(logs[:,:128],-1),torch.logsumexp(logs[:,128:192],-1),torch.logsumexp(logs[:,192:],-1)],-1)
        class_loss=-class_logs.gather(-1,cls[:,None]).squeeze(1)
        within=torch.empty_like(nll)
        for j,(lo,hi) in enumerate(((0,128),(128,192),(192,256))):
            mask=cls==j
            if bool(mask.any()):
                conditional=torch.log_softmax(active.double()[mask,lo:hi],-1)
                within[mask]=-conditional.gather(-1,(labels[mask]-lo)[:,None]).squeeze(1)
        identity_error=max(identity_error,float((nll-class_loss-within).abs().max()))
        if not torch.isfinite(nll).all() or float(within.min()) < -1e-12:
            raise ValueError('invalid probability decomposition')
        sums[0]+=float(nll.sum());sums[1]+=float(class_loss.sum());sums[2]+=float(within.sum())
        fp32+=float(F.cross_entropy(active,labels,reduction='none').double().sum())
        count+=len(labels)
    if count!=prepared[1]:raise ValueError('byte count')
    return {'nll_float64':sums[0]/count,'nll_original_float32':fp32/count,'class_nll':sums[1]/count,
            'within_class_nll':sums[2]/count,'target_bytes':count,'identity_error':identity_error}

def run(root,data,out):
    _,rows,_=r.read_data(data);prepared={l:r.make_eval(rows[l]) for l in ('ja','en')}
    checked=[];maxdiff=0.;classerr=0.;double_diff=0.
    for seed in r.SEEDS:
        p=root/f'seed{seed}'
        if not (p/'audit.json').exists():raise ValueError(f'incomplete seed {seed}')
        sm=json.loads((p/'summary.json').read_text())
        for row in sm:
            arm=row['arm']
            for stage,label,langs in [('phase1','base' if arm=='REF_EN' else 'phase1_'+arm,('ja','en')),
                                     ('target','target_'+arm,('ja','en')),
                                     ('end600','end600_'+arm,('ja','en'))]:
                ck=p/'checkpoints'/(label+'.pt')
                if not ck.exists():
                    if stage=='target' and row['target_update'] is None:continue
                    raise ValueError('checkpoint missing')
                torch.manual_seed(99009);m=r.core.Model();m.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True))
                m.train()
                for lang in langs:
                    a=independent_metric(m,prepared[lang]);classerr=max(classerr,a['identity_error'])
                    double_diff=max(double_diff,abs(a['nll_float64']-a['nll_original_float32']))
                    field={('phase1','ja'):'japanese_phase1_nll',('phase1','en'):'english_step0_nll',
                           ('target','ja'):'japanese_at_target_nll',('end600','ja'):'japanese_end600_nll',('end600','en'):'english_end600_nll'}.get((stage,lang))
                    discrepancy=abs(a['nll_original_float32']-row[field]) if field else 0.
                    maxdiff=max(maxdiff,discrepancy)
                    if discrepancy>1e-10:raise ValueError(f'checkpoint mismatch {seed} {arm} {stage} {lang}: {discrepancy}')
                    if stage=='target' and lang=='en' and a['nll_original_float32']>r.TARGET+1e-10:raise ValueError('checkpoint misses registered target')
                    checked.append({'seed':seed,'arm':arm,'stage':stage,'language':lang,**a,'record_discrepancy':discrepancy,'checkpoint_sha256':r.file_sha(ck)})
        print(f'checkpoint audit complete {seed}',flush=True)
    out.mkdir(parents=True,exist_ok=True)
    with (out/'checkpoint_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(checked[0]));w.writeheader();w.writerows(checked)
    result={'checkpoint_metric_rows':len(checked),'max_original_record_discrepancy':maxdiff,'max_float64_vs_float32_difference':double_diff,
            'max_decomposition_identity_error':classerr,'passed':True,'optimizer_updates':0,
            'scope':'byte probability factorization, not fractions of semantic knowledge; checkpoint remeasurement does not add independent seeds'}
    r.write_json(out/'verification.json',result);return result
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    print(json.dumps(run(a.results,a.data,a.out),indent=2))
