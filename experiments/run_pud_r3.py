#!/usr/bin/env python3
"""Fixed R3 trial. CPU/float32; original model and update kernel are reused.
No threshold, optimizer or training-budget adaptation is permitted.
"""
import argparse, copy, csv, hashlib, json, os, platform, sys, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import natural_ja_en_s2_hidden_teacher as core

SEEDS=tuple(range(33000,33010))
ARMS=('REF_EN','JA_nat','JA_TJA','JA_TEN','JA_TRU')
DATA_HASH='a70abd42d4ca57a9f27a6a784388559dc7ebb4366ced28d4aef7285a08e41236'
CORE_BLOB='aab161824538e29f71f2a5f00fd44364f2de9bd4'
PROTOCOL_COMMIT='afd7e82180ee856a4e4fc3b48b51f490721d41e1'
TARGET=3.0
HIGH=3.25
BMAX=600
core.HIDDEN_WEIGHT=.5

def sha(data): return hashlib.sha256(data).hexdigest()
def file_sha(path): return sha(path.read_bytes())
def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
    tmp.replace(path)

def read_data(root):
    if file_sha(root/'MANIFEST.json')!=DATA_HASH: raise ValueError('data manifest mismatch')
    manifest=json.loads((root/'MANIFEST.json').read_text())
    allrows={}
    for name,entry in manifest['files'].items():
        p=root/name
        if file_sha(p)!=entry['sha256']: raise ValueError(f'data hash mismatch {name}')
        # Reserve text is not parsed, fit, or evaluated.
        if 'reserve' not in name:
            allrows[name[:-6]]=[json.loads(s) for s in p.read_text().splitlines()]
    streams={l:torch.tensor(list(('\n'.join(r['text'] for r in allrows[l+'_train'])+'\n').encode()),dtype=torch.long) for l in ('ja','en','ru')}
    return streams,{l:allrows[l+'_eval'] for l in ('ja','en')},manifest

def make_eval(rows, batch_size=32):
    """Each target byte 1..len-1 once; context does not cross sentences.
    Padding inputs are byte zero. Padded targets are -100 (ignored), not zero.
    """
    blocks=[]
    for i,row in enumerate(rows):
        b=list(row['text'].encode('utf-8'))
        for start in range(0,len(b)-1,core.CTX):
            n=min(core.CTX,len(b)-1-start)
            x=b[start:start+n]+[0]*(core.CTX-n)
            y=b[start+1:start+1+n]+[-100]*(core.CTX-n)
            blocks.append((x,y,i,n))
    batches=[]
    for off in range(0,len(blocks),batch_size):
        sub=blocks[off:off+batch_size]
        batches.append((torch.tensor([a[0] for a in sub],dtype=torch.long),
                        torch.tensor([a[1] for a in sub],dtype=torch.long),
                        [a[2] for a in sub],[a[3] for a in sub]))
    n=sum(x[3] for x in blocks)
    assert n==sum(len(r['text'].encode())-1 for r in rows)
    return batches,n

@torch.no_grad()
def evaluate(model, prepared, details=False):
    batches,n=prepared; total=0.; doc=defaultdict(lambda:[0.,0])
    for x,y,ids,counts in batches:
        logits=model(x)
        loss=F.cross_entropy(logits.reshape(-1,core.VOCAB),y.reshape(-1),ignore_index=-100,reduction='none').view_as(y).double()
        if not torch.isfinite(loss).all(): raise ValueError('nonfinite evaluation')
        sums=loss.sum(1).tolist(); total+=sum(sums)
        if details:
            for idx,sm,ct in zip(ids,sums,counts):
                doc[idx][0]+=sm;doc[idx][1]+=ct
    return total/n,doc

class Observer:
    def __init__(self): self.counts=defaultdict(int);self.hashes={};self.last={}
    def update(self,key,source,z):
        if key not in self.hashes:self.hashes[key]=hashlib.sha256()
        self.hashes[key].update(source.encode()+b'\0'+z.numpy().astype('<i8',copy=False).tobytes())
        self.counts[key+'|'+source]+=1
    def export(self):return {'counts':dict(self.counts),'digests':{k:v.hexdigest() for k,v in self.hashes.items()}}

def optimize(m,opt,z,observer,key,source,teacher=None):
    observer.update(key,source,z)
    ce,hidden=core.optimize_step(m,opt,z,teacher)
    if not np.isfinite(ce) or not np.isfinite(hidden): raise ValueError('nonfinite update')
    return ce,hidden

def optimizer(m):return torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=0.)

def environment():
    cpu={}
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith(('model name','cpu MHz')):
                k,v=line.split(':',1);cpu.setdefault(k.strip(),v.strip())
    except OSError:pass
    return {'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),
            'cpu':cpu,'clock_fixed':False,'threads':torch.get_num_threads(),'dtype':'float32','cuda_used':False,
            'deterministic':torch.are_deterministic_algorithms_enabled(),'torch_build':torch.__config__.show()}

def run(seed,data_dir,out):
    if seed not in SEEDS:raise ValueError('unregistered seed')
    if (out/'STARTED.json').exists():raise ValueError('run exists; explicit new audit required, no silent overwrite')
    body=Path(core.__file__).read_bytes()
    if hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()!=CORE_BLOB:
        raise ValueError('original update kernel modified')
    streams,eval_rows,manifest=read_data(data_dir)
    prepared={l:make_eval(eval_rows[l]) for l in ('ja','en')}
    ck=out/'checkpoints';ck.mkdir(parents=True,exist_ok=True)
    started={'seed':seed,'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'protocol_commit':PROTOCOL_COMMIT,
             'manifest_sha256':DATA_HASH,'core_git_blob':CORE_BLOB,'runner_sha256':file_sha(Path(__file__)),
             'environment':environment(),'target':TARGET,'interval_high':HIGH,'status':'RUNNING'}
    write_json(out/'STARTED.json',started)
    obs=Observer();phase1loss={};endpoint_rows=[]; saved={}
    def save(m,label):
        p=ck/(label+'.pt');torch.save(m.state_dict(),p);saved[label]=file_sha(p)
    def log(msg):print(json.dumps({'seed':seed,'stage':msg,'elapsed_seconds':round(time.monotonic()-t0,2)}),flush=True)
    def metric(m,arm,stage,lang):
        val,doc=evaluate(m,prepared[lang],True)
        for idx,(loss,count) in doc.items():
            row=eval_rows[lang][idx]
            endpoint_rows.append({'seed':seed,'arm':arm,'stage':stage,'language':lang,
                                  'sentence_id':row['sentence_id'],'document_id':row['document_id'],
                                  'loss_sum':loss,'target_bytes':count})
        return val
    t0=time.monotonic()
    core.seed_all(5100000+seed);base=core.Model(); op=optimizer(base)
    if sum(p.numel() for p in base.parameters())!=141056:raise ValueError('model shape mismatch')
    for st in range(400):
        z=core.batch_from(streams['ja'],16,5110000+seed*1000+st)
        optimize(base,op,z,obs,'base','ja')
    save(base,'base');log('base_complete')
    teachers={name:copy.deepcopy(base) for name in ('T_JA','T_EN','T_RU')}
    ops={name:optimizer(m) for name,m in teachers.items()}
    codes={'ja':1,'en':2,'ru':3}
    for st in range(400):
        for name,m in teachers.items():
            lang='ja' if name=='T_JA' or st%2==0 else ('en' if name=='T_EN' else 'ru')
            z=core.batch_from(streams[lang],16,5120000+seed*10000+st*10+codes[lang])
            optimize(m,ops[name],z,obs,'teacher/'+name,lang)
        if (st+1)%100==0:log('teachers_'+str(st+1))
    for name,m in teachers.items():
        save(m,name);m.eval()
        for p in m.parameters():p.requires_grad_(False)
    models={name:copy.deepcopy(base) for name in ARMS[1:]}
    teachmap={'JA_nat':None,'JA_TJA':teachers['T_JA'],'JA_TEN':teachers['T_EN'],'JA_TRU':teachers['T_RU']}
    ops={name:optimizer(m) for name,m in models.items()}
    for st in range(200):
        z=core.batch_from(streams['ja'],16,5130000+seed*1000+st)
        for name,m in models.items():
            optimize(m,ops[name],z,obs,'phase1/'+name,'ja',teachmap[name])
    for name,m in models.items():save(m,'phase1_'+name)
    log('phase1_complete')
    models={'REF_EN':base,**models}
    summary=[]
    curve_path=out/'curves.csv'
    with curve_path.open('w',newline='') as cf:
        cw=csv.DictWriter(cf,fieldnames=['seed','arm','update','english_nll']);cw.writeheader()
        for name,m0 in models.items():
            ja0=metric(m0,name,'phase1','ja')
            m=copy.deepcopy(m0);opt=optimizer(m);hit=None;hit_ja=None;hi=None;en0=None
            for st in range(601):
                if st:
                    z=core.batch_from(streams['en'],16,5140000+seed*1000+st-1)
                    optimize(m,opt,z,obs,'phase2/'+name,'en')
                en,_=evaluate(m,prepared['en'])
                if st==0:en0=en
                cw.writerow({'seed':seed,'arm':name,'update':st,'english_nll':en})
                if hi is None and en<=HIGH:hi=st
                if hit is None and en<=TARGET:
                    hit=st;hit_ja=metric(m,name,'target','ja');metric(m,name,'target','en');save(m,'target_'+name)
                if st%100==0:
                    cf.flush();log(name+'_english_'+str(st))
            ja_end=metric(m,name,'end600','ja');en_end=metric(m,name,'end600','en');save(m,'end600_'+name)
            summary.append({'seed':seed,'arm':name,'japanese_phase1_nll':ja0,'english_step0_nll':en0,
                            'target_update':hit,'high_update':hi,'japanese_at_target_nll':hit_ja,
                            'english_end600_nll':en_end,'japanese_end600_nll':ja_end,
                            'interval_valid':bool(en0>HIGH and hit is not None and hi is not None and hit>hi),
                            'interval_span':hit-hi if en0>HIGH and hit is not None and hi is not None and hit>hi else None})
            write_json(out/'summary.json',summary)
    with (out/'endpoint_documents.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(endpoint_rows[0]));w.writeheader();w.writerows(endpoint_rows)
    exported=obs.export()
    d=exported['digests'];checks={
        'phase1_actual_inputs_equal':len({d['phase1/'+n] for n in ARMS[1:]})==1,
        'phase2_actual_inputs_equal':len({d['phase2/'+n] for n in ARMS})==1,
        'counts_correct':exported['counts']=={'base|ja':400,'teacher/T_JA|ja':400,'teacher/T_EN|ja':200,'teacher/T_EN|en':200,'teacher/T_RU|ja':200,'teacher/T_RU|ru':200,
                                           **{'phase1/'+n+'|ja':200 for n in ARMS[1:]},**{'phase2/'+n+'|en':600 for n in ARMS}},
        'evaluation_byte_counts_correct':all(prepared[l][1]==sum(len(r['text'].encode())-1 for r in eval_rows[l]) for l in prepared)}
    if not all(checks.values()):raise ValueError(checks)
    write_json(out/'audit.json',{**started,**exported,'checks':checks,'optimizer_updates':sum(exported['counts'].values()),
                                 'eval_target_bytes':{l:prepared[l][1] for l in prepared},'checkpoint_sha256':saved,
                                 'curve_sha256':file_sha(curve_path),'summary_sha256':file_sha(out/'summary.json'),
                                 'endpoint_document_sha256':file_sha(out/'endpoint_documents.csv'),
                                 'elapsed_seconds':time.monotonic()-t0,'status':'COMPLETE'})
    log('COMPLETE')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',required=True,type=int);ap.add_argument('--data-dir',required=True,type=Path);ap.add_argument('--outdir',required=True,type=Path);args=ap.parse_args()
    run(args.seed,args.data_dir,args.outdir)
