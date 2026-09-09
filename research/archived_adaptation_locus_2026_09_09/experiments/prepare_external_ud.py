#!/usr/bin/env python3
"""Prepare independent UD text when network access is available.

This preparer is NOT an executed external-corpus experiment. Immutable source
commits are pinned; text/license notices must not be confused with code license.
No outcome-dependent corpus selection is allowed. Annotations never enter text.
"""
from __future__ import annotations
import argparse,hashlib,json,re,unicodedata,urllib.request
from collections import Counter,defaultdict
from pathlib import Path

SOURCES={
 'ja':('UniversalDependencies/UD_Japanese-GSD','8e5794fae6d25796437a64a08c497a5c870b7b09','ja_gsd'),
 'en':('UniversalDependencies/UD_English-EWT','4c89b5833a70aa5ed3a00bad2f23f57992cc7df8','en_ewt'),
 'ru':('UniversalDependencies/UD_Russian-GSD','27975702bf9507f0a244007f47345239a2d39c2f','ru_gsd')}

def texts(raw:str) -> list[dict]:
 out=[];current={};doc=None
 for line in raw.splitlines()+['']:
  if line.startswith('# newdoc id = '):doc=line.partition(' = ')[2]
  elif line.startswith('# sent_id = '):current['id']=line.partition(' = ')[2]
  elif line.startswith('# text = '):current['text']=line.partition(' = ')[2]
  elif not line.strip():
   if current:
    if set(current)!={'id','text'}:raise ValueError('incomplete sentence metadata; no guessed reconstruction')
    current['doc_id']=doc or current['id'];out.append(current);current={}
 if len({r['id'] for r in out})!=len(out):raise ValueError('duplicate sentence IDs')
 if not out:raise ValueError('no text metadata')
 return out

def normalized(s:str) -> str:
 s=unicodedata.normalize('NFKC',s).casefold()
 s=re.sub(r'\d+','0',s)
 return ''.join(c for c in s if c.isalnum())

def grams(s:str):return {s[i:i+5] for i in range(max(0,len(s)-4))}

def clean(train,heldout):
 exact={normalized(r['text']) for r in train};rawseen=set();kept=[];rejected=[]
 sets=[];index=defaultdict(list)
 for r in train:
  g=grams(normalized(r['text']));sets.append(g)
  for token in g:index[token].append(len(sets)-1)
 for r in heldout:
  s=normalized(r['text']);reason=None
  if not s:reason='empty_normalized'
  elif s in exact:reason='train_exact_normalized'
  elif s in rawseen:reason='heldout_duplicate_normalized'
  elif len(s)>=20:
   g=grams(s);counts=Counter(j for tok in g for j in index[tok])
   for j,intersection in counts.items():
    union=len(g)+len(sets[j])-intersection
    if union and intersection/union>=.8:reason='train_character5gram_jaccard_ge_0.8';break
  rawseen.add(s)
  if reason:rejected.append({'id':r['id'],'reason':reason})
  else:kept.append(r)
 return kept,rejected

def prepare(raw_dir:Path,out:Path,download:bool):
 out.mkdir(parents=True,exist_ok=True);raw_dir.mkdir(parents=True,exist_ok=True);manifest={'status':'DATA_ONLY','sources':{},'outputs':{},'rejected':{},'warning':'CC licenses on annotations can differ from rights in original text. Keep upstream README/license; do not relabel data as Apache code.'}
 for lang,(repo,commit,prefix) in SOURCES.items():
  splits={};manifest['sources'][lang]={'repository':repo,'commit':commit,'files':{}}
  for split in ['train','dev','test']:
   name=f'{prefix}-ud-{split}.conllu';p=raw_dir/lang/name
   if download and not p.exists():
    p.parent.mkdir(parents=True,exist_ok=True)
    url=f'https://raw.githubusercontent.com/{repo}/{commit}/{name}'
    with urllib.request.urlopen(url,timeout=60) as response:body=response.read()
    p.write_bytes(body)
   body=p.read_bytes();splits[split]=texts(body.decode('utf-8'))
   manifest['sources'][lang]['files'][name]={'sha256':hashlib.sha256(body).hexdigest(),'bytes':len(body)}
  train=[];seen=set()
  for r in splits['train']:
   k=normalized(r['text'])
   if k and k not in seen:train.append(r);seen.add(k)
  dev,dr=clean(train,splits['dev']);test,tr=clean(train+dev,splits['test'])
  for split,records in [('train',train),('dev',dev),('test',test)]:
   data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records).encode('utf-8');p=out/f'{lang}_{split}.jsonl';p.write_bytes(data)
   manifest['outputs'][p.name]={'sentences':len(records),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
  manifest['rejected'][lang]={'dev':dr,'test':tr}
 (out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
 return manifest

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--raw-dir',type=Path,required=True);ap.add_argument('--outdir',type=Path,required=True);ap.add_argument('--download',action='store_true');a=ap.parse_args();print(json.dumps(prepare(a.raw_dir,a.outdir,a.download),indent=2))
