#!/usr/bin/env python3
"""Prepare one fixed document-grouped PUD split; never read model outcomes.
Code: Apache-2.0. Original and extracted corpus text: upstream CC BY-SA 3.0.
"""
import argparse, hashlib, json, re, unicodedata
from pathlib import Path
from collections import defaultdict

LANGS = ('en', 'ja', 'ru')
SALT = 'R3_PUD_2026_09_09:'

def digest(data):
    return hashlib.sha256(data).hexdigest()

def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold()
                   if c.isalnum())

def grams(text):
    t = normalize(text)
    return {t[i:i+5] for i in range(max(0,len(t)-4))} or {t}

def parse(path):
    rows = {}; doc = None; sid = None
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.startswith('# newdoc id = '):
            doc = line.split(' = ',1)[1]
        elif line.startswith('# sent_id = '):
            sid = line.split(' = ',1)[1]
        elif line.startswith('# text = '):
            # Deliberately exclude # text_en and # english_text; annotations unused.
            if sid is None or doc is None or sid in rows:
                raise ValueError('missing or duplicate sentence/document identity')
            text = unicodedata.normalize('NFC',line[len('# text = '):].strip())
            if len(text.encode()) < 2:
                raise ValueError('empty sentence')
            rows[sid] = {'sentence_id':sid, 'document_id':doc, 'text':text}
    if len(rows) != 1000:
        raise ValueError(f'{path}: expected 1000 sentences, found {len(rows)}')
    return rows

def prepare(source, out):
    source_manifest = json.loads((source/'SOURCE_MANIFEST.json').read_text())
    for name, entry in source_manifest['sources'].items():
        if digest((source/name).read_bytes()) != entry['sha256']:
            raise ValueError(f'source hash mismatch: {name}')
    data = {l:parse(source/l/f'{l}_pud-ud-test.conllu') for l in LANGS}
    ids = sorted(data['en'])
    if any(sorted(data[l]) != ids for l in LANGS):
        raise ValueError('parallel IDs mismatch')
    if any(data[l][s]['document_id'] != data['en'][s]['document_id']
           for l in LANGS for s in ids):
        raise ValueError('parallel document grouping mismatch')
    docs = sorted({r['document_id'] for r in data['en'].values()})
    parent = {d:d for d in docs}
    def find(d):
        while parent[d] != d:
            parent[d] = parent[parent[d]]; d = parent[d]
        return d
    def union(a,b):
        a,b = find(a),find(b)
        if a != b:
            lo,hi=sorted((a,b)); parent[hi]=lo
    related = []
    for l in LANGS:
        vectors = [grams(data[l][s]['text']) for s in ids]
        normalized = [normalize(data[l][s]['text']) for s in ids]
        for i in range(len(ids)):
            for j in range(i):
                if data[l][ids[i]]['document_id'] == data[l][ids[j]]['document_id']:
                    continue
                a,b = vectors[i],vectors[j]
                if min(len(a),len(b)) < 0.8*max(len(a),len(b)):
                    continue
                inter = len(a & b)
                score = inter/(len(a)+len(b)-inter)
                if normalized[i] == normalized[j] or score >= .8:
                    union(data[l][ids[i]]['document_id'],data[l][ids[j]]['document_id'])
                    related.append({'lang':l,'s1':ids[i],'s2':ids[j],'jaccard':score})
    split = {}
    for d in docs:
        fold = int(digest((SALT+find(d)).encode()),16) % 10
        split[d] = 'eval' if fold==0 else ('reserve' if fold==1 else 'train')
    out.mkdir(parents=True,exist_ok=True)
    manifest={'dataset':'UD-PUD r2.16', 'source_manifest':source_manifest,
              'text_license':'CC BY-SA 3.0; see bundled original licenses and READMEs',
              'split_rule':SALT+'canonical duplicate-cluster document ID; SHA256 modulo 10: 0 eval, 1 unused reserve, 2..9 train',
              'near_duplicate_rule':'within-language NFKC/casefold/alnum char-5gram Jaccard >=0.8: unite whole documents across all three languages before splitting',
              'doc_count':len(docs),'cluster_count':len({find(d) for d in docs}),
              'near_duplicate_links':related,'files':{},'cross_split_max_jaccard':{},
              'scope':'single prespecified corpus split conditional-seed study, not a 10-fold UD parsing benchmark; no parsing annotations are used'}
    split_ids={sp:[s for s in ids if split[data['en'][s]['document_id']]==sp] for sp in ('train','eval','reserve')}
    for l in LANGS:
        for sp in ('train','eval','reserve'):
            selected=[data[l][s] for s in split_ids[sp]]
            raw=('\n'.join(json.dumps(r,ensure_ascii=False,sort_keys=True) for r in selected)+'\n').encode()
            name=f'{l}_{sp}.jsonl'; (out/name).write_bytes(raw)
            manifest['files'][name]={'sha256':digest(raw),'bytes':len(raw),'sentences':len(selected),
                                    'text_bytes':sum(len(r['text'].encode()) for r in selected),
                                    'documents':len({r['document_id'] for r in selected})}
        train=[grams(data[l][s]['text']) for s in split_ids['train']]
        test=[grams(data[l][s]['text']) for s in split_ids['eval']]
        scores=[len(a&b)/len(a|b) for a in test for b in train]
        manifest['cross_split_max_jaccard'][l]=max(scores)
        if max(scores)>=.8:
            raise ValueError('near duplicate crosses train/eval split')
    for sp in split_ids:
        if len(split_ids[sp]) < (500 if sp=='train' else 40):
            raise ValueError('data split too small; no replacement split permitted')
    manifest['japanese_only_parser_check'] = 'exact # text = field only; English companion comments and all token annotations excluded'
    manifest['document_ids'] = {sp: sorted({data['en'][s]['document_id'] for s in sl}) for sp,sl in split_ids.items()}
    (out/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
    return manifest

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('out',type=Path);args=ap.parse_args()
    m=prepare(args.source,args.out)
    print(json.dumps({k:v for k,v in m.items() if k not in ('source_manifest','document_ids','near_duplicate_links')},ensure_ascii=False,indent=2))
