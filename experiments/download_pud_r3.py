#!/usr/bin/env python3
"""Download the pinned CC BY-SA 3.0 sources; verify every frozen source hash."""
import argparse, hashlib, json, urllib.request
from pathlib import Path

def download(manifest_path,out):
    manifest=json.loads(manifest_path.read_text())
    out.mkdir(parents=True,exist_ok=True)
    for name,entry in manifest['sources'].items():
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
        raw=p.read_bytes() if p.exists() else urllib.request.urlopen(entry['url'],timeout=90).read()
        if len(raw)!=entry['bytes'] or hashlib.sha256(raw).hexdigest()!=entry['sha256']:
            raise ValueError(f'frozen source mismatch: {name}')
        p.write_bytes(raw)
    (out/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();download(a.manifest,a.out)
