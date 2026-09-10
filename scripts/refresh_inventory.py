#!/usr/bin/env python3
"""Deliberately refresh public file hashes after staging a reviewed change."""
import json
import subprocess
from pathlib import Path
from verify_publication import check_snapshot, check_file_policy, check_text, digest, INVENTORY
ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    check_snapshot(ROOT)  # publication maintenance must not rewrite scientific files
    paths=subprocess.check_output(['git','-C',str(ROOT),'ls-files','-z']).decode('utf-8').split('\0')
    files={}
    for name in sorted(filter(None,paths)):
        if name==INVENTORY:continue
        check_file_policy(name)
        p=ROOT/name
        check_text(name,p.read_text(encoding='utf-8'))
        files[name]=digest(p)
    (ROOT/INVENTORY).write_text(json.dumps({'schema_version':1,'scope':'Complete reviewed compact snapshot except this self-describing inventory','files':files},indent=2)+'\n',encoding='utf-8')
    print(f'Inventoried {len(files)} files. Stage the inventory and run the full verifier.')
