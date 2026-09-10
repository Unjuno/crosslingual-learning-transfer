#!/usr/bin/env python3
"""Build a compact source/record archive only after full publication verification."""
from __future__ import annotations
import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(root=ROOT):
    root = Path(root).resolve()
    subprocess.run([sys.executable, str(root/'scripts/verify_publication.py')], cwd=root, check=True)
    release = json.loads((root/'publication/release.json').read_text(encoding='utf-8'))
    version = release['version']
    files = json.loads((root/'publication/files.sha256.json').read_text(encoding='utf-8'))['files']
    source_commit = os.environ.get('GITHUB_SHA')
    if not source_commit and (root/'.git').exists():
        source_commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    source_commit = source_commit or 'source-archive-no-git-metadata'
    if source_commit != 'source-archive-no-git-metadata' and re.fullmatch('[0-9a-f]{40}', source_commit) is None:
        raise ValueError('invalid source commit')
    dest=root/'dist'; dest.mkdir(exist_ok=True)
    archive=dest/f'crosslingual-learning-transfer-{version}.zip'
    prefix=f'crosslingual-learning-transfer-{version}/'
    checksums={}
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name in sorted([*files, 'publication/files.sha256.json']):
            data=(root/name).read_bytes()
            checksums[name]=hashlib.sha256(data).hexdigest()
            info=zipfile.ZipInfo(prefix+name, (2026,9,10,0,0,0)); info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644 << 16
            z.writestr(info,data)
        info=zipfile.ZipInfo(prefix+'SOURCE_COMMIT.txt',(2026,9,10,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
        z.writestr(info,(source_commit+'\n').encode())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None: raise ValueError('archive CRC failure')
        for name,sha in checksums.items():
            if hashlib.sha256(z.read(prefix+name)).hexdigest()!=sha: raise ValueError('archive hash failure')
    report=dest/'publication-verification.json'
    obj=json.loads(report.read_text(encoding='utf-8'));obj['package_source_commit']=source_commit
    obj['release_version']=version;obj['archive_files']=len(checksums)+1
    report.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    (dest/'SHA256SUMS.txt').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n'
                                           for p in (archive,report)),encoding='utf-8')
    print(json.dumps({'archive':archive.name,'source_commit':source_commit,'files':len(checksums)+1}))
    return archive


if __name__=='__main__': build()
