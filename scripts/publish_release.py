#!/usr/bin/env python3
"""Publish the verified main snapshot using runner credentials; never move tags."""
from __future__ import annotations
import json
import os
import re
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def gh(*args):
    return subprocess.check_output(['gh',*args],cwd=ROOT,text=True,encoding='utf-8')


def main():
    repo=os.environ.get('GITHUB_REPOSITORY','')
    sha=os.environ.get('GITHUB_SHA','')
    if repo!='Unjuno/crosslingual-learning-transfer' or os.environ.get('GITHUB_REF')!='refs/heads/main':
        raise SystemExit('release permitted only from the canonical main branch')
    if re.fullmatch('[0-9a-f]{40}',sha) is None: raise SystemExit('missing exact source SHA')
    release=json.loads((ROOT/'publication/release.json').read_text(encoding='utf-8'))
    report=json.loads((ROOT/'dist/publication-verification.json').read_text(encoding='utf-8'))
    version=release['version']
    if re.fullmatch(r'v\d+\.\d+\.\d+-research(?:\.\d+)?',version) is None: raise SystemExit('invalid tag')
    if report['status']!='PASS' or report.get('package_source_commit')!=sha: raise SystemExit('unverified package')
    refs=json.loads(gh('api',f'repos/{repo}/git/matching-refs/tags/{version}'))
    exact=[r for r in refs if r['ref']==f'refs/tags/{version}']
    if exact:
        if exact[0]['object']['sha']!=sha or exact[0]['object']['type']!='commit':
            raise SystemExit('existing tag differs; refusing to move or overwrite it')
    else:
        gh('api','--method','POST',f'repos/{repo}/git/refs','-f',f'ref=refs/tags/{version}','-f',f'sha={sha}')
    pages=json.loads(gh('api','--paginate','--slurp',f'repos/{repo}/releases?per_page=100'))
    existing=[r for page in pages for r in page if r['tag_name']==version]
    if existing:
        if existing[0]['draft']: raise SystemExit('existing draft needs explicit maintainer review')
        print(json.dumps({'status':'ALREADY_PUBLISHED_NO_OVERWRITE','url':existing[0]['html_url']}))
        return
    assets=[ROOT/'dist'/f'crosslingual-learning-transfer-{version}.zip',
            ROOT/'dist/publication-verification.json',ROOT/'dist/SHA256SUMS.txt']
    output=gh('release','create',version,*[str(p) for p in assets],
              '--repo',repo,'--verify-tag','--title',release['title'],
              '--notes-file','publication/RELEASE_NOTES.md','--prerelease')
    print(output.strip())


if __name__=='__main__': main()
