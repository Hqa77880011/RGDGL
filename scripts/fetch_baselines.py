"""Fetch pinned native sources without datasets or modifying existing checkouts."""
import json
import subprocess
from pathlib import Path

root=Path(__file__).resolve().parents[1]
sources=json.loads((root/'baseline_sources.json').read_text(encoding='utf-8'))
for name,source in sources.items():
    path=root/'.external'/name
    if (path/'.git').is_dir():
        actual=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        if actual!=source['commit']:raise RuntimeError(f'{name} exists at {actual}; use the pinned revision in a clean checkout')
        continue
    path.mkdir(parents=True,exist_ok=True)
    def git(*arguments):subprocess.run(['git','-C',str(path),*arguments],check=True)
    git('init');git('remote','add','origin',source['url'])
    git('config','core.longpaths','true')
    git('sparse-checkout','init','--cone')
    if source['directories']:git('sparse-checkout','set',*source['directories'])
    git('fetch','--depth','1','--filter=blob:none','origin',source['commit'])
    git('checkout','--detach','FETCH_HEAD')
    print(name,source['commit'])
