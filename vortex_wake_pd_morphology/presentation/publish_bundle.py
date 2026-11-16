"""Copy the portable bundle into a dedicated directory of the user's repository."""
import argparse
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('repository');a=p.parse_args()
repo=Path(a.repository).resolve()
if not (repo/'.git').is_dir():raise RuntimeError('Destination must be a Git checkout')
dest=repo/'vortex_wake_pd_morphology'
excluded={'.git','__pycache__','.venv'}
count=0
for src in ROOT.rglob('*'):
    rel=src.relative_to(ROOT)
    if any(part in excluded for part in rel.parts) or rel.as_posix().startswith('presentation/research-update/'):
        continue
    if src.is_file():
        target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target);count+=1
print(f'Copied {count} files into {dest}')
