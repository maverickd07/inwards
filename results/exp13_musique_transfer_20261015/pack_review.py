"""Export full exp13 review; no source weight files or destructive cleanup."""
import argparse,json,tarfile
from pathlib import Path
from common import sha

def pack(run):
 complete=json.loads((run/'complete.json').read_text())
 assert complete['models_evaluated']==13 and complete['training_runs']==0
 files=[p for p in sorted(run.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
 assert not any(p.suffix in ['.safetensors','.bin','.pt'] for p in files)
 assert sum(p.stat().st_size for p in files)<250*1024**2
 log=run.parent/f'exp13-{run.name.removeprefix("runs_")}.out'
 if log.is_file():files.append(log)
 receipt=run/'REVIEW_FILES_SHA256.txt'
 with receipt.open('x') as f:
  for p in files:
   if p.is_relative_to(run):f.write(sha(p)+'  '+str(p.relative_to(run))+'\n')
 files.append(receipt)
 dest=run.parent/f'{run.name}_review.tgz'
 with tarfile.open(dest,'x:gz') as tar:
  for p in files:tar.add(p,arcname=str(p.relative_to(run.parent)),recursive=False)
 with dest.with_suffix('.tgz.sha256').open('x') as f:f.write(sha(dest)+'  '+dest.name+'\n')
 print('Review archive:',dest,dest.stat().st_size,'bytes',flush=True)
 return dest
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);pack(ap.parse_args().run)
