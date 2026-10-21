"""Review archive excludes model/adapter weights; all originals remain on HPC."""
import argparse,tarfile
from common import *
def pack(run):
 done=json.loads((run/'complete.json').read_text());assert done['training_runs']==12
 files=[p for p in sorted(run.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and 'adapter' not in p.relative_to(run).parts]
 if any(p.suffix in ['.safetensors','.pt','.bin'] for p in files):raise ValueError('Unexpected weight in review files')
 if sum(p.stat().st_size for p in files)>300*1024**2:raise ValueError('Review package exceeds expected size')
 receipt=run/'REVIEW_FILES_SHA256.txt'
 with receipt.open('x') as f:
  for p in files:f.write(sha(p)+'  '+str(p.relative_to(run))+'\n')
 files.append(receipt)
 log=run.parent/f'exp14-{run.name.removeprefix("runs_")}.out'
 if log.is_file():files.append(log)
 dest=run.parent/f'{run.name}_review.tgz'
 with tarfile.open(dest,'x:gz') as tar:
  for p in files:tar.add(p,arcname=str(p.relative_to(run.parent)),recursive=False)
 with dest.with_suffix('.tgz.sha256').open('x') as f:f.write(sha(dest)+'  '+dest.name+'\n')
 print('Review archive:',dest,flush=True);return dest
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);pack(ap.parse_args().run)
