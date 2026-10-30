"""Archive completed scalar outputs and frozen protocol; no weights or tensors."""
import argparse,hashlib,json,tarfile
from pathlib import Path
H=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
assert json.loads((a.run/'complete.json').read_text())['analysis_complete']
archive=a.run.parent/(a.run.name+'_review.tgz');assert not archive.exists()
with tarfile.open(archive,'w:gz') as t:
 for f in sorted(a.run.rglob('*')):
  if f.is_file():t.add(f,arcname=a.run.name+'/'+str(f.relative_to(a.run)))
 for line in (H/'SHA256SUMS').read_text().splitlines():
  _,name=line.split('  ',1);t.add(H/name,arcname='frozen_protocol/'+name)
 for name in ['SHA256SUMS','LOCAL_VALIDATION.json','SMOKE_VERIFIED.sha256']:
  t.add(H/name,arcname='frozen_protocol/'+name)
archive.with_suffix(archive.suffix+'.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  '+archive.name+'\n')
print(archive)
