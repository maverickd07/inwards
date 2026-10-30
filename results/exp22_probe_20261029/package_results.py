"""Compact review archive, activations retained once on HPC and checksummed."""
import argparse,json,tarfile
from probe_utils import H,sha
p=argparse.ArgumentParser();p.add_argument('--run',type=__import__('pathlib').Path,required=True);a=p.parse_args()
assert json.loads((a.run/'complete.json').read_text())['analysis_complete']
dest=a.run.parent/(a.run.name+'_review.tgz');assert not dest.exists()
with tarfile.open(dest,'w:gz') as t:
 for f in sorted(a.run.rglob('*')):
  if f.is_file() and f.suffix!='.npy':t.add(f,arcname=a.run.name+'/'+str(f.relative_to(a.run)))
 for line in (H/'SHA256SUMS').read_text().splitlines():
  _,name=line.split('  ',1);t.add(H/name,arcname='exp22_protocol/'+name)
 for name in ['SHA256SUMS','LOCAL_VALIDATION.json']:t.add(H/name,arcname='exp22_protocol/'+name)
dest.with_suffix('.tgz.sha256').write_text(sha(dest)+'  '+dest.name+'\n')
print(dest)
