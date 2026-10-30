"""Bundle exp22 and byte-identical frozen exp21; never include model/cache weights."""
import shutil,tarfile,json
from pathlib import Path
from probe_utils import H,sha,verify
E=H.parent/'exp21_latent_emergence_20261022'
F=H/'frozen_exp21';F.mkdir(exist_ok=False)
for line in (E/'SHA256SUMS').read_text().splitlines():
 digest,name=line.split('  ',1);assert sha(E/name)==digest,name
 dest=F/name;dest.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(E/name,dest)
for name in ['SHA256SUMS','SMOKE_VERIFIED.sha256','LOCAL_VALIDATION.json']:shutil.copyfile(E/name,F/name)
for line in (E/'SMOKE_VERIFIED.sha256').read_text().splitlines():
 digest,name=line.split('  ',1);rel=name.split('results/exp21_latent_emergence_20261022/',1)[1];assert sha(E/rel)==digest
 dest=F/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(E/rel,dest)
js={'original_exp21_package_sha256':sha(E/'exp21_upload.tgz'),'exp21_files_byte_identical':True,'formal_hpc_submitted':False}
(H/'EXP21_PRESERVATION.json').write_text(json.dumps(js,indent=2)+'\n')
# Refresh manifest only after final validated sources are fixed; include frozen snapshot.
files=sorted([p for p in H.iterdir() if p.is_file() and p.suffix in ['.py','.sh','.txt','.json'] and p.name not in ['UPLOAD_MANIFEST.json'] and '_v1' not in p.stem]+[p for sub in ['inputs','frozen_exp21'] for p in (H/sub).rglob('*') if p.is_file()])
(H/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(H))+'\n' for p in files))
verify();dest=H/'exp21_exp22_upload.tgz';assert not dest.exists()
with tarfile.open(dest,'w:gz') as t:
 for p in files+[H/'SHA256SUMS']:t.add(p,arcname='exp22_probe_20261029/'+str(p.relative_to(H)))
dest.with_suffix('.tgz.sha256').write_text(sha(dest)+'  '+dest.name+'\n');print(dest, dest.stat().st_size)
