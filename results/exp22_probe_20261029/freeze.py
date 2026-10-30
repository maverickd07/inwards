import hashlib
from pathlib import Path
H=Path(__file__).resolve().parent
files=sorted([p for p in H.iterdir() if p.is_file() and p.suffix in ['.py','.sh','.txt','.json'] and p.name not in ['LOCAL_VALIDATION.json','UPLOAD_MANIFEST.json']]+[p for p in (H/'inputs').iterdir() if p.is_file()])
(H/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(H))+'\n' for p in files))
