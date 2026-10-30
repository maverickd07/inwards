"""Local packaging helper; freezes protocol files without modifying prior runs."""
import hashlib
from pathlib import Path
H=Path(__file__).resolve().parent
files=[H/n for n in ['PREREGISTRATION.txt','ADAPTER_MANIFEST.json','run.py','analyze.py','scoring.py','prompts.py','run_inference.sh','submit.sh','package_results.py','test_protocol.py'] if (H/n).exists()]
files+=sorted(p for p in (H/'inputs').iterdir() if p.is_file())
(H/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(H))+'\n' for p in files))
