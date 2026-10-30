"""Validate relocated byte-identical exp21, including its original smoke seal."""
from pathlib import Path
import hashlib
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
E=H/'frozen_exp21'
for line in (E/'SHA256SUMS').read_text().splitlines():
 digest,name=line.split('  ',1);assert sha(E/name)==digest,name
prefix='results/exp21_latent_emergence_20261022/'
for line in (E/'SMOKE_VERIFIED.sha256').read_text().splitlines():
 digest,name=line.split('  ',1);assert name.startswith(prefix);assert sha(E/name[len(prefix):])==digest,name
print('exp21 original source and smoke seals verified without protocol changes')
