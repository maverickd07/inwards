"""Stdlib IO and immutable plan checks. No model imports."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
NAMES=['baseline']+[f'{c}_s{s}' for s in range(6) for c in ['context','direct']]
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def js(p,obj):
 with Path(p).open('x') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)
def write(p,rows):
 with Path(p).open('x') as f:
  for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def digest(text):return hashlib.sha256(text.encode()).hexdigest()
def verify_freeze():
 freeze=json.loads((HERE/'FREEZE.json').read_text())
 for name,value in freeze['sha256'].items():
  if sha(HERE/name)!=value:raise ValueError('Frozen file changed: '+name)
 return freeze
