"""Small, dependency-free integrity and file helpers."""
import hashlib,json,math,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent
CONFIG=json.loads((HERE/'CONFIG.json').read_text())
SPLITS=['A_newPerson','B_newCategory','C_bothNew','D_reservedC']
METRICS={'composed':('pred','Answer'),'q1':('q1_pred','A1'),'q2':('q2_pred','A2')}
NAMES=['baseline']+[f'{c}_s{s}' for s in range(6) for c in ['context','direct']]
def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def key(r):return r['category'],r['person']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def js(p,value):
 with Path(p).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def write(p,rows):
 with Path(p).open('x') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def verify_manifest(base,manifest):
 for name,digest in manifest.items():
  p=Path(name)
  if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe manifest path')
  if sha(base/p)!=digest:raise ValueError('Changed frozen file: '+name)
def verify_freeze():
 f=json.loads((HERE/'FREEZE.json').read_text());verify_manifest(HERE,f['files']);return f

def moments(values):
 if len(values)!=6:raise ValueError('Exactly six paired seeds required')
 if any(x is None for x in values):return {'per_seed_pp':values,'mean_pp':None,'t95_pp':None}
 m=statistics.mean(values);h=CONFIG['t975_df5']*statistics.stdev(values)/math.sqrt(6)
 return {'per_seed_pp':values,'mean_pp':m,'t95_pp':[m-h,m+h]}

def verify_seal(run):
 seal=json.loads((run/'EVALUATION_FREEZE.json').read_text());verify_manifest(run,seal['files']);return seal
