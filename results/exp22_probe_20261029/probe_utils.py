import json,hashlib,unicodedata
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent

def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def js(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def verify():
 for line in (H/'SHA256SUMS').read_text().splitlines():
  digest,name=line.split('  ',1);assert sha(H/name)==digest,name

def split(rows):
 # One deterministic country-stratified PERSON split, no outcomes consulted.
 by={}
 for r in rows:
  g=r['group'];c=r['e']['birth_country'];assert g not in by or by[g]==c;by[g]=c
 train=set();test=set();rng=np.random.default_rng(22)
 for c in sorted(set(by.values())):
  gs=sorted(g for g in by if by[g]==c);gs=list(rng.permutation(gs));n=max(1,round(.3*len(gs))) if len(gs)>1 else 1
  test.update(gs[:n]);train.update(gs[n:])
 assert not train&test
 return np.array([r['group'] in train for r in rows]),{'train_groups':sorted(train),'test_groups':sorted(test),'seed':22}

def labels(rows,train,target):
 def val(r):
  if target=='country':return r['e']['birth_country']
  # reference snapshot exact category+canonical first reference; not predicted output
  return r['e']['category']+'::'+' '.join(unicodedata.normalize('NFKC',str(r['e']['Answer'][0])).casefold().split())
 groups={}
 for i,r in enumerate(rows):
  if train[i]:groups.setdefault(val(r),set()).add(r['group'])
 keep={k for k,g in groups.items() if len(g)>=5}
 yy=[val(r) if val(r) in keep else ('other' if target=='country' else r['e']['category']+'::other') for r in rows]
 return np.array(yy),{'retained':sorted(keep),'train_person_counts':{k:len(v) for k,v in groups.items()},'threshold':5,'definition':'unique training persons'}

def permuted_labels(rows,y,train,target):
 # One label per person per relation, so repeated questions cannot leak random labels.
 out=y.copy();rng=np.random.default_rng(2201 if target=='country' else 2202)
 cats=['all'] if target=='country' else sorted({r['e']['category'] for r in rows})
 for cat in cats:
  by={}
  for i,r in enumerate(rows):
   if train[i] and (cat=='all' or r['e']['category']==cat):by.setdefault(r['group'],[]).append(i)
  gs=sorted(by);vals=rng.permutation([y[by[g][0]] for g in gs])
  for g,v in zip(gs,vals):
   for i in by[g]:out[i]=v
 return out
