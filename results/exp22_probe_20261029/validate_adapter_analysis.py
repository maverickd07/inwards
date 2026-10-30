"""Synthetic branch coverage only, NOT adapter/model experiment results."""
import os,shutil,json
from pathlib import Path
from probe_utils import *
from analyze import analyze
h=H;src=h/'local_smoke30_final';d=h/'synthetic_adapter_validation';d.mkdir(exist_ok=False)
for n in ['roster.json','split.json']:shutil.copyfile(src/n,d/n)
base=read(src/'per_item.jsonl');manifest=json.loads((src/'CACHE_MANIFEST.json').read_text());variants=['base','context_s0','context_s1','direct_s0','direct_s1'];out=[]
for v in variants:
 os.link(src/'base.npy',d/(v+'.npy'))
 for i,r in enumerate(base):
  s=dict(r,variant=v,stratum='gap' if i%2 else 'correct');s['score']=dict(r['score'],correct=bool(v.startswith('context') or not i%2));out.append(s)
js(d/'CACHE_MANIFEST.json',{v:manifest['base'] for v in variants});(d/'per_item.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in out))
analyze(d,False)
c=json.loads((d/'contrasts.json').read_text());paired=[x for x in c if x['contrast'] in ['minus_base','minus_direct'] and x.get('mean') is not None];assert paired and all(abs(x['mean'])<1e-12 for x in paired)
js(d/'VALIDATION.json',{'synthetic_only':True,'five_variant_branches_passed':True,'identical_activations_paired_deltas_zero':True,'tested_contrasts':len(paired)})
