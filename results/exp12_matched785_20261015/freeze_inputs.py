"""One-shot allocation using frozen OLD labels; never reads revised labels or new outcomes."""
import json,random,shutil,hashlib,collections
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SOURCE=ROOT/'exp12_sources_20261014_v2/received_700454/extracted/runs_700454'
FEAS=ROOT/'exp12_accuracy_match_feasibility_20261015'
def read(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p,x):
 with p.open('x') as f: json.dump(x,f,ensure_ascii=False,indent=2)
def write(p,rows):
 with p.open('x') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def key(r):return r['category'],r['person']
if __name__=='__main__':
 out=HERE/'frozen_inputs';out.mkdir()
 classes=read(FEAS/'training_target_classes.jsonl');train=read(SOURCE/'plan/train.jsonl')
 assert list(map(key,classes))==list(map(key,train))
 rng=random.Random(20261015);selected=set()
 for cat in sorted({r['category'] for r in classes}):
  ids=[i for i,r in enumerate(classes) if r['category']==cat]
  selected.update(i for i in ids if classes[i]['old_class'] in ('both_correct','both_wrong'))
  groups=[[i for i in ids if classes[i]['old_class']==label] for label in ['context_only_correct','direct_only_correct']]
  n=min(map(len,groups))
  for group in groups:selected.update(group if len(group)==n else rng.sample(group,n))
 indices=sorted(selected);chosen=[classes[i] for i in indices]
 assert len(indices)==785 and sum(not r['targets_exactly_identical'] for r in chosen)==404
 assert all(sum(r['old'][c] for r in chosen)==359 for c in ['context','direct'])
 for cat in sorted({r['category'] for r in chosen}):assert sum(r['old']['context'] for r in chosen if r['category']==cat)==sum(r['old']['direct'] for r in chosen if r['category']==cat)
 (out/'plan').mkdir();(out/'paired_targets').mkdir()
 write(out/'plan/train.jsonl',[train[i] for i in indices])
 shutil.copyfile(SOURCE/'plan/items_D_reservedC.jsonl',out/'plan/items_D_reservedC.jsonl')
 targets_hash={}
 for c in ['context','direct']:
  rows=read(SOURCE/f'paired_targets/{c}_targets.jsonl');assert list(map(key,rows))==list(map(key,train))
  dest=out/f'paired_targets/{c}_targets.jsonl';write(dest,[rows[i] for i in indices]);targets_hash[c]=sha(dest)
 js(out/'paired_targets/complete.json',{'target_sha256':targets_hash,'cached_from_job':700454,'selected_without_regeneration':True})
 shutil.copyfile(FEAS/'primary_D303_keys.json',out/'primary_D303_keys.json')
 write(out/'selected_target_classes.jsonl',chosen)
 write(out/'all864_target_classes.jsonl',classes)
 js(out/'different404_keys.json',{'n':404,'identities':[list(key(r)) for r in chosen if not r['targets_exactly_identical']]})
 allocation={'selection_scorer':'original scoring_v2_v1; NOT revised training audit labels','selection_seed':20261015,'algorithm':'sorted categories; keep all agreement and smaller one-sided classes; random.Random(seed).sample larger class; restore original order','original_indices':indices,'identities':[list(key(r)) for r in chosen],'n':785,'different_target_n':404,'identical_target_n':381,'people':len({r['person'] for r in chosen}),'category_counts':dict(collections.Counter(r['category'] for r in chosen)),'old_correct_each':359,'epochs':2,'batch':8,'steps_per_fit':198,'exposures_per_fit':1570,'training_seeds':list(range(6)),'training_fits':12,'source_train_sha256':sha(SOURCE/'plan/train.jsonl'),'source_classes_sha256':sha(FEAS/'training_target_classes.jsonl')}
 js(out/'plan/exp12_manifest.json',allocation)
 ref=out/'reference_v3';ref.mkdir()
 old=ROOT/'exp12_rescore_v3_20261015/received/extracted_701713/runs_701713'
 for name in ['baseline']+[f'{c}_s{s}' for s in range(6) for c in ['context','direct']]:shutil.copyfile(old/f'{name}.jsonl',ref/f'{name}.jsonl')
 js(HERE/'ALLOCATION_FREEZE.json',{'frozen_before_new_training':True,'files':{str(p.relative_to(HERE)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()},'allocation':allocation})
 print(json.dumps({k:v for k,v in allocation.items() if k not in ['identities','original_indices']},indent=2))
