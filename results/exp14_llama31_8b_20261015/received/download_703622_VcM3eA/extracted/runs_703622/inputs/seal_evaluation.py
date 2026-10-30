"""Select using ONLY this Llama baseline; seal BEFORE any target training."""
import argparse,time
from common import *
from scorer_v3 import score

def scored(raw,items):
 if list(map(key,raw))!=list(map(key,items)):raise ValueError('Evaluation identities/order differ')
 return [{'category':r['category'],'person':r['person'],'original_prediction':r,
  'scores':{m:score(r[p],it['e'][g],it['category'],m) for m,(p,g) in METRICS.items()}} for r,it in zip(raw,items)]

def seal(run):
 if any((run/n).exists() for n in NAMES[1:]):raise ValueError('Cannot select after training starts')
 items=read(run/'plan/items_D_reservedC.jsonl');rows=scored(read(run/'baseline/eval_D_reservedC.jsonl'),items)
 known=[key(r) for r in rows if all(r['scores'][m]['correct'] and not r['scores'][m]['needs_review'] for m in ['q1','q2'])]
 clean=[key(r) for r in rows if key(r) in set(known) and not r['scores']['composed']['needs_review']]
 write(run/'baseline/scored_D_v3.jsonl',rows)
 js(run/'evaluation_ids.json',{'known':known,'clean':clean,'all_D':list(map(key,items)),
  'definition':'known = baseline q1 and q2 accepted, no review; clean = known AND baseline composed not flagged. No conditioning on composed correctness; no trained-model selection.',
  'baseline_composed_review_n':sum(r['scores']['composed']['needs_review'] for r in rows),
  'baseline_known_composed_wrong_n':sum(key(r) in set(known) and not r['scores']['composed']['correct'] for r in rows),
  'small_primary_n_warning':len(clean)<100})
 paths=list((run/'plan').glob('*.json*'))+list((run/'baseline').glob('*.json*'))+[run/'evaluation_ids.json']
 paths+=list((run/'paired_targets').glob('*.json*'))
 paths+=list((run/'inputs').rglob('*'))
 files={str(p.relative_to(run)):sha(p) for p in paths if p.is_file() and '__pycache__' not in p.parts}
 js(run/'EVALUATION_FREEZE.json',{'created_unix':time.time(),'before_training':True,'known_n':len(known),
  'clean_n':len(clean),'files':files,'criterion':'paired_six_seed_clean_context_minus_direct_t95_lower_pp > 0'})
 print(f'PRETRAIN SEALED: known={len(known)}, clean={len(clean)}; SHA256={sha(run/"EVALUATION_FREEZE.json")}',flush=True)
 if not clean:raise RuntimeError('Empty clean primary set: stop before training; no substitute test set or success claim')
 return clean
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);seal(ap.parse_args().run)
