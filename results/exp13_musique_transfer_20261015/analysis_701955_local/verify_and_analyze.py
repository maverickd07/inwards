"""Read-only audit of downloaded exp13; no inference or scoring-rule changes."""
from pathlib import Path
import json,hashlib,re,string,collections,math
import numpy as np
from scipy.stats import t
ROOT=Path(__file__).resolve().parent.parent
RUN=ROOT/'received/extracted_701955/runs_701955'
OUT=Path(__file__).resolve().parent
NAMES=['baseline']+[f'{c}_s{s}' for s in range(6) for c in ['context','direct']]
def read(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,value):
 with (OUT/name).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def norm(s):return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s.lower().translate(str.maketrans('','',string.punctuation))).split())
def pairscore(p,a):
 p,a=norm(p),norm(a);pt,at=p.split(),a.split();em=int(p==a)
 common=sum((collections.Counter(pt)&collections.Counter(at)).values())
 f1=(2*common/(len(pt)+len(at))) if pt and at else float(pt==at)
 return em,f1
def scores(p,aliases):
 vals=[pairscore(p,a) for a in aliases];return {'em':max(v[0] for v in vals),'f1':max(v[1] for v in vals)}
def close(a,b):
 assert set(a)==set(b)
 for k in a:assert math.isclose(a[k],b[k],rel_tol=1e-12,abs_tol=1e-12),(a,b)
freeze=json.loads((ROOT/'FREEZE.json').read_text());code=json.loads((RUN/'code_hashes.json').read_text())
assert sha(RUN/'inputs/FREEZE.json')==sha(ROOT/'FREEZE.json')
for name,digest in freeze['sha256'].items():
 assert sha(ROOT/name)==digest,name
 if name.startswith('data/'):assert sha(RUN/'inputs'/Path(name).name)==digest
 elif name.startswith('upstream/metrics/'):assert sha(RUN/'inputs/official_metrics'/Path(name).name)==digest
 elif name in code:assert code[name]==digest,name
for p in (RUN/'inputs').iterdir():
 if p.is_file() and (ROOT/p.name).is_file():assert sha(p)==sha(ROOT/p.name),p
manifest=json.loads((ROOT/'source_manifest.json').read_text());receipt=json.loads((RUN/'source_receipt.json').read_text())
for name,digest in manifest['files'].items():assert receipt['sha256'][name]==digest
items=read(RUN/'inputs/items.jsonl');queries=read(RUN/'inputs/queries.jsonl')
assert len(items)==2385 and len({r['id'] for r in items})==2385
assert len(queries)==5176 and len({r['query_id'] for r in queries})==5176
assert collections.Counter(r['n_hops'] for r in items)=={2:1229,3:751,4:405}
assert all(set(q)=={'query_id','question'} and hashlib.sha256(q['question'].encode()).hexdigest()==q['query_id'] for q in queries)
qh=sha(RUN/'inputs/queries.jsonl');cfg=sha(RUN/'inputs/inference_config.json')
groups={};preds={};scored_count=hop_count=0
for name in NAMES:
 folder=RUN/name;complete=json.loads((folder/'complete.json').read_text());runtime=json.loads((folder/'runtime.json').read_text())
 assert complete['name']==name and complete['queries']==5176 and not complete['smoke_test']
 assert complete['queries_sha256']==qh and complete['config_sha256']==cfg
 assert complete['predictions_sha256']==sha(folder/'predictions.jsonl')
 if name!='baseline':assert complete['adapter_sha256']==receipt['sha256'][name+'/adapter/adapter_model.safetensors']
 assert runtime['versions']==json.loads((RUN/'baseline/runtime.json').read_text())['versions'] and not runtime['smoke_test']
 rows=read(folder/'predictions.jsonl');assert [r['query_id'] for r in rows]==[q['query_id'] for q in queries]
 assert [r['question'] for r in rows]==[q['question'] for q in queries]
 by={r['query_id']:r for r in rows};preds[name]=by
 saved=read(folder/'scored_items.jsonl');assert [r['id'] for r in saved]==[r['id'] for r in items]
 assert json.loads((folder/'scoring_complete.json').read_text())['scored_sha256']==sha(folder/'scored_items.jsonl')
 verified=[]
 for it,rs in zip(items,saved):
  raw=by[it['query_id']];assert raw['question']==it['question'] and raw['prediction']==rs['prediction']
  assert rs['hit_token_limit']==raw['hit_token_limit'] and rs['contains_think_markup']==raw['contains_think_markup']
  close(scores(raw['prediction'],it['answers']),rs['scores']);scored_count+=1
  assert rs['n_hops']==it['n_hops'] and len(rs['hops'])==len(it['hops'])==it['n_hops']
  for hop,h in zip(it['hops'],rs['hops']):
   hp=by[hop['query_id']];assert hp['question']==hop['question'] and hp['prediction']==h['prediction'] and h['query_id']==hop['query_id']
   close(scores(hp['prediction'],hop['answers']),h['scores'])
   assert scores(hp['prediction'],[hop['canonical_answer']])['em']==h['canonical_em'];hop_count+=1
  assert rs['all_oracle_hops_em']==all(h['scores']['em']==1 for h in rs['hops'])
  verified.append(rs)
 groups[name]=verified
base=groups['baseline'];known=[i for i,r in enumerate(base) if r['all_oracle_hops_em']];gap=[i for i in known if base[i]['scores']['em']==0]
strata=json.loads((RUN/'baseline/baseline_strata.json').read_text())
assert strata['all_oracle_hops_correct_ids']==[items[i]['id'] for i in known]
assert strata['all_hops_correct_but_composed_wrong_ids']==[items[i]['id'] for i in gap]
assert strata['items_sha256']==sha(RUN/'inputs/items.jsonl')
report=json.loads((RUN/'analysis/report.json').read_text());out={}
def stats(values):
 mean=float(np.mean(values));half=float(t.ppf(.975,5)*np.std(values,ddof=1)/np.sqrt(6))
 return {'per_seed_pp':values,'mean_pp':mean,'t95_pp':[mean-half,mean+half]}
def comparison(ids,metric):
 if not ids:return None
 avg=lambda n:100*float(np.mean([groups[n][i]['scores'][metric] for i in ids]))
 c=[avg(f'context_s{s}') for s in range(6)];d=[avg(f'direct_s{s}') for s in range(6)];b=avg('baseline')
 return {'n':len(ids),'baseline_pct':b,'context_pct':float(np.mean(c)),'direct_pct':float(np.mean(d)),
 'context_minus_direct':stats([x-y for x,y in zip(c,d)]),'context_minus_base':stats([x-b for x in c]),'direct_minus_base':stats([x-b for x in d])}
def match(r,s):
 for k in ['n','baseline_pct','context_pct','direct_pct']:assert math.isclose(r[k],s[k],abs_tol=1e-10)
 for k in ['context_minus_direct','context_minus_base','direct_minus_base']:
  for field in ['per_seed_pp','mean_pp','t95_pp']:np.testing.assert_allclose(r[k][field],s[k][field],atol=1e-10)
for hop,label in [(2,'2hop'),(3,'3hop'),(4,'4hop'),(None,'all')]:
 ids=[i for i,it in enumerate(items) if hop is None or it['n_hops']==hop]
 out[label]={m:comparison(ids,m) for m in ['em','f1']}
 for m in ['em','f1']:match(out[label][m],report['groups'][label][m])
 for name,idx in [('fixed_baseline_all_hops_correct',known),('fixed_baseline_gap',gap)]:
  sub=[i for i in ids if i in set(idx)];calc=comparison(sub,'em');out[label][name]=calc
  if calc:match(calc,report['diagnostics'][label][name])
# Examine concentration in the predeclared known two-hop diagnostic, not a new endpoint.
known2=[i for i in known if items[i]['n_hops']==2];gap2=[i for i in gap if items[i]['n_hops']==2]
changes=[]
for i in known2:
 c=[groups[f'context_s{s}'][i]['scores']['em'] for s in range(6)]
 d=[groups[f'direct_s{s}'][i]['scores']['em'] for s in range(6)]
 if any(x!=base[i]['scores']['em'] for x in c+d):
  changes.append({'id':items[i]['id'],'question':items[i]['question'],'references':items[i]['answers'],
   'baseline_prediction':base[i]['prediction'],'baseline_em':base[i]['scores']['em'],
   'context_em':c,'direct_em':d,'context_predictions':[groups[f'context_s{s}'][i]['prediction'] for s in range(6)],
   'direct_predictions':[groups[f'direct_s{s}'][i]['prediction'] for s in range(6)]})
summary={'known_2hop_n':len(known2),'gap_2hop_n':len(gap2),
 'context_gap_repairs_per_seed':[sum(groups[f'context_s{s}'][i]['scores']['em'] for i in gap2) for s in range(6)],
 'direct_gap_repairs_per_seed':[sum(groups[f'direct_s{s}'][i]['scores']['em'] for i in gap2) for s in range(6)],
 'context_unique_gap_repairs':sum(any(groups[f'context_s{s}'][i]['scores']['em'] for s in range(6)) for i in gap2),
 'known2_changed_items':len(changes)}
health={n:{'unique_queries':5176,'empty_predictions':sum(not r['prediction'].strip() for r in preds[n].values()),
 'composed_cap_n':sum(r['hit_token_limit'] for r in groups[n]),'composed_think_n':sum(r['contains_think_markup'] for r in groups[n])} for n in NAMES}
for n in NAMES:
 assert health[n]['composed_cap_n']==report['diagnostics']['all']['models'][n]['composed_truncated']
 assert health[n]['composed_think_n']==report['diagnostics']['all']['models'][n]['composed_think_markup']
complete=json.loads((RUN/'complete.json').read_text());assert complete['models_evaluated']==13 and complete['training_runs']==0 and complete['source_unchanged'] and not complete['smoke_test']
save('independent_analysis.json',{'groups':out,'known_subset_concentration':summary,'generation_health':health})
save('known_twohop_changed_items.json',changes)
save('verification.json',{'passed':True,'archive_sha256':sha(ROOT/'received/download_IGA8gc/runs_701955_review.tgz'),
 'internal_files_verified':101,'frozen_inputs_and_code_verified':True,'adapter_receipts_verified':12,
 'model_outputs':13,'unique_predictions':13*5176,'composed_scores_verified':scored_count,'hop_scores_verified':hop_count,
 'six_seed_CI_independent_scipy_verified':True,'runtime_seconds':complete['seconds'],
 'limitations':['No adapter weights downloaded; checked production hash receipts against frozen manifest','No sacct record in archive; completion markers and successful log verified','No inference rerun or scoring rule changes']})
print(json.dumps(summary,indent=2));print(json.dumps(changes,indent=2,ensure_ascii=False))
print('VERIFIED',scored_count,'composed and',hop_count,'hop scores; 13 models; all reported contrasts agree.')
