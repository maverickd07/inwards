"""Frozen v3 analysis of D303 and D500, plus parallel training-label audit."""
import argparse,json,statistics
from pathlib import Path
from reuse import read,key,js,sha
from engine_io import write
from scorer_v3 import score
METRICS={'composed':('pred','Answer'),'q1':('q1_pred','A1'),'q2':('q2_pred','A2')}
T5=2.570581835636314

def moments(x):
 if len(x)!=6:raise ValueError('Exactly six training seeds required')
 if any(v is None for v in x):return {'per_seed_pp':x,'mean_pp':None,'t95_pp':None,'reason':'At least one seed has zero retained paired items; six-seed summary unavailable'}
 mean=statistics.mean(x);half=T5*statistics.stdev(x)/(6**.5)
 return {'per_seed_pp':x,'mean_pp':mean,'t95_pp':[mean-half,mean+half]}

def comparison(groups,indices,metric,policy):
 pairs=[]
 for seed in range(6):
  c,d=groups[f'context_s{seed}'],groups[f'direct_s{seed}']
  keep=[i for i in indices if policy!='exclude_pair_union' or not(c[i]['scores'][metric]['needs_review'] or d[i]['scores'][metric]['needs_review'])]
  def good(rows):return sum(rows[i]['scores'][metric]['correct'] or (policy=='all_correct' and rows[i]['scores'][metric]['needs_review']) for i in keep)
  nc,nd=good(c),good(d)
  pairs.append({'seed':seed,'n':len(keep),'context_correct':nc,'direct_correct':nd,'context_pct':100*nc/len(keep) if keep else None,'direct_pct':100*nd/len(keep) if keep else None,'difference_pp':100*(nc-nd)/len(keep) if keep else None,'context_review':sum(c[i]['scores'][metric]['needs_review'] for i in indices),'direct_review':sum(d[i]['scores'][metric]['needs_review'] for i in indices)})
 return {'pairs':pairs,**moments([p['difference_pp'] for p in pairs])}

def analyze(run,smoke=False):
 inputs=run/'inputs';out=run/'analysis_v3';out.mkdir()
 items=read(inputs/'plan/items_D_reservedC.jsonl');keys=list(map(key,items));assert len(keys)==len(set(keys))==(2 if smoke else 500)
 mask=json.loads((inputs/'primary_D303_keys.json').read_text());primary=set(map(tuple,mask['identities']))
 assert len(primary)==(2 if smoke else 303) and primary<=set(keys)
 names=['baseline']+[f'{c}_s{s}' for s in range(6) for c in ['context','direct']]
 groups={};ref={};review=[]
 for name in names:
  ref[name]=read(inputs/'reference_v3'/f'{name}.jsonl');assert list(map(key,ref[name]))==keys
  raw=read(run/name/'eval_D_reservedC.jsonl');assert list(map(key,raw))==keys
  if name=='baseline':
   assert [r['original_prediction'] for r in ref[name]]==raw
   groups[name]=ref[name]  # cached baseline scoring is reused, not recomputed
  else:
   scored=[]
   for row,item in zip(raw,items):
    scores={m:score(row[p],item['e'][g],item['category'],m) for m,(p,g) in METRICS.items()}
    record={'category':row['category'],'person':row['person'],'original_prediction':row,'scores':scores};scored.append(record)
   groups[name]=scored
  write(out/f'{name}.jsonl',groups[name])
  for row in groups[name]:
   for metric,s in row['scores'].items():
    if s['needs_review']:review.append({'model':name,'category':row['category'],'person':row['person'],'metric':metric,'score':s,'prediction':row['original_prediction'][METRICS[metric][0]]})
 report={'smoke_test':smoke,'primary_metric':'composed','primary_set':'fixed_D303 (selected from prior outputs; not untouched)','scorer':'frozen_v3','training_label_audit':json.loads((run/'training_label_audit/report.json').read_text()),'sets':{},'paired_initializations':{}}
 for seed in range(6):
  digests=[]
  for c in ['context','direct']:
   folder=run/f'{c}_s{seed}';cfg=json.loads((folder/'run_config.json').read_text())
   assert cfg['train_seed']==seed and cfg['epochs']==2 and cfg['batch']==8 and cfg['n_unique']==785 and cfg['optimizer_steps']==198 and cfg['sample_exposures']==1570
   loss=read(folder/'loss.jsonl');assert [x['step'] for x in loss]==list(range(1,199))
   assert json.loads((folder/'complete.json').read_text())['steps']==198
   assert list(map(key,read(folder/'train_examples.jsonl')))==list(map(key,read(inputs/'plan/train.jsonl')))
   assert [r['target'] for r in read(folder/'train_examples.jsonl')]==[r['target'] for r in read(inputs/f'paired_targets/{c}_targets.jsonl')]
   digests.append(json.loads((folder/'initialization.json').read_text())['lora_initial_sha256'])
  assert digests[0]==digests[1]
  report['paired_initializations'][str(seed)]=digests[0]
 assert len(set(report['paired_initializations'].values()))==6
 for label,indices in [('fixed_D303',[i for i,k in enumerate(keys) if k in primary]),('full_D500',list(range(len(keys))))]:
  report['sets'][label]={'n':len(indices),'metrics':{}}
  for metric in METRICS:
   table={policy:comparison(groups,indices,metric,policy) for policy in ['all_wrong','all_correct','exclude_pair_union']}
   old=comparison(ref,indices,metric,'all_wrong')
   table['unmatched_reference_all_wrong']=old
   table['matched_minus_unmatched_pp']=moments([a-b for a,b in zip(table['all_wrong']['per_seed_pp'],old['per_seed_pp'])])
   base=groups['baseline'];table['baseline_cached_pct']=100*sum(base[i]['scores'][metric]['correct'] for i in indices)/len(indices)
   report['sets'][label]['metrics'][metric]=table
 write(out/'review_queue.jsonl',review);js(out/'report.json',report)
 lines=['EXP12 matched785: fixed evaluation v3; 12 completed fits.','Primary: fixed303, composed, review=wrong. All intervals: six paired training seeds, df5 t95.']
 for label,info in report['sets'].items():
  lines.append(label+' n='+str(info['n']))
  for policy in ['all_wrong','all_correct','exclude_pair_union']:
   obj=info['metrics']['composed'][policy]
   lines.append(f"  {policy}: {obj['mean_pp']} pp; t95 {obj['t95_pp']}; per-seed {obj['per_seed_pp']}")
  obj=info['metrics']['composed']['matched_minus_unmatched_pp'];lines.append(f"  matched-unmatched change: {obj['mean_pp']} pp; t95 {obj['t95_pp']}")
 for label in ['selected785','different404']:
  a=report['training_label_audit'][label];lines.append(label+' revised training acceptance: '+json.dumps(a['groups'],ensure_ascii=False))
  lines.append('  sensitivity '+json.dumps(a['sensitivity'],ensure_ascii=False))
 lines.append('Frozen interpretation rules: inputs/PROTOCOL.txt. Residual teacher-accuracy differences must be disclosed. 404 analysis concerns training targets, not generalization. No automatic follow-up training.')
 with (out/'report.txt').open('x') as f:f.write('\n'.join(lines)+'\n')
 print('\n'.join(lines),flush=True)
 return report
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--smoke-test',action='store_true');a=ap.parse_args();analyze(a.run,a.smoke_test)
