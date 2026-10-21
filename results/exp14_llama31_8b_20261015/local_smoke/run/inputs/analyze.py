"""Frozen v3 primary, sensitivity, retention, diagnostics and blind flips."""
import argparse,collections,math,statistics
from common import *
from seal_evaluation import scored
from analysis_exp12_source import paired,cluster_interval,export_audit,holm

def compare(groups,indices,metric,policy,left='context',right='direct'):
 pairs=[]
 for seed in range(6):
  a=groups[f'{left}_s{seed}'];b=groups['baseline' if right=='baseline' else f'{right}_s{seed}']
  keep=[i for i in indices if policy!='exclude_pair_union' or not(a[i]['scores'][metric]['needs_review'] or b[i]['scores'][metric]['needs_review'])]
  def values(rows):
   return [bool(rows[i]['scores'][metric]['correct'] and not rows[i]['scores'][metric]['needs_review']) or
    (policy=='all_correct' and rows[i]['scores'][metric]['needs_review']) for i in keep]
  x,y=values(a),values(b)
  rec=paired(x,y) if keep else {'n':0,'gain_pp':None,'mcnemar_p':None}
  pairs.append({'seed':seed,'left_pct':100*statistics.mean(x) if x else None,
    'right_pct':100*statistics.mean(y) if y else None,**rec})
 return {'pairs':pairs,**moments([p['gain_pp'] for p in pairs])}

def analyze(run,smoke=False):
 verify_freeze();seal=verify_seal(run)
 out=run/'analysis_v3';out.mkdir(exist_ok=False)
 plan={s:read(run/'plan'/f'items_{s}.jsonl') for s in SPLITS};train=read(run/'plan/train.jsonl')
 raw={n:{s:read(run/n/f'eval_{s}.jsonl') for s in SPLITS} for n in NAMES}
 initial={};configs={}
 for seed in range(6):
  digests=[]
  for c in ['context','direct']:
   name=f'{c}_s{seed}';p=run/name;cfg=json.loads((p/'run_config.json').read_text());configs[name]=cfg
   steps=2*math.ceil(len(train)/8)
   assert (cfg['train_seed'],cfg['epochs'],cfg['batch'],cfg['n_unique'],cfg['optimizer_steps'],cfg['sample_exposures'])==(seed,2,8,len(train),steps,2*len(train))
   assert [r['step'] for r in read(p/'loss.jsonl')]==list(range(1,steps+1))
   done=json.loads((p/'complete.json').read_text());assert done['steps']==steps and done['adapter_sha256']==sha(p/'adapter/adapter_model.safetensors')
   actual=read(p/'train_examples.jsonl');targets=read(run/'paired_targets'/f'{c}_targets.jsonl')
   assert list(map(key,actual))==list(map(key,train))==list(map(key,targets))
   assert [r['target'] for r in actual]==[r['target'] for r in targets]
   assert cfg['targets_sha256']==sha(run/'paired_targets'/f'{c}_targets.jsonl')
   digests.append(json.loads((p/'initialization.json').read_text())['lora_initial_sha256'])
  assert digests[0]==digests[1];initial[str(seed)]=digests[0]
 assert len(set(initial.values()))==6
 for n in NAMES:
  for s in SPLITS:assert list(map(key,raw[n][s]))==list(map(key,plan[s]))
 groups={n:scored(raw[n]['D_reservedC'],plan['D_reservedC']) for n in NAMES}
 for n,rows in groups.items():write(out/f'{n}.jsonl',rows)
 ids=json.loads((run/'evaluation_ids.json').read_text());keys=list(map(key,plan['D_reservedC']))
 report={'primary_set':'clean','success_rule':CONFIG['success'],'scorer':'unchanged_v3',
  'smoke_test':smoke,'evaluation_freeze_sha256':sha(run/'EVALUATION_FREEZE.json'),
  'sets':{},'review_counts':{},'paired_initializations':initial,'training_configurations':configs,
  'seed_CI_scope':'training-seed variability conditional on one cached target pair and selected evaluation questions',
  'small_primary_n_warning':len(ids['clean'])<100}
 for label in ['clean','known','all_D']:
  selected=set(map(tuple,ids[label]));indices=[i for i,k in enumerate(keys) if k in selected]
  tables={}
  for metric in METRICS:
   tables[metric]={policy:compare(groups,indices,metric,policy) for policy in ['all_wrong','all_correct','exclude_pair_union']}
   for c in ['context','direct']:tables[metric][c+'_minus_base']=compare(groups,indices,metric,'all_wrong',c,'baseline')
  if indices:
   diff=[[int(groups[f'context_s{s}'][i]['scores']['composed']['correct'])-int(groups[f'direct_s{s}'][i]['scores']['composed']['correct']) for i in indices] for s in range(6)]
   tables['person_cluster']=cluster_interval(diff,[keys[i][1] for i in indices])
   tables['country_cluster']=cluster_interval(diff,[plan['D_reservedC'][i]['e']['A1'][0] for i in indices])
  report['sets'][label]={'n':len(indices),'metrics':tables,
   'baseline_composed_pct':100*statistics.mean(groups['baseline'][i]['scores']['composed']['correct'] for i in indices) if indices else None}
  report['review_counts'][label]={n:sum(rows[i]['scores']['composed']['needs_review'] for i in indices) for n,rows in groups.items()}
  report['review_counts'][label]['paired_overlap']=[]
  for seed in range(6):
   x={i for i in indices if groups[f'context_s{seed}'][i]['scores']['composed']['needs_review']}
   y={i for i in indices if groups[f'direct_s{seed}'][i]['scores']['composed']['needs_review']}
   report['review_counts'][label]['paired_overlap'].append({'seed':seed,'intersection':len(x&y),'union':len(x|y),'jaccard':len(x&y)/len(x|y) if x|y else None})
 primary=report['sets']['clean']['metrics']['composed']['all_wrong']
 ci=primary['t95_pp']
 report['replication_status']='positive_replication' if ci and ci[0]>0 else ('opposite_direction' if ci and ci[1]<0 else 'inconclusive')
 if smoke:report['replication_status']='software_smoke_only'
 # Retain exp12 multiplicity accounting as secondary checks: 3 contrasts x 6 seeds.
 for metric_group,metrics in [('composed_family',['composed']),('retention_family',['q1','q2'])]:
  records=[]
  for m in metrics:
   for contrast in ['all_wrong','context_minus_base','direct_minus_base']:
    records.extend(report['sets']['clean']['metrics'][m][contrast]['pairs'])
  if all(r['mcnemar_p'] is not None for r in records):holm(records)
  report[metric_group+'_tests']=len(records)
 # Export v3 flips against the baseline; raw predictions stay intact.
 auditbase={'D_reservedC':[]};auditfits={}
 for n,rows in groups.items():
  rs=[{**r['original_prediction'],'composed_ok':r['scores']['composed']['correct']} for r in rows]
  if n=='baseline':auditbase['D_reservedC']=rs
  else:
   c,s=n.split('_s');auditfits[c,int(s),'D_reservedC']=rs
 report['flip_audit']=export_audit(out,{'D_reservedC':plan['D_reservedC']},auditbase,auditfits)
 # A/B/C continuity: v3 B/C only; A includes unsupported capitals/multi-code gold.
 auxiliary={}
 for split in SPLITS[:3]:
  if split!='A_newPerson':
   g={n:scored(raw[n][split],plan[split]) for n in NAMES}
   for n,rs in g.items():write(out/f'{n}_{split}_v3.jsonl',rs)
   auxiliary[split]={'scorer':'v3','exploratory':True,'metrics':{m:{p:compare(g,list(range(len(plan[split]))),m,p) for p in ['all_wrong','all_correct','exclude_pair_union']} for m in METRICS}}
  else:
   auxiliary[split]={'scorer':'original_signed_floor_only','reason':'Frozen v3 does not cover capitals and some calling-code gold; no scorer extension.',
    'composed_context_minus_direct':moments([100*statistics.mean(int(x['composed_ok'])-int(y['composed_ok']) for x,y in zip(raw[f'context_s{s}'][split],raw[f'direct_s{s}'][split])) for s in range(6)])}
 report['auxiliary']=auxiliary
 teachers=read(run/'paired_targets/paired_diagnostics.jsonl');assert list(map(key,teachers))==list(map(key,train))
 report['teacher_advantage']={'scorer':'original_signed_floor_diagnostic_only',
  'context_pct':100*statistics.mean(r['context_ok'] for r in teachers),'direct_pct':100*statistics.mean(r['direct_ok'] for r in teachers),
  **paired([r['context_ok'] for r in teachers],[r['direct_ok'] for r in teachers]),
  'identical_targets_n':sum(r['context_target']==r['direct_target'] for r in teachers),
  'person_cluster':cluster_interval([[int(r['context_ok'])-int(r['direct_ok']) for r in teachers]],[r['person'] for r in teachers])}
 report['perplexity']={};report['same_answer_prompt_effect']={}
 for c in ['context','direct']:
  for p in ['context','direct']:
   name=c+'_under_'+p;valid=[r[name] for r in teachers if r[name]['n_tokens']];tokens=sum(r['n_tokens'] for r in valid)
   nll=sum(r['sum_nll'] for r in valid)/tokens if tokens else None
   report['perplexity'][name]={'answer_tokens':tokens,'empty_n':len(teachers)-len(valid),'corpus_nll':nll,'corpus_ppl':math.exp(nll) if nll is not None and nll<700 else None,'mean_item_nll':statistics.mean(r['mean_nll'] for r in valid) if valid else None}
  ds=[r[c+'_under_context']['mean_nll']-r[c+'_under_direct']['mean_nll'] for r in teachers if r[c+'_under_context']['n_tokens']]
  report['same_answer_prompt_effect'][c]={'n':len(ds),'context_minus_direct_nll':statistics.mean(ds) if ds else None}
 report['generation_caps']={n:sum(r['hit_token_cap'] for r in read(run/n/'generation_trace.jsonl')) for n in NAMES+['paired_targets']}
 write(out/'review_queue.jsonl',[{'model':n,**r} for n,rows in groups.items() for r in rows if any(s['needs_review'] for s in r['scores'].values())])
 js(out/'report.json',report)
 lines=['EXP14 Llama replication: '+report['replication_status'],f'Primary pretraining-sealed clean n={len(ids["clean"])}',
  'Six paired seeds; t95 df5; intervals do not include question-sampling uncertainty.']
 for label,info in report['sets'].items():
  for policy in ['all_wrong','all_correct','exclude_pair_union']:
   v=info['metrics']['composed'][policy];lines.append(f'{label} ({info["n"]}) {policy}: {v["mean_pp"]} pp; t95={v["t95_pp"]}; seeds={v["per_seed_pp"]}')
 lines+=['No automatic content/format interpretation; blinded audit pending.','Positive replication supports cross-family efficacy, not a unique knowledge-use mechanism.']
 with (out/'report.txt').open('x') as f:f.write('\n'.join(lines)+'\n')
 print('\n'.join(lines),flush=True);return report
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--smoke-test',action='store_true');a=ap.parse_args();analyze(a.run,a.smoke_test)
