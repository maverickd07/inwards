"""Read-only scientific verification of job701792; no torch, inference, or rule changes."""
import collections,hashlib,json,math,sys,tarfile
from pathlib import Path
import numpy as np
from scipy.stats import t,binomtest
OUT=Path(__file__).resolve().parent;PACKAGE=OUT.parent
RUN=PACKAGE/'received/extracted_701792/runs_701792'
sys.path.insert(0,str(PACKAGE))
from scorer_v3 import score
from training_label_scorer import score as training_score
from scoring_v2_v1 import is_correct,old_is_correct

def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def js(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(r):return r['category'],r['person']
def equal(a,b):
 if isinstance(a,dict):
  assert set(a)==set(b),(set(a),set(b))
  for k in a:equal(a[k],b[k])
 elif isinstance(a,list):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 elif isinstance(a,float):assert math.isclose(a,b,abs_tol=1e-10,rel_tol=1e-12),(a,b)
 else:assert a==b,(a,b)
def moments(vals):
 x=np.array(vals,dtype=float);mean=float(x.mean());half=float(t.ppf(.975,5)*x.std(ddof=1)/math.sqrt(6))
 return {'mean_pp':mean,'t95_pp':[mean-half,mean+half],'per_seed_pp':list(map(float,x))}

def main():
 inputs=RUN/'inputs';report=json.loads((RUN/'analysis_v3/report.json').read_text())
 before={str(p):sha(p) for p in RUN.rglob('*') if p.is_file()}
 archive=PACKAGE/'received/runs_701792_review.tgz'
 assert sha(archive)==archive.with_suffix('.tgz.sha256').read_text().split()[0]
 upload={}
 with tarfile.open(PACKAGE/'exp12_matched785_upload.tgz') as tar:
  for line in tar.extractfile('SHA256SUMS').read().decode().splitlines():
   digest,name=line.split('  ',1);upload[name]=digest
 recorded=json.loads((RUN/'source_hashes.json').read_text())
 for name,digest in recorded.items():
  if name not in upload:
   assert name in ['quota_login.txt','quota_ticket.json','submitted_job.txt'],name
   continue
  assert upload[name]==digest and sha(PACKAGE/name)==digest,name
 for name in ['ALLOCATION_FREEZE.json','DESIGN_FREEZE.json','PROTOCOL.txt','training_label_policy.json','training_label_scorer.py','scorer_v3.py','SCORER_FREEZE.json','aliases.json','reference_catalog.json']:
  if (inputs/name).exists():assert sha(inputs/name)==sha(PACKAGE/name)
  else:assert name.endswith('.py') and recorded[name]==sha(PACKAGE/name)
 frozen=json.loads((PACKAGE/'ALLOCATION_FREEZE.json').read_text())
 for rel,digest in frozen['files'].items():assert sha(inputs/Path(rel).relative_to('frozen_inputs'))==digest,rel
 source=json.loads((PACKAGE/'source_700454_manifest.json').read_text())
 for rel,digest in source['files'].items():assert sha(RUN/'source_700454'/rel)==digest
 original=PACKAGE.parent/'exp12_sources_20261014_v2/received_700454/extracted/runs_700454'
 extension=PACKAGE.parent/'exp12_seed345_20261014/received/extracted_700972/runs_700972'
 items=read(inputs/'plan/items_D_reservedC.jsonl');keys=list(map(key,items));assert len(keys)==len(set(keys))==500
 primary=set(map(tuple,json.loads((inputs/'primary_D303_keys.json').read_text())['identities']));assert len(primary)==303
 train=read(inputs/'plan/train.jsonl');assert len(train)==len(set(map(key,train)))==785
 assert not {r['person'] for r in train}&{r['person'] for r in items}
 assert not {r['family'] for r in train}&{r['family'] for r in items}
 names=['baseline']+[f'{c}_s{s}' for s in range(6) for c in ['context','direct']]
 metrics={'composed':('pred','Answer'),'q1':('q1_pred','A1'),'q2':('q2_pred','A2')}
 scored={};raw={};review=[];cfgs={};nlabels=0
 for name in names:
  raw[name]=read(RUN/name/'eval_D_reservedC.jsonl');scored[name]=read(RUN/'analysis_v3'/f'{name}.jsonl')
  assert list(map(key,raw[name]))==keys==list(map(key,scored[name]))
  for r,s,item in zip(raw[name],scored[name],items):
   assert r==s['original_prediction']
   for metric,(pred,gold) in metrics.items():
    calculated=score(r[pred],item['e'][gold],item['category'],metric)
    for k,v in calculated.items():assert s['scores'][metric][k]==v,(name,key(r),metric)
    assert bool(is_correct(r[pred],item['e'][gold]))==r[metric+'_ok']
    if calculated['needs_review']:review.append({'model':name,'category':r['category'],'person':r['person'],'metric':metric,'score':s['scores'][metric],'prediction':r[pred]})
    nlabels+=1
   assert bool(old_is_correct(r['pred'],item['e']['Answer']))==r['legacy_composed_ok']
  if name=='baseline':assert raw[name]==read(original/'baseline/eval_D_reservedC.jsonl');continue
  c,seed=name.split('_s');seed=int(seed);folder=RUN/name
  cfg=json.loads((folder/'run_config.json').read_text());cfgs[name]=cfg
  for k,v in {'condition':c,'train_seed':seed,'epochs':2,'batch':8,'lr':.0001,'rank':16,'n_unique':785,'sample_exposures':1570,'optimizer_steps':198,'init_adapter_sha256':None,'optimizer_reset':True,'scheduler_reset':True,'smoke_test':False,'model':'models/Qwen3-14B'}.items():assert cfg[k]==v,(name,k)
  assert cfg['targets_sha256']==sha(inputs/f'paired_targets/{c}_targets.jsonl') and cfg['plan_sha256']==sha(inputs/'plan/exp12_manifest.json')
  losses=read(folder/'loss.jsonl');assert len(losses)==198 and [r['step'] for r in losses]==list(range(1,199)) and [r['epoch'] for r in losses]==[1]*99+[2]*99
  assert all(math.isfinite(r['loss']) for r in losses)
  assert json.loads((folder/'complete.json').read_text())['steps']==198
  examples=read(folder/'train_examples.jsonl');targets=read(inputs/f'paired_targets/{c}_targets.jsonl')
  assert list(map(key,examples))==list(map(key,train));assert [r['target'] for r in examples]==[r['target'] for r in targets]
  rt=json.loads((folder/'runtime.json').read_text());oldrt=json.loads((original/'context_s0/runtime.json').read_text())
  for k in ['python','versions','cuda','gpu']:assert rt[k]==oldrt[k]
  oldrun=original if seed<3 else extension
  assert json.loads((folder/'initialization.json').read_text())==json.loads((oldrun/name/'initialization.json').read_text())
 equal(review,read(RUN/'analysis_v3/review_queue.jsonl'))
 for s in range(6):assert json.loads((RUN/f'context_s{s}/initialization.json').read_text())==json.loads((RUN/f'direct_s{s}/initialization.json').read_text())
 summary={'job':701792,'sets':{},'training_labels':{},'per_run_tokens':{name:{k:v for k,v in cfg.items() if k in ['trained_target_tokens_including_eos','presented_prompt_tokens','max_sequence_tokens']} for name,cfg in cfgs.items()}}
 for setname,ids in [('fixed_D303',[i for i,k in enumerate(keys) if k in primary]),('full_D500',list(range(500)))]:
  summary['sets'][setname]={}
  for metric in metrics:
   table={};old_diffs=report['sets'][setname]['metrics'][metric]['unmatched_reference_all_wrong']['per_seed_pp']
   for policy in ['all_wrong','all_correct','exclude_pair_union']:
    diffs=[];pairs=[]
    for seed in range(6):
     arrays=[np.array([scored[f'{c}_s{seed}'][i]['scores'][metric]['correct'] for i in ids],dtype=bool) for c in ['context','direct']]
     flags=[np.array([scored[f'{c}_s{seed}'][i]['scores'][metric]['needs_review'] for i in ids],dtype=bool) for c in ['context','direct']]
     keep=~(flags[0]|flags[1]) if policy=='exclude_pair_union' else np.ones(len(ids),dtype=bool)
     if policy=='all_correct':arrays=[a|f for a,f in zip(arrays,flags)]
     cc,dd=[int(a[keep].sum()) for a in arrays];n=int(keep.sum());diff=100*(cc-dd)/n;diffs.append(diff)
     pair={'seed':seed,'n':n,'context_correct':cc,'direct_correct':dd,'context_pct':100*cc/n,'direct_pct':100*dd/n,'difference_pp':diff,'context_review':int(flags[0].sum()),'direct_review':int(flags[1].sum())}
     equal(pair,report['sets'][setname]['metrics'][metric][policy]['pairs'][seed]);pairs.append(pair)
    obj=moments(diffs)
    for k,v in obj.items():equal(v,report['sets'][setname]['metrics'][metric][policy][k])
    table[policy]={**obj,'pairs':pairs,'context_mean_pct':float(np.mean([p['context_pct'] for p in pairs])),'direct_mean_pct':float(np.mean([p['direct_pct'] for p in pairs]))}
   # Independently recompute previous unmatched effects from frozen prior predictions.
   old_diffs=[]
   for seed in range(6):
    old=[read(inputs/'reference_v3'/f'{c}_s{seed}.jsonl') for c in ['context','direct']]
    assert all(list(map(key,rows))==keys for rows in old)
    old_diffs.append(100*sum(int(old[0][i]['scores'][metric]['correct'])-int(old[1][i]['scores'][metric]['correct']) for i in ids)/len(ids))
   old_summary=moments(old_diffs)
   for k,v in old_summary.items():equal(v,report['sets'][setname]['metrics'][metric]['unmatched_reference_all_wrong'][k])
   changes=moments([a-b for a,b in zip(table['all_wrong']['per_seed_pp'],old_diffs)])
   equal(changes,report['sets'][setname]['metrics'][metric]['matched_minus_unmatched_pp'])
   table['unmatched_reference']=old_summary;table['matched_minus_unmatched']=changes
   base=np.array([scored['baseline'][i]['scores'][metric]['correct'] for i in ids],dtype=bool)
   table['baseline_pct']=100*float(base.mean())
   for c in ['context','direct']:table[c+'_minus_baseline_pp']=moments([p[c+'_pct']-table['baseline_pct'] for p in table['all_wrong']['pairs']])
   summary['sets'][setname][metric]=table
 # Independently score training targets and verify cached audit reproduces exactly.
 classes=read(inputs/'all864_target_classes.jsonl');audit=read(RUN/'training_label_audit/all864_labels.jsonl');assert list(map(key,classes))==list(map(key,audit))
 capitals=[a for r in classes if r['category']=='birthplace_capital' for a in r['gold']]
 for r,s in zip(classes,audit):
  for c in ['context','direct']:equal(training_score(r[c+'_target'],r['gold'],r['category'],capitals),s['revised'][c])
 selected=[r for r in audit if key(r) in set(map(key,train))];different=[r for r in selected if r['context_target']!=r['direct_target']]
 assert len(selected)==785 and len(different)==404
 equal(selected,read(RUN/'training_label_audit/selected785_labels.jsonl'));equal(different,read(RUN/'training_label_audit/different404_labels.jsonl'))
 audit_report=json.loads((RUN/'training_label_audit/report.json').read_text());equal(audit_report,report['training_label_audit'])
 for label,rows in [('all864',audit),('selected785',selected),('different404',different),('identical381',[r for r in selected if r['targets_exactly_identical']])]:
  n=len(rows);a=np.array([[r['revised'][c]['correct'] for c in ['context','direct']] for r in rows],dtype=bool);f=np.array([[r['revised'][c]['needs_review'] for c in ['context','direct']] for r in rows],dtype=bool)
  for j,c in enumerate(['context','direct']):assert int(a[:,j].sum())==audit_report[label]['groups'][c]['revised_correct'] and int(f[:,j].sum())==audit_report[label]['groups'][c]['needs_review']
  for policy in ['all_wrong','all_correct','exclude_pair_union']:
   keep=~f.any(axis=1) if policy=='exclude_pair_union' else np.ones(n,dtype=bool);values=a|f if policy=='all_correct' else a
   diff=100*float(np.mean(values[keep,0].astype(int)-values[keep,1].astype(int)))
   equal(diff,audit_report[label]['sensitivity'][policy]['difference_pp']);assert int(keep.sum())==audit_report[label]['sensitivity'][policy]['n']
  wins=int((a[:,0]&~a[:,1]).sum());losses=int((~a[:,0]&a[:,1]).sum());expected=float(binomtest(wins,wins+losses).pvalue) if wins+losses else 1.
  equal(expected,audit_report[label]['mcnemar_exact_p'])
  summary['training_labels'][label]={**audit_report[label],'review_union':int(f.any(axis=1).sum()),'review_intersection':int(f.all(axis=1).sum())}
 summary['training_labels']['by_category']=audit_report['selected_by_category'];summary['training_labels']['different404_by_category']=audit_report['different404_by_category'];summary['training_labels']['cached_nll404']=audit_report['different404_cached_nll']
 diag={key(r):r for r in read(RUN/'source_700454/paired_targets/paired_diagnostics.jsonl')}
 for field,value in audit_report['different404_cached_nll'].items():equal(float(np.mean([diag[key(r)][field]['mean_nll'] for r in different])),value)
 equal(before,{str(p):sha(p) for p in RUN.rglob('*') if p.is_file()})
 complete=json.loads((RUN/'complete.json').read_text());assert complete['training_runs']==12 and not complete['smoke_test'] and not complete['baseline_regenerated'] and not complete['targets_regenerated']
 summary['seconds']=complete['seconds'];js(OUT/'independent_analysis.json',summary)
 js(OUT/'verification.json',{'job':701792,'archive_sha256':sha(archive),'input_and_protocol_frozen':True,'production_code_hashes_match_uploaded_package':True,'prediction_sets':13,'items_per_prediction_set':500,'v3_labels_recomputed':nlabels,'old_labels_recomputed':nlabels,'legacy_labels_recomputed':6500,'training_target_labels_recomputed':1728,'steps_per_fit':198,'training_fits':12,'runtime_matches_job700454':True,'paired_initializations_equal':True,'initializations_match_previous_same_seeds':True,'independent_numpy_scipy_statistics_match':True,'source_files_unchanged':before,'no_inference_or_training':True,'limitations':['No final sacct export in archive; completion evidenced by program receipts and log.','Adapter weights intentionally remain on HPC; completion hashes present but weight bytes not downloaded.','Intervals describe six training seeds on fixed benchmark, not independent population samples.','Training labels are automated against existing references, not blinded factual adjudication.']})
 print('VERIFIED job701792: 13x500 predictions; 19500 v3 labels; 12x198 updates; frozen design/source preserved.')
 print(json.dumps({s:summary['sets'][s]['composed'] for s in summary['sets']},indent=2))
if __name__=='__main__':main()
