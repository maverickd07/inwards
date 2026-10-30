import json,hashlib,tarfile,sys,collections,importlib.util
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parents[1];R=H/'received/download_SaGWid';O=Path(__file__).resolve().parent
sys.path.insert(0,str(H));from probe_utils import labels,split

def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
checks={};archive_checks=[]
for exp,job,protocol in [('exp21_latent_emergence_20261022','712907','frozen_protocol'),('exp22_probe_20261029','726630','exp22_protocol')]:
 d=R/exp;a=d/f'runs_{job}_review.tgz';digest=(d/(a.name+'.sha256')).read_text().split()[0];assert sha(a)==digest
 with tarfile.open(a) as t:
  count=0
  for m in t.getmembers():
   if m.isfile():assert hashlib.sha256(t.extractfile(m).read()).hexdigest()==sha(d/m.name),m.name;count+=1
 proto=d/protocol
 for line in (proto/'SHA256SUMS').read_text().splitlines():
  digest,name=line.split('  ',1);assert sha(proto/name)==digest,name
 for name in ['run.py','analyze.py','PREREGISTRATION.txt','ADAPTER_MANIFEST.json']:
  assert sha(proto/name)==sha(H.parent/exp/name),name
 archive_checks.append({'experiment':exp,'job':job,'archive_sha256':sha(a),'all_extracted_members_match':count,'frozen_protocol_matches_local':True})
p=R/'exp22_probe_20261029/runs_726630';rows=json.loads((p/'roster.json').read_text());rs=read(p/'per_item.jsonl');train,sp=split(rows);assert sp==json.loads((p/'split.json').read_text());assert not set(sp['train_groups'])&set(sp['test_groups'])
assert len(rows)==2402 and len(rs)==12010 and len({(x['variant'],x['id']) for x in rs})==12010
by={(x['variant'],x['id']):x for x in rs};rowby={x['id']:x for x in rows};variants=['base','context_s0','context_s1','direct_s0','direct_s1']
for x in rs:
 assert x['first_top1_verified'] and x['generated_ids'] and x['positions'][0]<x['positions'][1]==len(x['prompt_ids'])-1
 assert x['train']==bool(train[x['index']])
 if x['variant']!='base':assert x['stratum']==by['base',x['id']]['stratum']
labelsets={t:labels(rows,train,t)[0] for t in ['country','answer']};band=json.loads((p/'band_summary.json').read_text());ag=collections.defaultdict(lambda:collections.defaultdict(list));hist=collections.defaultdict(collections.Counter);count=0
for x in read(p/'probe_predictions.jsonl'):
 count+=1;r=by[x['variant'],x['id']];assert not r['train'];i=r['index'];assert x['truth']==labelsets[x['target']][i]
 assert len(x['predictions'])==len(x['hit'])==len(x['true_probability'])==len(x['shuffled_hit'])==41
 assert x['hit']==[float(v==x['truth']) for v in x['predictions']];assert all(0<=v<=1 for v in x['true_probability'])
 sn=['new400'] if r['source']=='new400' else [s for s in ['fixed303','known313'] if r[s]]
 subsets=['all',r['stratum']]+(['rescued'] if r['stratum']=='gap' and r['score']['correct'] else [])
 if 'other' not in x['truth']:subsets+=['non_other']
 for s in sn:
  for sub in subsets:
   for metric,field in [('hit','hit'),('prob','true_probability')]:
    k=(x['target'],x['position'],x['variant'],s,sub,metric);ag[k][r['group']].append(float(np.mean(x[field][14:27])))
  if r['stratum']=='gap' and x['target']=='country':
   hist[s,x['position'],x['variant']].update(x['predictions'][14:27])
assert count==sum(~train)*5*4
n=0
for b in band:
 if b['band']!='middle' or b['stratum'] not in ['all','correct','gap','missing_q1','missing_q2','rescued','non_other']:continue
 k=tuple(b[a] for a in ['target','position','variant','set','stratum','metric']);gg=ag[k];expected=float(np.mean([np.mean(v) for v in gg.values()])) if gg else None
 assert (b['mean'] is None and expected is None) or abs(b['mean']-expected)<1e-12,(k,b['mean'],expected);n+=1
# Fresh explicit scoring import; no changes to preserved records.
for exp,job,protocol in [('exp21_latent_emergence_20261022','712907','frozen_protocol'),('exp22_probe_20261029','726630','exp22_protocol')]:
 spec=importlib.util.spec_from_file_location('score_'+job,R/exp/protocol/'scoring.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 rr=read(R/exp/f'runs_{job}/per_item.jsonl')
 for x in rr:
  e=x.get('e') or rowby[x['id']]['e'];country,answer,malformed=m.parts(x['raw'],x.get('condition','C0'));s=m.score_answer(answer,e)
  if malformed or x['truncated']:s={'correct':False,'status':'REVIEW','reason':'truncated' if x['truncated'] else 'malformed_C2'}
  assert s==x['score'],(job,x['id'])
 checks[job+'_scoring_records_recomputed']=len(rr)
old=read(R/'exp21_latent_emergence_20261022/runs_712907/per_item.jsonl');d21={(x['variant'],x['id']):x for x in old if x['condition']=='C0'};different=[]
for x in rs:
 if x['source']=='old':
  y=d21[x['variant'],int(x['id'][4:])]
  if x['raw']!=y['raw']:different.append({'id':x['id'],'variant':x['variant'],'exp21_raw':y['raw'],'exp22_raw':x['raw'],'exp21_correct':y['score']['correct'],'exp22_correct':x['score']['correct']})
for x in old:
 for t in x['targets']:
  first=next((i for i in range(1,41) if x['layers'][i]['ranks'][t]<=10),None);assert first==x['emergence'][t]
assert json.loads((p/'complete.json').read_text())['analysis_complete']
checks.update({'archives':archive_checks,'exp22_unique_rows':len(rs),'probe_rows_checked':count,'band_means_independently_recomputed':n,'person_groups_train_test':[len(sp['train_groups']),len(sp['test_groups'])],'cross_run_old_C0_raw_differences':len(different),'cross_run_old_C0_score_differences':sum(x['exp21_correct']!=x['exp22_correct'] for x in different),'raw_activation_files_not_downloaded':True,'probe_fit_or_activation_values_not_independently_refitted':True,'remote_activation_bytes_recorded':sum(v['bytes'] for v in json.loads((p/'CACHE_MANIFEST.json').read_text()).values())})
(O/'VERIFICATION.json').write_text(json.dumps(checks,indent=2)+'\n');(O/'CROSS_RUN_DIFFERENCES.json').write_text(json.dumps(different,ensure_ascii=False,indent=2)+'\n')
(O/'GAP_PREDICTION_HISTOGRAM.json').write_text(json.dumps([{'set':k[0],'position':k[1],'variant':k[2],'counts':dict(v)} for k,v in hist.items()],indent=2)+'\n')
print(json.dumps(checks,indent=2))
