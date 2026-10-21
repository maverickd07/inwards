"""Frozen local analysis; outputs are exclusive-create, never historical mutations."""
import sys
sys.dont_write_bytecode=True
from consistency import *
import statistics,datetime
METHODS=['attribute_floor','attribute_round','model_exp12','model_exp1','model_extended','model_exp12_round','model_exp1_round','model_extended_round']
SCOPES=['all864','selected785','different404']
TARGETS=['all','correct','wrong','strict_wrong','review']
ZSTRATA=['all','correct','wrong','strict_wrong','review']

def matches(label,stratum):
 if stratum=='all':return True
 if stratum=='correct':return label['correct']
 if stratum=='wrong':return not label['correct']
 if stratum=='strict_wrong':return not label['correct'] and not label['needs_review']
 if stratum=='review':return label['needs_review']
 raise ValueError(stratum)
def scope_rows(rows,scope):return rows if scope=='all864' else [r for r in rows if r[scope]]
def stat(rs,method,cond):
 n=len(rs);counts={s:sum(r['consistency'][method][cond]['status']==s for r in rs) for s in STATUSES}
 assert sum(counts.values())==n
 known=n-counts['unknown']
 return {'n':n,'counts':counts,'pct':{s:100*v/n if n else None for s,v in counts.items()},
 'consistent_among_judgeable_pct':100*counts['consistent']/known if known else None,
 'target_review_n':sum(r['labels'][cond]['needs_review'] for r in rs),
 'unknown_reasons':dict(collections.Counter(r['consistency'][method][cond]['reason'] for r in rs if r['consistency'][method][cond]['status']=='unknown'))}
def verdict(d):
 if d is None:return '不可计算'
 return '自洽是候选机制' if d>=20 else '自洽解释不了' if d<10 else '不确定'
def delta(c,d):
 if not c['n'] or not d['n']:return {'delta_pp':None,'verdict':'不可计算','unknown_bounds_pp':None}
 diff=c['pct']['consistent']-d['pct']['consistent']
 return {'delta_pp':diff,'verdict':verdict(diff),'unknown_bounds_pp':[diff-d['pct']['unknown'],diff+c['pct']['unknown']],
 'judgeable_delta_pp':c['consistent_among_judgeable_pct']-d['consistent_among_judgeable_pct'] if c['consistent_among_judgeable_pct'] is not None and d['consistent_among_judgeable_pct'] is not None else None}
def table(rows,methods=METHODS,scopes=SCOPES,conds=CONDS):
 out=[]
 for scope in scopes:
  rs=scope_rows(rows,scope)
  for cat in ['all']+CATS:
   cs=rs if cat=='all' else [r for r in rs if r['category']==cat]
   for zstr in ZSTRATA:
    zs=[r for r in cs if matches(r['z_label'],zstr)]
    for ts in TARGETS:
     for m in methods:
      for c in conds:
       subset=[r for r in zs if matches(r['labels'][c],ts)]
       out.append({'scope':scope,'category':cat,'z_stratum':zstr,'target_stratum':ts,'method':m,'condition':c,**stat(subset,m,c)})
 return out

def exp10_rows(world):
 allrows=[];checks=[];frozen={key(r):r for r in read(path('exp10_targets'))}
 for group in ['self_wrong','self_correct','shuffled_wrong']:
  seed_rows=[]
  for seed in range(3):
   prefix=f'exp10_{group}_s{seed}_';assign=read(path(prefix+'target_assignment'));kept=[r for r in assign if r['kept']];examples=read(path(prefix+'train_examples'));assert len(kept)==len(examples)==228
   for a,b in zip(kept,examples):
    e=world.by[key(a)];clear=e['Question'].replace(f"the birthplace of {e['person']}",f"the country where {e['person']} was born")
    assert b['target']==a['assigned_target'] and b['question']==clear
    assert a['original_target']==frozen[key(a)]['target']
   seed_rows.append(kept)
  if group!='shuffled_wrong':assert seed_rows[0]==seed_rows[1]==seed_rows[2]
  for seed in (range(3) if group=='shuffled_wrong' else [0]):
   for a in seed_rows[seed]:
    e=world.by[key(a)];z=world.exp1[key(a)]['pred_q1'];target=a['assigned_target'];lab=labeler.score(target,e['Answer'],e['category'],world.capitals)
    r={'group':group,'seed':seed if group=='shuffled_wrong' else 'shared_0_1_2','category':e['category'],'person':e['person'],
     'question':e['Question'],'Q1':e['Q1'],'Q2':e['Q2'],'z_tilde':z,'z_country':world.country(z),'gold_first':e['A1'],'gold_final':e['Answer'],
     'gold_country':e['birth_country'],'context_target':target,'direct_target':target,'labels':{'context':lab,'direct':lab},
     'z_label':v3.score(z,e['A1'],e['category'],'q1'),'original_target_ok':a['diagnostic_assigned_target_ok'],
     'donor_person':a['donor_person'],'selected785':False,'different404':False}
    allrows.append(r)
  checks.append({'group':group,'n_each_seed':228,'self_targets_identical_across_seeds':seed_rows[0]==seed_rows[1]==seed_rows[2]})
 return add_consistency(world,allrows),checks

def candidates(rows):
 out=[];counts=[]
 for scope in SCOPES:
  for definition in ['attribute_floor','model_exp12','model_extended','intersection_exp12','intersection_extended']:
   selected=[]
   for r in scope_rows(rows,scope):
    if not all(matches(r['labels'][c],'strict_wrong') for c in CONDS):continue
    a,b=[parse(r['category'],r[c+'_target']) for c in CONDS]
    if a is None or b is None or a==b:continue
    methods=['attribute_floor','model_'+definition.split('_')[1]] if definition.startswith('intersection') else [definition]
    statuses=[[r['consistency'][m][c]['status'] for c in CONDS] for m in methods]
    if not all(set(pair)=={'consistent','contradictory'} for pair in statuses):continue
    if len(statuses)>1 and statuses[0]!=statuses[1]:continue
    coherent='context' if statuses[0][0]=='consistent' else 'direct'
    selected.append(r);out.append({'scope':scope,'definition':definition,'category':r['category'],'person':r['person'],
      'consistent_source':coherent,'inconsistent_source':'direct' if coherent=='context' else 'context',
      'consistent_target':r[coherent+'_target'],'inconsistent_target':r[('direct' if coherent=='context' else 'context')+'_target'],
      'z_tilde':r['z_tilde'],'z_correct':r['z_label']['correct'],'gold_final':r['gold_final'],
      'consistent_chars':len(r[coherent+'_target']),'inconsistent_chars':len(r[('direct' if coherent=='context' else 'context')+'_target'])})
   subset=[x for x in out if x['scope']==scope and x['definition']==definition]
   counts.append({'scope':scope,'definition':definition,'n':len(selected),
     'by_category':{c:sum(r['category']==c for r in selected) for c in CATS},
     'consistent_source':dict(collections.Counter(x['consistent_source'] for x in subset)),
     'z_correct':sum(r['z_label']['correct'] for r in selected),
     'mean_consistent_chars':statistics.mean(x['consistent_chars'] for x in subset) if subset else None,
     'mean_inconsistent_chars':statistics.mean(x['inconsistent_chars'] for x in subset) if subset else None})
 return out,counts

def choose_examples(rows):
 available=[]
 for r in rows:
  if not r['different404']:continue
  for method in ['attribute_floor','model_exp12','model_extended']:
   statuses=[r['consistency'][method][c]['status'] for c in CONDS]
   if set(statuses)=={'consistent','contradictory'}:
    available.append((not all(matches(r['labels'][c],'strict_wrong') for c in CONDS),r,method));break
 chosen=[]
 for tier in [False,True]:
  queues={c:sorted([(r,m) for t,r,m in available if t==tier and r['category']==c],key=lambda x:key(x[0])) for c in sorted(CATS)}
  while any(queues.values()) and len(chosen)<10:
   for cat in sorted(CATS):
    if queues[cat] and len(chosen)<10:
     r,m=queues[cat].pop(0);chosen.append({'definition':m,'both_strict_wrong':not tier,**r})
 return chosen

def main():
 verify_inputs();assert (HERE/'COVERAGE.json').exists(),'Coverage must be disclosed first'
 world=World();rows=add_consistency(world,base_rows(world));tables=table(rows);write(HERE/'ALL864_ENRICHED.jsonl',rows);js(HERE/'TABLES.json',tables)
 verdicts=[];paired=[]
 for method in METHODS:
  for ts in ['wrong','strict_wrong']:
   s={c:stat([r for r in rows if r['different404'] and matches(r['labels'][c],ts)],method,c) for c in CONDS}
   verdicts.append({'method':method,'target_stratum':ts,**s,**delta(s['context'],s['direct'])})
  for strict in [False,True]:
   rs=[r for r in rows if r['different404'] and all(matches(r['labels'][c],'strict_wrong' if strict else 'wrong') for c in CONDS)]
   for joint in [False,True]:
    sub=[r for r in rs if all(r['consistency'][method][c]['status']!='unknown' for c in CONDS)] if joint else rs
    s={c:stat(sub,method,c) for c in CONDS};paired.append({'method':method,'both_strict_wrong':strict,'jointly_judgeable':joint,**s,**delta(s['context'],s['direct'])})
 js(HERE/'VERDICTS.json',verdicts);js(HERE/'PAIRED_SENSITIVITY.json',paired)
 ex10,checks=exp10_rows(world);write(HERE/'EXP10_ENRICHED.jsonl',ex10);js(HERE/'EXP10_CHECKS.json',checks)
 t10=[];cov10=[]
 for group,seed in sorted(set((r['group'],r['seed']) for r in ex10)):
  rs=[r for r in ex10 if r['group']==group and r['seed']==seed]
  t10.extend([{'group':group,'seed':seed,**t} for t in table(rs,scopes=['all864'],conds=['context'])])
  cov10.append({'group':group,'seed':seed,'coverage':coverage(world,rs)['all864']})
 js(HERE/'EXP10_TABLES.json',t10);js(HERE/'EXP10_COVERAGE.json',cov10)
 cr,cc=candidates(rows);write(HERE/'EXP17_CANDIDATES.jsonl',cr);js(HERE/'EXP17_COUNTS.json',cc)
 examples=choose_examples(rows);js(HERE/'EXAMPLES_10.json',examples)
 evidence={source:[{'country':c,'category':cat,'records':records} for (c,cat),records in sorted(index.items())] for source,index in world.indexes.items()}
 js(HERE/'SAVED_Q2_EVIDENCE.json',evidence)
 # Descriptive validation: exact identical target pairs have identical consistency classifications.
 for r in rows:
  if r['context_target']==r['direct_target']:
   for method in METHODS:assert r['consistency'][method]['context']==r['consistency'][method]['direct']
 for r in tables+t10:
  assert sum(r['counts'].values())==r['n']
 verify_inputs()
 js(HERE/'ANALYSIS_COMPLETE.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'n864':len(rows),'n785':sum(r['selected785'] for r in rows),'n404':sum(r['different404'] for r in rows),'historical_input_hashes_unchanged':True,'protocol_unchanged':True,'inference_runs':0,'hpc_calls':0,'examples':len(examples)})
 print(json.dumps({'verdicts':verdicts,'candidates':cc},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
