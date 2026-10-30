"""Saved-data-only feasibility; no training or submission code."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
P=Path(__file__).resolve().parent;OLD=P.parent/'expxx_self_consistency_20261016'
sys.path.insert(0,str(OLD))
from consistency import *
import math,datetime
verify_inputs()
freeze=json.loads((P/'PRECOUNT_FREEZE.json').read_text())
for f,h in freeze['files'].items():assert sha(P/f)==h
w=World();rs=read(OLD/'ALL864_ENRICHED.jsonl');people={}
for r in rs:
 if r['person'] in people:assert people[r['person']]==r['gold_country']
 people[r['person']]=r['gold_country']
freq=collections.Counter(people.values())
binof=lambda c:-1 if not freq[c] else freq[c].bit_length()-1

def lookup(c,cat,source):
 l=w.lookup(cat,c,'exp12' if source=='extended' else source)
 if source=='extended' and l['reason']=='missing_saved_q2':l=w.lookup(cat,c,'exp1')
 return l

def canonical_answer(l):
 return min(r['prediction'] for r in l['records']) if l['reason']=='available' else None

def wrong(target,e):
 label=labeler.score(target,e['Answer'],e['category'],w.capitals)
 return not label['correct'] and not label['needs_review'],label
summary={};output=[]
for source in ['exp12','exp1','extended']:
 stages=collections.Counter();bycat={c:collections.Counter() for c in CATS};eligiblepeople=set();nonsamefreq=0;allq2available=0
 for r in rs:
  cat=r['category'];e=w.by[key(r)];z=r['z_country'];real=r['gold_country'];local={};donors=[]
  stages['all864']+=1;bycat[cat]['all864']+=1
  if z and lookup(z,cat,source)['reason']=='available':allq2available+=1
  if not r['z_label']['correct']:stages['v3_q1_marked_wrong']+=1;bycat[cat]['v3_q1_marked_wrong']+=1
  if not z or z==real:continue
  local['first_wrong_resolved']=True
  l=lookup(z,cat,source);target=canonical_answer(l)
  if target is not None:
   local['own_q2_available']=True;ownwrong,ownlabel=wrong(target,e)
   if ownwrong:local['consistent_target_strict_wrong']=True
   for donor in sorted(w.table):
    if donor in [z,real]:continue
    d=lookup(donor,cat,source);dt=canonical_answer(d)
    if dt is None:continue
    incompatible=not w.compatible(set(l['values']),set(d['values']),cat)
    dw,dl=wrong(dt,e);matched=binof(z)==binof(donor)
    donors.append({'country':donor,'target':dt,'q2_source':d['source'],'freq':freq[donor],'frequency_bin':binof(donor),
     'freq_log_distance':abs(math.log1p(freq[z])-math.log1p(freq[donor])),'frequency_matched':matched,'attribute_incompatible':incompatible,'strict_wrong':dw,'label':dl})
   matched=[d for d in donors if d['frequency_matched']]
   if matched:local['country_rule_and_frequency_matched']=True
   if any(d['attribute_incompatible'] for d in matched):local['also_attribute_incompatible']=True
   valid=[d for d in matched if d['attribute_incompatible'] and d['strict_wrong'] and ownwrong]
   if valid:local['full_strict_eligible']=True;eligiblepeople.add(r['person'])
   elif ownwrong and any(d['attribute_incompatible'] and d['strict_wrong'] for d in donors):nonsamefreq+=1
   valid.sort(key=lambda d:(d['freq_log_distance'],d['country']))
   output.append({'source':source,'category':cat,'person':r['person'],'z_tilde':r['z_tilde'],'z_country':z,'true_country':real,'z_freq':freq[z],'frequency_bin':binof(z),
    'consistent_target':target,'consistent_source':l['source'],'consistent_label':ownlabel,'stages':local,'eligible_donors':valid,'all_country_candidates':donors})
  for name in local:stages[name]+=1;bycat[cat][name]+=1
 summary[source]={'stages':dict(stages),'by_category':{c:dict(x) for c,x in bycat.items()},'unique_eligible_people':len(eligiblepeople),'additional_if_frequency_relaxed_not_selected':nonsamefreq,'all864_stable_q2_n':allq2available}

# Dataset capacity, keeping the old person split and all evaluation identities disjoint.
full_people=collections.defaultdict(set)
for e in w.data:
 if e['category'] in CATS:full_people[e['person']].add(e['category'])
complete={p for p,cs in full_people.items() if set(CATS)<=cs}
testpeople=set()
for s in ['A_newPerson','C_bothNew','D_reservedC']:testpeople.update(r['person'] for r in read(path('exp12_plan_'+s)))
oldpersons=set(people);remaining=complete-oldpersons-testpeople
even_remaining={p for p in remaining if int(hashlib.md5(p.encode()).hexdigest(),16)%2==0}
variants=collections.defaultdict(set)
for r in rs:variants[r['person']].add(r['z_tilde'])
pool={'current_people':len(people),'all_dataset_people_with_all_four_categories':len(complete),'evaluation_people_union_A_C_D':len(testpeople),
'new_people_after_current_and_eval_exclusion':len(remaining),'new_people_preserving_even_md5_train_split':len(even_remaining),'new_people_names':sorted(even_remaining),
'current_people_with_multiple_saved_q1_strings':sum(len(x)>1 for x in variants.values()),'original_country_count':len(w.table),'frequency_by_training_person':dict(sorted(freq.items()))}

# Historical timings used as observations, not generation-only benchmarks.
run=path('train864').parent.parent
paired_time=json.loads((run/'paired_targets/timing.json').read_text());base_time=json.loads((run/'baseline/timing.json').read_text())
base_n=sum(len(read(path('exp12_baseline_'+s))) for s in ['A_newPerson','B_newCategory','C_bothNew','D_reservedC'])
times={'paired_generation_3x864_plus_4x864_NLL_seconds':paired_time['seconds'],'baseline_3xN_generation_plus_N_NLL_seconds':base_time['seconds'],'baseline_items_N':base_n,
'paired_total_over_generation_request_seconds':paired_time['seconds']/(3*864),'baseline_total_over_generation_request_seconds':base_time['seconds']/(3*base_n),
'caveat':'Includes model loading and NLL, not pure inference speed; batch length and hardware matter. Planning range 0.1–0.5 seconds/request plus 5–10 minutes loading/IO, excluding queue and reference curation.'}

third={}
for source in ['exp12','exp1','extended']:
 available=0;unparseable=0;missingkeys=set();q1unknown=0
 for r in rs:
  z=r['z_country']
  if not z:q1unknown+=1;continue
  l=lookup(z,r['category'],source)
  if l['reason']=='available':available+=1
  else:missingkeys.add((z,r['category']))
 third[source]={'available864':available,'missing_or_conflicting_unique_country_category_queries':len(missingkeys),'unresolved_firsthop_rows':q1unknown}

for name,obj in [('FEASIBILITY.json',summary),('POOL_CAPACITY.json',pool),('TIMING_EVIDENCE.json',times),('OPTIONAL_THIRD.json',third)]:js(P/name,obj)
write(P/'CANDIDATE_AUDIT.jsonl',output)
inputs={str(f.relative_to(ROOT)):sha(f) for f in [OLD/'ALL864_ENRICHED.jsonl',OLD/'INPUTS.json',OLD/'consistency.py',run/'paired_targets/timing.json',run/'baseline/timing.json']}
for spec in INPUTS.values():inputs[spec['path']]=spec['sha256']
js(P/'INPUT_HASHES.json',inputs)
verify_inputs()
for f,h in freeze['files'].items():assert sha(P/f)==h
js(P/'COUNT_COMPLETE.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_inference':True,'no_hpc':True,'no_submission_script':True,'historical_input_hashes_verified':True})
print(json.dumps({'feasibility':summary,'pool':pool,'timings':times,'third':third},ensure_ascii=False,indent=2))
