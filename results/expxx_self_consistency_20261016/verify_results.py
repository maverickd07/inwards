"""Independent count reconstruction and source/provenance assertions."""
import sys
sys.dont_write_bytecode=True
from consistency import *
import datetime,math
verify_inputs();P=HERE;load=lambda n:json.loads((P/n).read_text())
rows=read(P/'ALL864_ENRICHED.jsonl');ex10=read(P/'EXP10_ENRICHED.jsonl');by={key(r):r for r in rows}
train=read(path('train864'));paired=read(path('paired864'));labels=read(path('labels864'));assert len(rows)==864 and len(by)==864
for r,t,p,l in zip(rows,train,paired,labels):
 assert key(r)==key(t)==key(p)==key(l)
 assert r['z_tilde']==p['own_q1'] and r['labels']==l['revised']
 assert r['gold_first']==t['e']['A1'] and r['gold_final']==t['e']['Answer']
 for c in CONDS:assert r[c+'_target']==p[c+'_target']
 assert r['Q1']==t['e']['Q1'] and r['Q2']==t['e']['Q2']
assert sum(r['selected785'] for r in rows)==785
assert sum(r['different404'] for r in rows)==404
assert len(ex10)==5*228

def filt(l,s):
 return s=='all' or s=='correct' and l['correct'] or s=='wrong' and not l['correct'] or s=='strict_wrong' and not l['correct'] and not l['needs_review'] or s=='review' and l['needs_review']
# Reconstruct every exported count without using analyze.stat/table/matches.
ntables=0
for filename,rsall in [('TABLES.json',rows),('EXP10_TABLES.json',ex10)]:
 for tab in load(filename):
  rs=[r for r in rsall if (tab['scope']=='all864' or r[tab['scope']]) and (tab['category']=='all' or r['category']==tab['category']) and ('group' not in tab or r['group']==tab['group'] and r['seed']==tab['seed']) and filt(r['z_label'],tab['z_stratum']) and filt(r['labels'][tab['condition']],tab['target_stratum'])]
  counter=collections.Counter(r['consistency'][tab['method']][tab['condition']]['status'] for r in rs)
  assert len(rs)==tab['n']
  for s in STATUSES:
   assert counter[s]==tab['counts'][s]
   if rs:assert math.isclose(tab['pct'][s],100*counter[s]/len(rs),abs_tol=1e-12)
  ntables+=1
world=World();nver=0
for r in rows+ex10:
 for method,conds in r['consistency'].items():
  for cond,result in conds.items():
   cat=r['category'];mode='round' if method.endswith('_round') else 'floor'
   if method.startswith('attribute'):
    if result['status']!='unknown':
     assert result['z_country'] and result['implied_countries']
     assert (result['z_country'] in result['implied_countries'])==(result['status']=='consistent')
     assert result['country_set_size']==len(set(result['implied_countries']))
   else:
    source=result['lookup_source'];evidence=world.indexes[source].get(tuple(result['lookup_key']),[])
    assert result['saved_q2_n']==len(evidence)
    if result['status']!='unknown':
     assert evidence
     values={parse(cat,e['prediction'],mode) for e in evidence};assert len(values)==1 and None not in values
     assert set(result['saved_q2_values'])==set(next(iter(values)))
   nver+=1
for x in read(P/'EXP17_CANDIDATES.jsonl'):
 r=by[key(x)]
 assert all(not r['labels'][c]['correct'] and not r['labels'][c]['needs_review'] for c in CONDS)
 assert r[x['consistent_source']+'_target']==x['consistent_target']
 assert r[x['inconsistent_source']+'_target']==x['inconsistent_target']
 assert parse(r['category'],x['consistent_target'])!=parse(r['category'],x['inconsistent_target'])
 methods=['attribute_floor','model_'+x['definition'].split('_')[1]] if x['definition'].startswith('intersection') else [x['definition']]
 for m in methods:
  assert r['consistency'][m][x['consistent_source']]['status']=='consistent'
  assert r['consistency'][m][x['inconsistent_source']]['status']=='contradictory'
for x in load('EXAMPLES_10.json'):
 assert x['different404'] and x['both_strict_wrong']
 assert {x['consistency'][x['definition']][c]['status'] for c in CONDS}=={'consistent','contradictory'}
 for c in CONDS:assert x[c+'_target']==by[key(x)][c+'_target']
for freeze in ['PREREGISTRATION_FREEZE.json','CODE_FREEZE.json','COMPLETION_CODE_FREEZE.json']:
 for n,d in load(freeze)['files'].items():assert sha(P/n)==d
verify_inputs()
js(P/'VERIFICATION.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'tables_recounted':ntables,'item_method_condition_records_checked':nver,'raw_targets_and_labels_match_saved_sources':True,'frozen_inputs_protocol_and_code_unchanged':True,'all_candidate_and_example_constraints_verified':True})
print('Verification passed:',ntables,'tables;',nver,'item-method-condition records.')
