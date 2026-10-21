"""Resume after empty exp10-only scope coverage division. Frozen statistical rules unchanged."""
import sys
sys.dont_write_bytecode=True
from analyze import *
verify_inputs()
world=World();rows=read(HERE/'ALL864_ENRICHED.jsonl');ex10=read(HERE/'EXP10_ENRICHED.jsonl')
t10=[];cov10=[]
for group,seed in sorted(set((r['group'],r['seed']) for r in ex10)):
 rs=[r for r in ex10 if r['group']==group and r['seed']==seed]
 t10.extend([{'group':group,'seed':seed,**t} for t in table(rs,scopes=['all864'],conds=['context'])])
 cov={}
 for source in ['exp12','exp1','extended']:
  counts=collections.Counter()
  for r in rs:
   look=world.lookup(r['category'],r['z_tilde'],'exp12' if source=='extended' else source)
   if source=='extended' and look['reason']=='missing_saved_q2':look=world.lookup(r['category'],r['z_tilde'],'exp1')
   counts[look['reason']]+=1
  cov[source]={'n':len(rs),'counts':dict(counts),'available_pct':100*counts['available']/len(rs) if rs else None}
 cov10.append({'group':group,'seed':seed,'coverage':cov})
js(HERE/'EXP10_TABLES.json',t10);js(HERE/'EXP10_COVERAGE.json',cov10)
cr,cc=candidates(rows);write(HERE/'EXP17_CANDIDATES.jsonl',cr);js(HERE/'EXP17_COUNTS.json',cc)
examples=choose_examples(rows);js(HERE/'EXAMPLES_10.json',examples)
evidence={source:[{'country':c,'category':cat,'records':records} for (c,cat),records in sorted(index.items())] for source,index in world.indexes.items()}
js(HERE/'SAVED_Q2_EVIDENCE.json',evidence)
for r in rows:
 if r['context_target']==r['direct_target']:
  for method in METHODS:assert r['consistency'][method]['context']==r['consistency'][method]['direct']
for r in json.loads((HERE/'TABLES.json').read_text())+t10:assert sum(r['counts'].values())==r['n']
verify_inputs()
js(HERE/'ANALYSIS_COMPLETE.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'n864':len(rows),'n785':sum(r['selected785'] for r in rows),'n404':sum(r['different404'] for r in rows),'historical_input_hashes_unchanged':True,'protocol_unchanged':True,'inference_runs':0,'hpc_calls':0,'examples':len(examples),'implementation_note':'Initial analyze.py stopped on division by zero for unused empty exp10 selected785 scope. Existing output retained. This script completes remaining outputs using exp10 all864 scope only; no classification, denominator or decision rule changed.'})
print(json.dumps(cc,ensure_ascii=False,indent=2))
