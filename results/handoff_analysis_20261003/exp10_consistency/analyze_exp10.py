"""Recompute all metrics from predictions. Never pool repeated seeds as independent questions."""
import argparse,json,math,statistics,itertools
from pathlib import Path
from scoring_v2 import is_correct,old_is_correct
C=['self_wrong','shuffled_wrong','self_correct'];S=['A_newPerson','B_newCategory','C_bothNew']
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def key(r):return r['category'],r['person']
def pval(b,c):
 n=b+c
 return min(1.,2*sum(math.comb(n,i) for i in range(min(b,c)+1))/2**n) if n else 1.
def moments(v):
 m=statistics.mean(v);sd=statistics.stdev(v);h=4.302652729911275*sd/math.sqrt(3)
 return {'values':v,'mean':m,'sample_sd':sd,'seed_mean_t95':[m-h,m+h]}
def paired(x,y):
 assert len(x)==len(y) and x
 wins=sum(a and not b for a,b in zip(x,y));loss=sum(b and not a for a,b in zip(x,y))
 return {'n':len(x),'wins':wins,'losses':loss,'difference_pp':100*(wins-loss)/len(x),'mcnemar_p':pval(wins,loss)}
def holm(tests):
 running=0.
 for i,t in enumerate(sorted(tests,key=lambda t:t['mcnemar_p'])):
  running=max(running,min(1.,(len(tests)-i)*t['mcnemar_p']));t['mcnemar_p_holm']=running

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--runs',type=Path,required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 by={key(r):r for r in json.load(open(a.data))['data']};records={};initial={};diag={};baselines={};assign={};ref=[]
 for seed in range(3):
  for condition in C:
   name=f'{condition}_s{seed}';d=a.runs/name;cfg=json.load(open(d/'run_config.json'));assert cfg['condition']==condition and cfg['train_seed']==seed and cfg['seed']==0
   initial[name]=json.load(open(d/'initialization.json'))['lora_initial_sha256'];diag[name]=json.load(open(d/'target_diagnostic.json'));assign[name]=[r for r in read(d/'target_assignment.jsonl') if r['kept']]
   assert diag[name]['uses_gold_for_allocation'] and diag[name]['allocation_scorer']=='scoring_v2_floor'
   assert len(assign[name])==diag[name]['n_kept'];assert all(r['diagnostic_assigned_target_ok']==(condition=='self_correct') for r in assign[name])
   if (d/'reference_base_check.jsonl').exists():ref+=read(d/'reference_base_check.jsonl')
   for subset in S:
    rows=read(d/f'eval_{subset}.jsonl');assert len(set(map(key,rows)))==len(rows)
    assert len(rows)==(2 if cfg['smoke_test'] else 500)
    bp=[(key(r),r['base_pred'],r['base_q1_pred'],r['base_q2_pred']) for r in rows]
    if subset not in baselines:baselines[subset]=bp
    assert bp==baselines[subset],('base predictions differ',name,subset)
    assert all(r['base_q1_pred'] is not None and r['trained_q2_pred'] is not None for r in rows)
    for r in rows:
     assert r['base_ok']==is_correct(r['base_pred'],by[key(r)]['Answer'])
     assert r['trained_ok']==is_correct(r['trained_pred'],by[key(r)]['Answer'])
    records[condition,seed,subset]=rows
  assert len({initial[f'{c}_s{seed}'] for c in C})==1
  x=assign[f'self_wrong_s{seed}'];y=assign[f'shuffled_wrong_s{seed}'];assert [key(r) for r in x]==[key(r) for r in y]
  from collections import Counter
  for cat in {r['category'] for r in x}:
   assert Counter(r['assigned_target'] for r in x if r['category']==cat)==Counter(r['assigned_target'] for r in y if r['category']==cat)
  assert all(r['person']!=r['donor_person'] and r['original_target']!=r['assigned_target'] for r in y)
  for c in C:
   assert {k:v['kept'] for k,v in diag[f'{c}_s{seed}']['category_counts'].items()}=={k:v['kept'] for k,v in diag[f'self_wrong_s{seed}']['category_counts'].items()}
 for c in ['self_wrong','self_correct']:assert len({json.dumps(assign[f'{c}_s{s}'],sort_keys=True) for s in range(3)})==1
 assert len({d['target_cache_sha256'] for d in diag.values()})==1
 output={}
 for scoring,scorer in [('floor',is_correct),('truncate',lambda p,g:is_correct(p,g,'truncate')),('legacy',old_is_correct)]:
  per=[];groups=[];comparisons=[];vectors={}
  for subset in S:
   for metric,field,goldfield in [('composed','pred','Answer'),('hop1','q1_pred','A1'),('hop2','q2_pred','A2')]:
    for c in C:
     for seed in range(3):
      rows=records[c,seed,subset];b=[scorer(r['base_'+field],by[key(r)][goldfield]) for r in rows];t=[scorer(r['trained_'+field],by[key(r)][goldfield]) for r in rows];vectors[c,seed,subset,metric]=t
      rec={'condition':c,'seed':seed,'subset':subset,'metric':metric,'base_accuracy_pct':100*sum(b)/len(b),'trained_accuracy_pct':100*sum(t)/len(t),**paired(t,b)};per.append(rec)
     rs=[r for r in per if r['condition']==c and r['subset']==subset and r['metric']==metric]
     groups.append({'condition':c,'subset':subset,'metric':metric,'accuracy_pct':moments([r['trained_accuracy_pct'] for r in rs]),'gain_pp':moments([r['difference_pp'] for r in rs])})
    for c,control in itertools.combinations(C,2):
     ps=[{'seed':seed,**paired(vectors[c,seed,subset,metric],vectors[control,seed,subset,metric])} for seed in range(3)]
     comparisons.append({'condition':c,'control':control,'subset':subset,'metric':metric,'paired_seeds':ps,'difference_pp':moments([r['difference_pp'] for r in ps])})
  primary=[r for r in per if r['subset']=='C_bothNew' and r['metric']=='composed']+[r for c in comparisons if c['subset']=='C_bothNew' and c['metric']=='composed' for r in c['paired_seeds']]
  holm(primary) # 18 prespecified C composed tests (9 against baseline + 9 between conditions).
  hops=[r for r in per if r['subset']=='C_bothNew' and r['metric'] in ['hop1','hop2']];holm(hops)
  output[scoring]={'per_run':per,'groups':groups,'comparisons':comparisons}
 report={'scoring':output,'diagnostics':diag,'reference_exp8_prediction_disagreements':sum(r['base_prediction_disagreements'] for r in ref),'limitations':['Gold selects subsets and constrains shuffled targets; this is not gold-free training allocation.','Correct and wrong groups contain different questions; category matching does not match difficulty or person distribution.','Self-output correspondence is not direct evidence of internal beliefs.','Only three seeds; t intervals assume normally distributed seed effects.','Failure to reject a difference does not prove equivalence or no forgetting.','Matching is randomized but not uniform over all valid permutations.','Legacy/truncate are rescoring sensitivity analyses; allocation always used floor.']}
 a.out.mkdir(exist_ok=False);(a.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 for g in output['floor']['groups']:
  if g['subset']=='C_bothNew':print(g)
if __name__=='__main__':main()
