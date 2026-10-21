"""Analyze paired predictions without pooling repeated seeds as independent items."""
import argparse,json,csv,math,statistics
from pathlib import Path
CONDITIONS=['self_full','shuffled_full','category_confidence50','category_random50']
def read(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def key(r):return r['category'],r['person']
def pval(b,c):
 n=b+c
 return min(1.,2*sum(math.comb(n,i) for i in range(min(b,c)+1))/2**n) if n else 1.
def moments(v):
 m=statistics.mean(v);sd=statistics.stdev(v) if len(v)>1 else None
 half=4.302652729911275*sd/math.sqrt(3) if len(v)==3 else None
 return {'values':v,'mean':m,'sample_sd':sd,'seed_mean_t95':[m-half,m+half] if half is not None else None}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--runs',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(exist_ok=False)
 per=[];maps={};config={};initial={};baseline={};diagnostics={};reference=[]
 for seed in range(3):
  for condition in CONDITIONS:
   name=f'{condition}_s{seed}';d=a.runs/name
   cfg=json.load(open(d/'run_config.json'));assert cfg['condition']==condition and cfg['train_seed']==seed and cfg['seed']==0
   config[name]=cfg;initial[name]=json.load(open(d/'initialization.json'))['lora_initial_sha256'];diagnostics[name]=json.load(open(d/'target_diagnostic.json'))
   if (d/'reference_base_check.jsonl').exists():reference+=read(d/'reference_base_check.jsonl')
   for subset in ['A_newPerson','B_newCategory','C_bothNew']:
    rows=read(d/f'eval_{subset}.jsonl');assert len({key(r) for r in rows})==len(rows)
    bp={key(r):(r['base_pred'],r['base_ok']) for r in rows}
    if subset not in baseline:baseline[subset]=bp
    assert bp==baseline[subset],('different evaluation/base',name,subset)
    maps[(condition,seed,subset)]={key(r):r for r in rows}
    fixed=sum(not r['base_ok'] and r['trained_ok'] for r in rows);broke=sum(r['base_ok'] and not r['trained_ok'] for r in rows);n=len(rows)
    per.append({'condition':condition,'seed':seed,'subset':subset,'n':n,'base_accuracy':sum(r['base_ok'] for r in rows)/n,'trained_accuracy':sum(r['trained_ok'] for r in rows)/n,'gain_pp':100*(fixed-broke)/n,'fixed':fixed,'broke':broke,'mcnemar_p_unadjusted':pval(fixed,broke)})
  assert len({initial[f'{c}_s{seed}'] for c in CONDITIONS})==1,('LoRA initialization differs',seed)
  full=diagnostics[f'self_full_s{seed}'];shuffled=diagnostics[f'shuffled_full_s{seed}'];assert full['n_kept']==shuffled['n_kept']
  assert diagnostics[f'category_confidence50_s{seed}']['category_counts'].keys()==diagnostics[f'category_random50_s{seed}']['category_counts'].keys()
  for cat, v in diagnostics[f'category_confidence50_s{seed}']['category_counts'].items():assert v['kept']==diagnostics[f'category_random50_s{seed}']['category_counts'][cat]['kept']
 assert len({v['target_cache_sha256'] for v in diagnostics.values()})==1
 groups=[];comparisons=[]
 for subset in ['A_newPerson','B_newCategory','C_bothNew']:
  for c in CONDITIONS:
   rs=[r for r in per if r['condition']==c and r['subset']==subset]
   groups.append({'condition':c,'subset':subset,'accuracy_pct':moments([100*r['trained_accuracy'] for r in rs]),'gain_pp':moments([r['gain_pp'] for r in rs])})
  for c,control in [(c,'self_full') for c in CONDITIONS if c!='self_full']+[('category_confidence50','category_random50')]:
   pairs=[]
   for seed in range(3):
    x=maps[(c,seed,subset)];y=maps[(control,seed,subset)];wins=sum(x[k]['trained_ok'] and not y[k]['trained_ok'] for k in x);loss=sum(y[k]['trained_ok'] and not x[k]['trained_ok'] for k in x)
    pairs.append({'seed':seed,'wins':wins,'losses':loss,'delta_pp':100*(wins-loss)/len(x),'mcnemar_p_unadjusted':pval(wins,loss)})
   comparisons.append({'condition':c,'control':control,'subset':subset,'paired_seeds':pairs,'difference_pp':moments([r['delta_pp'] for r in pairs])})
 # Holm adjustment for six prespecified C tests: shuffled-vs-full and balanced-confidence-vs-random.
 primary=[]
 for comp in comparisons:
  if comp['subset']=='C_bothNew' and (comp['condition'],comp['control']) in [('shuffled_full','self_full'),('category_confidence50','category_random50')]:
   primary.extend(comp['paired_seeds'])
 running=0.
 for rank,row in enumerate(sorted(primary,key=lambda r:r['mcnemar_p_unadjusted'])):
  running=max(running,min(1.,(len(primary)-rank)*row['mcnemar_p_unadjusted']))
  row['mcnemar_p_holm_C_primary']=running
 report={'groups':groups,'paired_comparisons':comparisons,'diagnostics':diagnostics,'exp8_base_prediction_disagreements':sum(r['base_prediction_disagreements'] for r in reference),'limitations':['Three-seed t intervals assume normal seed effects; do not quantify task sampling uncertainty.','Per-seed McNemar p values are unadjusted; repeated seeds are not pooled as independent items.','No significant difference is not equivalence. No predeclared equivalence margin, so no equivalence claim.','Shuffled labels can remain correct or unchanged; use assignment diagnostics.','Category matching does not establish a unique internal mechanism.']}
 (a.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 with open(a.out/'per_run.csv','w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=per[0].keys());w.writeheader();w.writerows(per)
 for g in groups:
  if g['subset']=='C_bothNew':print(g)
if __name__=='__main__':main()
