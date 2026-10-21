import json,csv,math,statistics,hashlib,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent;RUNS=HERE/'results/filtered_680959';OUT=HERE/'analysis';OUT.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('scoring_v2',HERE.parent/'scoring_v2.py');sv=importlib.util.module_from_spec(spec);spec.loader.exec_module(sv)
DATA=json.load(open(HERE.parents[2]/'inward/data/compositional_celebrities.json'))['data'];by={(r['category'],r['person']):r for r in DATA}
def read(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def key(r):return r['category'],r['person']
def pval(b,c):
 n=b+c
 return min(1.,2*sum(math.comb(n,i) for i in range(min(b,c)+1))/2**n) if n else 1.
def auc(s,y):
 pos=[x for x,a in zip(s,y) if a];neg=[x for x,a in zip(s,y) if not a]
 return sum((a>b)+.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg)) if pos and neg else None
names=[f'self_{m}_top{k}_s{s}' for m,k in [('confidence','1.0'),('confidence','0.5'),('confidence','0.7'),('random','0.5'),('random','0.7')] for s in range(3)]
assert set(d.name for d in RUNS.iterdir() if d.is_dir())==set(names)
configs={};evals={};targets={};baseline={};training=[];perrun=[];categories=[]
for name in names:
 d=RUNS/name;cfg=json.load(open(d/'run_config.json'));configs[name]=cfg;summary=json.load(open(d/'inward_summary.json'));diag=json.load(open(d/'confidence_diagnostic.json'));ts=read(d/'target_scores.jsonl');targets[name]=ts
 assert not cfg['smoke_test'] and cfg['seed']==0 and cfg['target']=='self'
 assert len(ts)==864 and len({key(r) for r in ts})==864
 assert math.isclose(auc([r['mean_logprob'] for r in ts],[r['diagnostic_target_ok'] for r in ts]),diag['auroc'])
 kept=[r for r in ts if r['kept']];assert len(kept)==diag['n_kept']==len(read(d/'train_examples.jsonl'))==summary['n_train']
 mode=cfg['selection'];top=cfg['keep_top'];seed=cfg['train_seed']
 if mode=='confidence':
  ranked=sorted(range(len(ts)),key=lambda i:(-ts[i]['mean_logprob'],i));assert {i for i,r in enumerate(ts) if r['kept']}==set(ranked[:math.ceil(top*len(ts))])
 training.append(dict(run=name,selection=mode,keep_top=top,seed=seed,n_kept=len(kept),target_correct_n=sum(r['diagnostic_target_ok'] for r in kept),target_correct_share=sum(r['diagnostic_target_ok'] for r in kept)/len(kept),auroc=diag['auroc']))
 for cat in sorted({r['category'] for r in ts}):
  rs=[r for r in ts if r['category']==cat];ks=[r for r in rs if r['kept']]
  categories.append(dict(run=name,category=cat,n_all=len(rs),n_kept=len(ks),correct_all=sum(r['diagnostic_target_ok'] for r in rs)/len(rs),correct_kept=sum(r['diagnostic_target_ok'] for r in ks)/len(ks) if ks else None,auroc=auc([r['mean_logprob'] for r in rs],[r['diagnostic_target_ok'] for r in rs])))
 for subset in ['A_newPerson','B_newCategory','C_bothNew']:
  rs=read(d/f'eval_{subset}.jsonl');assert len(rs)==500 and len({key(r) for r in rs})==500;maps={key(r):r for r in rs};evals[(name,subset)]=maps
  bp={k:(r['base_pred'],r['base_ok']) for k,r in maps.items()}
  if subset not in baseline:baseline[subset]=bp
  assert bp==baseline[subset],('baseline mismatch',name,subset)
  v=summary['evals'][subset];fixed=sum(not r['base_ok'] and r['trained_ok'] for r in rs);broke=sum(r['base_ok'] and not r['trained_ok'] for r in rs)
  assert fixed==v['fixed'] and broke==v['broke'];assert math.isclose(pval(fixed,broke),v['p']);assert math.isclose(sum(r['trained_ok'] for r in rs)/500,v['acc_trained'])
  for scoring in ['original','numeric_v2']:
   if scoring=='original':bs=[r['base_ok'] for r in rs];ys=[r['trained_ok'] for r in rs]
   else:
    bs=[sv.is_correct(r['base_pred'],by[key(r)]['Answer']) for r in rs];ys=[sv.is_correct(r['trained_pred'],by[key(r)]['Answer']) for r in rs]
   f=sum(not b and y for b,y in zip(bs,ys));b=sum(a and not y for a,y in zip(bs,ys))
   perrun.append(dict(run=name,selection=mode,keep_top=top,seed=seed,subset=subset,scoring=scoring,n=500,base_accuracy=sum(bs)/500,trained_accuracy=sum(ys)/500,gain_pp=(sum(ys)-sum(bs))/5,fixed=f,broke=b,mcnemar_p=pval(f,b),q1_base=v['q1_base'],q1_trained=v['q1_trained'],q2_base=v['q2_base'],q2_trained=v['q2_trained']))
first=targets[names[0]]
assert all([(key(r),r['target'],r['mean_logprob'],r['diagnostic_target_ok']) for r in ts]==[(key(r),r['target'],r['mean_logprob'],r['diagnostic_target_ok']) for r in first] for ts in targets.values())
grouped=[]
for scoring in ['original','numeric_v2']:
 for subset in ['A_newPerson','B_newCategory','C_bothNew']:
  for mode,top in [('confidence',1.),('confidence',.5),('confidence',.7),('random',.5),('random',.7)]:
   rs=[r for r in perrun if r['scoring']==scoring and r['subset']==subset and r['selection']==mode and r['keep_top']==top];assert len(rs)==3
   vals=[r['trained_accuracy']*100 for r in rs];mean=statistics.mean(vals);sd=statistics.stdev(vals);half=4.302652729911275*sd/math.sqrt(3)
   grouped.append(dict(selection=mode,keep_top=top,subset=subset,scoring=scoring,seed_accuracies=vals,mean_accuracy_pct=mean,sample_sd_pp=sd,seed_mean_t95_low=mean-half,seed_mean_t95_high=mean+half,mean_gain_pp=statistics.mean(r['gain_pp'] for r in rs)))
comparisons=[]
for scoring in ['original','numeric_v2']:
 for subset in ['A_newPerson','B_newCategory','C_bothNew']:
  for top in [.5,.7]:
   differences=[];seed_results=[]
   for seed in range(3):
    cn=f'self_confidence_top{top}_s{seed}';rn=f'self_random_top{top}_s{seed}';c=evals[(cn,subset)];r=evals[(rn,subset)]
    def correct(row):return row['trained_ok'] if scoring=='original' else sv.is_correct(row['trained_pred'],by[key(row)]['Answer'])
    wins=sum(correct(c[k]) and not correct(r[k]) for k in c);loss=sum(correct(r[k]) and not correct(c[k]) for k in c);delta=(wins-loss)/5;differences.append(delta)
    seed_results.append(dict(seed=seed,confidence_wins=wins,random_wins=loss,delta_pp=delta,mcnemar_p=pval(wins,loss)))
   mean=statistics.mean(differences);sd=statistics.stdev(differences);half=4.302652729911275*sd/math.sqrt(3)
   comparisons.append(dict(scoring=scoring,subset=subset,keep_top=top,mean_confidence_minus_random_pp=mean,paired_seed_t95=[mean-half,mean+half],per_seed=seed_results))
def savecsv(name,rs):
 with open(OUT/name,'w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rs[0].keys());w.writeheader();w.writerows(rs)
savecsv('per_run.csv',perrun);savecsv('training_quality.csv',training);savecsv('training_categories.csv',categories)
(OUT/'summary.json').write_text(json.dumps({'groups':grouped,'confidence_vs_random':comparisons,'training_quality':training,'validation':{'complete_runs':15,'evaluations_per_run':3,'items_per_evaluation':500,'identical_base_predictions':True,'identical_generated_targets_and_scores':True,'auc_recomputed':True,'summary_recomputed':True},'limitations':['Original scorer is primary; numeric_v2 rescoring checks only composed predictions. Single-hop predictions are not saved in eval files, so new single-hop accuracies/gaps cannot be reconstructed here.','t95 over 3 seeds uses Student t df=2 normality assumption; not population/task uncertainty. No pooling repeated seed predictions as independent observations.','Per-run McNemar p values are unadjusted; many comparisons.','Gold and hops eval files from previous experiments remain missing.','New keep-all seed0 is 53.2%, whereas handoff reported 54.2%; cause unknown, do not merge runs.','Exp8 logit lens uses original exp6 logic, without explicitly filtering bridge/final token overlap; avoid treating as clean mechanism verification.'],'archive_sha256':hashlib.sha256((HERE.parents[2]/'exp8_680959_review.tgz').read_bytes()).hexdigest()},ensure_ascii=False,indent=2))
for x in training:print('QUALITY',x)
for g in grouped:
 if g['scoring']=='original':print('GROUP',g)
for x in comparisons:
 if x['subset']=='C_bothNew':print('COMPARE',x)
