"""Read-only evidence audit and vector figures. Run from any working directory."""
from pathlib import Path
import json, tarfile, hashlib, statistics as st, math, sys, importlib.util, os
O=Path(__file__).resolve().parent; R=O.parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(O/'mpl_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
manifest={}
def read(p):
 p=Path(p);p=p if p.is_absolute() else R/p
 b=p.read_bytes();manifest[str(p.relative_to(R))]=hashlib.sha256(b).hexdigest();return json.loads(b)
def jl(p):
 p=R/p;b=p.read_bytes();manifest[str(p.relative_to(R))]=hashlib.sha256(b).hexdigest();return [json.loads(l) for l in b.decode().splitlines() if l.strip()]
old=read('results/report_20261005/report_data.json')
oldhash=read('results/report_20261005/source_sha256.json');checked=0
for p,h in oldhash.items():
 if p.startswith('tokenizer/'):continue
 b=(R/p).read_bytes();assert hashlib.sha256(b).hexdigest()==h,(p,'changed');manifest[p]=h;checked+=1
# Old report's overlap-filtered lens result is a derived artifact, checked against unchanged raw input hashes.
e8=read(old['sources']['S5']);e9=read(old['sources']['S6']);e10=read(old['sources']['S7'])
arc=R/'exp11_685985_review.tgz';manifest[arc.name]=hashlib.sha256(arc.read_bytes()).hexdigest()
prefix='results/handoff_analysis_20261003/exp11_iterate/runs_685985/'
with tarfile.open(arc) as t:
 def loadmember(rel):return json.loads(t.extractfile(prefix+rel).read())
 def rowsmember(rel):return [json.loads(l) for l in t.extractfile(prefix+rel).read().decode().splitlines() if l.strip()]
 e11=loadmember('analysis/report.json');stages=loadmember('stages.json');done=loadmember('complete.json');alloc=loadmember('plan/allocation.json')
 assert done['training_runs']==len(stages)==21 and not done['smoke_test']
 spec=importlib.util.spec_from_file_location('score',R/'results/handoff_analysis_20261003/exp11_iterate/scoring_v2_v1.py');score=importlib.util.module_from_spec(spec);spec.loader.exec_module(score)
 base={s:rowsmember(f'baseline/eval_{s}.jsonl') for s in ['A_newPerson','B_newCategory','C_bothNew']}
 keys=lambda rs:[(x['category'],x['person']) for x in rs]
 vectors={};nchecked=0
 for s,rs in base.items():
  gold=rowsmember(f'plan/items_{s}.jsonl');assert len(rs)==500 and keys(rs)==keys(gold)
  for pred,g in zip(rs,gold):
   for field,pfield,gfield in [('composed_ok','pred','Answer'),('q1_ok','q1_pred','A1'),('q2_ok','q2_pred','A2')]:assert pred[field]==score.is_correct(pred[pfield],g['e'][gfield])
  for stage in stages:
   rr=rowsmember(stage['path']+f'/eval_{s}.jsonl');assert len(rr)==500 and keys(rr)==keys(rs)
   for pred,g in zip(rr,gold):
    for field,pfield,gfield in [('composed_ok','pred','Answer'),('q1_ok','q1_pred','A1'),('q2_ok','q2_pred','A2')]:assert pred[field]==score.is_correct(pred[pfield],g['e'][gfield]);nchecked+=1
   vectors[stage['condition'],stage['round'],stage['seed'],s]=rr
 def pair(a,b):
  w=sum(x and not y for x,y in zip(a,b));l=sum(y and not x for x,y in zip(a,b));n=w+l
  p=min(1.,2*sum(math.comb(n,i) for i in range(min(w,l)+1))/2**n) if n else 1.
  return w,l,p
 for g in e11['groups']:
  field={'composed':'composed_ok','hop1':'q1_ok','hop2':'q2_ok'}[g['metric']]
  vv=[100*st.mean(z[field] for z in vectors[g['condition'],g['round'],k,g['subset']]) for k in range(3)]
  assert np.allclose(vv,g['accuracy_pct']['values']) and np.isclose(st.stdev(vv),g['accuracy_pct']['sample_sd'])
 for c in e11['C_paired_comparisons']:
  for p in c['paired_seeds']:
   a=vectors[*c['condition'],p['seed'],'C_bothNew'];b=vectors[*c['control'],p['seed'],'C_bothNew'];w,l,pv=pair([z['composed_ok'] for z in a],[z['composed_ok'] for z in b]);assert (w,l)==(p['wins'],p['losses']) and math.isclose(pv,p['mcnemar_p'])
 # Preserve only new copies; original result files remain untouched.
 for m in t.getmembers():
  if not m.isfile():continue
  if m.name.startswith(prefix):dest=O/'raw_exp11'/m.name[len(prefix):]
  elif m.name.endswith('exp11-685985.out'):dest=O/'raw_exp11'/'exp11-685985.out'
  else:continue
  dest.parent.mkdir(parents=True,exist_ok=True);b=t.extractfile(m).read()
  if dest.exists():assert dest.read_bytes()==b
  else:dest.write_bytes(b)
 oldref=jl('results/handoff_analysis_20261003/exp11_iterate/reference_exp8/eval_C_bothNew.jsonl')
 nb=base['C_bothNew'];assert keys(nb)==keys(oldref)
 drift={'n':len(nb),'same_item_order':True,'old_correct':sum(x['base_ok'] for x in oldref),'new_correct':sum(x['composed_ok'] for x in nb),'new_legacy_correct':sum(x['legacy_composed_ok'] for x in nb),'changed_text':sum(x['base_pred']!=y['pred'] for x,y in zip(oldref,nb)),'old_wrong_new_right':sum(not x['base_ok'] and y['legacy_composed_ok'] for x,y in zip(oldref,nb)),'old_right_new_wrong':sum(x['base_ok'] and not y['legacy_composed_ok'] for x,y in zip(oldref,nb))}
 timing=sum(loadmember(m.name[len(prefix):])['total_seconds'] for m in t.getmembers() if m.name.startswith(prefix) and m.name.endswith('/timing.json'))
# Verify exp9/10 C and old self consistency from raw prediction files.
roots={'e9':'results/handoff_analysis_20261003/exp9_681194/results/handoff_analysis_20261003/exp9_shuffled/runs_681194','e10':'results/handoff_analysis_20261003/exp10_681386/results/handoff_analysis_20261003/exp10_consistency/resumed_681332_681386'}
for tag,rep,cs in [('e9',e9['groups'],['self_full','shuffled_full','category_confidence50','category_random50']),('e10',e10['scoring']['floor']['groups'],['self_wrong','self_correct','shuffled_wrong'])]:
 for c in cs:
  vals=[]
  for k in range(3):
   rr=jl(f'{roots[tag]}/{c}_s{k}/eval_C_bothNew.jsonl');assert len(rr)==500
   vals.append(100*st.mean(z['trained_ok'] for z in rr))
  g=next(z for z in rep if z['condition']==c and z['subset']=='C_bothNew' and z.get('metric','composed')=='composed');assert np.allclose(vals,g['accuracy_pct']['values'])
# Recompute shared-known subsets across scales from source rows.
rs={s:{(z['category'],z['person']):z for z in jl(f'results/handoff_analysis_20261003/tasks56_verified/rescored_{s}.jsonl') if z['category'].startswith('birthplace') and 'common_name' not in z['category']} for s in ['1.7B','8B','14B']}
common=set.intersection(*[{k for k,z in v.items() if z['floor']['q1'] and z['floor']['q2']} for v in rs.values()])
scale=[{'size':s,'n':len(common),'gap_clear':100*st.mean(not v[k]['floor']['clear'] for k in common),'gap_ctx':100*st.mean(not v[k]['floor']['ctx'] for k in common)} for s,v in rs.items()]
def g11(c,r,s='C_bothNew',m='composed'):return next(g for g in e11['groups'] if (g['condition'],g['round'],g['subset'],g['metric'])==(c,r,s,m))
def g9(c,s='C_bothNew'):return next(g for g in e9['groups'] if (g['condition'],g['subset'])==(c,s))
def g10(c,m='composed'):return next(g for g in e10['scoring']['floor']['groups'] if (g['condition'],g['subset'],g['metric'])==(c,'C_bothNew',m))
D={'exp8':e8,'exp9':e9,'exp10':e10,'exp11':e11,'gap_clean':old['gap_clean'],'scale':scale,'scale_pair':old['scale_pair'],'lens_filtered':old['lens_filtered'],'labels':old['label_summary'],'drift':drift,'allocation':alloc,'timing_seconds':timing,'old_checked_files':checked,'exp11_verified_labels':nchecked,'baseline':{s:{f:100*st.mean(z[f] for z in rs) for f in ['composed_ok','q1_ok','q2_ok']} for s,rs in base.items()}}
(O/'evidence.json').write_text(json.dumps(D,ensure_ascii=False,indent=2));(O/'source_sha256.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#243849','text.color':'#243849'})
colors=['#155e75','#bb683c','#82939d']
def finish(fig,name):
 fig.tight_layout();fig.savefig(O/(name+'.pdf'),bbox_inches='tight');fig.savefig(O/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10.1,3.1))
ax=axs[0];ss=['A_newPerson','B_newCategory','C_bothNew'];vals=[g9('self_full',s)['gain_pp']['mean'] for s in ss];sd=[g9('self_full',s)['gain_pp']['sample_sd'] for s in ss]
ax.bar(range(3),vals,yerr=sd,color=colors[0],capsize=3,width=.55);ax.set_xticks(range(3),['A: new people','B: new relations','C: both new']);ax.set_ylabel('Gain over baseline (pp)');ax.set_ylim(0,12);ax.set_title('(a) Self-target training: exp8/9',loc='left',fontsize=10)
for i,v in enumerate(vals):ax.text(i,v+sd[i]+.35,f'+{v:.2f}',ha='center')
ax=axs[1]
for c,l,col in [('reset','Reset + refreshed targets',colors[0]),('continue','Continue + refreshed targets',colors[1]),('frozen_budget','Frozen targets, matched steps',colors[2])]:
 gs=[g11('shared',1),g11(c,2),g11(c,3)];ax.errorbar([1,2,3],[g['accuracy_pct']['mean'] for g in gs],yerr=[g['accuracy_pct']['sample_sd'] for g in gs],marker='o',capsize=3,color=col,label=l,lw=1.5)
ax.axhline(50,color='#888',ls='--',lw=1,label='Base: 50.0%');ax.set_xticks([1,2,3],['R1','R2 / 4 epochs','R3 / 6 epochs']);ax.set_ylim(48.7,57.2);ax.set_ylabel('C accuracy (%)');ax.set_title('(b) Iteration and training-budget controls: exp11',loc='left',fontsize=10);ax.legend(fontsize=6.8,frameon=False,loc='upper left');finish(fig,'fig1_overview')
fig,ax=plt.subplots(figsize=(8.8,3.1));x=np.arange(3)
for off,f,lab,col in [(-.18,'gap_clear','Direct composed question',colors[0]),(.18,'gap_ctx','Intermediate fact supplied',colors[2])]:
 vv=[z[f] for z in scale];ax.bar(x+off,vv,width=.34,label=lab,color=col)
 for xx,v in zip(x+off,vv):ax.text(xx,v+1,f'{v:.1f}',ha='center')
ax.set_xticks(x,[z['size'] for z in scale]);ax.set_ylabel('Conditional failure rate (%)');ax.set_ylim(0,88);ax.legend(frameon=False);finish(fig,'fig2_gap')
fig,axs=plt.subplots(1,2,figsize=(10,3.2))
for ax,conds,func,title in [(axs[0],['self_full','category_confidence50','category_random50','shuffled_full'],g9,'exp9: full / within-category controls'),(axs[1],['self_wrong','self_correct','shuffled_wrong'],g10,'exp10: 228 matched-count examples')]:
 labels={'self_full':'Self full\n864','category_confidence50':'Confidence\n432','category_random50':'Random\n432','shuffled_full':'Shuffled\n864','self_wrong':'Self wrong','self_correct':'Self correct','shuffled_wrong':'Shuffled wrong'}
 for i,c in enumerate(conds):
  g=func(c)['accuracy_pct'];ax.bar(i,g['mean'],yerr=g['sample_sd'],capsize=3,color=colors[1] if 'shuffled' in c else colors[0],width=.6);ax.text(i,g['mean']+g['sample_sd']+1.4,f"{g['mean']:.1f}",ha='center')
 ax.axhline(49.6,ls='--',color='#888',lw=1);ax.set_xticks(range(len(conds)),[labels[c] for c in conds],fontsize=8);ax.set_ylim(0,63);ax.set_ylabel('C accuracy (%)');ax.set_title(title,loc='left',fontsize=10)
finish(fig,'fig3_controls')
fig,axs=plt.subplots(1,2,figsize=(10,3.0))
for c,col in [('reset',colors[0]),('continue',colors[1])]:
 gs=[g11('shared',1,m='hop2'),g11(c,2,m='hop2'),g11(c,3,m='hop2')];axs[0].errorbar([1,2,3],[g['accuracy_pct']['mean'] for g in gs],yerr=[g['accuracy_pct']['sample_sd'] for g in gs],marker='o',capsize=3,color=col,label=c)
 ds=e11['target_diagnostics'];vv=[100*ds['base_targets']['target_correct_share']]+[100*st.mean(ds[f'{name}_targets_r{r}_s{k}']['target_correct_share'] for k in range(3)) for name,r in [('shared',1),(c,2),(c,3)]];ee=[0]+[100*st.stdev(ds[f'{name}_targets_r{r}_s{k}']['target_correct_share'] for k in range(3)) for name,r in [('shared',1),(c,2),(c,3)]];axs[1].errorbar(range(4),vv,yerr=ee,marker='o',capsize=3,color=col,label=c)
axs[0].axhline(D['baseline']['C_bothNew']['q2_ok'],ls='--',color='#888',label='Base');axs[0].set_xticks([1,2,3],['R1','R2','R3']);axs[0].set_ylabel('C single-hop 2 accuracy (%)');axs[0].set_ylim(73.5,80);axs[0].legend(frameon=False);axs[0].set_title('(a) Retention cost',loc='left',fontsize=10)
axs[1].set_xticks(range(4),['Base','R1','R2','R3']);axs[1].set_ylabel('Generated target accuracy (%)');axs[1].set_ylim(48.5,53);axs[1].legend(frameon=False);axs[1].set_title('(b) No clear target-quality improvement',loc='left',fontsize=10);finish(fig,'fig4_retention')
print(json.dumps({'old_source_hashes_verified':checked,'exp11_training_runs':len(stages),'exp11_prediction_labels_verified':nchecked,'baseline_drift':drift,'accounted_stage_hours':timing/3600,'source_files':len(manifest)},ensure_ascii=False,indent=2))
