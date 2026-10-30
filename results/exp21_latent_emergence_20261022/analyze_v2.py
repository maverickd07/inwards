"""Frozen item-bootstrap analysis; never-hit has an explicit censored bin."""
import argparse,json
from pathlib import Path
import numpy as np

def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def interval(x):return [float(v) for v in np.percentile(x,[2.5,97.5])]
def stats(x,B,rng):
 x=np.asarray(x,dtype=float)
 if not len(x):return {'n':0,'mean':None,'median':None,'mean_ci95':None,'median_ci95':None}
 b=x[rng.integers(0,len(x),(B,len(x)))]
 return {'n':len(x),'mean':float(x.mean()),'median':float(np.median(x)),'mean_ci95':interval(b.mean(1)),'median_ci95':interval(np.median(b,axis=1))}
def emergence_stats(rr,target,B,rng):
 L=rr[0]['num_layers'] if rr else None
 hits=[r['emergence'][target] for r in rr if r['emergence'][target] is not None]
 return {'censored_L_plus_1':stats([r['emergence'][target] if r['emergence'][target] is not None else r['num_layers']+1 for r in rr],B,rng),'hit_only':stats(hits,B,rng),'never_hit_n':len(rr)-len(hits),'never_hit_fraction':(len(rr)-len(hits))/len(rr) if rr else None,'sentinel':L+1 if L else None}
def paired(a,b,target,B,rng):
 assert len(a)==len(b) and all(x['id']==y['id'] for x,y in zip(a,b))
 if not a:return {'n':0}
 x=np.array([r['emergence'][target] if r['emergence'][target] is not None else r['num_layers']+1 for r in a]); y=np.array([r['emergence'][target] if r['emergence'][target] is not None else r['num_layers']+1 for r in b])
 ix=rng.integers(0,len(x),(B,len(x))); bsmean=(y[ix]-x[ix]).mean(1);bsmedian=np.median(y[ix],axis=1)-np.median(x[ix],axis=1)
 return {'n':len(x),'delta_mean':float((y-x).mean()),'delta_mean_ci95':interval(bsmean),'difference_of_medians':float(np.median(y)-np.median(x)),'difference_of_medians_ci95':interval(bsmedian),'paired_delta_median':float(np.median(y-x)),'earlier_n':int((y<x).sum()),'same_n':int((y==x).sum()),'later_n':int((y>x).sum()),'both_hit_n':int(((x<=a[0]['num_layers'])&(y<=a[0]['num_layers'])).sum())}
def analyze(path,out,B=10000):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 out=Path(out);rows=read(path);assert rows
 lookup={(r['variant'],r['condition'],r['id']):r for r in rows};assert len(lookup)==len(rows)
 variants=list(dict.fromkeys(r['variant'] for r in rows));L=rows[0]['num_layers'];assert all(r['num_layers']==L and len(r['layers'])==L+1 and r['final_top1_verified'] for r in rows)
 roster={r['id']:r for r in rows};rng=np.random.default_rng(21)
 summary={'bootstrap':B,'seed':21,'never_hit_sentinel':L+1,'smoke':rows[0]['smoke'],'sets':{}}
 lines=['exp21 latent emergence','ENGINEERING SMOKE ONLY' if rows[0]['smoke'] else 'Qwen3-14B bf16; seeds 0,1; inference only','Negative paired shift = earlier. Never-hit uses L+1 (right-censored, not an observed layer).','Primary nonoverlap; all-item and leading-space sensitivity. Frozen exp19 scoring; REVIEW incorrect.','CIs: item bootstrap; rescued subsets are outcome-selected descriptive comparisons.','No causal inference from a first-token logit lens; no automatic seed expansion.','']
 for setname in ['fixed303','D500_known313']:
  setids=sorted(i for i,r in roster.items() if r[setname]);info={};summary['sets'][setname]=info
  for sensitivity in ['primary_nonoverlap','all_items','leading_space_nonoverlap','exclude_review_nonoverlap']:
   suffix='_space' if sensitivity=='leading_space_nonoverlap' else ''
   ids=[i for i in setids if sensitivity=='all_items' or roster[i]['targets']['bridge'+suffix]['id']!=roster[i]['targets']['answer'+suffix]['id']]
   block={'n':len(ids),'excluded_overlap_n':len(setids)-len(ids),'models':{},'paired':{}};info[sensitivity]=block
   for variant in variants:
    for condition in (['C0','C2'] if variant=='base' else ['C0']):
     rr=[lookup[variant,condition,i] for i in ids];label=variant+'_'+condition
     if sensitivity=='exclude_review_nonoverlap':rr=[r for r in rr if r['score']['status']!='REVIEW']
     groups={'all':rr,'C0_correct':[r for r in rr if lookup[variant,'C0',r['id']]['score']['correct']],'C0_wrong':[r for r in rr if not lookup[variant,'C0',r['id']]['score']['correct']]}
     ms={'n':len(rr),'accuracy':stats([r['score']['correct'] for r in rr],B,rng),'review_n':sum(r['score']['status']=='REVIEW' for r in rr),'historical_n':sum(r['score']['status']=='HISTORICAL' for r in rr),'groups':{},'correctness_relation':{}}
     block['models'][label]=ms
     for g,gr in groups.items():ms['groups'][g]={t:emergence_stats(gr,t+suffix,B,rng) for t in ['bridge','answer']}
     for target in ['bridge','answer']:
      rel={}
      for name,pred in [('early',lambda v:v is not None and v<=L//2),('late',lambda v:v is not None and v>L//2),('never',lambda v:v is None)]:
       gr=[r for r in rr if pred(r['emergence'][target+suffix])];rel[name]=stats([r['score']['correct'] for r in gr],B,rng)
      ms['correctness_relation'][target]=rel
     if sensitivity=='primary_nonoverlap':
      lines.append('Grouped correctness (early / late / never), accuracy and bootstrap CI: '+json.dumps(ms['correctness_relation']))
      lines.append(f'{setname} {label} n={len(rr)} correct={sum(r["score"]["correct"] for r in rr)} REVIEW={ms["review_n"]}')
      for g in groups:
       for t in ['bridge','answer']:
        ss=ms['groups'][g][t];st=ss['censored_L_plus_1'];lines.append(f'  {g}/{t}: n={st["n"]} median={st["median"]} CI={st["median_ci95"]} mean={st["mean"]} CI={st["mean_ci95"]} never={ss["never_hit_n"]}')
   for variant in variants:
    if variant=='base':continue
    pp={};block['paired'][variant+'_minus_base']=pp
    for g in ['all','base_correct','base_wrong','rescued']:
     ii=[i for i in ids if (g=='all' or g=='base_correct' and lookup['base','C0',i]['score']['correct'] or g=='base_wrong' and not lookup['base','C0',i]['score']['correct'] or g=='rescued' and not lookup['base','C0',i]['score']['correct'] and lookup[variant,'C0',i]['score']['correct'])]
     if sensitivity=='exclude_review_nonoverlap':ii=[i for i in ii if all(lookup[v,'C0',i]['score']['status']!='REVIEW' for v in ['base',variant])]
     pp[g]={t:paired([lookup['base','C0',i] for i in ii],[lookup[variant,'C0',i] for i in ii],t+suffix,B,rng) for t in ['bridge','answer']}
     if sensitivity=='primary_nonoverlap':lines.append(f'{setname} {variant}-base {g}: '+json.dumps(pp[g],ensure_ascii=False))
   for seed in [0,1]:
    c=f'context_s{seed}';d=f'direct_s{seed}'
    if c in variants and d in variants:
     ii=ids
     if sensitivity=='exclude_review_nonoverlap':ii=[i for i in ii if all(lookup[v,'C0',i]['score']['status']!='REVIEW' for v in [c,d])]
     block['paired'][c+'_minus_'+d]={t:paired([lookup[d,'C0',i] for i in ii],[lookup[c,'C0',i] for i in ii],t+suffix,B,rng) for t in ['bridge','answer']}
   if sensitivity=='primary_nonoverlap':
    for seed in [0,1]:
     key=f'context_s{seed}_minus_direct_s{seed}'
     if key in block['paired']:lines.append(f'{setname} {key}: '+json.dumps(block['paired'][key]))
  # All and own-C0-correct/wrong distributions; primary nonoverlap.
  primary=info['primary_nonoverlap'];pi=[i for i in setids if roster[i]['targets']['bridge']['id']!=roster[i]['targets']['answer']['id']]
  labels=list(primary['models'])
  for label in labels:
   variant,condition=label.rsplit('_',1);rr=[lookup[variant,condition,i] for i in pi]
   fig,axes=plt.subplots(2,3,figsize=(14,7),sharex=True)
   for j,(group,gr) in enumerate([('all',rr),('C0 correct',[r for r in rr if lookup[variant,'C0',r['id']]['score']['correct']]),('C0 wrong',[r for r in rr if not lookup[variant,'C0',r['id']]['score']['correct']])]):
    for k,t in enumerate(['bridge','answer']):
     ax=axes[k,j];xx=[r['emergence'][t] if r['emergence'][t] is not None else L+1 for r in gr];counts=np.bincount(xx,minlength=L+2)[1:L+2] if xx else np.zeros(L+1)
     ax.bar(np.arange(1,L+2),counts,color=['#4979b5']*L+['#ba6355']);ax.set_title(f'{t}: {group} (n={len(gr)})');ax.set_xlabel('First top-10 decoder layer; N = never');ax.set_ylabel('Items');ticks=sorted(set([1,*range(5,L+1,5),L,L+1]));ax.set_xticks(ticks,[str(x) if x<=L else 'N' for x in ticks])
   fig.suptitle(f'{setname} | {label} | nonoverlap | '+('SMOKE' if rows[0]['smoke'] else 'Qwen3-14B'));fig.tight_layout();fig.savefig(out/f'distribution_{setname}_{label}.png',dpi=150);plt.close(fig)
 lines+=['','Full numerical results, conditional-hit summaries, sensitivities and grouped correctness CIs: summary.json.','C2 groups use base C0 correctness, C2 accuracy itself scores only the Answer field.','Unresolved scoring/reference issues remain REVIEW/HISTORICAL; do not tune references or expand experiments.']
 (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');(out/'report.txt').write_text('\n'.join(lines)+'\n')
 return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();analyze(a.run/'per_item.jsonl',a.run)
