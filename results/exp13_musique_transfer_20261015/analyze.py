"""Fixed primary two-hop EM and descriptive secondary outcomes; six paired seeds."""
import argparse,json,statistics,math
from pathlib import Path
from common import read,write,js,NAMES
T5=2.570581835636314

def moments(values):
 assert len(values)==6
 if any(x is None for x in values):return {'per_seed_pp':values,'mean_pp':None,'t95_pp':None}
 mean=statistics.mean(values);half=T5*statistics.stdev(values)/math.sqrt(6)
 return {'per_seed_pp':values,'mean_pp':mean,'t95_pp':[mean-half,mean+half]}

def compare(groups,ids,metric):
 if not ids:return {'n':0,'baseline_pct':None,'context_pct':None,'direct_pct':None,'context_minus_direct':moments([None]*6),'context_minus_base':moments([None]*6),'direct_minus_base':moments([None]*6)}
 avg=lambda name:100*statistics.mean(groups[name][i]['scores'][metric] for i in ids)
 base=avg('baseline');cs=[avg(f'context_s{s}') for s in range(6)];ds=[avg(f'direct_s{s}') for s in range(6)]
 return {'n':len(ids),'baseline_pct':base,'context_pct':statistics.mean(cs),'direct_pct':statistics.mean(ds),'context_per_seed_pct':cs,'direct_per_seed_pct':ds,'context_minus_direct':moments([c-d for c,d in zip(cs,ds)]),'context_minus_base':moments([c-base for c in cs]),'direct_minus_base':moments([d-base for d in ds])}

def analyze(run):
 items=read(run/'inputs/items.jsonl');ids=[r['id'] for r in items];groups={}
 for name in NAMES:
  rows=read(run/name/'scored_items.jsonl');assert [r['id'] for r in rows]==ids
  groups[name]={r['id']:r for r in rows}
 strata=json.loads((run/'baseline/baseline_strata.json').read_text());known=set(strata['all_oracle_hops_correct_ids']);gap=set(strata['all_hops_correct_but_composed_wrong_ids'])
 report={'primary':'full pre-frozen two-hop EM; context-direct, with context-base also required for positive transfer','interval_scope':'six paired training seeds on a fixed dataset; no question-sampling/pretraining/label uncertainty','groups':{},'diagnostics':{},'baseline_strata_counts':{},'no_new_training':True}
 for hop in [2,3,4,'all']:
  subset=[r['id'] for r in items if hop=='all' or r['n_hops']==hop]
  label=str(hop)+'hop' if hop!='all' else 'all'
  report['groups'][label]={metric:compare(groups,subset,metric) for metric in ['em','f1']}
  report['baseline_strata_counts'][label]={'n':len(subset),'all_oracle_hops_correct':len(set(subset)&known),'all_hops_correct_composed_wrong':len(set(subset)&gap)}
  report['diagnostics'][label]={
   'fixed_baseline_all_hops_correct':compare(groups,[i for i in subset if i in known],'em'),
   'fixed_baseline_gap':compare(groups,[i for i in subset if i in gap],'em'),
   'small_known_stratum':len(set(subset)&known)<100,
   'models':{}}
  for name in NAMES:
   rows=[groups[name][i] for i in subset];hops=[h for r in rows for h in r['hops']]
   report['diagnostics'][label]['models'][name]={'composed_truncated':sum(r['hit_token_limit'] for r in rows),'composed_think_markup':sum(r['contains_think_markup'] for r in rows),'all_oracle_hops_em_pct':100*statistics.mean(r['all_oracle_hops_em'] for r in rows) if rows else None,'per_hop_em_pct':100*statistics.mean(h['scores']['em'] for h in hops) if hops else None,'per_hop_canonical_only_em_pct':100*statistics.mean(h['canonical_em'] for h in hops) if hops else None,'per_hop_f1_pct':100*statistics.mean(h['scores']['f1'] for h in hops) if hops else None}
 primary=report['groups']['2hop']['em'];effect=primary['context_minus_direct'];base_effect=primary['context_minus_base']
 if effect['t95_pp'] is None:verdict='No eligible primary items; no conclusion.'
 elif effect['t95_pp'][0]>0 and base_effect['t95_pp'][0]>0:verdict='Positive transfer supported in this predefined closed-book setting; not a knowledge-use mechanism proof.'
 elif effect['t95_pp'][0]>0:verdict='Context beats direct, but positive improvement over baseline is not established; do not call it successful transfer.'
 elif effect['t95_pp'][1]<0:verdict='Context underperforms direct on the primary endpoint; negative relative transfer.'
 else:verdict='No clear primary advantage; not equivalence or proof that the broader method fails.'
 report['predeclared_interpretation']=verdict
 (run/'analysis').mkdir();js(run/'analysis/report.json',report)
 lines=['EXP13 MuSiQue closed-book transfer; no new training.','PRIMARY: fixed full two-hop EM; EM/F1 are official answer metrics, not official open-book leaderboard results.',verdict]
 for group in ['2hop','3hop','4hop','all']:
  for metric in ['em','f1']:
   obj=report['groups'][group][metric];diff=obj['context_minus_direct']
   lines.append(f"{group} {metric}: n={obj['n']}; base={obj['baseline_pct']}; context={obj['context_pct']}; direct={obj['direct_pct']}; delta_pp={diff['mean_pp']}; t95={diff['t95_pp']}; per_seed={diff['per_seed_pp']}")
 lines.append('Baseline oracle-hop strata: '+json.dumps(report['baseline_strata_counts']))
 lines.append('Secondary strata never redefine the primary set. Oracle single-hop questions expose reference bridge entities; these diagnostics do not prove internal knowledge activation. Alias limitations and public-data pretraining overlap remain.')
 lines.append('STOP: fixed13-model evaluation complete; no automatic extra seeds, datasets, tuning or training.')
 with (run/'analysis/report.txt').open('x') as f:f.write('\n'.join(lines)+'\n')
 print('\n'.join(lines),flush=True)
 return report
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);analyze(ap.parse_args().run)
