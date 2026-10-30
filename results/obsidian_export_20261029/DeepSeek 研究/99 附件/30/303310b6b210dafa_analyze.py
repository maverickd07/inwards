"""Paired item bootstrap; no training-seed uncertainty is claimed."""
import argparse,json
from pathlib import Path
import numpy as np
from prompts import CONDITIONS
from scoring import score_row,reference_audit
SETS=['D500_known313','fixed303']
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def write(p,rows):
 with Path(p).open('x') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def ci(values):return np.quantile(values,[.025,.975]).tolist()
def summarize(scored,bootstrap=20000):
 lookup={(r['id'],r['condition']):r for r in scored};ids=sorted({r['id'] for r in scored})
 assert len(lookup)==len(scored)==len(ids)*5
 result={'bootstrap':{'B':bootstrap,'seed':20261022,'unit':'paired question','method':'percentile'},'sets':{}}
 for label in SETS:
  keys=[i for i in ids if lookup[i,'C0'][label]]
  if not keys:continue
  records=[[lookup[i,c] for c in CONDITIONS] for i in keys];n=len(keys)
  mat=np.array([[r['score']['correct'] for r in rs] for rs in records],dtype=float);acc=mat.mean(axis=0);failure=mat[:,0]==0;nf=int(failure.sum())
  rng=np.random.default_rng(20261022);samples=[]
  for start in range(0,bootstrap,256):
   idx=rng.integers(0,n,size=(min(256,bootstrap-start),n));samples.append(mat[idx].mean(axis=1))
  bs=np.concatenate(samples);den=acc[4]-acc[0];bd=bs[:,4]-bs[:,0];positive=bd>0;bad=int((~positive).sum())
  info={'n':n,'C0_failure_n':nf,'denominator_C3_minus_C0':float(den),'denominator_ci95':ci(bd),'bootstrap_nonpositive_denominator_n':bad,'bootstrap_nonpositive_denominator_fraction':bad/bootstrap,'ratio_CI_unstable':bool(bad),'conditions':{}}
  for j,c in enumerate(CONDITIONS):
   rr=[rs[j] for rs in records];r_failure=[rr[i] for i in np.flatnonzero(failure)]
   known_wrong=sum(r['country_correct'] is False for r in r_failure);unknown=sum(r['country_correct'] is None for r in r_failure)
   entry={'correct_n':int(mat[:,j].sum()),'accuracy':float(acc[j]),'accuracy_ci95':ci(bs[:,j]),'repair_n':int(((mat[:,j]==1)&failure).sum()),'regression_n':int(((mat[:,j]==0)&(~failure)).sum()),'historical_n':sum(r['score']['status']=='HISTORICAL' for r in rr),'review_n':sum(r['score']['status']=='REVIEW' for r in rr),'truncated_n':sum(r['truncated'] for r in rr),'C0_failure_wrong_country_proxy_n':sum(bool(r['wrong_country_proxy_candidates']) for r in r_failure),'explicit_wrong_country_n':known_wrong if c=='C2' else None,'explicit_unknown_country_n':unknown if c=='C2' else None,'explicit_wrong_country_rate':known_wrong/nf if nf and c=='C2' else None,'explicit_country_error_bounds': [known_wrong/nf,(known_wrong+unknown)/nf] if nf and c=='C2' else None,'internal_wrong_country_rate':None}
   entry['C0_failure_wrong_country_proxy_rate']=entry['C0_failure_wrong_country_proxy_n']/nf if nf else None
   entry['recovery']=float((acc[j]-acc[0])/den) if den>0 else None
   ratios=(bs[positive,j]-bs[positive,0])/bd[positive]
   entry['recovery_ci95']=ci(ratios) if den>0 and bad==0 else None
   entry['recovery_positive_denominator_conditional_ci95']=ci(ratios) if den>0 and bad and len(ratios) else None
   info['conditions'][c]=entry
  diff=acc[1]-acc[2];diffbs=bs[:,1]-bs[:,2]
  info['C1_minus_ctrl']={'accuracy_difference':float(diff),'accuracy_difference_ci95':ci(diffbs),'recovery_difference':float(diff/den) if den>0 else None,'recovery_difference_ci95':ci(diffbs/bd) if den>0 and bad==0 else None}
  x=info['conditions']['C1'];clear=info['C1_minus_ctrl']['accuracy_difference_ci95'][0]>0
  info['prediction_point_threshold_and_control_met']=bool(x['recovery'] is not None and x['recovery']>=.5 and clear)
  info['strong_selection_consistent_support']=bool(x['recovery_ci95'] and x['recovery_ci95'][0]>=.5 and clear)
  info['low_recovery_threshold_support']=bool(x['recovery_ci95'] and x['recovery_ci95'][1]<=.2)
  # Same item set; uncertain answers are all counted correct in sensitivity only.
  reviewmat=np.array([[r['score']['needs_review'] if 'needs_review' in r['score'] else r['score']['status']=='REVIEW' for r in rs] for rs in records])
  sens=np.maximum(mat,reviewmat).mean(axis=0);sd=sens[4]-sens[0]
  info['all_review_correct_sensitivity']={c:{'accuracy':float(sens[j]),'recovery':float((sens[j]-sens[0])/sd) if sd>0 else None} for j,c in enumerate(CONDITIONS)}
  result['sets'][label]=info
 return result

def analyze(raw_path,out,bootstrap=20000):
 out=Path(out);out.mkdir(exist_ok=False)
 raw=read(raw_path);scored=[score_row(r) for r in raw];summary=summarize(scored,bootstrap)
 write(out/'per_item.jsonl',scored);write(out/'review_queue.jsonl',[r for r in scored if r['score']['status']=='REVIEW'])
 roster=list({r['id']:r for r in raw}.values());audit=reference_audit(roster);write(out/'reference_discrepancies.jsonl',audit)
 summary['reference_discrepancy_item_n']=len(audit)
 summary['smoke_test']=any(r.get('smoke_test',False) for r in raw)
 (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 lines=['exp19 pointer — '+('LOCAL TINY RANDOM MODEL SMOKE, NOT QWEN3-14B RESULTS' if summary['smoke_test'] else 'Qwen3-14B inference results'),'Paired question bootstrap B='+str(bootstrap)+', seed=20261022. No ratio clipping.','Country errors: final-only conditions are unobservable (NA); proxy is answer consistency, not causal evidence.','Historical currency is incorrect; REVIEW primary=incorrect. Reference snapshot discrepancies='+str(len(audit))+'.']
 for label,info in summary['sets'].items():
  lines+=['',f"{label}: n={info['n']}, C0 failures={info['C0_failure_n']}",f"C3-C0={info['denominator_C3_minus_C0']:.6f}, 95% CI={info['denominator_ci95']}; nonpositive bootstrap denominators={info['bootstrap_nonpositive_denominator_fraction']:.4%}"]
  for c,x in info['conditions'].items():
   lines.append(f"{c}: accuracy={x['accuracy']:.4%} ({x['correct_n']}/{info['n']}), 95% CI={x['accuracy_ci95']}; recovery={x['recovery']}, 95% CI={x['recovery_ci95']}; historical={x['historical_n']}, review={x['review_n']}, repairs={x['repair_n']}, regressions={x['regression_n']}; on C0 failures explicit wrong-country={x['explicit_wrong_country_rate'] if c=='C2' else 'NA'}, unknown={x['explicit_unknown_country_n']}, proxy={x['C0_failure_wrong_country_proxy_rate']}")
  lines+=['C1 minus control: '+json.dumps(info['C1_minus_ctrl']), 'Prediction point threshold plus control: '+str(info['prediction_point_threshold_and_control_met']), 'Strong selection-consistent support: '+str(info['strong_selection_consistent_support']), 'Low recovery threshold support: '+str(info['low_recovery_threshold_support']), 'All REVIEW-correct sensitivity: '+json.dumps(info['all_review_correct_sensitivity'])]
  if info['ratio_CI_unstable']:lines.append('Ratio CI UNSTABLE: conditional interval only in summary.json; no unconditional 95% claim.')
 lines+=['','Stop at this stage. Prompt effects do not uniquely distinguish internal selection from retrieval. Fixed303 contains prior-selected items; the two sets overlap.']
 (out/'report.txt').write_text('\n'.join(lines)+'\n');return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--out',required=True);p.add_argument('--bootstrap',type=int,default=20000);a=p.parse_args();analyze(a.raw,a.out,a.bootstrap)
