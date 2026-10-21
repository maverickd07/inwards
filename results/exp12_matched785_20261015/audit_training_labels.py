"""CPU-only parallel audit; selection does not depend on corrected labels."""
import argparse,collections,json,math,statistics
from pathlib import Path
from reuse import read,key,js,sha
from training_label_scorer import score

def summary(rows):
 n=len(rows);out={'n':n,'groups':{}}
 for c in ['context','direct']:
  good=sum(r['revised'][c]['correct'] for r in rows);review=sum(r['revised'][c]['needs_review'] for r in rows)
  out['groups'][c]={'old_correct':sum(r['old'][c] for r in rows),'revised_correct':good,'revised_accuracy_pct':100*good/n if n else None,'needs_review':review,'mean_target_characters':statistics.mean(len(r[c+'_target']) for r in rows) if rows else None,'changed_labels':sum(r['old'][c]!=r['revised'][c]['correct'] for r in rows)}
 def classify(r):
  c,d=(r['revised'][x]['correct'] for x in ['context','direct'])
  return ('both_correct' if d else 'context_only_correct') if c else ('direct_only_correct' if d else 'both_wrong')
 counts=collections.Counter(map(classify,rows));out['four_classes']={s:counts[s] for s in ['both_correct','both_wrong','context_only_correct','direct_only_correct']}
 wins,losses=counts['context_only_correct'],counts['direct_only_correct'];m=wins+losses
 out['mcnemar_exact_p']=min(1.,2*sum(math.comb(m,i) for i in range(min(wins,losses)+1))/(1<<m)) if m else 1.
 out['sensitivity']={}
 for policy in ['all_wrong','all_correct','exclude_pair_union']:
  retained=[r for r in rows if policy!='exclude_pair_union' or not any(r['revised'][c]['needs_review'] for c in ['context','direct'])]
  scores={c:sum(r['revised'][c]['correct'] or (policy=='all_correct' and r['revised'][c]['needs_review']) for r in retained) for c in ['context','direct']}
  out['sensitivity'][policy]={'n':len(retained),'context_correct':scores['context'],'direct_correct':scores['direct'],'difference_pp':100*(scores['context']-scores['direct'])/len(retained) if retained else None}
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,required=True);ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 a.out.mkdir(parents=True,exist_ok=False)
 allrows=read(a.inputs/'all864_target_classes.jsonl');selected={key(r) for r in read(a.inputs/'selected_target_classes.jsonl')}
 capitals=[x for r in allrows if r['category']=='birthplace_capital' for x in r['gold']]
 rows=[]
 for r in allrows:
  item={k:v for k,v in r.items() if k not in ('v3','v3_class','v3_status')}
  item['selected']=key(r) in selected
  item['revised']={c:score(r[c+'_target'],r['gold'],r['category'],capitals) for c in ['context','direct']}
  if r['targets_exactly_identical']:assert item['revised']['context']==item['revised']['direct']
  rows.append(item)
 chosen=[r for r in rows if r['selected']];different=[r for r in chosen if not r['targets_exactly_identical']]
 assert len(rows)==864 and len(chosen)==785 and len(different)==404
 from engine_io import write
 write(a.out/'all864_labels.jsonl',rows);write(a.out/'selected785_labels.jsonl',chosen);write(a.out/'different404_labels.jsonl',different)
 write(a.out/'review_queue.jsonl',[r for r in rows if any(r['revised'][c]['needs_review'] for c in ['context','direct'])])
 report={'note':'Automatic accepted-answer accuracy, not independent factual truth; audit labels never change training allocation. 404 are training targets, not evaluation cases.', 'all864':summary(rows),'selected785':summary(chosen),'different404':summary(different),'identical381':summary([r for r in chosen if r['targets_exactly_identical']]),'selected_by_category':{cat:summary([r for r in chosen if r['category']==cat]) for cat in sorted({r['category'] for r in chosen})},'different404_by_category':{cat:summary([r for r in different if r['category']==cat]) for cat in sorted({r['category'] for r in different})}}
 diag={key(r):r for r in read(a.source/'paired_targets/paired_diagnostics.jsonl')}
 if selected<=set(diag):
  report['different404_cached_nll']={field:statistics.mean(diag[key(r)][field]['mean_nll'] for r in different) for field in ['context_under_context','context_under_direct','direct_under_context','direct_under_direct']}
 else:raise ValueError('Missing cached 404 diagnostics')
 js(a.out/'report.json',report)
 js(a.out/'complete.json',{'rows':864,'selected':785,'different':404,'scorer_sha256':sha(Path(__file__).parent/'training_label_scorer.py'),'policy_sha256':sha(Path(__file__).parent/'training_label_policy.json'),'no_inference':True})
 print('Training label audit complete: 864 cached pairs, including frozen785/different404.',flush=True)
if __name__=='__main__':main()
