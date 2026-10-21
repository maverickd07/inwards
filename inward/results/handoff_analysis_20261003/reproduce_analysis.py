"""Run from any directory. New output directory required; no overwrites."""
import argparse,csv,json,random,hashlib,collections,pathlib,math
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[1]
LABELS=['GENUINE','AMBIGUOUS_GOLD','FORMAT','SCORER_ERROR']
def read(p):
 return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def key(r): return (r['category'],r['person'])
def ident(r): return hashlib.sha256(json.dumps(key(r),ensure_ascii=False).encode()).hexdigest()[:16]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=pathlib.Path,required=True);a=ap.parse_args();a.out.mkdir(exist_ok=False)
 src=ROOT/'results/exp1v2_Qwen3-14B/items.jsonl'
 rows=[dict(r,source_line=i) for i,r in enumerate(read(src),1)]
 eligible=[r for r in rows if r['category'].startswith('birthplace') and 'common_name' not in r['category'] and r['q1'] is True and r['q2'] is True and r['clear'] is False]
 sample=random.Random(7).sample(eligible,120)
 assert sample==json.load(open(HERE/'sample_120.json'))
 decisions=json.load(open(HERE/'label_decisions.json'))
 data={key(r):r for r in json.load(open(ROOT/'data/compositional_celebrities.json'))['data']}
 labeled=[]
 for i,r in enumerate(sample):
  label,reason=decisions.get(str(i),['GENUINE',f"明确出生国问题回答 {r['pred_clear']!r}，与标准答案 {r['gold']!r} 不符；未找到足以确认别名、历史解释或取整合理性的依据，按保守规则归为真错。"])
  e=data[key(r)]
  labeled.append(dict(id=ident(r),category=r['category'],person=r['person'],q1_answer=r['pred_q1'],q2_answer=r['pred_q2'],composed_answer=r['pred_clear'],gold=json.dumps(r['gold'],ensure_ascii=False),label=label,reason=reason,source_line=r['source_line'],clear_question=e['Question'].replace('the birthplace of '+r['person'],'the country where '+r['person']+' was born'),q1_gold=json.dumps(e['A1'],ensure_ascii=False),q2_gold=json.dumps(e['A2'],ensure_ascii=False)))
 with open(a.out/'failures_120_labeled.csv','w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=labeled[0].keys());w.writeheader();w.writerows(labeled)
 counts=collections.Counter(r['label'] for r in labeled)
 by={}
 for cat in sorted({r['category'] for r in labeled}):
  rs=[r for r in labeled if r['category']==cat];c=collections.Counter(r['label'] for r in rs)
  by[cat]={'n':len(rs),**{l:{'n':c[l],'share':c[l]/len(rs)} for l in LABELS}}
 summary={'source':str(src),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'seed':7,'eligible':len(eligible),'sample_n':120,'counts':{l:counts[l] for l in LABELS},'shares':{l:counts[l]/120 for l in LABELS},'by_category':by,'labeling':'Individual model review; conservative GENUINE for uncertainty; historical alternative accepted when question has no time constraint. Not independent human validation.'}
 json.dump(summary,open(a.out/'label_summary.json','w'),ensure_ascii=False,indent=2)
 methods=['self','self_s1','gold','hops'];sets=['A_newPerson','B_newCategory','C_bothNew'];missing=[];overlap={}
 for subset in sets:
  files={m:ROOT/f'results/inward_{m}_Qwen3-14B/eval_{subset}.jsonl' for m in methods}
  absent=[str(p) for p in files.values() if not p.is_file()];missing+=absent
  if absent:continue
  ms={m:read(p) for m,p in files.items()}
  for m,rs in ms.items():
   assert len({key(r) for r in rs})==len(rs),('duplicate identity',m,subset)
   assert all(type(r['base_ok']) is bool and type(r['trained_ok']) is bool for r in rs)
  ks={m:{key(r) for r in rs} for m,rs in ms.items()}
  assert all(v==ks['self'] for v in ks.values()),('unmatched evaluation sets',subset)
  maps={m:{key(r):r for r in rs} for m,rs in ms.items()}
  assert all(len({maps[m][k]['base_ok'] for m in methods})==1 for k in ks['self']),('baseline disagreement',subset)
  result={}
  for cat in ['ALL']+sorted({r['category'] for r in ms['self']}):
   fixed={m:{key(r) for r in rs if (cat=='ALL' or r['category']==cat) and not r['base_ok'] and r['trained_ok']} for m,rs in ms.items()}
   broke={m:{key(r) for r in rs if (cat=='ALL' or r['category']==cat) and r['base_ok'] and not r['trained_ok']} for m,rs in ms.items()}
   entry={'methods':{m:{'fixed':len(fixed[m]),'broke':len(broke[m])} for m in methods},'pairs':{}}
   for x,y in [('self','gold'),('self_s1','gold'),('self','self_s1')]:
    u=fixed[x]|fixed[y];entry['pairs'][x+'__'+y]={'intersection':len(fixed[x]&fixed[y]),'union':len(u),'jaccard':len(fixed[x]&fixed[y])/len(u) if u else None}
   result[cat]=entry
  overlap[subset]=result
 cross={}; pending=[]
 labels_by_key={key(r):r for r in labeled}
 for m in ['self','self_s1','gold']:
  path=ROOT/f'results/inward_{m}_Qwen3-14B/eval_C_bothNew.jsonl'
  if not path.is_file():continue
  fixed=[r for r in read(path) if not r['base_ok'] and r['trained_ok']]
  classified=[]
  for r in fixed:
   prior=labels_by_key.get(key(r))
   if prior and prior['composed_answer']==r['base_pred']:
    classified.append(prior['label'])
   else:pending.append(dict(method=m,id=ident(r),**r))
  cross[m]={'fixed_n':len(fixed),'matched_label_n':len(classified),'matched_counts':dict(collections.Counter(classified)),'needs_review':len(fixed)-len(classified)}
 json.dump({'status':'blocked_missing_files' if missing else 'review_pending' if pending else 'complete','results':cross,'note':'Only reuse a label if category, person and exact base prediction all match; classify unmatched base failures before drawing conclusions.'},open(a.out/'C_repair_labels.json','w'),ensure_ascii=False,indent=2)
 with open(a.out/'C_repairs_to_review.jsonl','w') as f:
  for r in pending:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 json.dump({'missing':missing,'results':overlap},open(a.out/'repair_overlap.json','w'),indent=2)
 print(json.dumps(summary['counts'],ensure_ascii=False));print('Missing evaluation files:',len(missing))
if __name__=='__main__':main()
