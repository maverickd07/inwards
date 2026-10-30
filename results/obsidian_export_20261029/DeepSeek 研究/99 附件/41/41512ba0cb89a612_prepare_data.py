"""One-shot outcome-independent data preparation; never invokes a model."""
import collections,json,re,sys,unicodedata
from pathlib import Path
from common import HERE,read,write,js,sha,digest
sys.path.insert(0,str(HERE/'upstream'))
from metrics.answer import normalize_answer

def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())
def resolve_steps(row,aliases):
 steps=row['question_decomposition'];out=[]
 context_by_id={p['idx']:p for p in row['paragraphs']}
 for i,step in enumerate(steps):
  def replace(m):
   j=int(m[1])-1
   if not 0<=j<i:raise ValueError('Invalid/forward reference: '+row['id'])
   return steps[j]['answer']
  resolved=re.sub(r'#(\d+)',replace,step['question'])
  if re.search(r'#\d+',resolved):raise ValueError('Unresolved placeholder')
  if '>>' in resolved:
   if resolved.count('>>')!=1:raise ValueError('Unsupported relation query')
   entity,relation=[s.strip() for s in resolved.split('>>')]
   question=f'What is the {relation} of {entity}?'
  else:question=resolved
  # Same upstream alias source, restricted to the hop's supporting paragraph.
  paragraph=context_by_id[step['paragraph_support_idx']]
  context=(paragraph['title']+' '+paragraph['paragraph_text']).lower()
  refs=[step['answer']]+[a for a in aliases.get(str(step['id']),[]) if a.lower().strip() in context]
  refs=list(dict.fromkeys(refs))
  out.append({'step':i+1,'source_singlehop_id':step['id'],'source_question':step['question'],'question':question,'answers':refs,'canonical_answer':step['answer'],'query_id':digest(question),'oracle_previous_answers':bool(re.search(r'#\d+',step['question']))})
 return out

def main():
 out=HERE/'data';out.mkdir()
 source=HERE.parent/'exp12_matched785_20261015/frozen_inputs/plan/train.jsonl'
 old=read(source);people=sorted({norm(r['person']) for r in old})
 pattern=re.compile(r'(?<!\w)(?:'+'|'.join(map(re.escape,people))+r')(?!\w)')
 raw=read(HERE/'raw_data/musique_ans_v1.0_dev.jsonl');aliases=json.loads((HERE/'upstream/.answer_aliases.json').read_text())
 dup=collections.Counter(norm(r['question']) for r in raw);kept=[];excluded=[]
 def make(r):
  assert r['answerable'] and len(r['question_decomposition']) in (2,3,4)
  hops=resolve_steps(r,aliases)
  return {'id':r['id'],'question':r['question'],'answers':list(dict.fromkeys([r['answer']]+r['answer_aliases'])),'n_hops':len(hops),'shape':r['id'].split('__')[0],'query_id':digest(r['question']),'hops':hops}
 for r in raw:
  prepared=make(r)
  text=norm(' '.join([r['question']]+[s['question'] for s in prepared['hops']]))
  matched=sorted(set(pattern.findall(text)))
  reasons=[]
  if dup[norm(r['question'])]>1:reasons.append('duplicate_normalized_composed_question_exclude_entire_group')
  if matched:reasons.append('exact_name_overlap_with_matched785_training_people')
  if reasons:excluded.append({'id':r['id'],'question':r['question'],'reasons':reasons,'matched_people':matched});continue
  kept.append(prepared)
 assert len({r['id'] for r in kept})==len(kept)
 write(out/'items.jsonl',kept);write(out/'excluded.jsonl',excluded)
 queries={}
 for r in kept:
  for q in [r]+r['hops']:
   if q['query_id'] in queries:assert queries[q['query_id']]['question']==q['question']
   else:queries[q['query_id']]={'query_id':q['query_id'],'question':q['question']}
 write(out/'queries.jsonl',list(queries.values()))
 # Timing fixture comes only from training, not the primary evaluation pool.
 dev_subids={s['source_singlehop_id'] for r in kept for s in r['hops']}
 dev_questions={norm(q['question']) for q in queries.values()}
 pilot=[]
 for r in sorted(read(HERE/'raw_data/musique_ans_v1.0_train.jsonl'),key=lambda r:digest('exp13-pilot-'+r['id'])):
  if any(s['id'] in dev_subids for s in r['question_decomposition']):continue
  a=make(r)
  if norm(a['question']) in dev_questions:continue
  if pattern.search(norm(' '.join([a['question']]+[s['question'] for s in a['hops']]))):continue
  pilot.append({'query_id':digest('pilot:'+a['question']),'question':a['question']})
  if len(pilot)==64:break
 assert len(pilot)==64
 write(out/'timing_queries.jsonl',pilot)
 groups={}
 for r in kept:groups.setdefault(r['n_hops'],0);groups[r['n_hops']]+=1
 manifest={'dataset':'MuSiQue-Ans v1.0 dev','upstream':json.loads((HERE/'upstream_commit.json').read_text()),'raw_dev_sha256':sha(HERE/'raw_data/musique_ans_v1.0_dev.jsonl'),'raw_zip_sha256':sha(HERE/'raw_data/musique_v1.0.zip'),'raw_train_sha256':sha(HERE/'raw_data/musique_ans_v1.0_train.jsonl'),'official_alias_source_sha256':sha(HERE/'upstream/.answer_aliases.json'),'original_dev_n':len(raw),'original_hops':dict(collections.Counter(len(r['question_decomposition']) for r in raw)),'retained_n':len(kept),'retained_by_hop':groups,'excluded_n':len(excluded),'exclusion_reasons':dict(collections.Counter(reason for r in excluded for reason in r['reasons'])),'matched785_train_sha256':sha(source),'deduplicated_inference_queries_per_model':len(queries),'composed_records_per_model':len(kept),'hop_records_per_model':sum(r['n_hops'] for r in kept),'model_count':13,'total_main_inference_queries':13*len(queries),'timing_queries_baseline_only':64,'selection_uses_model_predictions':False,'known_overlap_limit':'Exact training-person names only. Does not establish entity-disjointness for aliases or absence of pretraining contamination.','hop_alias_policy':'Canonical answer plus upstream single-hop aliases occurring in that hop supporting paragraph. Hop diagnostic only; composed scoring uses official released answer_aliases exactly.'}
 js(out/'manifest.json',manifest)
 print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
