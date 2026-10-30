"""Score saved answers after inference; reference answers never enter main prompts."""
import argparse,json
from pathlib import Path
from common import read,write,js,sha
from scoring import scores

def score_model(folder,items_path,baseline=False):
 items=read(items_path);preds=read(folder/'predictions.jsonl');by={r['query_id']:r for r in preds}
 assert len(by)==len(preds)
 rows=[]
 for item in items:
  p=by[item['query_id']];assert p['question']==item['question']
  hops=[]
  for hop in item['hops']:
   hp=by[hop['query_id']];assert hp['question']==hop['question']
   hops.append({'step':hop['step'],'source_singlehop_id':hop['source_singlehop_id'],'query_id':hop['query_id'],'prediction':hp['prediction'],'scores':scores(hp['prediction'],hop['answers']),'canonical_em':scores(hp['prediction'],[hop['canonical_answer']])['em']})
  rows.append({'id':item['id'],'n_hops':item['n_hops'],'shape':item['shape'],'prediction':p['prediction'],'scores':scores(p['prediction'],item['answers']),'hit_token_limit':p['hit_token_limit'],'contains_think_markup':p['contains_think_markup'],'hops':hops,'all_oracle_hops_em':all(h['scores']['em']==1 for h in hops)})
 write(folder/'scored_items.jsonl',rows)
 if baseline:
  js(folder/'baseline_strata.json',{'derived_from':'baseline only; before any adapter inference','items_sha256':sha(items_path),'all_oracle_hops_correct_ids':[r['id'] for r in rows if r['all_oracle_hops_em']],'all_hops_correct_but_composed_wrong_ids':[r['id'] for r in rows if r['all_oracle_hops_em'] and not r['scores']['em']],'policy':'Secondary diagnostic, never replace fixed full two-hop primary denominator; public aliases and oracle bridge inputs limit knowledge interpretation.'})
 js(folder/'scoring_complete.json',{'items':len(rows),'scored_sha256':sha(folder/'scored_items.jsonl')})
 print('Scored',folder.name,len(rows),'items',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--folder',type=Path,required=True);ap.add_argument('--items',type=Path,required=True);ap.add_argument('--baseline',action='store_true');a=ap.parse_args();score_model(a.folder,a.items,a.baseline)
