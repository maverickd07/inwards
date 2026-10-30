"""Independent numpy/scipy arithmetic check on existing production predictions."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.stats import t
from analyze import comparison
from reuse import HERE,read,key,js

def main():
 inputs=HERE/'frozen_inputs'
 groups={f'{c}_s{s}':read(inputs/'reference_v3'/f'{c}_s{s}.jsonl') for s in range(6) for c in ['context','direct']}
 primary=set(map(tuple,json.loads((inputs/'primary_D303_keys.json').read_text())['identities']))
 checks=0
 for subset in ['primary303','full500']:
  ids=[i for i,r in enumerate(groups['context_s0']) if subset=='full500' or key(r) in primary]
  for metric in ['composed','q1','q2']:
   for policy in ['all_wrong','all_correct','exclude_pair_union']:
    actual=comparison(groups,ids,metric,policy);diffs=[]
    for seed in range(6):
     scores=[];flags=[]
     for c in ['context','direct']:
      scores.append(np.array([groups[f'{c}_s{seed}'][i]['scores'][metric]['correct'] for i in ids],dtype=bool))
      flags.append(np.array([groups[f'{c}_s{seed}'][i]['scores'][metric]['needs_review'] for i in ids],dtype=bool))
     keep=~(flags[0]|flags[1]) if policy=='exclude_pair_union' else np.ones(len(ids),dtype=bool)
     c,d=[scores[j]|flags[j] if policy=='all_correct' else scores[j] for j in range(2)]
     diffs.append(100*np.mean(c[keep].astype(float)-d[keep].astype(float)))
     assert actual['pairs'][seed]['n']==int(keep.sum())
    mean=float(np.mean(diffs));half=float(t.ppf(.975,5)*np.std(diffs,ddof=1)/np.sqrt(6))
    assert np.allclose(actual['per_seed_pp'],diffs,rtol=0,atol=1e-10)
    assert np.allclose(actual['t95_pp'],[mean-half,mean+half],rtol=0,atol=1e-10)
    checks+=1
 # Future all-review predictions must not change the fixed primary; exclusion has no estimate.
 import copy
 edge=copy.deepcopy(groups)
 for rows in edge.values():
  for r in rows:r['scores']['composed'].update(correct=False,needs_review=True)
 ids=list(range(303))
 wrong=comparison(edge,ids,'composed','all_wrong');right=comparison(edge,ids,'composed','all_correct');excluded=comparison(edge,ids,'composed','exclude_pair_union')
 assert all(p['n']==303 for p in wrong['pairs']+right['pairs'])
 assert all(p['context_correct']==0 for p in wrong['pairs'])
 assert all(p['context_correct']==303 for p in right['pairs'])
 assert excluded['mean_pp'] is None and all(p['n']==0 for p in excluded['pairs'])
 js(HERE/'statistics_validation.json',{'independent_numpy_scipy_scenarios':checks,'t95_df5_verified':True,'all_review_denominator_edge_verified':True,'atol':1e-10})
 print('Independent statistics check passed:',checks,'scenarios and future-review edge case.')
if __name__=='__main__':main()
