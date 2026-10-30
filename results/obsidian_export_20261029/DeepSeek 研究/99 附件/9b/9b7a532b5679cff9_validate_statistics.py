"""Independent SciPy check with nonzero synthetic outcomes; no model inference."""
import json
import numpy as np
from scipy.stats import t
from common import HERE,js,NAMES
from analyze import compare,moments
if __name__=='__main__':
 rng=np.random.default_rng(13);groups={};ids=[str(i) for i in range(137)]
 for name in NAMES:groups[name]={i:{'scores':{'em':int(rng.random()<.45),'f1':float(rng.random())}} for i in ids}
 for metric in ['em','f1']:
  result=compare(groups,ids,metric)
  c=np.array([100*np.mean([groups[f'context_s{s}'][i]['scores'][metric] for i in ids]) for s in range(6)])
  d=np.array([100*np.mean([groups[f'direct_s{s}'][i]['scores'][metric] for i in ids]) for s in range(6)])
  base=100*np.mean([groups['baseline'][i]['scores'][metric] for i in ids])
  for key,values in [('context_minus_direct',c-d),('context_minus_base',c-base),('direct_minus_base',d-base)]:
   m=float(values.mean());half=float(t.ppf(.975,5)*values.std(ddof=1)/np.sqrt(6));a=result[key]
   assert np.allclose(a['per_seed_pp'],values,rtol=0,atol=1e-10)
   assert np.allclose(a['t95_pp'],[m-half,m+half],rtol=0,atol=1e-10)
   assert abs(a['mean_pp']-m)<1e-10
 assert compare(groups,[],'em')['context_minus_direct']['mean_pp'] is None
 js(HERE/'statistics_validation.json',{'independent_scipy_verified':True,'metrics':['EM','F1'],'comparisons':['context-direct','context-base','direct-base'],'seed_n':6,'atol':1e-10,'empty_stratum_handled':True})
 print('Independent statistics verified on nonzero synthetic EM/F1 outcomes.')
