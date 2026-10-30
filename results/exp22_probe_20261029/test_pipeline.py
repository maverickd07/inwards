"""Engineering tests: planted signal, absent signal, persona grouping, heldout labels."""
import numpy as np
from probe_utils import split,labels,permuted_labels
from analyze import fit_probe

def test():
 rng=np.random.default_rng(222);n=1500;X=rng.normal(size=(n,12));y=np.where(X[:,0]+X[:,1]>0,'a','b');p=fit_probe(X[:1000],y[:1000]);planted=float(np.mean(p.predict(X[1000:])==y[1000:]));assert planted>.9
 noise=rng.choice(['a','b'],n);q=fit_probe(X[:1000],noise[:1000]);null=float(np.mean(q.predict(X[1000:])==noise[1000:]));assert .4<null<.6
 rows=[{'group':str(i),'e':{'birth_country':'abc'[i//10],'category':'rel','Answer':['x'+str(i//10)]}} for i in range(30) for _ in range(3)]
 t,sp=split(rows);assert not set(sp['train_groups'])&set(sp['test_groups']);assert len(sp['test_groups'])==9
 for target in ['country','answer']:
  y,meta=labels(rows,t,target);assert len(meta['retained'])==3
  yp=permuted_labels(rows,y,t,target);assert np.all(yp[~t]==y[~t]);assert sorted(yp[t])==sorted(y[t])
  for g in sp['train_groups']:assert len(set(yp[[r['group']==g for r in rows]]))==1
 # Poison test references: training class mapping must remain unchanged.
 poisoned=[dict(r,e=dict(r['e'])) for r in rows]
 for i in np.flatnonzero(~t):poisoned[i]['e']['birth_country']='poison';poisoned[i]['e']['Answer']=['poison']
 for target in ['country','answer']:assert labels(poisoned,t,target)[1]==labels(rows,t,target)[1]
 return {'planted_accuracy':planted,'null_accuracy':null,'group_and_label_leakage_checks_passed':True}
if __name__=='__main__':
 import json;print(json.dumps(test(),indent=2))
