import sys
sys.dont_write_bytecode=True
from pathlib import Path
P=Path(__file__).resolve().parent;OLD=P.parent/'expxx_self_consistency_20261016';sys.path.insert(0,str(OLD))
from consistency import *
import math
w=World();rows=read(OLD/'ALL864_ENRICHED.jsonl');audit=read(P/'CANDIDATE_AUDIT.jsonl')
pred_by_person={}
for r in rows:
 if r['person'] in pred_by_person:assert pred_by_person[r['person']]==r['z_country']
 pred_by_person[r['person']]=r['z_country']
freq=collections.Counter(c for c in pred_by_person.values() if c)
binof=lambda c:-1 if not freq[c] else freq[c].bit_length()-1
out={}
# Hungarian minimum-cost assignment. Diagonal cost=1 means drop this item from both arms.
# Legal donor cost=0; forbidden cost large. Non-diagonal assignments form closed cycles.
def assignment(cost):
 n=len(cost)
 if not n:return []
 u=[0]*(n+1);v=[0]*(n+1);p=[0]*(n+1);way=[0]*(n+1)
 for i in range(1,n+1):
  p[0]=i;j0=0;minv=[float('inf')]*(n+1);used=[False]*(n+1)
  while True:
   used[j0]=True;i0=p[j0];d=float('inf');j1=0
   for j in range(1,n+1):
    if not used[j]:
     cur=cost[i0-1][j-1]-u[i0]-v[j]
     if cur<minv[j]:minv[j]=cur;way[j]=j0
     if minv[j]<d:d=minv[j];j1=j
   for j in range(n+1):
    if used[j]:u[p[j]]+=d;v[j]-=d
    else:minv[j]-=d
   j0=j1
   if p[j0]==0:break
  while True:
   j1=way[j0];p[j0]=p[j1];j0=j1
   if j0==0:break
 result=[0]*n
 for j in range(1,n+1):result[p[j]-1]=j-1
 return result
# Validate algorithm against brute force on synthetic small matrices, independent of real counts.
import itertools,random
rr=random.Random(17)
for n in range(1,7):
 for _ in range(10):
  c=[[1 if i==j else rr.choice([0,999]) for j in range(n)] for i in range(n)]
  a=assignment(c);assert sum(c[i][j] for i,j in enumerate(a))==min(sum(c[i][j] for i,j in enumerate(perm)) for perm in itertools.permutations(range(n)))
selected=[]
for source in ['exp12','exp1','extended']:
 own=[r for r in audit if r['source']==source]
 model_common=[]
 for r in own:
  if not r['stages'].get('consistent_target_strict_wrong'):continue
  ds=[d for d in r['all_country_candidates'] if d['attribute_incompatible'] and d['strict_wrong'] and binof(d['country'])==binof(r['z_country'])]
  if ds:model_common.append(r)
 category_counts={};details=[]
 for cat in CATS:
  xs=sorted([r for r in own if r['category']==cat and r['stages'].get('full_strict_eligible')],key=lambda r:r['person']);n=len(xs);cost=[]
  for i,r in enumerate(xs):
   donor_countries={d['country'] for d in r['eligible_donors']}
   cost.append([1 if i==j else 0 if d['z_country'] in donor_countries else 999 for j,d in enumerate(xs)])
  a=assignment(cost);keep=[i for i,j in enumerate(a) if i!=j]
  assert all(cost[i][a[i]]==0 for i in keep)
  assert set(keep)=={a[i] for i in keep}
  assert collections.Counter(xs[i]['consistent_target'] for i in keep)==collections.Counter(xs[a[i]]['consistent_target'] for i in keep)
  category_counts[cat]=len(keep)
  for i in keep:
   r=xs[i];d=xs[a[i]];selected.append({'source':source,'category':cat,'person':r['person'],'donor_person':d['person'],'z_country':r['z_country'],'true_country':r['true_country'],'donor_country':d['z_country'],'consistent_target':r['consistent_target'],'inconsistent_target':d['consistent_target']})
 out[source]={'q1_output_frequency_proxy_eligible':len(model_common),'q1_frequency_by_category':dict(collections.Counter(r['category'] for r in model_common)),
 'same_target_multiset_max_eligible':sum(category_counts.values()),'same_target_multiset_by_category':category_counts}
js(P/'ADDITIONAL_MATCHING.json',out);write(P/'PERMUTATION_FEASIBILITY.jsonl',selected)
print(json.dumps(out,ensure_ascii=False,indent=2))
