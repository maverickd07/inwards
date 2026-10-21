from shared import *

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

def build_pairs(people,q1,q2,k,frequency):
 freq=collections.Counter(frequency);binof=lambda c:-1 if not freq[c] else freq[c].bit_length()-1
 first={r['person_id']:r['prediction'] for r in q1};second={(r['country'],r['category']):r['prediction'] for r in q2}
 stages=collections.Counter();audit=[];candidates=[];selected=[]
 for p in people:
  stages['people']+=1;zraw=first[p['person_id']];z=k.country(zraw)
  if not z:stages['unresolved_q1_people']+=1;continue
  if z==p['birth_country']:stages['correct_q1_people']+=1;continue
  stages['wrong_q1_people']+=1
  for cat in CATS:
   target=second.get((z,cat));v=parse(cat,target or '')
   r={'person_id':p['person_id'],'person':p['person'],'qid':p['qid'],'source':p['source'],'category':cat,'z_tilde':zraw,'z_country':z,'birth_country':p['birth_country'],'question':k.question(p['person'],cat),'consistent_target':target,'frequency':freq[z],'frequency_bin':binof(z)}
   stages['wrong_q1_items']+=1
   if not v:r['excluded']='unparseable_q2'
   elif not k.strict_wrong(target,p['birth_country'],cat):r['excluded']='consistent_target_correct_or_review'
   else:r['normalized_target']=sorted(v);candidates.append(r);stages['consistent_target_strict_wrong_items']+=1
   audit.append(r)
 for cat in CATS:
  xs=sorted([r for r in candidates if r['category']==cat],key=lambda r:r['person_id']);cost=[]
  for i,r in enumerate(xs):
   row=[]
   for j,d in enumerate(xs):
    valid=(d['z_country'] not in [r['z_country'],r['birth_country']] and d['frequency_bin']==r['frequency_bin'] and not k.compatible(set(r['normalized_target']),set(d['normalized_target']),cat) and k.strict_wrong(d['consistent_target'],r['birth_country'],cat))
    row.append(1 if i==j else 0 if valid else 999999)
   cost.append(row)
  a=assignment(cost);keep=[i for i,j in enumerate(a) if i!=j]
  assert all(cost[i][a[i]]==0 for i in keep)
  assert set(keep)=={a[i] for i in keep}
  for i in keep:
   r=xs[i];d=xs[a[i]]
   selected.append({**r,'donor_person_id':d['person_id'],'donor_country':d['z_country'],'inconsistent_target':d['consistent_target'],'consistent_label':k.label(r['consistent_target'],r['birth_country'],cat),'inconsistent_label':k.label(d['consistent_target'],r['birth_country'],cat)})
  stages[cat+'_matched']=len(keep)
 for cat in CATS:
  subset=[r for r in selected if r['category']==cat]
  assert collections.Counter(r['consistent_target'] for r in subset)==collections.Counter(r['inconsistent_target'] for r in subset)
 matched={(r['person_id'],r['category']) for r in selected}
 for r in audit:r['matched']=(r['person_id'],r['category']) in matched
 stages['matched_pairs']=len(selected)
 return selected,audit,dict(stages)
