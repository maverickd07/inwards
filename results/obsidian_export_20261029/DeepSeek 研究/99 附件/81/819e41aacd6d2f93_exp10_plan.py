"""Gold-stratified allocation. Targets always come from frozen self outputs.
Subset seed 7 is fixed across training seeds; only matching and training vary.
Constrained bipartite matching preserves the selected wrong-target multiset.
"""
import collections,random,hashlib,math
CONDITIONS=['self_wrong','shuffled_wrong','self_correct']
def identity(r):return r['category'],r['person']
def stream_seed(seed,cat,purpose):return int.from_bytes(hashlib.sha256(f'{seed}|{cat}|{purpose}'.encode()).digest()[:8],'big')
def make_plan(rows,targets,scores,condition,selection_seed,scorer):
 if condition not in CONDITIONS:raise ValueError(condition)
 assert len(rows)==len(targets)==len(scores)
 assert len(set(map(identity,rows)))==len(rows)
 assert all(t and s is not None and math.isfinite(s) for t,s in zip(targets,scores))
 groups=collections.defaultdict(list)
 for i,r in enumerate(rows):groups[r['category']].append(i)
 donors=list(range(len(rows)));kept=set();counts={}
 for cat,idx in sorted(groups.items()):
  correct=[i for i in idx if scorer(targets[i],rows[i]['e']['Answer'])]
  wrong=[i for i in idx if i not in set(correct)]
  n=min(len(correct),len(wrong))
  if n<2:raise ValueError('Insufficient correctness classes for matched category '+cat)
  selected={}
  for name,pool in [('correct',correct),('wrong',wrong)]:
   pool=pool.copy();random.Random(stream_seed(7,cat,name)).shuffle(pool);selected[name]=pool[:n]
  ids=selected['correct'] if condition=='self_correct' else selected['wrong']
  if condition=='shuffled_wrong':
   rng=random.Random(stream_seed(selection_seed,cat,'constrained_matching'))
   edges={}
   for i in ids:
    eligible=[j for j in ids if rows[i]['person']!=rows[j]['person'] and targets[i]!=targets[j] and not scorer(targets[j],rows[i]['e']['Answer'])]
    rng.shuffle(eligible);edges[i]=eligible
   # Randomized augmenting-path matching; not a uniform draw over all valid matchings.
   owner={}
   def assign(i,seen):
    for j in edges[i]:
     if j in seen:continue
     seen.add(j)
     if j not in owner or assign(owner[j],seen):owner[j]=i;return True
    return False
   order=ids.copy();rng.shuffle(order)
   for i in order:
    if not assign(i,set()):raise ValueError('No perfect changed-and-still-wrong matching for '+cat+'; do not silently relax constraints')
   for j,i in owner.items():donors[i]=j
   assert sorted(donors[i] for i in ids)==sorted(ids)
  kept.update(ids);counts[cat]={'available':len(idx),'original_correct':len(correct),'original_wrong':len(wrong),'kept':n,'subset_seed':7}
 assert kept
 for i in kept:
  assert scorer(targets[donors[i]],rows[i]['e']['Answer'])==(condition=='self_correct')
 return donors,kept,counts
