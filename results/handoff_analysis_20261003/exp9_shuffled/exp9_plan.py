"""Gold-free allocation: category derangement or size-matched category selection."""
import collections,random,hashlib,math

def identity(r): return r['category'],r['person']
def stream_seed(seed,category,purpose):
    return int.from_bytes(hashlib.sha256(f'{seed}|{category}|{purpose}'.encode()).digest()[:8],'big')
def make_plan(rows,targets,scores,condition,selection_seed):
    if condition not in ['self_full','shuffled_full','category_confidence50','category_random50']:
        raise ValueError(condition)
    if not len(rows)==len(targets)==len(scores):raise ValueError('length mismatch')
    if len({identity(r) for r in rows})!=len(rows):raise ValueError('duplicate identity')
    if any(not t or s is None or not math.isfinite(s) for t,s in zip(targets,scores)):
        raise ValueError('Every original row must have a nonempty finite-scored target; do not silently change training pool')
    groups=collections.defaultdict(list)
    for i,r in enumerate(rows):groups[r['category']].append(i)
    donors=list(range(len(rows)));kept=set();counts={}
    for cat,idx in sorted(groups.items()):
        n=len(idx)
        if condition=='shuffled_full':
            if n<2:raise ValueError('Cannot derange singleton category '+cat)
            # Rejection sampling gives a uniform random permutation conditional on no fixed points.
            rng=random.Random(stream_seed(selection_seed,cat,'derangement'))
            for attempt in range(10000):
                perm=idx.copy();rng.shuffle(perm)
                if all(a!=b for a,b in zip(idx,perm)):break
            else:raise RuntimeError('derangement rejection limit')
            for i,j in zip(idx,perm):donors[i]=j
            selected=idx
        elif condition=='category_confidence50':
            selected=sorted(idx,key=lambda i:(-scores[i],i))[:math.ceil(n*.5)]
        elif condition=='category_random50':
            selected=idx.copy();random.Random(stream_seed(selection_seed,cat,'random_selection')).shuffle(selected)
            selected=selected[:math.ceil(n*.5)]
        else:selected=idx
        kept.update(selected);counts[cat]={'available':n,'kept':len(selected)}
    if condition=='shuffled_full':
        assert sorted(donors)==list(range(len(rows)))
        assert all(i!=j and rows[i]['category']==rows[j]['category'] and rows[i]['person']!=rows[j]['person'] for i,j in enumerate(donors))
    return donors,kept,counts
