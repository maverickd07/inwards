"""Check real frozen data allocation, evaluation identities, and RNG independence."""
import json,random,collections
from pathlib import Path
import exp9_shuffled as m
P=Path(__file__).resolve().parent
ROOT=next(a for a in P.parents if (a/'inward/results/exp1v2_Qwen3-14B/items.jsonl').exists())
fam={c:f for f,cs in m.FAMILIES.items() for c in cs}
rows=[json.loads(x) for x in open(ROOT/'inward/results/exp1v2_Qwen3-14B/items.jsonl')];rows=[r for r in rows if r['category'] in fam]
train=[r for r in rows if fam[r['category']] in m.TRAIN_FAMILIES and m.person_split(r['person'])=='train']
cache=[json.loads(x) for x in (P/'frozen_targets.jsonl').read_text().splitlines()];assert [m.identity(r) for r in train]==[m.identity(r) for r in cache]
pools={'A_newPerson':[r for r in rows if fam[r['category']] in m.TRAIN_FAMILIES and m.person_split(r['person'])=='test'],'B_newCategory':[r for r in rows if fam[r['category']] in m.TEST_FAMILIES and m.person_split(r['person'])=='train'],'C_bothNew':[r for r in rows if fam[r['category']] in m.TEST_FAMILIES and m.person_split(r['person'])=='test']}
rng=random.Random(0)
for k,pool in pools.items():
 sampled=rng.sample(pool,min(500,len(pool)));ref=[json.loads(x) for x in (P/'reference_exp8'/f'eval_{k}.jsonl').read_text().splitlines()];assert [m.identity(r) for r in sampled]==[m.identity(r) for r in ref]
for seed in range(3):
 for condition in ['self_full','shuffled_full','category_confidence50','category_random50']:
  state=m.torch.random.get_rng_state().clone();donors,kept,counts=m.make_plan(train,[r['target'] for r in cache],[r['mean_logprob'] for r in cache],condition,seed);assert m.torch.equal(state,m.torch.random.get_rng_state())
  assert all(v['kept']==(216 if condition.endswith('full') else 108) for v in counts.values())
  if condition=='shuffled_full':
   assert all(i!=j and m.identity(train[i])!=m.identity(train[j]) and train[i]['category']==train[j]['category'] for i,j in enumerate(donors))
   for cat in counts:
    assert collections.Counter(cache[i]['target'] for i,r in enumerate(train) if r['category']==cat)==collections.Counter(cache[donors[i]]['target'] for i,r in enumerate(train) if r['category']==cat)
print('PASS: train identities, A/B/C exact identity/order, category sample counts, derangement, target histograms, and torch RNG isolation')
