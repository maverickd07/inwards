"""Real-pool invariants: matched categories, still-wrong changed permutations, frozen subsets, RNG isolation."""
import sys,json,collections,hashlib
from pathlib import Path
import exp10_consistency as m
P=Path(__file__).resolve().parent
ROOT=next(a for a in P.parents if (a/'inward/data/compositional_celebrities.json').exists())
by={m.identity(r):r for r in json.load(open(ROOT/'inward/data/compositional_celebrities.json'))['data']}
cache=[json.loads(x) for x in (P/'frozen_targets.jsonl').read_text().splitlines()]
fam={c:f for f,cs in m.FAMILIES.items() for c in cs}
train=[r for r in map(json.loads,open(ROOT/'inward/results/exp1v2_Qwen3-14B/items.jsonl')) if r['category'] in fam and fam[r['category']] in m.TRAIN_FAMILIES and m.person_split(r['person'])=='train']
assert [m.identity(r) for r in train]==[m.identity(r) for r in cache]
rows=[{**r,'e':by[m.identity(r)]} for r in cache];targets=[r['target'] for r in cache];scores=[r['mean_logprob'] for r in cache]
plans={};report={}
for seed in range(3):
 for c in ['self_wrong','shuffled_wrong','self_correct']:
  state=m.torch.random.get_rng_state().clone();d,k,counts=m.make_plan(rows,targets,scores,c,seed,m.is_correct);assert m.torch.equal(state,m.torch.random.get_rng_state())
  assert len(k)==228 and [counts[c]['kept'] for c in sorted(counts)]==[51,60,71,46]
  plans[c,seed]=(d,k,counts)
  assert all(m.is_correct(targets[d[i]],rows[i]['e']['Answer'])==(c=='self_correct') for i in k)
  if c=='shuffled_wrong':
   assert all(i!=d[i] and rows[i]['category']==rows[d[i]]['category'] and targets[i]!=targets[d[i]] for i in k)
   for cat in counts:assert collections.Counter(targets[i] for i in k if rows[i]['category']==cat)==collections.Counter(targets[d[i]] for i in k if rows[i]['category']==cat)
  report[f'{c}_s{seed}']={'n':len(k),'counts':counts,'changed':sum(targets[i]!=targets[d[i]] for i in k),'floor_correct':sum(m.is_correct(targets[d[i]],rows[i]['e']['Answer']) for i in k),'legacy_correct':sum(m.is_correct.__globals__['old_is_correct'](targets[d[i]],rows[i]['e']['Answer']) for i in k),'truncate_correct':sum(m.is_correct(targets[d[i]],rows[i]['e']['Answer'],'truncate') for i in k)}
 for c in ['self_wrong','self_correct']:assert plans[c,seed][1]==plans[c,0][1]
 assert plans['self_wrong',seed][1]==plans['shuffled_wrong',seed][1]
 assert not plans['self_wrong',seed][1] & plans['self_correct',seed][1]
# Do not overwrite validation reports.
out=P/'verified_allocation.json';assert not out.exists();out.write_text(json.dumps(report,indent=2))
print('PASS: 9 real allocations, fixed subsets, same categories, preserved target histograms, 100% changed/still-wrong shuffled targets, RNG isolation')
