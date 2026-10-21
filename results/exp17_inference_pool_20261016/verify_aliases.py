import json,unicodedata
from pathlib import Path
P=Path(__file__).resolve().parent;I=P/'package/inputs'
def norm(s):return ''.join(c for c in unicodedata.normalize('NFKD',s).casefold() if c.isalnum())
r=json.loads((P/'ORIGINAL_ID_RESOLUTION.json').read_text())['resolved'];r.update(json.loads((P/'IDENTITY_SUPPLEMENT.json').read_text()));red=json.loads((P/'ORIGINAL_REDIRECT_ALIASES_COMPLETE.json').read_text());ens=json.loads((P/'PERSON_ENTITIES.json').read_text());ens.update(json.loads((P/'IDENTITY_SUPPLEMENT_ENTITIES.json').read_text()))
plan=P.parents[1]/'results/exp12_sources_20261014_v2/received_700454/extracted/runs_700454/plan';evalnames=set()
for split in ['A_newPerson','C_bothNew','D_reservedC']:evalnames.update(json.loads(x)['person'] for x in (plan/f'items_{split}.jsonl').read_text().splitlines())
assert len(evalnames)==252 and evalnames<=set(r)
idx=[]
for name in sorted(evalnames):
 rr=r[name];e=ens[rr['qid']];aliases={name,rr['canonical_title'],*rr['redirect_aliases'],*red.get(rr['canonical_title'],[])};aliases.update(x['value'] for x in e.get('aliases',{}).get('en',[]));aliases.add(e.get('labels',{}).get('en',{}).get('value',''))
 for prop in ['P1477','P1449']:
  for claim in e.get('claims',{}).get(prop,[]):
   value=claim.get('mainsnak',{}).get('datavalue',{}).get('value',{})
   if isinstance(value,dict):aliases.add(value.get('text',''))
 aliases.discard('');idx.append({'person':name,'qid':rr['qid'],'aliases':sorted(aliases)})
ids={x['qid'] for x in idx};alias={norm(a) for x in idx for a in x['aliases']}
for name in ['new_people.jsonl','old_people_eligible.jsonl']:
 for line in (I/name).read_text().splitlines():
  x=json.loads(line);assert x['qid'] not in ids and not ({norm(a) for a in [x['person'],*x['aliases']]}&alias),x['person']
with (I/'evaluation_exclusion_index.json').open('x') as f:json.dump({'n_eval_people':252,'all_resolved':True,'entries':idx},f,ensure_ascii=False,indent=2)
print('Independent final identity/alias exclusion verification passed')
