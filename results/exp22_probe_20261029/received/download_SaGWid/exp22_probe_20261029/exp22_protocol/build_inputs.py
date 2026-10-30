"""Build frozen roster without model outputs; never modify prior experiments."""
import json, hashlib, shutil, unicodedata
from pathlib import Path
H=Path(__file__).resolve().parent; P=H.parent; E=P/'exp21_latent_emergence_20261022'
def read(p): return [json.loads(s) for s in p.read_text().splitlines() if s]
def key(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())
def write(p,x):p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in x))
(H/'inputs').mkdir(exist_ok=True)
for p in (E/'inputs').iterdir():
 if p.name!='items.jsonl' and p.is_file():shutil.copyfile(p,H/'inputs'/p.name)
shutil.copyfile(E/'ADAPTER_MANIFEST.json',H/'ADAPTER_MANIFEST.json')
new=read(P/'exp17_inference_pool_20261016/package/inputs/new_people.jsonl');assert len(new)==400
shutil.copyfile(P/'exp17_inference_pool_20261016/package/inputs/new_people.jsonl',H/'inputs/new_people_source.jsonl')
old=read(E/'inputs/items.jsonl');templates={r['category']:r['e'] for r in old}
rows=[];oldkeys={key(r['person']) for r in old};seen={}
for r in old:
 e=r['e'];person=key(r['person']);assert person not in seen or seen[person]==e['birth_country'];seen[person]=e['birth_country']
 rows.append(dict(id='old:'+str(r['id']),e=e,person=r['person'],group=person,source='old',fixed303=r['fixed303'],known313=r['D500_known313']))
attrs=json.loads((H/'inputs/country_attributes.json').read_text())
attrs['New Zealand']={'birthplace_currency':['New Zealand dollar'],'birthplace_currency_short':['NZD'],'birthplace_currency_symbol':['$','NZ$'],'birthplace_tld':['.nz'],'birthplace_ccn3':['554']}
(H/'inputs/country_attributes.json').write_text(json.dumps(attrs,ensure_ascii=False,indent=2)+'\n')
for p in new:
 assert key(p['person']) not in oldkeys
 # Join any recognized aliases to the same person, rather than randomizing rows.
 aliases={key(x) for x in p['aliases']}|{key(p['person'])};assert not aliases&oldkeys,'new/old identity overlap'
 for cat,t in sorted(templates.items()):
  country=p['birth_country'];a=attrs[country][cat];assert a
  e=dict(t,person=p['person'],person_id=p['person_id'],birth_country=country,A1=[country],A2=a,Answer=a,
   Q1=f"What is the birthplace (country only) of {p['person']}?",
   Q2=t['Q2'].replace(t['birth_country'],country),Question=t['Question'].replace(t['person'],p['person']))
  rows.append(dict(id=p['person_id']+':'+cat,e=e,person=p['person'],group=key(p['person']),source='new400',fixed303=False,known313=False))
# Collapse alias-connected identities, including disambiguated source aliases.
parent={r['group']:r['group'] for r in rows}
def root(x):
 while parent[x]!=x:x=parent[x]
 return x
alias_owner={}
for p in new:
 g=key(p['person'])
 for a in set(map(key,p['aliases']))|{g}:
  if a in alias_owner:
   u,v=root(g),root(alias_owner[a]);assert seen.get(u,p['birth_country'])==p['birth_country'];parent[max(u,v)]=min(u,v)
  alias_owner[a]=g
for r in rows:r['group']=root(r['group'])
assert len(rows)==2402 and len({r['id'] for r in rows})==2402
write(H/'inputs/items.jsonl',rows)
from collections import Counter
counts=Counter(p['birth_country'] for p in new);countries=sorted(counts,key=lambda c:(-counts[c],c))[:3]
smoke=[]
for c in countries:
 people=sorted([p for p in new if p['birth_country']==c],key=lambda p:p['person_id'])[:10]
 for p in people:smoke.append(next(r for r in rows if r['id']==p['person_id']+':birthplace_currency_short'))
assert len(smoke)==30 and len({r['group'] for r in smoke})==30
write(H/'inputs/smoke30.jsonl',smoke)
(H/'INPUT_SUMMARY.json').write_text(json.dumps({'rows':len(rows),'old':402,'new_people':400,'new_rows':2000,'categories':sorted(templates),'smoke_countries':countries,'unique_groups':len(parent)},indent=2)+'\n')
