"""Offline observational analysis. No network, model loading, inference or historical writes."""
import sys
sys.dont_write_bytecode=True
import json,hashlib,re,collections,unicodedata
from decimal import Decimal,ROUND_FLOOR,ROUND_HALF_UP
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE/'vendor'))
import scorer_v3 as v3
import training_label_scorer as labeler
import scoring_v2_v1 as numeric
INPUTS=json.loads((HERE/'INPUTS.json').read_text())
CATS=['birthplace_callingcode','birthplace_capital','birthplace_rounded_lat','birthplace_rounded_lng']
CONDS=['context','direct'];STATUSES=['consistent','contradictory','unknown']
def path(k):return ROOT/INPUTS[k]['path']
def read(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def key(r):return r['category'],r['person']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def write(p,rows):
 with Path(p).open('x') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def verify_inputs():
 for n,v in json.loads((HERE/'PREREGISTRATION_FREEZE.json').read_text())['files'].items():assert sha(HERE/n)==v,n
 for k,v in INPUTS.items():assert sha(ROOT/v['path'])==v['sha256'],k
 if (HERE/'CODE_FREEZE.json').exists():
  for n,d in json.loads((HERE/'CODE_FREEZE.json').read_text())['files'].items():assert sha(HERE/n)==d,n

def country_norm(s):return v3.canonical(v3.unwrap(v3.clean(s) or '', 'country'),'country')
def clean_cap(s):
 s=v3.clean(s)
 if not s:return None
 s=re.sub(r'^answer\s*:\s*','',s,flags=re.I)
 s=re.sub(r'^(?:the\s+)?capital(?:\s+city)?\s*(?::|is\b)\s*','',s,flags=re.I).strip()
 # Whole place-name-shaped answer only. No substring extraction from prose.
 if not all(ch.isalpha() or ch.isspace() or ch in ",.'’‐–-()" for ch in s):return None
 words=set(re.findall(r'\w+',s.casefold()))
 if words & {'unknown','sorry','cannot','unable','answer','born','know','unsure'}:return None
 if re.search(r'\b(?:is|was|are|were|because|however)\b',s,re.I):return None
 n=labeler.capital(s)
 if not n:return None
 for group in labeler.POLICY['capital_alias_groups']:
  norms={labeler.capital(v) for v in group}
  if n in norms:return min(norms)
 return n

def parse(cat,s,mode='floor'):
 text=v3.clean(s)
 if not text:return None
 if cat=='birthplace_capital':
  cap=clean_cap(text);return frozenset([cap]) if cap else None
 if cat=='birthplace_callingcode':
  text=v3.unwrap(text,cat).rstrip('.')
  codes=labeler.codes(text)
  if not codes:return None
  for group in labeler.POLICY['calling_code_groups']:
   if codes & set(group):codes|=set(group)
  return frozenset(codes)
 if cat in ['birthplace_rounded_lat','birthplace_rounded_lng']:
  clean=numeric.norm(text);m=numeric.FULL.fullmatch(clean)
  if not m:return None
  direction=(m[2] or '').upper();lat=cat.endswith('_lat')
  if direction and direction not in ('NS' if lat else 'EW'):return None
  v=numeric.value(text)
  if v is None or not v.is_finite() or abs(v)>(90 if lat else 180):return None
  value=v.to_integral_value(rounding=ROUND_FLOOR if mode=='floor' else ROUND_HALF_UP)
  return frozenset([str(int(value))])
 return None

class World:
 def __init__(self):
  self.table=json.loads(path('country_table').read_text());self.data=json.loads(path('dataset').read_text())['data']
  self.by={key(e):e for e in self.data};assert len(self.by)==len(self.data)
  rebuilt=collections.defaultdict(lambda:collections.defaultdict(set))
  for e in self.data:
   if e['category'].startswith('birthplace'):
    rebuilt[e['birth_country']][e['category']].update(str(x) for x in e['A2'])
  assert {c:{k:sorted(v) for k,v in attrs.items()} for c,attrs in rebuilt.items()}==self.table
  self.country_map=collections.defaultdict(set)
  for country in self.table:self.country_map[country_norm(country)].add(country)
  for e in self.data:
   if e['category'] in CATS:
    for alias in e['A1']:self.country_map[country_norm(alias)].add(e['birth_country'])
  for group in json.loads(path('country_aliases').read_text())['country']:
   countries=set().union(*(self.country_map[country_norm(x)] for x in group))
   for x in group:self.country_map[country_norm(x)].update(countries)
  self.attrs={mode:{c:{cat:set().union(*(parse(cat,v,mode) or set() for v in attrs.get(cat,[]))) for cat in CATS} for c,attrs in self.table.items()} for mode in ['floor','round']}
  self.code_roots=set().union(*(d['birthplace_callingcode'] for d in self.attrs['floor'].values()))
  self.capitals=[x for a in self.table.values() for x in a.get('birthplace_capital',[])]
  self.indexes={'exp12':collections.defaultdict(list),'exp1':collections.defaultdict(list)}
  for split in ['A_newPerson','B_newCategory','C_bothNew','D_reservedC']:
   rows=read(path('exp12_baseline_'+split));plan=read(path('exp12_plan_'+split));assert list(map(key,rows))==list(map(key,plan))
   for r,p in zip(rows,plan):
    if r['category'] in CATS:self.indexes['exp12'][p['e']['birth_country'],r['category']].append({'prediction':r['q2_pred'],'question':p['e']['Q2'],'source':INPUTS['exp12_baseline_'+split]['path'],'person':r['person']})
  self.exp1={key(r):r for r in read(path('exp1_14b'))}
  for k,r in self.exp1.items():
   if r['category'] in CATS:
    e=self.by[k];self.indexes['exp1'][e['birth_country'],r['category']].append({'prediction':r['pred_q2'],'question':e['Q2'],'source':INPUTS['exp1_14b']['path'],'person':r['person']})
  self.lookup_cache={}
 def country(self,raw):
  s=v3.clean(raw)
  matches=self.country_map.get(country_norm(s),set()) if s else set()
  return next(iter(matches)) if len(matches)==1 else None
 def compatible(self,a,b,cat):
  if a & b:return True
  if cat=='birthplace_callingcode':
   return any((x in self.code_roots and y.startswith(x)) or (y in self.code_roots and x.startswith(y)) for x in a for y in b)
  return False
 def attribute(self,cat,z,target,mode='floor'):
  parsed=parse(cat,target,mode);country=self.country(z)
  countries=sorted(c for c,attrs in self.attrs[mode].items() if parsed and attrs[cat] and self.compatible(parsed,attrs[cat],cat))
  if not country:status,reason='unknown','z_unresolved'
  elif not parsed:status,reason='unknown','target_unparseable'
  elif not countries:status,reason='unknown','target_not_in_attribute_table'
  elif not self.attrs[mode][country][cat]:status,reason='unknown','z_attribute_missing_in_table'
  else:status,reason=('consistent','z_in_country_set') if country in countries else ('contradictory','z_outside_country_set')
  return {'status':status,'reason':reason,'z_country':country,'target_values':sorted(parsed) if parsed else None,'implied_countries':countries,'country_set_size':len(countries)}
 def lookup(self,cat,z,source,mode='floor'):
  country=self.country(z);k=(cat,country,source,mode)
  if k in self.lookup_cache:return self.lookup_cache[k]
  records=self.indexes[source].get((country,cat),[]) if country else []
  vals=[parse(cat,r['prediction'],mode) for r in records]
  if not country:reason='z_unresolved'
  elif not records:reason='missing_saved_q2'
  elif any(v is None for v in vals):reason='saved_q2_unparseable'
  elif len(set(vals))!=1:reason='saved_q2_conflicting'
  else:reason='available'
  answer={'reason':reason,'values':sorted(vals[0]) if reason=='available' else None,'source':source,'records':records,'z_country':country}
  self.lookup_cache[k]=answer;return answer
 def model(self,cat,z,target,source='exp12',mode='floor'):
  lookup=self.lookup(cat,z,'exp12' if source=='extended' else source,mode)
  if source=='extended' and lookup['reason']=='missing_saved_q2':lookup=self.lookup(cat,z,'exp1',mode)
  parsed=parse(cat,target,mode)
  if lookup['reason']!='available':status,reason='unknown',lookup['reason']
  elif parsed is None:status,reason='unknown','target_unparseable'
  else:status,reason=('consistent','matches_saved_q2') if self.compatible(parsed,set(lookup['values']),cat) else ('contradictory','differs_from_saved_q2')
  return {'status':status,'reason':reason,'lookup_source':lookup['source'],'lookup_key':[lookup['z_country'],cat],
   'saved_q2_values':lookup['values'],'target_values':sorted(parsed) if parsed else None,'saved_q2_n':len(lookup['records'])}

def base_rows(world):
 train=read(path('train864'));paired=read(path('paired864'));labels=read(path('labels864'))
 assert list(map(key,train))==list(map(key,paired))==list(map(key,labels)) and len(train)==864
 selected=set(map(key,read(path('train785'))));different=set(map(tuple,json.loads(path('different404').read_text())['identities']))
 assert len(selected)==785 and len(different)==404 and different<=selected
 rows=[]
 for i,(t,p,l) in enumerate(zip(train,paired,labels)):
  assert p['context_target']==l['context_target'] and p['direct_target']==l['direct_target']
  assert l['selected']==(key(t) in selected)
  e=t['e'];rows.append({'index864':i,'category':t['category'],'person':t['person'],'question':e['Question'],
   'Q1':e['Q1'],'Q2':e['Q2'],'z_tilde':p['own_q1'],'z_country':world.country(p['own_q1']),
   'gold_first':e['A1'],'gold_final':e['Answer'],'gold_country':e['birth_country'],
   'context_target':p['context_target'],'direct_target':p['direct_target'],'labels':l['revised'],
   'z_label':v3.score(p['own_q1'],e['A1'],t['category'],'q1'),
   'selected785':key(t) in selected,'different404':key(t) in different})
 assert {key(r) for r in rows if r['selected785'] and r['context_target']!=r['direct_target']}==different
 return rows

def coverage(world,rows):
 result={}
 for scope,rs in [('all864',rows),('selected785',[r for r in rows if r['selected785']]),('different404',[r for r in rows if r['different404']])]:
  result[scope]={}
  for source in ['exp12','exp1','extended']:
   counts=collections.Counter()
   for r in rs:
    lookup=world.lookup(r['category'],r['z_tilde'],'exp12' if source=='extended' else source)
    if source=='extended' and lookup['reason']=='missing_saved_q2':lookup=world.lookup(r['category'],r['z_tilde'],'exp1')
    counts[lookup['reason']]+=1
   result[scope][source]={'n':len(rs),'counts':dict(counts),'available_pct':100*counts['available']/len(rs)}
 return result

def add_consistency(world,rows):
 for r in rows:
  r['consistency']={}
  for method in ['attribute_floor','attribute_round','model_exp12','model_exp1','model_extended','model_exp12_round','model_exp1_round','model_extended_round']:
   r['consistency'][method]={}
   for cond in CONDS:
    if method.startswith('attribute'):answer=world.attribute(r['category'],r['z_tilde'],r[cond+'_target'],'round' if method.endswith('_round') else 'floor')
    else:answer=world.model(r['category'],r['z_tilde'],r[cond+'_target'],method.split('_')[1],'round' if method.endswith('_round') else 'floor')
    r['consistency'][method][cond]=answer
 return rows

if __name__=='__main__':
 verify_inputs();world=World();rows=base_rows(world)
 if sys.argv[1]=='coverage':
  cov=coverage(world,rows);js(HERE/'COVERAGE.json',cov);print(json.dumps(cov,ensure_ascii=False,indent=2))
 else:raise ValueError('Only coverage entrypoint; use analyze.py after coverage disclosure')
