import json,hashlib,collections,math,re
from pathlib import Path
from decimal import Decimal,ROUND_FLOOR,ROUND_HALF_UP
import scorer_v3 as v3
import training_label_scorer as labeler
import scoring_v2_v1 as numeric
HERE=Path(__file__).resolve().parent
CATS=['birthplace_callingcode','birthplace_capital','birthplace_rounded_lat','birthplace_rounded_lng']
def read(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def write(p,rs):
 with Path(p).open('x') as f:
  for r in rs:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def js(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
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

class Knowledge:
 def __init__(self,inputs):
  self.catalog=json.loads((inputs/'catalog.json').read_text())
  self.capitals=[v for a in self.catalog['attributes'].values() for v in a.get('birthplace_capital',[])]
  self.code_roots=set().union(*(parse('birthplace_callingcode',v) or set() for a in self.catalog['attributes'].values() for v in a.get('birthplace_callingcode',[])))
 def country(self,text):
  values=self.catalog['country_alias_map'].get(country_norm(text),[])
  return values[0] if len(values)==1 else None
 def compatible(self,a,b,cat):
  if a & b:return True
  return cat=='birthplace_callingcode' and any((x in self.code_roots and y.startswith(x)) or(y in self.code_roots and x.startswith(y)) for x in a for y in b)
 def label(self,target,country,cat):return labeler.score(target,self.catalog['attributes'][country][cat],cat,self.capitals)
 def strict_wrong(self,target,country,cat):
  s=self.label(target,country,cat);return not s['correct'] and not s['needs_review']
 def question(self,person,cat):return self.catalog['question_templates'][cat].format(person=person)
 def q2(self,country,cat):return self.catalog['q2_templates'][cat].format(country=country)

def verify_package(root=HERE):
 manifest=json.loads((root/'PACKAGE_FREEZE.json').read_text())
 for name,digest in manifest['files'].items():
  if sha(root/name)!=digest:raise ValueError('Package changed: '+name)
 return manifest
