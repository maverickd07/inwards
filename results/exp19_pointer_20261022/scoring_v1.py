"""Frozen format-equivalent scoring; source identities are archived, not web-updated."""
import json,re,unicodedata,xml.etree.ElementTree as ET
from pathlib import Path
I=Path(__file__).resolve().parent/'inputs'
TABLE=json.loads((I/'country_attributes.json').read_text())
ALIASES=json.loads((I/'aliases.json').read_text())
HIST=json.loads((I/'historical_rules.json').read_text())
CURRENCY=['birthplace_currency','birthplace_currency_short','birthplace_currency_symbol']
def norm(x):
 s=unicodedata.normalize('NFKC',str(x)).casefold().strip()
 s=re.sub(r'^(?:the\s+)?(?:final answer|answer|currency code|currency symbol|currency|country)\s*:\s*','',s)
 s=s.strip('`*"“”\' ')
 return ' '.join(s.split()).rstrip('.,;!')
def variants(x):
 s=norm(x);v={s}
 # One-word singular/plural and last noun; same operation on references and answers.
 if s.endswith('s') and len(s)>2:v.add(s[:-1])
 if s.endswith('ies'):v.add(s[:-3]+'y')
 return v

def country_names(c):
 names={norm(c)}
 for group in ALIASES['country']:
  if names & set(map(norm,group)):names.update(map(norm,group))
 return names
ISO_CURRENT={}
for e in ET.parse(I/'iso4217_current.xml').getroot().iter('CcyNtry'):
 c=norm(e.findtext('CtryNm') or ''); code=e.findtext('Ccy');name=e.findtext('CcyNm')
 if code:ISO_CURRENT.setdefault(c,[]).append((code,name))
HIST_NAMES={}
for e in HIST['source_entries']:
 if not e['Ccy']:continue
 HIST_NAMES.setdefault(e['Ccy'],set()).update([e['Ccy'],e['CcyNm']]+HIST['name_aliases'].get(e['Ccy'],[]))

def current_for(c):
 names=[norm(HIST['entity_aliases'].get(c,c)),norm(c)]
 return [pair for n in dict.fromkeys(names) for pair in ISO_CURRENT.get(n,[])]
# Build currency identities by original code, then add current ISO names.
LEX={}
for c,a in TABLE.items():
 for code in a.get('birthplace_currency_short',[]):
  lex=LEX.setdefault(code,set());lex.add(code)
  for cat in CURRENCY:lex.update(a.get(cat,[]))
for pairs in ISO_CURRENT.values():
 for code,name in pairs:LEX.setdefault(code,{code}).add(name)
for group in ALIASES.get('birthplace_currency',[]):
 for lex in LEX.values():
  if set(map(norm,group)) & set(map(norm,lex)):lex.update(group)
LEX.setdefault('EUR',{'EUR'}).update(['euro','euros','€'])
LEX.setdefault('INR',{'INR'}).update(['Indian rupee','Indian rupees','₹'])
# A bare denomination (dollar, rupee, lira, mark) is format-equivalent but
# ambiguous: retain all compatible identities instead of guessing a country.
for lex in LEX.values():
 for name in list(lex):
  words=name.split()
  if len(words)>1 and words[-1].casefold() in {'dollar','rupee','lira','mark','franc','peso','dinar','pound','shilling','rial','riyal','leu','lev','ruble','rouble','won','yen','yuan','krona','krone'}:lex.add(words[-1])
LEX={code:set().union(*(variants(v) for v in lex)) for code,lex in LEX.items()}
# Historical names indexed separately. Live names take precedence globally;
# correct-country historical aliases take precedence when NOT live there.
HL={code:set().union(*(variants(v) for v in names if v)) for code,names in HIST_NAMES.items()}
def current_codes(c):
 pairs=current_for(c)
 return {code for code,_ in pairs} if pairs else set(TABLE.get(c,{}).get('birthplace_currency_short',[]))
def expected_codes(e):return current_codes(e['birth_country'])
def atoms(raw):
 s=norm(raw)
 # Accept exact alias first; otherwise equivalent redundant representations only.
 return [s]+[norm(v) for v in re.split(r'\s*[()\[\]\n]\s*|\s+(?:=|/|or)\s+|\s*=\s*',s) if norm(v)]
def resolve_currency(raw):
 v=variants(raw)
 direct={k for k,lex in LEX.items() if v&lex}
 if direct:return direct,False
 chunks=atoms(raw)[1:]
 sets=[{k for k,lex in LEX.items() if variants(s)&lex} for s in chunks]
 if len(sets)>=2 and all(sets):return set.intersection(*sets),not bool(set.intersection(*sets))
 return set(),False

def historical(raw,e):
 v=variants(raw); live,_=resolve_currency(raw);expected=expected_codes(e)
 if live&expected:return False
 # Old ISO codes never become accepted merely through old dataset labels.
 if any(norm(code)==norm(raw) and code not in expected for code in HL):return True
 country=norm(HIST['entity_aliases'].get(e['birth_country'],e['birth_country']))
 codes={r['Ccy'] for r in HIST['source_entries'] if norm(r['CtryNm'])==country}
 if any(v & HL.get(code,set()) for code in codes):return True
 if norm(raw) in {'lira','mark','marks','lire'}:
  return not (live & expected)
 # Globally obsolete names with no live interpretation.
 return not live and any(v&lex for lex in HL.values())

def parts(raw,c):
 if c!='C2':return None,raw,False
 match=re.fullmatch(r'\s*Country\s*:\s*([^\n]+)\s*\n\s*Answer\s*:\s*(.+?)\s*',raw,re.I|re.S)
 if not match:return None,'',True
 return match[1].strip(),match[2].strip(),False

def country_score(raw,e):
 if raw is None:return None
 names=country_names(e['birth_country'])|set(map(norm,e['A1']))
 if norm(raw) in names:return True
 if any(norm(raw) in country_names(c) for c in TABLE):return False
 return None

def score_answer(raw,e):
 cat=e['category'];n=norm(raw)
 if not n:return {'correct':False,'status':'REVIEW','reason':'empty_or_missing_answer'}
 if '<think>' in raw or '</think>' in raw:return {'correct':False,'status':'REVIEW','reason':'unexpected_reasoning_markup'}
 ambiguous=False
 if cat in CURRENCY:
  codes,conflict=resolve_currency(raw);expected=expected_codes(e)
  if historical(raw,e):return {'correct':False,'status':'HISTORICAL','reason':'obsolete_currency','currency_candidates':sorted(codes)}
  ok=bool(codes&expected) and not conflict
  status='CORRECT' if ok else 'INCORRECT' if codes and not conflict else 'REVIEW'
  return {'correct':ok,'status':status,'reason':'currency_identity' if codes else 'unknown_or_multiple_currency','currency_candidates':sorted(codes),'expected_currency_codes':sorted(expected),'ambiguous_form':len(codes)>1}
 if cat=='birthplace_ccn3':
  def numeric(s):return str(int(s)) if re.fullmatch(r'\d{1,3}',s) else None
  val=numeric(n);want={numeric(norm(x)) for x in e['Answer']}
  ok=val is not None and val in want
  return {'correct':ok,'status':'CORRECT' if ok else 'INCORRECT' if val is not None else 'REVIEW','reason':'numeric_code'}
 if cat=='birthplace_tld':
  def domain(s):
   s=s.strip('.');
   try:s=s.encode('idna').decode('ascii')
   except UnicodeError:return None
   return s if re.fullmatch(r'[a-z0-9](?:[a-z0-9-]*[a-z0-9])?',s) else None
  val=domain(n);ok=val is not None and val in {domain(norm(x)) for x in e['Answer']}
  return {'correct':ok,'status':'CORRECT' if ok else 'INCORRECT' if val is not None else 'REVIEW','reason':'tld'}
 raise ValueError(cat)

def proxy_countries(raw,e,score):
 if score['correct'] or score['status'] in ['HISTORICAL','REVIEW']:return []
 countries=[]
 for country,attrs in TABLE.items():
  if country==e['birth_country']:continue
  if e['category'] in CURRENCY:
   codes,_=resolve_currency(raw)
   match=bool(codes&current_codes(country))
  else:
   other=dict(e,birth_country=country,Answer=attrs.get(e['category'],[]))
   match=bool(other['Answer']) and score_answer(raw,other)['correct']
  if match:countries.append(country)
 return countries

def score_row(row):
 e=row['e'];country,answer,malformed=parts(row['raw'],row['condition']);s=score_answer(answer,e)
 if malformed or row.get('truncated'):
  s={'correct':False,'status':'REVIEW','reason':'truncated' if row.get('truncated') else 'malformed_C2'}
 return dict(row,final_answer=answer,explicit_country=country,country_correct=country_score(country,e),score=s,wrong_country_proxy_candidates=proxy_countries(answer,e,s))

def reference_audit(rows):
 out=[]
 for r in rows:
  e=r['e']
  if e['category'] in CURRENCY:
   old=set(TABLE[e['birth_country']].get('birthplace_currency_short',[]));now=expected_codes(e)
   if old!=now:out.append({'id':r['id'],'country':e['birth_country'],'old_codes':sorted(old),'current_snapshot_codes':sorted(now),'dataset_gold':e['Answer']})
 return out
