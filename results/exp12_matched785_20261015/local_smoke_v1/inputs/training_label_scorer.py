"""Training-only coverage extension. Frozen evaluation v3 is imported, never mutated."""
import json,re
from pathlib import Path
import scorer_v3 as v3
POLICY=json.loads((Path(__file__).parent/'training_label_policy.json').read_text())

def capital(s):
 return v3.canonical(s.replace(',', ''),'country')

def codes(s):
 # Whole response only. Spaces/hyphens within a code are presentation;
 # comma, semicolon, slash and explicit conjunctions separate complete codes.
 parts=re.split(r'\s*(?:[,;/]|\band\b|\bor\b)\s*',s,flags=re.I)
 values=set()
 for part in parts:
  part=part.strip()
  if not re.fullmatch(r'\+?[0-9](?:[0-9 -]*[0-9])?',part):return None
  value=re.sub(r'[ -]','',part).lstrip('+')
  if not re.fullmatch(r'[1-9][0-9]{0,3}',value):return None
  values.add(value)
 return values

def score(pred,refs,category,known_capitals=()):
 if category not in ('birthplace_capital','birthplace_callingcode'):
  return v3.score(pred,refs,category)
 def result(ok,reason,review=False):
  return {'correct':bool(ok),'needs_review':bool(review),'reason':reason,'answer_type':category,'reference_policy':'unchanged_dataset_plus_declared_aliases_training_only'}
 text=v3.clean(pred)
 if text is None:return result(False,'unclosed_reasoning_markup',True)
 if not text:return result(False,'empty_answer')
 text=re.sub(r'^answer\s*:\s*','',text,flags=re.I)
 if category=='birthplace_capital':
  text=re.sub(r'^(?:the\s+)?capital(?:\s+city)?\s*(?::|is\b)\s*','',text,flags=re.I).strip()
  want={capital(v3.clean(x)) for x in refs}
  for group in POLICY['capital_alias_groups']:
   norms={capital(x) for x in group}
   if want & norms:want|=norms
  value=capital(text)
  if value in want:return result(True,'whole_capital_or_declared_alias')
  known={capital(x) for x in known_capitals}
  for group in POLICY['capital_alias_groups']:known.update(map(capital,group))
  return result(False,'different_known_capital' if value in known else 'unmatched_capital_or_prose',value not in known)
 text=v3.unwrap(text,category).rstrip('.')
 want=set()
 for ref in refs:
  value=codes(v3.clean(ref))
  if value is None:raise ValueError('Unparseable calling-code reference: '+str(ref))
  want|=value
 for group in POLICY['calling_code_groups']:
  if want & set(group):want.update(group)
 value=codes(text)
 if value is None:return result(False,'unsupported_or_multiple_code_format',True)
 if value and value<=want:return result(True,'whole_code_set_all_accepted')
 if '1' in value and any(x.startswith('1') and len(x)==4 for x in want):
  return result(False,'shared_plus1_vs_specific_territory_code',True)
 return result(False,'different_calling_code')
