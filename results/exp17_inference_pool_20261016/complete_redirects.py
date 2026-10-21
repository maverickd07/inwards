"""Complete API pagination for the previously resolved original identities; public aliases only."""
import json,urllib.parse,collections
from pathlib import Path
from collect_public import api,save
P=Path(__file__).resolve().parent
aliases=collections.defaultdict(set);queries=[]
for f in sorted((P/'public_cache').glob('*.json')):
 d=json.loads(f.read_text());params=dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(d['url']).query))
 if params.get('prop')=='pageprops|redirects' and 'rdcontinue' not in params:queries.append((params,d['response']))
for i,(params,d) in enumerate(queries):
 while True:
  for page in d.get('query',{}).get('pages',[]):aliases[page['title']].update(x['title'] for x in page.get('redirects',[]))
  if 'continue' not in d:break
  d=api('en.wikipedia.org',{**params,**d['continue']})
 print('original redirect pagination complete',i+1,'/',len(queries),flush=True)
save('ORIGINAL_REDIRECT_ALIASES_COMPLETE.json',{t:sorted(a) for t,a in aliases.items()})
print('Complete titles',len(aliases),'aliases',sum(map(len,aliases.values())),flush=True)
