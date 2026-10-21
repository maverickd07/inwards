"""Predeclared bounded second public-candidate window, still exactly400 selected people."""
import json
from pathlib import Path
from collect_public import api,entities,qids,save
P=Path(__file__).resolve().parent
old=json.loads((P/'CANDIDATE_ENUMERATION.json').read_text());more={}
for year in range(1985,2000):
 params={'action':'query','generator':'categorymembers','gcmtitle':f'Category:{year}_births','gcmtype':'page','gcmsort':'timestamp','gcmdir':'desc','gcmlimit':500,'prop':'pageprops'}
 first=api('en.wikipedia.org',params)
 if 'continue' not in first:continue
 d=api('en.wikipedia.org',{**params,'gcmlimit':200,**first['continue']})
 for r in d.get('query',{}).get('pages',[]):
  q=r.get('pageprops',{}).get('wikibase_item')
  if q and q not in old:more[q]={'qid':q,'title':r['title'],'category_year':year,'pageid':r['pageid'],'candidate_window':2}
 print('additional candidate year',year,'cumulative',len(more),flush=True)
save('CANDIDATE_ENUMERATION_EXPANDED.json',{**old,**more})
ens=json.loads((P/'PERSON_ENTITIES.json').read_text());new=entities(set(more)-set(ens));ens.update(new);save('PERSON_ENTITIES_EXPANDED.json',ens)
locs=json.loads((P/'LOCATION_ENTITIES.json').read_text());white={'Q30','Q145','Q142','Q16','Q17','Q408','Q664','Q183','Q40','Q38','Q29','Q45','Q155','Q414','Q96','Q298','Q739','Q419','Q77','Q33','Q34','Q20','Q35','Q189','Q39','Q41','Q884'}
possible=[e for q,e in new.items() if len(qids(e,'P27'))==1 and qids(e,'P27')<=white]
places=set().union(*(qids(e,'P19') for e in possible))-set(locs);added=entities(places);locs.update(added)
frontier=set().union(*(qids(e,'P131') for e in added.values()))-set(locs)
for depth in range(5):
 if not frontier:break
 added=entities(frontier);locs.update(added);frontier=set().union(*(qids(e,'P131') for e in added.values()))-set(locs)
save('LOCATION_ENTITIES_EXPANDED.json',locs)
print('Expanded collection complete',len(old)+len(more),'new',len(more),'locations',len(locs),flush=True)
