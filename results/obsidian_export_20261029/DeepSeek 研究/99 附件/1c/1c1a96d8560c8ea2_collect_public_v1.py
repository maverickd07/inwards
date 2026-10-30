"""Public biographical metadata collection; never reads model predictions for selection."""
import json,hashlib,time,urllib.request,urllib.parse,sys,datetime
from pathlib import Path
P=Path(__file__).resolve().parent;C=P/'public_cache';C.mkdir(exist_ok=True)
UA='INWARD-Research/1.0 (public biographical metadata, no model training requests)'
def api(host,params):
 params={'format':'json','formatversion':2,**params};url='https://'+host+'/w/api.php?'+urllib.parse.urlencode(params);name=hashlib.sha256(url.encode()).hexdigest();f=C/(name+'.json')
 if f.exists():return json.loads(f.read_text())['response']
 for attempt in range(4):
  try:
   req=urllib.request.Request(url,headers={'User-Agent':UA})
   with urllib.request.urlopen(req,timeout=60) as r:data=json.load(r)
   if 'error' in data:raise RuntimeError(str(data['error']))
   with f.open('x') as out:json.dump({'url':url,'fetched_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'response':data},out,ensure_ascii=False)
   time.sleep(.15);return data
  except Exception as e:
   if attempt==3:raise
   time.sleep(2*(attempt+1))
def entities(ids):
 result={};ids=sorted(set(ids))
 for i in range(0,len(ids),40):
  got=api('www.wikidata.org',{'action':'wbgetentities','ids':'|'.join(ids[i:i+40]),'props':'labels|aliases|claims|sitelinks','languages':'en','sitefilter':'enwiki'})
  result.update(got['entities'])
  if i%400==0:print('entities',i,'/',len(ids),flush=True)
 return result

def vals(e,p):
 return [x['mainsnak']['datavalue']['value'] for x in e.get('claims',{}).get(p,[]) if x.get('rank')!='deprecated' and x.get('mainsnak',{}).get('snaktype')=='value']
def qids(e,p):return {v['id'] for v in vals(e,p) if isinstance(v,dict) and 'id'in v}
def save(n,x):
 f=P/n
 if not f.exists():
  with f.open('x') as o:json.dump(x,o,ensure_ascii=False,indent=2)

def main():
 # Fixed acquisition window: 500 most-recently-added category members per birth year.
 candidates={}
 for year in range(1985,2000):
  data=api('en.wikipedia.org',{'action':'query','generator':'categorymembers','gcmtitle':f'Category:{year}_births','gcmtype':'page','gcmsort':'timestamp','gcmdir':'desc','gcmlimit':500,'prop':'pageprops'})
  for r in data.get('query',{}).get('pages',[]):
   q=r.get('pageprops',{}).get('wikibase_item')
   if q:candidates[q]={'qid':q,'title':r['title'],'category_year':year,'pageid':r['pageid']}
  print('year',year,'cumulative candidates',len(candidates),flush=True)
 save('CANDIDATE_ENUMERATION.json',candidates)
 # Resolve every original dataset identity (train + evaluation + other original people).
 original=json.loads((P.parents[1]/'inward/data/compositional_celebrities.json').read_text())['data'];names=sorted({e['person'] for e in original})
 resolved={};unresolved=[]
 for start in range(0,len(names),40):
  chunk=names[start:start+40];d=api('en.wikipedia.org',{'action':'query','titles':'|'.join(chunk),'redirects':1,'prop':'pageprops|redirects','rdlimit':500})
  normal={x['from']:x['to'] for x in d.get('query',{}).get('normalized',[])};normal.update({x['from']:x['to'] for x in d.get('query',{}).get('redirects',[])})
  pages={x['title']:x for x in d.get('query',{}).get('pages',[])}
  for name in chunk:
   title=name;seen=set()
   while title in normal and title not in seen:seen.add(title);title=normal[title]
   page=pages.get(title,{});q=page.get('pageprops',{}).get('wikibase_item')
   if q and 'disambiguation' not in page.get('pageprops',{}):resolved[name]={'qid':q,'canonical_title':title,'redirect_aliases':[x['title'] for x in page.get('redirects',[])]}
   else:unresolved.append(name)
 save('ORIGINAL_ID_RESOLUTION.json',{'resolved':resolved,'unresolved':unresolved})
 all_entities=entities(set(candidates)|{r['qid'] for r in resolved.values()});save('PERSON_ENTITIES.json',all_entities)
 places=set().union(*(qids(e,'P19') for e in all_entities.values()));places|=set().union(*(qids(e,'P27') for e in all_entities.values()))
 locs=entities(places)
 # Ancestor locations are used to flag disputed/historical/overseas entities, never infer nationality.
 frontier=set().union(*(qids(e,'P131') for e in locs.values()))-set(locs)
 for depth in range(5):
  if not frontier:break
  new=entities(frontier);locs.update(new);frontier=set().union(*(qids(e,'P131') for e in new.values()))-set(locs)
 save('LOCATION_ENTITIES.json',locs)
 print('collection complete',len(candidates),len(resolved),'unresolved',len(unresolved),'locations',len(locs),flush=True)
if __name__=='__main__':main()
