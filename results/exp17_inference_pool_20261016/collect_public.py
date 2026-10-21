"""Public biographical metadata collection; never reads model predictions for selection."""
import json,hashlib,time,urllib.request,urllib.parse,sys,datetime
from pathlib import Path
P=Path(__file__).resolve().parent;C=P/'public_cache';C.mkdir(exist_ok=True)
import concurrent.futures,threading
RATE_LOCK=threading.Lock();LAST_REQUEST=[0.]
UA='INWARD-Research/1.0 (public biographical metadata, no model training requests)'
def api(host,params):
 params={'format':'json','formatversion':2,**params};url='https://'+host+'/w/api.php?'+urllib.parse.urlencode(params);name=hashlib.sha256(url.encode()).hexdigest();f=C/(name+'.json')
 if f.exists():return json.loads(f.read_text())['response']
 for attempt in range(6):
  try:
   with RATE_LOCK:
    pause=max(0,6-(time.monotonic()-LAST_REQUEST[0]))
    if pause:time.sleep(pause)
    LAST_REQUEST[0]=time.monotonic()
   req=urllib.request.Request(url,headers={'User-Agent':UA})
   with urllib.request.urlopen(req,timeout=60) as r:data=json.load(r)
   if 'error' in data:raise RuntimeError(str(data['error']))
   with f.open('x') as out:json.dump({'url':url,'fetched_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'response':data},out,ensure_ascii=False)
   time.sleep(2);return data
  except Exception as e:
   if attempt==5:raise
   print("API retry", host, type(e).__name__, str(e), flush=True)
   time.sleep(min(60,20*(attempt+1)))
def entities(ids):
 result={};ids=sorted(set(ids));batches=[ids[i:i+40] for i in range(0,len(ids),40)]
 def fetch(chunk):return api('www.wikidata.org',{'action':'wbgetentities','ids':'|'.join(chunk),'props':'labels|aliases|claims|sitelinks','languages':'en','sitefilter':'enwiki'})['entities']
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
  futures=[executor.submit(fetch,chunk) for chunk in batches]
  for j,future in enumerate(concurrent.futures.as_completed(futures)):
   result.update(future.result())
   if j%10==0:print('entity batches',j+1,'/',len(batches),flush=True)
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
 white={'Q30','Q145','Q142','Q16','Q17','Q408','Q664','Q183','Q40','Q38','Q29','Q45','Q155','Q414','Q96','Q298','Q739','Q419','Q77','Q33','Q34','Q20','Q35','Q189','Q39','Q41','Q884'}
 possible=[e for e in all_entities.values() if len(qids(e,'P27'))==1 and qids(e,'P27')<=white]
 places=set().union(*(qids(e,'P19') for e in possible))|white
 locs=entities(places)
 # Ancestor locations are used to flag disputed/historical/overseas entities, never infer nationality.
 frontier=set().union(*(qids(e,'P131') for e in locs.values()))-set(locs)
 for depth in range(5):
  if not frontier:break
  new=entities(frontier);locs.update(new);frontier=set().union(*(qids(e,'P131') for e in new.values()))-set(locs)
 save('LOCATION_ENTITIES.json',locs)
 print('collection complete',len(candidates),len(resolved),'unresolved',len(unresolved),'locations',len(locs),flush=True)
if __name__=='__main__':main()
