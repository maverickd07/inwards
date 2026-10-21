from collect_public import api,entities,save
mapping={'Drake':'Drake (musician)','Hussein, Crown Prince':'Hussein, Crown Prince of Jordan','Kim Taehyung':'V (singer)','Luis Suarez':'Luis Suárez','Dhee':'Dhee (singer)','Sade':'Sade (singer)'}
d=api('en.wikipedia.org',{'action':'query','titles':'|'.join(mapping.values()),'redirects':1,'prop':'pageprops|redirects','rdlimit':500})
redir={r['from']:r['to'] for r in d.get('query',{}).get('redirects',[])};pages={p['title']:p for p in d['query']['pages']};out={}
for name,title in mapping.items():
 title=redir.get(title,title);p=pages.get(title,{})
 q=p.get('pageprops',{}).get('wikibase_item')
 if q and 'disambiguation' not in p.get('pageprops',{}):out[name]={'qid':q,'canonical_title':title,'redirect_aliases':[r['title'] for r in p.get('redirects',[])]}
assert len(out)==6,out
save('IDENTITY_SUPPLEMENT.json',out);save('IDENTITY_SUPPLEMENT_ENTITIES.json',entities(r['qid'] for r in out.values()))
print(out)
