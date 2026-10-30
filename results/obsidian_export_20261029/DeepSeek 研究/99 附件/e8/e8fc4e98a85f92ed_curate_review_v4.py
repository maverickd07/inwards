"""Conservative metadata screen and review evidence. No model prediction used for new selection."""
import sys,json,re,unicodedata,hashlib,collections
from pathlib import Path
from collect_public import vals,qids,entities,api,save
P=Path(__file__).resolve().parent
WHITE={'Q30':'United States','Q145':'United Kingdom','Q142':'France','Q16':'Canada','Q17':'Japan','Q408':'Australia','Q664':'New Zealand','Q183':'Germany','Q40':'Austria','Q38':'Italy','Q29':'Spain','Q45':'Portugal','Q155':'Brazil','Q414':'Argentina','Q96':'Mexico','Q298':'Chile','Q739':'Colombia','Q419':'Peru','Q77':'Uruguay','Q33':'Finland','Q34':'Sweden','Q20':'Norway','Q35':'Denmark','Q189':'Iceland','Q39':'Switzerland','Q41':'Greece','Q884':'South Korea'}
RISK_WORDS=['disputed territory','disputed region','unrecognized state','partially recognized','former country','former state','occupied territory','dependent territory','overseas territory','overseas collectivity','soviet union','yugoslavia','czechoslovakia','east germany','german democratic republic','crimea','kosovo','gaza','west bank','kashmir','abkhazia','south ossetia','nagorno','donetsk','luhansk','northern cyprus','falkland','gibraltar','new caledonia','guam','puerto rico','french polynesia','mayotte','réunion','greenland','faroe islands','hong kong','macau','taiwan']
DUAL=re.compile(r'\b(?:dual\s+(?:citizen\w*|national\w*|passport\w*)|double\s+(?:citizen\w*|national\w*)|multiple\s+citizenship|natural[is]ed|natural[is]ation|acquir(?:ed|ing)\s+(?:\w+\s+){0,3}citizenship|obtained\s+(?:\w+\s+){0,3}citizenship|holds?\s+(?:both|two)\s+(?:\w+\s+){0,3}(?:passport|citizen))',re.I)
def norm(s):return ''.join(c for c in unicodedata.normalize('NFKD',s).casefold() if c.isalnum())
def names(e):
 out={e.get('labels',{}).get('en',{}).get('value','')};out.update(a['value'] for a in e.get('aliases',{}).get('en',[]));out.update(v.get('text','') for v in vals(e,'P1477') if isinstance(v,dict));out.update(v.get('text','') for v in vals(e,'P1449') if isinstance(v,dict))
 out.add(e.get('sitelinks',{}).get('enwiki',{}).get('title',''));return out-{''}
def ancestry(place,locs):
 seen=set();todo=[place]
 for _ in range(7):
  fresh=[x for x in todo if x not in seen];seen.update(fresh);todo=[p for x in fresh for p in qids(locs.get(x,{}),'P131')]
 return sorted(seen)
def main():
 ens=json.loads((P/'PERSON_ENTITIES.json').read_text());locs=json.loads((P/'LOCATION_ENTITIES.json').read_text());enum=json.loads((P/'CANDIDATE_ENUMERATION.json').read_text());original=json.loads((P/'ORIGINAL_ID_RESOLUTION.json').read_text());resolved={**original['resolved'],**json.loads((P/'IDENTITY_SUPPLEMENT.json').read_text())};ens.update(json.loads((P/'IDENTITY_SUPPLEMENT_ENTITIES.json').read_text()))
 redirect_complete=json.loads((P/'ORIGINAL_REDIRECT_ALIASES_COMPLETE.json').read_text())
 for rr in resolved.values():rr['redirect_aliases']=sorted(set(rr['redirect_aliases'])|set(redirect_complete.get(rr['canonical_title'],[])))
 dataset=json.loads((P.parents[1]/'inward/data/compositional_celebrities.json').read_text())['data'];oldnames={r['person'] for r in dataset};oldids={r['qid'] for r in resolved.values()};oldaliases=set(map(norm,oldnames))
 for r in resolved.values():oldaliases.update(map(norm,[r['canonical_title'],*r['redirect_aliases'],*names(ens.get(r['qid'],{}))]))
 oldtrainpath=P.parents[1]/'results/exp12_sources_20261014_v2/received_700454/extracted/runs_700454/plan/train.jsonl';oldtrain={r['person']:r['e']['birth_country'] for r in map(json.loads,oldtrainpath.read_text().splitlines())}
 evalnames=set()
 for split in ['A_newPerson','C_bothNew','D_reservedC']:
  f=oldtrainpath.parent/('items_'+split+'.jsonl');evalnames.update(json.loads(x)['person'] for x in f.read_text().splitlines())
 evalids={resolved[n]['qid'] for n in evalnames if n in resolved};evalaliases=set(map(norm,evalnames))
 for name in evalnames:
  rr=resolved.get(name)
  if rr:evalaliases.update(map(norm,[rr['canonical_title'],*rr['redirect_aliases'],*names(ens.get(rr['qid'],{}))]))
 candidates=[];audits=[]
 display_ids=collections.defaultdict(set)
 for q in enum:display_ids[norm(ens.get(q,{}).get('labels',{}).get('en',{}).get('value',''))].add(q)
 def screen(q,mode,name_override=None):
  e=ens.get(q,{});ns=names(e);name=name_override or e.get('labels',{}).get('en',{}).get('value','');reasons=[];flags=[]
  if 'Q5' not in qids(e,'P31'):reasons.append('not_confirmed_human')
  if not name or not e.get('sitelinks',{}).get('enwiki'):reasons.append('identity_or_english_biography_missing')
  if mode=='new' and len(display_ids.get(norm(name),set()))>1:reasons.append('ambiguous_shared_display_name')
  if mode=='new' and(q in oldids or set(map(norm,ns)) & oldaliases):reasons.append('existing_person_or_alias')
  if mode=='old' and(q in evalids or set(map(norm,ns)) & evalaliases):reasons.append('evaluation_entity_or_alias_leakage')
  dates=vals(e,'P569');years={int(v['time'][1:5]) for v in dates if isinstance(v,dict) and v.get('time','').startswith('+')}
  if len(years)!=1:reasons.append('birth_date_missing_or_conflicting')
  year=next(iter(years)) if len(years)==1 else None
  # The candidate category window is not applied to old people. An old DOB before 1955 is conservatively outside this stable-location review window.
  if mode=='new' and (year is None or not 1985<=year<=1999):reasons.append('birth_year_outside_candidate_window')
  if mode=='old' and year is not None and year<1955:reasons.append('old_person_before_conservative_1955_window')
  cit=qids(e,'P27');births=qids(e,'P19');country=None;place=None;countryids=set();an=[]
  if any(x.get('rank')!='deprecated' and x.get('mainsnak',{}).get('snaktype')!='value' for prop in ['P19','P27'] for x in e.get('claims',{}).get(prop,[])):reasons.append('uncertain_birthplace_or_citizenship_statement')
  if len(cit)>1:reasons.append('multiple_citizenship_recorded');flags.append('multiple_citizenship_recorded')
  elif not cit:reasons.append('citizenship_missing')
  elif not cit<=set(WHITE):reasons.append('citizenship_outside_conservative_scope')
  if len(births)!=1:reasons.append('birthplace_missing_or_multiple')
  else:
   place=next(iter(births));countryids=qids(locs.get(place,{}),'P17');an=ancestry(place,locs)
   if any(x.get('rank')!='deprecated' and x.get('mainsnak',{}).get('snaktype')!='value' for x in locs.get(place,{}).get('claims',{}).get('P17',[])):reasons.append('uncertain_birth_country_statement')
   if len(countryids)>1:reasons.append('historical_or_multiple_birth_country');flags.append('historical_or_multiple_birth_country')
   elif not countryids:reasons.append('birth_country_missing_or_not_loaded_after_prior_exclusion')
   else:
    countryqid=next(iter(countryids));country=WHITE.get(countryqid)
    if country is None:reasons.append('birth_country_outside_conservative_scope')
    if countryqid=='Q183' and year is not None and year<1991:reasons.append('germany_before_1991');flags.append('changed_country_time_window')
    if len(cit)==1 and next(iter(cit))!=countryqid:reasons.append('citizenship_differs_from_birth_country');flags.append('cross_country_identity_risk')
   risky=[]
   for ancestor in an:
    item=locs.get(ancestor,{});label=item.get('labels',{}).get('en',{}).get('value','').casefold()
    if any((label=='jersey' if s=='jersey' else s in label) for s in RISK_WORDS) or qids(item,'P31') & {'Q15239622','Q3024240','Q6256'} and ancestor not in WHITE and ancestor!=place:
     if any((label=='jersey' if s=='jersey' else s in label) for s in RISK_WORDS) or qids(item,'P31') & {'Q15239622','Q3024240'}:risky.append(ancestor)
   if risky:reasons.append('territorial_or_historical_location_risk');flags.append('territorial_or_historical_location_risk')
  if mode=='old' and country!=oldtrain.get(name):reasons.append('old_reference_birth_country_not_confirmed')
  if mode=='new' and int(hashlib.md5(name.encode()).hexdigest(),16)%2:reasons.append('not_even_md5_training_partition')
  rec={'qid':q,'person':name,'mode':mode,'aliases':sorted(ns),'birth_year':year,'birthplace_qid':place,'birth_country_qids':sorted(countryids),'birth_country':country,'citizenship_qids':sorted(cit),'ancestor_qids':an,'metadata_reasons':reasons,'risk_flags':flags,'entity_revision':e.get('lastrevid'),'wikidata_url':'https://www.wikidata.org/wiki/'+q,'enwiki_title':e.get('sitelinks',{}).get('enwiki',{}).get('title'),'birthplace_statement_ids':[x.get('id') for x in e.get('claims',{}).get('P19',[])],'citizenship_statement_ids':[x.get('id') for x in e.get('claims',{}).get('P27',[])]}
  return rec
 for q in sorted(enum):audits.append(screen(q,'new'))
 for name in sorted(oldtrain):
  q=resolved.get(name,{}).get('qid')
  if q:audits.append(screen(q,'old',name))
  else:audits.append({'person':name,'mode':'old','qid':None,'metadata_reasons':['original_identity_unresolved'],'risk_flags':[]})
 preliminary=[r for r in audits if not r['metadata_reasons']]
 save('METADATA_SCREEN.json',audits);print('metadata screen eligible',collections.Counter(r['mode'] for r in preliminary),'countries',collections.Counter(r['birth_country'] for r in preliminary if r['mode']=='new'),flush=True)
 # All Wikipedia language sitelinks measure prominence for deterministic ranking; not model outputs.
 sitecounts={}
 for i in range(0,len(preliminary),40):
  qs=[r['qid'] for r in preliminary[i:i+40]];d=api('www.wikidata.org',{'action':'wbgetentities','ids':'|'.join(qs),'props':'sitelinks'})
  for q,e in d['entities'].items():sitecounts[q]=len(e.get('sitelinks',{}))
 save('SITELINK_COUNTS.json',sitecounts)
 # Fetch full original English wikitext, not summaries. Retain evidence but never upload entire biographies to HPC.
 bios={}
 for i in range(0,len(preliminary),20):
  chunk=preliminary[i:i+20];d=api('en.wikipedia.org',{'action':'query','titles':'|'.join(r['enwiki_title'] for r in chunk),'redirects':1,'prop':'revisions|redirects|pageprops','rvprop':'ids|content','rvslots':'main','rdlimit':500})
  pages={page['pageid']:page for page in d.get('query',{}).get('pages',[])}
  while 'continue' in d:
   d=api('en.wikipedia.org',{'action':'query','titles':'|'.join(r['enwiki_title'] for r in chunk),'redirects':1,'prop':'revisions|redirects|pageprops','rvprop':'ids|content','rvslots':'main','rdlimit':500,**d['continue']})
   for page in d.get('query',{}).get('pages',[]):
    prior=pages.setdefault(page['pageid'],{'pageid':page['pageid'],'title':page['title']})
    prior.setdefault('redirects',[]).extend(page.get('redirects',[]))
    for key in ['revisions','pageprops']:
     if key in page:prior[key]=page[key]
  for page in pages.values():
   q=page.get('pageprops',{}).get('wikibase_item');revs=page.get('revisions',[])
   if q and revs:bios[q]={'title':page['title'],'revision':revs[0]['revid'],'text':revs[0].get('slots',{}).get('main',{}).get('content',''),'redirects':[x['title'] for x in page.get('redirects',[])]}
  print('biography checks',min(i+20,len(preliminary)),'/',len(preliminary),flush=True)
 save('BIOGRAPHY_EVIDENCE.json',bios)
 for r in audits:
  r['all_reasons']=list(r['metadata_reasons'])
  if r['metadata_reasons']:continue
  bio=bios.get(r['qid']);r['sitelinks']=sitecounts.get(r['qid'],0)
  if not bio or not bio['text']:r['all_reasons'].append('biography_unavailable');continue
  r['biography_revision']=bio['revision'];r['redirect_aliases']=bio['redirects']
  if r['mode']=='new' and set(map(norm,bio['redirects'])) & oldaliases:r['all_reasons'].append('existing_person_redirect_alias')
  text=bio['text'];hits=[m.group(0) for m in DUAL.finditer(text)]
  # Citizenship/nationality infobox lists or compound-national descriptors are conservatively excluded.
  fields=re.findall(r'^\s*\|\s*(?:citizenship|nationality)\s*=([^\n]+)',text,re.M|re.I)
  mult=[f for f in fields if re.search(r'<br|hlist|plainlist|ubl\||\s/\s|\band\b|[,;]',f,re.I)]
  adjs='American|British|Canadian|French|German|Italian|Spanish|Portuguese|Australian|Japanese|Korean|Brazilian|Argentine|Mexican|Chilean|Colombian|Peruvian|Uruguayan|Swiss|Swedish|Norwegian|Danish|Finnish|Icelandic|Greek|Irish|Russian|Chinese|Indian|Pakistani|Nigerian|Ghanaian|Jamaican|Dominican|Dutch|Belgian|Polish|Austrian|Turkish|Israeli|South African'
  compounds=re.findall(r'\b(?:'+adjs+r')[–-](?:'+adjs+r')\b',text[:12000],re.I)
  compounds+=re.findall(r'\b(?:'+adjs+r')\s+and\s+(?:'+adjs+r')\b',text[:12000],re.I)
  r['compound_nationality_signals']=compounds
  if hits or mult or compounds:r['all_reasons'].append('biography_multiple_or_naturalised_citizenship_signal');r['risk_flags'].append('biography_citizenship_signal')
  r['citizenship_screen_hits']=hits;r['nationality_infobox_fields']=fields
 eligible=[r for r in audits if r['mode']=='new' and not r['all_reasons']];by=collections.defaultdict(list)
 for r in eligible:by[r['birth_country']].append(r)
 for c in by:by[c].sort(key=lambda r:(-r['sitelinks'],int(r['qid'][1:])))
 chosen=[]
 for roundno in range(40):
  for country in sorted(by):
   if len(chosen)<400 and roundno<len(by[country]):chosen.append(by[country][roundno])
 selected={r['qid'] for r in chosen}
 for r in audits:
  r['selected_new']=r['mode']=='new' and r['qid'] in selected
  r['eligible_old']=r['mode']=='old' and not r['all_reasons']
  r['primary_exclusion']=r['all_reasons'][0] if r['all_reasons'] else ('eligible_not_selected_by_fixed_cap' if r['mode']=='new' and not r['selected_new'] else None)
 summary={}
 for mode in ['new','old']:
  rs=[r for r in audits if r['mode']==mode];summary[mode]={'audited':len(rs),'eligible':sum(not r['all_reasons'] for r in rs),'selected':sum(r['selected_new'] if mode=='new' else r['eligible_old'] for r in rs),'exclusive_exclusions':dict(collections.Counter(r['primary_exclusion'] for r in rs if r['primary_exclusion'])),'overlapping_risk_flags':dict(collections.Counter(x for r in rs for x in set(r['risk_flags'])))}
 summary['selected_countries']=dict(collections.Counter(r['birth_country'] for r in chosen));summary['selection_complete']=len(chosen)==400
 save('CURATION_AUDIT.json',audits);save('CURATION_SUMMARY.json',summary);save('SELECTED_NEW400.json',chosen);save('ELIGIBLE_OLD_PEOPLE.json',[r for r in audits if r.get('eligible_old')]);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
 if len(chosen)!=400:raise RuntimeError('Not enough screened people; no runnable package may be released')
RISK_WORDS.extend(['guadeloupe','martinique','french guiana','saint pierre and miquelon','wallis and futuna','northern mariana','american samoa','christmas island','cocos islands','norfolk island','bermuda','guernsey','jersey','isle of man','saint helena'])
if __name__=='__main__':main()
