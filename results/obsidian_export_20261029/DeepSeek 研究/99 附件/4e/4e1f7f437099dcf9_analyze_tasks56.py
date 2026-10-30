"""Deterministic tasks 5/6; requires fresh --out and frozen historical XML."""
import csv,json,collections,hashlib,argparse,math,xml.etree.ElementTree as ET
from pathlib import Path
from scoring_v2 import is_correct,old_is_correct,norm,value
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];DATA=ROOT/'inward/data/compositional_celebrities.json'
AUTO=['A_OTHER_COUNTRY','B_OTHER_ATTRIBUTE','C_HISTORICAL_CURRENCY','D_UNDETERMINED']
MANUAL=['GENUINE','AMBIGUOUS_GOLD','FORMAT','SCORER_ERROR']
def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def key(r):return (r['category'],r['person'])
def canon(v):
 n=value(v)
 return ('NUMBER',str(n.normalize())) if n is not None else ('TEXT',norm(v))
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def csvout(p,rs):
 if not rs:return
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def paired_p(b,c):
 n=b+c
 if n==0:return 1.0
 k=min(b,c);lps=[math.lgamma(n+1)-math.lgamma(i+1)-math.lgamma(n-i+1)-n*math.log(2) for i in range(k+1)]
 m=max(lps)
 return min(1.0,2*math.exp(m)*sum(math.exp(x-m) for x in lps))
def wilson(k,n):
 if not n:return [None,None]
 z=1.96;p=k/n;d=1+z*z/n;m=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return [max(0,m-h),min(1,m+h)]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(exist_ok=False)
 data=json.load(open(DATA))['data'];by={}
 for e in data:
  if key(e) in by:raise ValueError('duplicate data identity')
  by[key(e)]=e
 table=collections.defaultdict(lambda:collections.defaultdict(set))
 for e in data:
  if e['category'].startswith('birthplace'):
   table[e['birth_country']][e['category']].update(str(v) for v in e['A2'])
 dump(a.out/'country_attributes.json',{c:{cat:sorted(vs) for cat,vs in sorted(attrs.items())} for c,attrs in sorted(table.items())})
 reverse=collections.defaultdict(list)
 for c,attrs in table.items():
  for cat,vs in attrs.items():
   for v in vs:reverse[(cat,canon(v))].append(c)
 # ISO entity naming differences are fixed explicit aliases, not fuzzy or model matching.
 iso_alias={'United States':'UNITED STATES','Russia':'RUSSIAN FEDERATION','South Korea':'KOREA, REPUBLIC OF','North Korea':'KOREA, DEMOCRATIC PEOPLE’S REPUBLIC OF','Vietnam':'VIET NAM','Venezuela':'VENEZUELA (BOLIVARIAN REPUBLIC OF)','Iran':'IRAN (ISLAMIC REPUBLIC OF)','Bolivia':'BOLIVIA (PLURINATIONAL STATE OF)','Tanzania':'TANZANIA, UNITED REPUBLIC OF','Turkey':'TURKEY','Taiwan':'TAIWAN (PROVINCE OF CHINA)','Moldova':'MOLDOVA, REPUBLIC OF','Czech Republic':'CZECHIA','United Kingdom':'UNITED KINGDOM','Syria':'SYRIAN ARAB REPUBLIC','Laos':'LAO PEOPLE’S DEMOCRATIC REPUBLIC'}
 # Fixed common qualified names not present in ISO's short names; no new adjudication per answer.
 name_alias={'DEM':['German mark','Deutsche mark','Deutschmark'],'FRF':['French franc'],'ITL':['Italian lira'],'ESP':['Spanish peseta'],'GRD':['Greek drachma'],'ATS':['Austrian schilling'],'EEK':['Estonian kroon'],'TRL':['Turkish lira','old Turkish lira'],'ROL':['Romanian leu','old Romanian leu'],'LTL':['Lithuanian litas'],'LVL':['Latvian lats'],'NLG':['Dutch guilder'],'HRK':['Croatian kuna'],'BEF':['Belgian franc'],'FIM':['Finnish markka'],'PTE':['Portuguese escudo'],'IEP':['Irish pound']}
 entries=[]
 for e in ET.parse(HERE/'iso4217_historical.xml').getroot().iter('HstrcCcyNtry'):
  entries.append({tag:(e.findtext(tag) or '').strip() for tag in ['CtryNm','CcyNm','Ccy','WthdrwlDt']})
 current_names=collections.defaultdict(set)
 for ce in ET.parse(HERE/'iso4217_current.xml').getroot().iter('CcyNtry'):
  entity=norm(ce.findtext('CtryNm') or '')
  for c in table:
   if entity in {norm(c),norm(iso_alias.get(c,c))}:
    current_names[c].add(norm(ce.findtext('CcyNm') or ''))
 hist=collections.defaultdict(list)
 for c in table:
  country_names={norm(c),norm(iso_alias.get(c,c))}
  for e in entries:
   if norm(e['CtryNm']) in country_names:
    for x in [e['Ccy'],e['CcyNm']]+name_alias.get(e['Ccy'],[]):
     # Bare/qualified historical names also used by a current currency cannot prove obsolescence.
     xn=norm(x)
     name_collision=x!=e['Ccy'] and any(cn==xn or cn.endswith(' '+xn) or xn.endswith(' '+cn) for cn in current_names[c] if cn)
     if x and not name_collision:hist[(c,canon(x))].append(e)
 dump(a.out/'historical_rules.json',{'entity_aliases':iso_alias,'name_aliases':name_alias,'source_entries':entries,'priority':'correct same property -> D; historical correct-country -> C; correct-country other attribute -> B; other-country same attribute -> A; else D','normalization':'NFKC/casefold/whitespace; full string equality; numeric exact Decimal with sign, no rounding; no fuzzy matching','limits':'A proves a table value mismatch only; shared numeric/currency values cannot prove a wrong intermediate country. C only matches a historical currency of the dataset country. Symbols not provided by ISO are not guessed.'})
 r14=read(ROOT/'inward/results/exp1v2_Qwen3-14B/items.jsonl')
 failures=[r for r in r14 if r['category'].startswith('birthplace') and 'common_name' not in r['category'] and r['q1'] is True and r['q2'] is True and r['clear'] is False]
 assert len(failures)==818
 automatic=[]
 for r in failures:
  e=by[key(r)];c=e['birth_country'];cat=r['category'];p=canon(r['pred_clear']);same=c in reverse.get((cat,p),[])
  hs=hist.get((c,p),[]) if 'currency' in cat else []
  otherattrs=[k for k,vs in table[c].items() if k!=cat and any(canon(v)==p for v in vs)]
  othercountries=sorted(set(reverse.get((cat,p),[]))-{c})
  label='D_UNDETERMINED' if same else 'C_HISTORICAL_CURRENCY' if hs else 'B_OTHER_ATTRIBUTE' if otherattrs else 'A_OTHER_COUNTRY' if othercountries else 'D_UNDETERMINED'
  automatic.append(dict(category=cat,person=r['person'],country=c,pred_clear=r['pred_clear'],gold=json.dumps(r['gold'],ensure_ascii=False),auto_label=label,correct_same_property_match=same,other_countries=json.dumps(othercountries,ensure_ascii=False),other_attributes=json.dumps(otherattrs,ensure_ascii=False),historical_matches=json.dumps(hs,ensure_ascii=False)))
 csvout(a.out/'automatic_818.csv',automatic)
 counts=collections.Counter(r['auto_label'] for r in automatic)
 dump(a.out/'automatic_summary.json',{'n':len(automatic),'counts':{l:counts[l] for l in AUTO},'shares':{l:counts[l]/len(automatic) for l in AUTO},'wilson95':{l:wilson(counts[l],len(automatic)) for l in AUTO}})
 manualpath=ROOT/'inward/results/handoff_analysis_20261003/analysis_outputs/failures_120_labeled.csv'
 manual=list(csv.DictReader(manualpath.open(encoding='utf-8-sig')));am={key(r):r for r in automatic};cm={m:{l:0 for l in AUTO} for m in MANUAL};matched=[]
 semantic={'A_OTHER_COUNTRY':'GENUINE','B_OTHER_ATTRIBUTE':'FORMAT','C_HISTORICAL_CURRENCY':'AMBIGUOUS_GOLD'}
 for r in manual:
  t=am[key(r)];assert r['composed_answer']==t['pred_clear'];cm[r['label']][t['auto_label']]+=1
  matched.append(dict(id=r['id'],category=r['category'],person=r['person'],manual_label=r['label'],auto_label=t['auto_label'],agreement=(semantic.get(t['auto_label'])==r['label']) if t['auto_label'] in semantic else '',pred_clear=t['pred_clear']))
 classified=[r for r in matched if r['auto_label'] in semantic];agree=sum(r['agreement'] for r in classified)
 dump(a.out/'confusion_120.json',{'rows_manual_columns_auto':cm,'covered_n':len(classified),'agreement_n':agree,'conditional_agreement':agree/len(classified) if classified else None,'wilson95':wilson(agree,len(classified)),'note':'D is abstention, not SCORER_ERROR. No four-way accuracy claim. Manual labels are prior model-review labels.'});csvout(a.out/'matched_120.csv',matched)
 allstats=[];changed=[];replay=collections.Counter()
 for size in ['0.6B','1.7B','8B','14B']:
  path=ROOT/f'inward/results/exp1v2_Qwen3-{size}/items.jsonl';rows=read(path);scored=[]
  for line,r in enumerate(rows,1):
   e=by[key(r)];new={};trunc={}
   for field,gold in [('q1','A1'),('q2','A2'),('clear','Answer'),('ctx','Answer')]:
    pred=r['pred_'+field];old=old_is_correct(pred,e[gold]);replay[(size,field)]+=old!=r[field]
    new[field]=is_correct(pred,e[gold],'floor');trunc[field]=is_correct(pred,e[gold],'truncate')
    if new[field]!=r[field]:changed.append(dict(size=size,source_line=line,category=r['category'],person=r['person'],field=field,prediction=pred,gold=json.dumps(e[gold],ensure_ascii=False),old_ok=r[field],floor_ok=new[field],truncate_ok=trunc[field]))
   scored.append(dict(category=r['category'],person=r['person'],old={f:r[f] for f in new},floor=new,truncate=trunc))
  with open(a.out/f'rescored_{size}.jsonl','w') as f:
   for r in scored:f.write(json.dumps(r,ensure_ascii=False)+'\n')
  categories=['ALL','CLEAN']+sorted({r['category'] for r in scored})
  for cat in categories:
   rs=[r for r in scored if cat=='ALL' or (cat=='CLEAN' and r['category'].startswith('birthplace') and 'common_name' not in r['category']) or r['category']==cat]
   modes={}
   for mode in ['old','floor','truncate']:
    known=[r for r in rs if r[mode]['q1'] and r[mode]['q2']];fail=sum(not r[mode]['clear'] for r in known);ctxfail=sum(not r[mode]['ctx'] for r in known)
    modes[mode]={'known_n':len(known),'failed_clear_n':fail,'gap_clear':fail/len(known) if known else None,'gap_context':ctxfail/len(known) if known else None,'gap_clear_wilson95':wilson(fail,len(known)),**{f+'_accuracy':sum(r[mode][f] for r in rs)/len(rs) for f in ['q1','q2','clear','ctx']}}
   fixedold=[r for r in rs if r['old']['q1'] and r['old']['q2']]
   tests={}
   for field in ['q1','q2','clear','ctx']:
    b=sum(r['old'][field] and not r['floor'][field] for r in rs);c=sum(not r['old'][field] and r['floor'][field] for r in rs)
    tests[field]={'old_true_new_false':b,'old_false_new_true':c,'mcnemar_p':paired_p(b,c)}
   allstats.append(dict(size=size,category=cat,n=len(rs),coordinate_category=cat in ['birthplace_rounded_lat','birthplace_rounded_lng'],modes=modes,floor_gap_on_old_known=sum(not r['floor']['clear'] for r in fixedold)/len(fixedold) if fixedold else None,paired_label_changes=tests))
 assert not any(replay.values()),f'Original flags not reproducible: {replay}'
 dump(a.out/'gap_comparison.json',allstats);csvout(a.out/'score_changes.csv',changed)
 flat=[]
 for r in allstats:
  x={k:r[k] for k in ['size','category','n','coordinate_category','floor_gap_on_old_known']}
  for mode,v in r['modes'].items():
   for k,val in v.items():x[mode+'_'+k]=json.dumps(val) if isinstance(val,list) else val
  x['gap_delta_pp']=100*(r['modes']['floor']['gap_clear']-r['modes']['old']['gap_clear']) if r['modes']['floor']['gap_clear'] is not None and r['modes']['old']['gap_clear'] is not None else None
  flat.append(x)
 csvout(a.out/'gap_comparison.csv',flat)
 sources=[DATA,HERE/'iso4217_historical.xml',HERE/'iso4217_current.xml',manualpath,ROOT/'inward/exp1_gap_v2.py']+[ROOT/f'inward/results/exp1v2_Qwen3-{s}/items.jsonl' for s in ['0.6B','1.7B','8B','14B']]
 dump(a.out/'provenance.json',{'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'command':f'python3 {HERE}/analyze_tasks56.py --out NEW_DIRECTORY','scoring':'Every numeric alias: one signed numeric expression, floor compare; truncate sensitivity. Text scorer unchanged. Gap = clear failures among q1 && q2 under the same scoring mode. ALL includes date/year/translation; CLEAN birthplace excluding common_name. Wilson intervals per proportion; McNemar on paired scoring flags, not on gaps with changing denominators.','original_replay_mismatches':{str(k):v for k,v in replay.items()}})
 print('automatic',dict(counts));print('sample confusion',cm,'covered',len(classified),'agree',agree)
 for r in allstats:
  if r['category'] in ['CLEAN','birthplace_rounded_lat','birthplace_rounded_lng']:
   print(r['size'],r['category'],[(m,r['modes'][m]['known_n'],r['modes'][m]['gap_clear']) for m in ['old','floor','truncate']])
if __name__=='__main__':main()
