import sys,json,hashlib,collections,shutil,datetime
sys.dont_write_bytecode=True
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[1];PKG=P/'package';I=PKG/'inputs';sys.path.insert(0,str(PKG))
from shared import read,write,js,sha
summary=json.loads((P/'CURATION_SUMMARY.json').read_text());new=json.loads((P/'SELECTED_NEW400.json').read_text());old=json.loads((P/'ELIGIBLE_OLD_PEOPLE.json').read_text())
assert summary['selection_complete'] and len(new)==400 and len({r['qid'] for r in new})==400
oldtrain=read(ROOT/'results/exp12_sources_20261014_v2/received_700454/extracted/runs_700454/plan/train.jsonl');traincountry={r['person']:r['e']['birth_country'] for r in oldtrain}
paired=read(ROOT/'results/exp12_sources_20261014_v2/received_700454/extracted/runs_700454/paired_targets/paired_diagnostics.jsonl');first=collections.defaultdict(set)
for r in paired:first[r['person']].add(r['own_q1'])
assert all(len(v)==1 for v in first.values())
def record(r,source):
 return {'person_id':source+':'+r['qid'],'qid':r['qid'],'person':r['person'],'birth_country':r['birth_country'],'birth_year':r['birth_year'],'source':source,'birthplace_qid':r['birthplace_qid'],'citizenship_qids':r['citizenship_qids'],'aliases':r['aliases'],'wikidata_url':r['wikidata_url'],'enwiki_title':r['enwiki_title'],'wikidata_revision':r['entity_revision'],'biography_revision':r['biography_revision'],'birthplace_statement_ids':r['birthplace_statement_ids'],'citizenship_statement_ids':r['citizenship_statement_ids'],'source_review':'single-record citizenship, consistent birth-country, no flagged biography signals; absence of second citizenship is not legally certified'}
np=[record(r,'new') for r in new];op=[record(r,'old') for r in old];assert len({r['qid'] for r in op})==len(op)
assert not {r['qid'] for r in np}&{r['qid'] for r in op}
for r in op:assert r['birth_country']==traincountry[r['person']]
write(I/'new_people.jsonl',np);write(I/'old_people_eligible.jsonl',op);write(I/'old_q1.jsonl',[{'person_id':r['person_id'],'person':r['person'],'prediction':next(iter(first[r['person']])),'source':'frozen_job700454'} for r in op])
freq=collections.Counter(traincountry.values());freq.update(r['birth_country'] for r in np)
js(I/'country_frequency.json',{'definition':'all original216 reference countries plus new400, unique original person names; not pretraining frequency','total_people':sum(freq.values()),'frequency':dict(sorted(freq.items()))});assert sum(freq.values())==616
js(I/'roster_summary.json',{'new_people':400,'original_people':216,'eligible_original_people':len(op),'original_people_excluded':216-len(op),'source_review_complete':True,'new_country_counts':summary['selected_countries'],'new_q1_requests':400,'max_q2_requests':472,'minimum_matched_pairs_per_group':300,'no_automatic_training':True,'source_review_limit':'No public database can certify the absence of every second citizenship; all declared screens and biography checks passed.'})
# Deliver the exclusion audit and primary evidence fields; full copyrighted biographies stay in local review cache.
shutil.copyfile(P/'CURATION_AUDIT.json',I/'curation_audit.json');shutil.copyfile(P/'CURATION_SUMMARY.json',I/'curation_summary.json')
for n in ['CURATION_POLICY_ZH.txt','CURATION_POLICY_FREEZE.json']:
 shutil.copyfile(P/n,PKG/n)
if (P/'CURATION_IMPLEMENTATION_ADDENDUM.json').exists():shutil.copyfile(P/'CURATION_IMPLEMENTATION_ADDENDUM.json',PKG/'CURATION_IMPLEMENTATION_ADDENDUM.json')
lines=['新增400位人物名单（已封存资料筛查；不是训练完成结果）','姓名 | 出生国 | 出生年 | Wikidata实体 | 英文传记版本']
for r in sorted(np,key=lambda r:(r['birth_country'],r['person'])):lines.append(f"{r['person']} | {r['birth_country']} | {r['birth_year']} | {r['qid']} | {r['biography_revision']}")
(P/'NEW400_ZH.txt').write_text('\n'.join(lines)+'\n')
js(P/'ROSTER_FREEZE.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'before_real_model_inference':True,'files':{str(f.relative_to(P)):sha(f) for f in sorted(I.iterdir()) if f.is_file()},'new_people':400,'eligible_old_people':len(op),'source_selection_did_not_use_new_model_predictions':True})
print('Roster built: new400 + clean old',len(op))
