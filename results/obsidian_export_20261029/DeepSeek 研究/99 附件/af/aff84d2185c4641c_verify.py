"""Read-only validation of received exp19 artifacts; no model generation or rescoring changes."""
import collections,hashlib,json,sys,tarfile
from pathlib import Path
P=Path(__file__).resolve().parents[1];D=P/'received/download_IWlTO9';R=D/'runs_712816';OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
from scoring import score_row
from analyze import analyze

def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(ok,message):
 if not ok:raise AssertionError(message)
checks={}
archive=D/'runs_712816_review.tgz';check(digest(archive)==(D/'runs_712816_review.tgz.sha256').read_text().split()[0],'archive hash');checks['download_archive_sha256']=digest(archive)
with tarfile.open(archive) as tf:
 for m in tf.getmembers():
  check(not m.issym() and not m.islnk(),'Unexpected archive link')
  check(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts,'Unsafe archive path')
  if m.isfile():check(hashlib.sha256(tf.extractfile(m).read()).hexdigest()==digest(D/m.name),'Extracted file changed: '+m.name)
checks['all_extracted_files_match_archive']=True
for line in (D/'frozen_protocol/SHA256SUMS').read_text().splitlines():
 d,n=line.split('  ',1);check(digest(P/n)==d,'Frozen local package changed: '+n)
checks['all_51_local_frozen_package_files_match']=True
cfg=json.loads((R/'config.json').read_text());check(cfg['preregistration_sha256']==digest(P/'PREREGISTRATION.txt'),'preregistration hash');check(cfg['input_roster_sha256']==digest(P/'inputs/items.jsonl'),'roster hash')
check((D/'frozen_protocol/PREREGISTRATION.txt').read_bytes()==(P/'PREREGISTRATION.txt').read_bytes(),'received prereg changed')
check(cfg['do_sample'] is False and cfg['enable_thinking'] is False and cfg['smoke_test'] is False and cfg['max_new_tokens']==96,'inference settings')
check(cfg['model_config']['hidden_size']==5120 and cfg['model_config']['num_hidden_layers']==40,'Qwen3-14B architecture')
checks['configuration_and_preregistration_match']=True
raw=read(R/'raw.jsonl');scored=read(R/'analysis/per_item.jsonl');prompts=read(R/'prompts.jsonl');roster=read(P/'inputs/items.jsonl');conditions=['C0','C1','C1-ctrl','C2','C3']
expected={(r['id'],c) for r in roster for c in conditions}
for name,rows in [('raw',raw),('scored',scored),('prompts',prompts)]:
 check(len(rows)==2010 and {(r['id'],r['condition']) for r in rows}==expected,'coverage '+name)
check(all({k:r[k] for k in roster[r['id']]}==roster[r['id']] for r in raw),'all input fields match')
check(all(score_row(a)==b for a,b in zip(raw,scored)),'scored rows not reproduced')
pp={(r['id'],r['condition']):r for r in prompts}
check(all(pp[r['id'],'C1']['prompt_tokens']==pp[r['id'],'C1-ctrl']['prompt_tokens'] for r in roster),'length matching')
check(all(r['generated_tokens']==len(r['generated_ids']) and not r['truncated'] and not r['smoke_test'] for r in raw),'token/truncation/smoke flags')
check(json.loads((R/'complete.json').read_text())=={'smoke_test':False,'items':402,'generations':2010,'conditions':conditions,'analysis_complete':True},'completion marker')
checks.update({'complete_unique_generations':2010,'complete_unique_questions':402,'all_scored_rows_reproduce_exactly':True,'all_402_prompt_length_pairs_match':True,'truncated_generations':0,'all_generated_token_counts_match':True})
# Independent local rerun of frozen scorer and 20,000-bootstrap analysis.
recomputed=analyze(R/'raw.jsonl',OUT/'recomputed_analysis')
check(recomputed==json.loads((R/'analysis/summary.json').read_text()),'summary differs from independent local recomputation')
checks['full_summary_and_20000_bootstrap_reproduce_exactly']=True
lookup={(r['id'],r['condition']):r for r in scored};groups={}
for label,predicate in [('D500_known313',lambda r:r['D500_known313']),('fixed303',lambda r:r['fixed303']),('posthoc_fixed303_exp1_known214',lambda r:r['fixed303'] and r['D500_known313']),('posthoc_fixed303_not_exp1_known89',lambda r:r['fixed303'] and not r['D500_known313'])]:
 ids=[r['id'] for r in roster if predicate(r)];fails=[i for i in ids if not lookup[i,'C0']['score']['correct']]
 cross=collections.Counter()
 for i in fails:
  x=lookup[i,'C2'];cs=x['country_correct'];cross[('correct_country' if cs is True else 'wrong_country' if cs is False else 'unknown_country')+('_correct_final' if x['score']['correct'] else '_wrong_final')]+=1
 groups[label]={'n':len(ids),'C0_failure_n':len(fails),'accuracy':{c:sum(lookup[i,c]['score']['correct'] for i in ids)/len(ids) for c in conditions},'C2_on_C0_failures_country_by_final':dict(cross)}
checks['descriptive_posthoc_strata_and_C2_cross_tabs']=groups
checks['summed_generation_seconds']=sum(r['seconds'] for r in raw)
checks['review_unresolved_n']=sum(r['score']['status']=='REVIEW' for r in scored)
checks['review_unique_raw_answers']=len({r['final_answer'] for r in scored if r['score']['status']=='REVIEW'})
checks['slurm_final_state']='NOT independently verified: downloaded archive has no sacct output or Slurm stdout'
(OUT/'VERIFICATION.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
with (OUT/'unresolved_review.jsonl').open('x') as f:
 for r in scored:
  if r['score']['status']=='REVIEW':f.write(json.dumps({k:r[k] for k in ['id','condition','person','category','raw','final_answer','score','e']},ensure_ascii=False)+'\n')
print(json.dumps(checks,ensure_ascii=False,indent=2))
