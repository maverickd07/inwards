"""Validate and copy completed seed-0 evaluations into a fresh run root; never alter source."""
import argparse,json,shutil,hashlib
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--previous',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--expected-n',type=int,default=500);a=ap.parse_args()
 assert not a.out.exists(), 'Resume output must be fresh'
 sources=[]
 for condition in ['self_wrong','shuffled_wrong']:
  d=a.previous/(condition+'_s0');cfg=json.load(open(d/'run_config.json'));summary=json.load(open(d/'inward_summary.json'));diagnostic=json.load(open(d/'target_diagnostic.json'))
  assert cfg['condition']==condition and cfg['train_seed']==0 and cfg['seed']==0
  assert summary['condition']==condition and diagnostic['allocation_scorer']=='scoring_v2_floor'
  assert diagnostic['n_kept']==(8 if a.expected_n==2 else 228)
  for subset in ['A_newPerson','B_newCategory','C_bothNew']:
   rows=[json.loads(s) for s in (d/f'eval_{subset}.jsonl').read_text().splitlines() if s.strip()]
   assert len(rows)==a.expected_n and len({(r['category'],r['person']) for r in rows})==a.expected_n
   assert all(r['base_q1_pred'] is not None and r['trained_q1_pred'] is not None and r['base_q2_pred'] is not None and r['trained_q2_pred'] is not None for r in rows)
   assert summary['evals'][subset]['n']==a.expected_n
  for name in ['initialization.json','target_assignment.jsonl','train_examples.jsonl','evaluation_manifest.json']:assert (d/name).is_file()
  sources.append(d)
 a.out.mkdir()
 manifest={}
 for d in sources:
  shutil.copytree(d,a.out/d.name,ignore=shutil.ignore_patterns('adapter','__pycache__'))
  hashes={str(f.relative_to(d)):hashlib.sha256(f.read_bytes()).hexdigest() for f in d.rglob('*') if f.is_file() and 'adapter' not in f.relative_to(d).parts}
  manifest[d.name]={'source':str(d.resolve()),'copied_sha256':hashes,'adapter_note':'Original adapter retained in prior run root; not copied.'}
 (a.out/'resume_sources.json').write_text(json.dumps(manifest,indent=2))
 print('Validated/copied two completed conditions. Prior files unchanged.',flush=True)
if __name__=='__main__':main()
