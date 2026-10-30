"""Meaningful local checks: selection/provenance, actual budgets, score boundaries and export."""
import json,math,os,shlex,shutil,subprocess,sys,tarfile,tempfile
from pathlib import Path
from reuse import HERE,sha,read,key,verify_source,js
from protocol import verify_freeze
from training_label_scorer import score,codes
from analyze import moments,comparison
from pack_review import pack

def rejects(fn,exception=ValueError):
 try:fn()
 except exception:return
 raise AssertionError('Expected rejection')

def checks():
 freeze=verify_freeze();inp=HERE/'frozen_inputs';rows=read(inp/'selected_target_classes.jsonl')
 assert len(rows)==785 and sum(not r['targets_exactly_identical'] for r in rows)==404
 assert [sum(r['old'][c] for r in rows) for c in ['context','direct']]==[359,359]
 for pred,refs,cat,ok in [
  ('Washington, D.C.',['Washington D.C.'],'birthplace_capital',True),
  ('Kiev',['Kyiv'],'birthplace_capital',True),
  ('Ulaanbaatar',['Ulan Bator'],'birthplace_capital',True),
  ('Bogota',['Bogotá'],'birthplace_capital',True),
  ('Paris and London',['Paris'],'birthplace_capital',False),
  ('Delhi',['New Delhi'],'birthplace_capital',False),
  ('La Paz',['Sucre'],'birthplace_capital',False),
  ('+1 809',['+1809'],'birthplace_callingcode',True),
  ('+1-829 / +1 849',['+1 809'],'birthplace_callingcode',True),
  ('+1 876',['+1 658'],'birthplace_callingcode',True),
  ('+1 939',['+1 787'],'birthplace_callingcode',True),
  ('+44, +33',['+44'],'birthplace_callingcode',False),
  ('+1',['+1 809'],'birthplace_callingcode',False),
  ('+44 and +33',['+44','+33'],'birthplace_callingcode',True),
  ('+4.4',['+44'],'birthplace_callingcode',False),
  ('+4412345678',['+44'],'birthplace_callingcode',False)]:
  assert score(pred,refs,cat)['correct']==ok,(pred,refs)
 assert score('+1',['+1 809'],'birthplace_callingcode')['needs_review']
 allrows=read(inp/'all864_target_classes.jsonl')
 capitals=[v for r in allrows if r['category']=='birthplace_capital' for v in r['gold']]
 for r in allrows:
  for gold in r['gold']:assert score(gold,r['gold'],r['category'],capitals)['correct']
 # No condition/person/seed argument; identical texts must produce identical labels.
 for r in rows:
  if r['targets_exactly_identical']:assert score(r['context_target'],r['gold'],r['category'],capitals)==score(r['direct_target'],r['gold'],r['category'],capitals)
 # Frozen evaluation rules are byte-identical to original v3; train extension cannot mutate them.
 prior=HERE.parent/'exp12_rescore_v3_20261015'
 for name in ['scorer_v3.py','aliases.json','reference_catalog.json','scoring_v2_v1.py']:assert sha(HERE/name)==sha(prior/name)
 refs={name:read(inp/'reference_v3'/f'{name}.jsonl') for name in ['baseline']+[f'{c}_s{s}' for s in range(6) for c in ['context','direct']]}
 mask=set(map(tuple,json.loads((inp/'primary_D303_keys.json').read_text())['identities']))
 ids=[i for i,r in enumerate(refs['baseline']) if key(r) in mask]
 assert len(ids)==303
 baseline=comparison(refs,ids,'composed','all_wrong');assert abs(baseline['mean_pp']-3.1353135313531353)<1e-10
 assert abs(baseline['t95_pp'][0]-2.525504253)<1e-8
 for name in ['submit.sh','run_exp12.sh']:subprocess.run(['bash','-n',str(HERE/name)],check=True)
 for queue in ['', '123\n', '123\n456\n']:scheduler_case(queue)
 scheduler_case('',existing=True);scheduler_case('',preflight_failure=True)
 # Random allocation independently reconstructed from frozen original class labels.
 import random
 rng=random.Random(20261015);indices=[]
 for cat in sorted({r['category'] for r in allrows}):
  buckets={label:[i for i,r in enumerate(allrows) if r['category']==cat and r['old_class']==label] for label in ['both_correct','both_wrong','context_only_correct','direct_only_correct']}
  indices+=buckets['both_correct']+buckets['both_wrong'];n=min(len(buckets['context_only_correct']),len(buckets['direct_only_correct']))
  for label in ['context_only_correct','direct_only_correct']:
   group=buckets[label];indices+=group if len(group)==n else rng.sample(group,n)
 assert sorted(indices)==freeze['allocation']['original_indices']
 print('Integrity checks passed: allocation, labels, v3 regression, CI, scheduler guards.',flush=True)

def smoke_checks():
 run=HERE/'local_smoke';complete=json.loads((run/'complete.json').read_text())
 assert complete['training_runs']==12 and complete['steps_per_fit']==198 and complete['smoke_test']
 assert not complete['baseline_regenerated'] and not complete['targets_regenerated']
 configs=[]
 for s in range(6):
  for c in ['context','direct']:
   folder=run/f'{c}_s{s}';loss=read(folder/'loss.jsonl')
   assert len(loss)==198 and [x['epoch'] for x in loss]==[1]*99+[2]*99
   configs.append(json.loads((folder/'run_config.json').read_text()))
 assert len(configs)==12 and sum(c['optimizer_steps'] for c in configs)==2376
 report=json.loads((run/'analysis_v3/report.json').read_text());assert len(set(report['paired_initializations'].values()))==6
 assert json.loads((run/'training_label_audit/complete.json').read_text())['different']==404
 result=subprocess.run([sys.executable,str(HERE/'run_pipeline.py'),'--source','unused','--model','unused','--out',str(run)],capture_output=True,text=True)
 assert result.returncode!=0 and 'No overwrite' in result.stderr
 with tempfile.TemporaryDirectory(prefix='matched-export-') as tmp:
  copied=Path(tmp)/'runs_smoke';shutil.copytree(run,copied);archive=pack(copied)
  with tarfile.open(archive) as tar:
   names=tar.getnames();assert not any('/adapter/' in n or n.endswith('.safetensors') for n in names)
   assert sum(n.endswith('/eval_D_reservedC.jsonl') and '/source_700454/' not in n for n in names)==13
   assert any(n.endswith('/different404_labels.jsonl') for n in names)
  rejects(lambda:pack(copied),FileExistsError)
 print('12 real tiny-model fits passed: 785 items, 198 updates each; paired init, analysis and export verified.',flush=True)

def scheduler_case(queue, existing=False, preflight_failure=False):
    with tempfile.TemporaryDirectory(prefix='seed345-submit-') as tmp:
        root = Path(tmp)
        exp = root / 'results' / HERE.name
        exp.mkdir(parents=True)
        (exp / 'quota_login.txt').write_text('USED=60GB LIMIT=100GB\n')
        if existing:
            (exp / 'submitted_job.txt').write_text('previous\n')
        binary = root / 'bin'
        binary.mkdir()
        common = '''#!/bin/bash
case "${0##*/}" in
conda) if [ "${1:-}" = shell.bash ]; then echo ':'; fi ;;
flock) exit 0 ;;
squeue) if [ "${1:-}" = -h ]; then printf '%s' "$TEST_QUEUE"; fi ;;
python) if [ "$TEST_PREFLIGHT_FAILURE" = 1 ]; then exit 7; fi ;;
sbatch) printf '%s\\n' "$*" > "$TEST_SBATCH_LOG"; echo 987654 ;;
esac
'''
        for name in ['conda', 'flock', 'squeue', 'python', 'sbatch']:
            script = binary / name
            script.write_text(common)
            script.chmod(0o755)
        script = root / 'submit_test.sh'
        script.write_text((HERE / 'submit.sh').read_text().replace('cd "$HOME/inward"', 'cd ' + shlex.quote(str(root))))
        env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'],
                   TEST_QUEUE=queue, TEST_PREFLIGHT_FAILURE=str(int(preflight_failure)),
                   TEST_SBATCH_LOG=str(root / 'sbatch.args'))
        result = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
        submitted = (root / 'sbatch.args').exists()
        should_submit = not existing and not preflight_failure and len(queue.split()) < 2
        assert submitted == should_submit, (result.stdout, result.stderr)
        assert (result.returncode == 0) == should_submit
        if submitted:
            args = (root / 'sbatch.args').read_text()
            assert ('--dependency=afterany:123' in args) == (queue.strip() == '123')
            assert (exp / 'submitted_job.txt').read_text().strip() == '987654'

if __name__=='__main__':
 checks()
 if '--smoke' in sys.argv:smoke_checks()
