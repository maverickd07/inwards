"""Local verification of immutable plan, score behavior, guards and13-process smoke."""
import json,os,shlex,shutil,subprocess,sys,tarfile,tempfile,statistics,ast
from pathlib import Path
from common import HERE,NAMES,read,sha,verify_freeze,digest
from scoring import scores
from metrics.answer import AnswerMetric
from analyze import moments,compare
from check_source import check
from pack_review import pack

def rejects(fn,exception=ValueError):
 try:fn()
 except exception:return
 raise AssertionError('Expected rejection')

def test_plan():
 verify_freeze();items=read(HERE/'data/items.jsonl');queries=read(HERE/'data/queries.jsonl');manifest=json.loads((HERE/'data/manifest.json').read_text())
 assert len(items)==2385 and len(queries)==5176 and len(set(r['id'] for r in items))==2385
 assert {n:sum(r['n_hops']==n for r in items) for n in [2,3,4]}=={2:1229,3:751,4:405}
 by={q['query_id']:q for q in queries};assert len(by)==len(queries)
 for r in items:
  for q in [r]+r['hops']:
   assert by[q['query_id']]['question']==q['question'] and digest(q['question'])==q['query_id']
 for q in queries:assert set(q)=={'query_id','question'}
 for r in items:
  for s in r['hops']:assert not __import__('re').search(r'#\d+',s['question'])
 old=read(HERE.parent/'exp12_matched785_20261015/frozen_inputs/plan/train.jsonl')
 from prepare_data import norm
 import re
 pattern=re.compile(r'(?<!\w)(?:'+'|'.join(map(re.escape,sorted({norm(r['person']) for r in old})))+r')(?!\w)')
 for r in items:assert not pattern.search(norm(' '.join([r['question']]+[s['question'] for s in r['hops']])))
 # Released composed aliases are preserved exactly, with no score-specific additions.
 raw={r['id']:r for r in read(HERE/'raw_data/musique_ans_v1.0_dev.jsonl')}
 for r in items:assert set(r['answers'])==set([raw[r['id']]['answer']]+raw[r['id']]['answer_aliases'])
 fixture=[('The Paris.',['Paris']),('Answer: Paris',['Paris']),('New York',['New York City','NYC']),('',['Paris']),('Kyiv',['Kiev']),('U.S.',['US','United States'])]
 official=AnswerMetric();all_em=[];all_f1=[]
 for pred,refs in fixture:
  official(pred,refs);s=scores(pred,refs);all_em.append(s['em']);all_f1.append(s['f1'])
 assert all(__import__('math').isclose(x,y,abs_tol=1e-12) for x,y in zip(official.get_metric(),(statistics.mean(all_em),statistics.mean(all_f1))))
 assert scores('Answer: Paris',['Paris'])['em']==0 and scores('Kyiv',['Kiev'])['em']==0
 assert scores('Paris',['Paris'])['em']==1
 cfg=json.loads((HERE/'inference_config.json').read_text());tree=ast.parse((HERE/'infer.py').read_text())
 system=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='SYSTEM' for x in node.targets))
 assert system==cfg['system'] and cfg['batch']==8 and cfg['max_new_tokens']==32
 assert compare({},[],'em')['n']==0
 assert moments([1]*6)['t95_pp']==[1,1]
 for file in ['submit.sh','run_exp13.sh']:subprocess.run(['bash','-n',str(HERE/file)],check=True)
 for q in ['', '123\n', '123\n456\n']:scheduler_case(q)
 scheduler_case('',existing=True);scheduler_case('',preflight_failure=True)
 print('Plan, official scoring, no-label-in-prompt schema, overlap and scheduler checks passed.')

def test_smoke():
 run=HERE/'local_smoke';complete=json.loads((run/'complete.json').read_text())
 assert complete['models_evaluated']==13 and complete['training_runs']==0 and complete['smoke_test']
 items=read(run/'inputs/items.jsonl');assert [r['n_hops'] for r in items]==[2,3,4]
 source=HERE.parent/'exp12_matched785_20261015/local_smoke';hashes=check(source,smoke=True,weights=True)
 assert json.loads((run/'source_receipt.json').read_text())['sha256']==hashes
 commands=json.loads((run/'commands.json').read_text());assert len(commands)==13 and all('infer.py' in cmd[2] for cmd in commands)
 for name in NAMES:
  folder=run/name;meta=json.loads((folder/'complete.json').read_text());rows=read(folder/'predictions.jsonl')
  assert len(rows)==complete['queries_per_model'] and meta['predictions_sha256']==sha(folder/'predictions.jsonl')
  assert [r['query_id'] for r in rows]==[r['query_id'] for r in read(run/'inputs/queries.jsonl')]
  assert (folder/'scoring_complete.json').is_file() and not (folder/'adapter').exists()
  assert ('--adapter' in commands[NAMES.index(name)])==(name!='baseline')
  if name!='baseline':assert meta['adapter_sha256']==sha(source/name/'adapter/adapter_model.safetensors')
  else:assert len(read(folder/'timing_predictions.jsonl'))==2
 result=subprocess.run([sys.executable,str(HERE/'run_pipeline.py'),'--model','unused','--source','unused','--out',str(run)],capture_output=True,text=True)
 assert result.returncode!=0 and 'No overwrite' in result.stderr
 rejects(lambda:check(source,smoke=False,weights=False))
 with tempfile.TemporaryDirectory(prefix='exp13-tamper-') as tmp:
  copied=Path(tmp)/'source';shutil.copytree(source,copied)
  p=copied/'context_s0/adapter/adapter_model.safetensors'
  with p.open('ab') as f:f.write(b'TAMPER')
  rejects(lambda:check(copied,smoke=True,weights=True))
 with tempfile.TemporaryDirectory(prefix='exp13-pack-') as tmp:
  copied=Path(tmp)/'runs_smoke';shutil.copytree(run,copied);archive=pack(copied)
  with tarfile.open(archive) as tar:
   names=tar.getnames();assert not any(n.endswith('.safetensors') for n in names)
   assert sum(n.endswith('/predictions.jsonl') for n in names)==13
   for line in tar.extractfile('runs_smoke/REVIEW_FILES_SHA256.txt').read().decode().splitlines():
    digest,name=line.split('  ',1)
    assert __import__('hashlib').sha256(tar.extractfile('runs_smoke/'+name).read()).hexdigest()==digest
  rejects(lambda:pack(copied),FileExistsError)
 print('13-model tiny CPU smoke, actual adapter loading, scoring, preservation, tamper rejection and export passed.')

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
 test_plan()
 if '--smoke' in sys.argv:test_smoke()
