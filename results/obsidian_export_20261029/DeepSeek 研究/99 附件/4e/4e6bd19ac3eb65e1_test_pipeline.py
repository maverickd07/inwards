"""Validate full outputs and fail-closed reuse/freeze/no-overwrite guards."""
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from rescore import HERE, MODELS, METRICS, verify, read, sha, js
from scorer_v3 import score


def main():
    run = HERE/'local_validation'
    paths = verify(Path('.'),True)
    report = json.loads((run/'report.json').read_text())
    plan = read(paths['plan'])
    queue_expected = []
    for name in MODELS:
        rows = read(run/(name+'.jsonl'))
        raw = read(paths[name])
        assert len(rows) == 500
        for i,(row,prior,item) in enumerate(zip(rows,raw,plan)):
            assert row['original_prediction'] == prior
            for metric,(field,answer,oldkey) in METRICS.items():
                new = score(prior[field],item['e'][answer],item['category'],metric)
                record = row['scores'][metric]
                assert all(record[k] == v for k,v in new.items())
                assert record['old'] == prior[oldkey]
                assert record['changed'] == (record['old'] != record['v3'])
                if record['changed'] or record['needs_review']:
                    queue_expected.append((name,item['category'],item['person'],metric))
        for metric in METRICS:
            summary = report['models'][name][metric]
            for label, count in [('old','old_correct'),('v3','new_correct'),('needs_review','needs_review')]:
                assert summary[count] == sum(r['scores'][metric][label] for r in rows)
    queue = read(run/'review_queue.jsonl')
    assert [(r['model'],r['category'],r['person'],r['metric']) for r in queue] == queue_expected
    assert len(queue) == report['review_queue_records']
    # Independently established previous old-score numbers must remain identical.
    old = json.loads((HERE.parent/'exp12_seed345_20261014/analysis_700972_local/original_scorer_recomputed.json').read_text())
    for metric,(_,_,oldkey) in METRICS.items():
        assert math.isclose(report['models']['baseline'][metric]['old_accuracy_pct'],old['baseline_accuracy_pct'][oldkey],abs_tol=1e-9)
        for group in old['groups']:
            a = report['groups'][group][metric]['old']['context_minus_direct_pp']
            b = old['groups'][group][oldkey]['context_minus_direct_pp']
            for x,y in zip(a['values']+a['seed_t95'],b['values']+b['seed_t95']):
                assert math.isclose(x,y,abs_tol=1e-9)
    no_overwrite = subprocess.run([sys.executable,str(HERE/'rescore.py'),'--local-inputs','--out',str(run)],capture_output=True,text=True)
    assert no_overwrite.returncode and 'No overwrite' in no_overwrite.stderr
    # Changes to rules or inputs must stop before creating an output directory.
    for tamper in ['aliases.json','input']:
        with tempfile.TemporaryDirectory(prefix='v3-integrity-') as temp:
            fixture=Path(temp)
            for p in HERE.iterdir():
                if p.is_file() and p.suffix in ('.py','.json','.txt'):
                    shutil.copyfile(p,fixture/p.name)
            if tamper == 'aliases.json':
                with (fixture/tamper).open('a') as f:f.write('\n')
            else:
                mapping=json.loads((fixture/'local_inputs.json').read_text())
                altered=fixture/'altered_input.jsonl'
                altered.write_text(Path(mapping['baseline']).read_text()+'\n')
                mapping['baseline']=str(altered)
                (fixture/'local_inputs.json').write_text(json.dumps(mapping))
            proc=subprocess.run([sys.executable,str(fixture/'rescore.py'),'--local-inputs','--out',str(fixture/'output')],capture_output=True,text=True)
            assert proc.returncode and ('changed' in proc.stderr or 'checksum mismatch' in proc.stderr)
            assert not (fixture/'output').exists()
    for script in ['submit.sh','run_rescore.sh']:
        subprocess.run(['bash','-n',str(HERE/script)],check=True)
    # Reuse the established mock scheduler harness in an isolated Python process.
    harness=HERE.parent/'exp12_seed345_20261014'
    program='''import sys, pathlib
sys.path.insert(0, sys.argv[1])
import test_integrity as harness
harness.HERE=pathlib.Path(sys.argv[2])
for queue in ['', '123\\n', '123\\n456\\n']:
    harness.scheduler_case(queue)
harness.scheduler_case('', existing=True)
harness.scheduler_case('', preflight_failure=True)
'''
    subprocess.run([sys.executable,'-c',program,str(harness),str(HERE)],check=True)
    # Guard against accidentally requesting a GPU or invoking training.
    batch=(HERE/'run_rescore.sh').read_text()
    assert '--gres' not in batch and 'train_seed.py' not in batch
    from pack_review import pack
    with tempfile.TemporaryDirectory(prefix='v3-export-') as temp:
        copied=Path(temp)/'runs_fixture'
        shutil.copytree(run,copied)
        pack(copied)
        with tarfile.open(Path(temp)/'runs_fixture_review.tgz') as archive:
            names=archive.getnames()
            assert all('adapter' not in n for n in names)
            assert all('runs_fixture/'+n+'.jsonl' in names for n in MODELS)
    js(HERE/'PIPELINE_VALIDATION.json',{'passed':True,'models':13,'items_per_model':500,
        'metric_labels_checked':19500,'old_report_reproduced':True,'raw_predictions_preserved':True,
        'review_queue_verified':True,'frozen_rules_tampering_rejected':True,
        'changed_inputs_rejected':True,'existing_output_rejected':True,
        'scheduler_guards':'5 mocked cases passed; actual Slurm/flock not exercised',
        'shell_syntax_passed':True,'cpu_only':True,'HPC_submitted':False})
    print('All pipeline validation checks passed.')


if __name__ == '__main__':
    main()
