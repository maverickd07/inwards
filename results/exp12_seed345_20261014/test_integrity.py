"""Bounded, local checks of reuse, guardrails and review completeness."""
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import patch

import reuse
from pack_review import pack
from summarize import checked_predictions, moments

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'exp12_sources_20261014_v2'
SOURCE = OLD / 'received_700454/extracted/runs_700454'
SMOKE = HERE / 'local_smoke'


def rejects(fn, exception=ValueError):
    try:
        fn()
    except exception:
        return
    raise AssertionError('Expected rejection')


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


def main():
    source_hashes = reuse.verify_source(SOURCE)
    reuse.verify_source(OLD / 'local_smoke_noscipy', smoke=True)
    rejects(lambda: reuse.verify_source(SOURCE, smoke=True))
    rejects(lambda: reuse.verify_source(OLD / 'local_smoke_noscipy'))
    for file in ['submit.sh', 'run_exp12.sh']:
        subprocess.run(['bash', '-n', str(HERE / file)], check=True)
    with tempfile.TemporaryDirectory(prefix='seed345-tamper-') as tmp:
        fixture = Path(tmp) / 'source'
        for name in source_hashes:
            dest = fixture / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE / name, dest)
        target = fixture / 'paired_targets/context_targets.jsonl'
        target.write_text(target.read_text() + '\n')
        rejects(lambda: reuse.verify_source(fixture))
    expected = json.loads((HERE / 'source_700454_manifest.json').read_text())['expected_versions']
    old_python = json.loads((SOURCE / 'context_s0/runtime.json').read_text())['python']
    with patch('reuse.importlib.metadata.version', side_effect=lambda p: expected[p]), patch('reuse.sys.version', old_python):
        assert reuse.verify_environment(SOURCE) == expected
    with patch('reuse.importlib.metadata.version', return_value='wrong'):
        rejects(lambda: reuse.verify_environment(SOURCE))
    completion = json.loads((SMOKE / 'complete.json').read_text())
    assert completion['added_seeds'] == [3, 4, 5] and completion['training_runs'] == 6
    assert not completion['baseline_regenerated'] and not completion['targets_regenerated']
    commands = json.loads((SMOKE / 'commands.json').read_text())
    assert len(commands) == 6 and all('train_seed.py' in c[2] for c in commands)
    for seed in [3, 4, 5]:
        for condition in reuse.CONDITIONS:
            folder = SMOKE / f'{condition}_s{seed}'
            assert [p.name for p in folder.glob('eval_*.jsonl')] == ['eval_D_reservedC.jsonl']
    receipt = json.loads((SMOKE / 'reuse_receipt.json').read_text())
    for name, digest in receipt['copied_sha256'].items():
        assert reuse.sha(SMOKE / 'reused_700454' / name) == digest
        assert reuse.sha(OLD / 'local_smoke_noscipy' / name) == digest
    # Existing output fails before model loading or a write.
    result = subprocess.run([sys.executable, str(HERE / 'run_pipeline.py'),
        '--source', str(SOURCE), '--model', 'unused-model', '--out', str(SMOKE)], capture_output=True, text=True)
    assert result.returncode != 0 and 'No overwrite' in result.stderr
    assert not list(SMOKE.glob('baseline')) and not list(SMOKE.glob('paired_targets'))
    # Verify all 13 production model outputs already available locally, where present.
    rows = reuse.read(SOURCE / 'plan/items_D_reservedC.jsonl')
    for folder in ['baseline'] + [f'{c}_s{s}' for s in range(3) for c in reuse.CONDITIONS]:
        checked_predictions(SOURCE / folder, rows)
    assert moments([2, 1.8, 4.8])['mean'] == 2.8666666666666667
    report = json.loads((SMOKE / 'analysis_original_scorer/report.json').read_text())
    assert len(report['per_seed']) == 12 and len(report['groups']) == 3
    for q in ['', '123\n', '123\n456\n']:
        scheduler_case(q)
    scheduler_case('', existing=True)
    scheduler_case('', preflight_failure=True)
    # Test export on a temporary copy; all twelve fits plus base, no adapters.
    with tempfile.TemporaryDirectory(prefix='seed345-export-') as tmp:
        copied = Path(tmp) / 'runs_smoke'
        shutil.copytree(SMOKE, copied)
        archive = pack(copied)
        with tarfile.open(archive) as t:
            names = t.getnames()
            assert sum(n.endswith('/eval_D_reservedC.jsonl') for n in names) == 13
            assert not any('/adapter/' in n or n.endswith('.safetensors') for n in names)
        rejects(lambda: pack(copied), FileExistsError)
    assert reuse.verify_source(SOURCE) == source_hashes
    results = {'exact_production_source_matches': True, 'source_tamper_rejected': True,
               'smoke_production_isolation': True, 'environment_match_and_drift_guard': True,
               'six_fit_smoke_passed': True, 'source_and_reused_predictions_unchanged': True,
               'no_regeneration_commands': True, 'no_overwrite_rejected': True,
               'D_only_all_item_predictions': True, 'thirteen_prediction_files_in_review': True,
               'scheduler_cases': ['empty', 'one_afterany', 'two_rejected', 'receipt_rejected', 'preflight_failure'],
               'scheduler_test_scope': 'Mocked Slurm, conda, flock and preflight commands; real scheduler/lock behavior not exercised.',
               'bash_syntax': True, 'production_GPU_tested_locally': False}
    reuse.js(HERE / 'integrity_checks.json', results)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
