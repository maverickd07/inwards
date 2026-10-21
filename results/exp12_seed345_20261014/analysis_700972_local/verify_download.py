"""Read-only verification of downloaded job 700972; original scoring only."""
import json
import math
import statistics
import sys
from pathlib import Path
from scipy.stats import t

OUT = Path(__file__).resolve().parent
PACKAGE = OUT.parent
sys.path.insert(0, str(PACKAGE))
from reuse import sha, read, key, verify_source, METRICS, CONDITIONS, js
from summarize import checked_predictions

RUN = PACKAGE / 'received/extracted_700972/runs_700972'
OLD = PACKAGE.parent / 'exp12_sources_20261014_v2/received_700454/extracted/runs_700454'


def interval(values):
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = float(t.ppf(.975, len(values) - 1)) * sd / math.sqrt(len(values))
    return {'values': values, 'mean': mean, 'sample_sd': sd, 'seed_t95': [mean-half, mean+half]}


def equal(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9), (a, b)
    else:
        assert a == b, (a, b)


def main():
    complete = json.loads((RUN / 'complete.json').read_text())
    assert complete['training_runs'] == 6 and complete['added_seeds'] == [3, 4, 5]
    assert not any(complete[k] for k in ['smoke_test', 'baseline_regenerated', 'targets_regenerated', 'new_scorer_applied'])
    snapshot = RUN / 'reused_700454'
    original_hashes = verify_source(OLD)
    assert verify_source(snapshot) == original_hashes
    receipt = json.loads((RUN / 'reuse_receipt.json').read_text())
    assert receipt['source_job'] == '700454' and receipt['copied_sha256'] == original_hashes
    production_hashes = json.loads((RUN / 'source_hashes.json').read_text())
    matched_code = []
    for line in (PACKAGE / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        assert production_hashes[name] == digest == sha(PACKAGE / name), name
        matched_code.append(name)
    stages = json.loads((RUN / 'stages.json').read_text())
    assert stages == [{'condition': c, 'seed': s, 'path': f'{c}_s{s}'} for s in [3,4,5] for c in CONDITIONS]
    commands = json.loads((RUN / 'commands.json').read_text())
    assert len(commands) == 6
    assert all(Path(c[2]).name == 'train_seed.py' and '--smoke-test' not in c for c in commands)
    rows = read(snapshot / 'plan/items_D_reservedC.jsonl')
    train = read(snapshot / 'plan/train.jsonl')
    assert len(rows) == 500 and len(train) == 864
    baseline = checked_predictions(snapshot / 'baseline', rows)
    base = {m: 100 * sum(v) / len(v) for m,v in baseline.items()}
    labels, per_seed, initials = {}, [], {}
    for seed in range(6):
        for condition in CONDITIONS:
            folder = (snapshot if seed < 3 else RUN) / f'{condition}_s{seed}'
            cfg = json.loads((folder / 'run_config.json').read_text())
            ref = json.loads((OLD / f'{condition}_s0/run_config.json').read_text())
            for k in ref:
                if k not in ['train_seed', 'training_seconds', 'evaluation_seconds']:
                    assert cfg[k] == ref[k], (folder, k)
            assert cfg['train_seed'] == seed
            assert cfg['optimizer_steps'] == 216 and cfg['sample_exposures'] == 1728
            marker = json.loads((folder / 'complete.json').read_text())
            assert marker['steps'] == 216 and len(marker['adapter_sha256']) == 64
            init = json.loads((folder / 'initialization.json').read_text())
            assert init['train_seed'] == seed and init['continued_from'] is None
            initials[condition, seed] = init['lora_initial_sha256']
            runtime = json.loads((folder / 'runtime.json').read_text())
            prior_runtime = json.loads((OLD / 'context_s0/runtime.json').read_text())
            for k in ['versions', 'cuda', 'gpu', 'python', 'smoke_test']:
                assert runtime[k] == prior_runtime[k], (folder, k)
            if seed >= 3:
                assert [f.name for f in folder.glob('eval_*.jsonl')] == ['eval_D_reservedC.jsonl']
                examples = read(folder / 'train_examples.jsonl')
                assert examples == read(OLD / f'{condition}_s0/train_examples.jsonl')
                loss = read(folder / 'loss.jsonl')
                assert len(loss) == 216
                assert all(r['step'] == i+1 and math.isfinite(r['loss']) for i,r in enumerate(loss))
            labels[condition, seed] = checked_predictions(folder, rows)
            per_seed.append({'seed': seed, 'condition': condition, **{
                m: 100 * sum(v) / len(v) for m,v in labels[condition,seed].items()}})
        assert initials['context',seed] == initials['direct',seed]
    assert len({initials['context', s] for s in range(6)}) == 6
    groups = {}
    for name, seeds in [('original_012', range(3)), ('added_345', range(3,6)), ('all_012345', range(6))]:
        groups[name] = {}
        for metric in METRICS:
            acc = {c: [100 * sum(labels[c,s][metric]) / 500 for s in seeds] for c in CONDITIONS}
            groups[name][metric] = {'accuracy_pct': {c: interval(v) for c,v in acc.items()},
                'gain_over_base_pp': {c: interval([x-base[metric] for x in v]) for c,v in acc.items()},
                'context_minus_direct_pp': interval([x-y for x,y in zip(acc['context'],acc['direct'])])}
    saved = json.loads((RUN / 'analysis_original_scorer/report.json').read_text())
    equal(base, saved['baseline_accuracy_pct'])
    equal(per_seed, saved['per_seed'])
    equal(groups, saved['groups'])
    log = (RUN.parent / 'exp12-700972.out').read_text()
    assert 'SEED EXTENSION COMPLETE.' in log and 'Traceback (most recent call last)' not in log
    verification = {'job_id': '700972', 'complete': complete, 'source_files_matched': len(original_hashes),
        'package_files_matched': matched_code, 'all_six_budgets_and_losses_checked': True,
        'training_examples_identical_to_original_condition': True, 'runtime_matches_700454': True,
        'paired_initializations_and_six_distinct_seeds': True, 'D_count':500,
        'prediction_files_verified': 13, 'primary_and_hop_labels_recomputed':19500,
        'legacy_composed_labels_recomputed':6500, 'all_report_values_recomputed':True,
        'intervals_independently_checked_with_scipy':True, 'new_scorer_applied':False,
        'limitations':['Adapter weights excluded from download; their recorded hashes were not rechecked against binaries locally.',
                      'No final sacct export in archive; computation completion marker and successful final log verified.',
                      'Saved token likelihoods were not rerun on the 14B model.']}
    js(OUT / 'verification.json', verification)
    js(OUT / 'original_scorer_recomputed.json', {'baseline_accuracy_pct':base,'per_seed':per_seed,'groups':groups})
    print(json.dumps({'completion_seconds':complete['seconds'],'baseline':base,'per_seed':per_seed,
                      'composed_groups':{k:v['composed_ok'] for k,v in groups.items()}},indent=2))


if __name__ == '__main__':
    main()
