"""Recompute the received text results without pretending adapters were downloaded."""
import collections
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
PACKAGE = OUT.parent
ROOT = PACKAGE / 'received_700454/extracted/runs_700454'
sys.path.insert(0, str(PACKAGE))
import analyze as stats
from scoring_v2_v1 import is_correct, old_is_correct


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def same(x, y, location='root'):
    if isinstance(x, dict):
        assert set(x) == set(y), location
        for k in x:
            same(x[k], y[k], location + '/' + k)
    elif isinstance(x, list):
        assert len(x) == len(y), location
        for i, (a, b) in enumerate(zip(x, y)):
            same(a, b, f'{location}/{i}')
    elif isinstance(x, float):
        assert math.isclose(x, y, abs_tol=1e-10, rel_tol=1e-10), (location, x, y)
    else:
        assert x == y, (location, x, y)


def main():
    complete = json.loads((ROOT / 'complete.json').read_text())
    assert complete['training_runs'] == 6 and complete['smoke_test'] is False
    assert (ROOT.parent / 'SHA256SUMS').read_bytes() == (PACKAGE / 'SHA256SUMS').read_bytes()
    sources = json.loads((ROOT / 'source_hashes.json').read_text())
    source_verified = []
    for line in (PACKAGE / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        assert sha(PACKAGE / name) == digest
        if name in sources:
            assert sources[name] == digest, name
            source_verified.append(name)
    for name in ['exp12.py', 'engine_exp11_v1.py', 'analyze.py', 'run_pipeline.py',
                 'scoring_v2_v1.py', 'exp1_gap_v2_v1.py']:
        assert name in source_verified
    report = json.loads((ROOT / 'analysis/report.json').read_text())
    manifest = json.loads((ROOT / 'plan/exp12_manifest.json').read_text())
    sealed = json.loads((PACKAGE.parent / 'exp12_sources_20261014/local_full_plan/exp12_manifest.json').read_text())
    assert manifest['reserved_full_keys'] == sealed['reserved_full_keys']
    items = {s: stats.read(ROOT / 'plan' / f'items_{s}.jsonl') for s in stats.SPLITS}
    train = stats.read(ROOT / 'plan/train.jsonl')
    assert len(train) == 864 and len({stats.key(r) for r in train}) == 864
    assert all(len(v) == len({stats.key(r) for r in v}) == 500 for v in items.values())
    assert [list(stats.key(r)) for r in items['D_reservedC']] == manifest['reserved_full_keys']
    assert not {r['person'] for r in train} & {r['person'] for r in items['D_reservedC']}
    assert not {r['family'] for r in train} & {r['family'] for r in items['D_reservedC']}
    prior_keys = {tuple(k) for k in json.loads((PACKAGE / 'historical_C_exclusion.json').read_text())['keys']}
    assert not prior_keys & {stats.key(r) for r in items['D_reservedC']}
    for s in stats.SPLITS[:3]:
        assert list(map(stats.key, items[s])) == list(map(stats.key, stats.read(PACKAGE / 'reference_exp8' / f'eval_{s}.jsonl')))
    predictions, configs, initial = {}, {}, {}
    checked_labels = 0
    for condition in ['baseline', *[f'{c}_s{k}' for c in ['context', 'direct'] for k in range(3)]]:
        assert json.loads((ROOT / condition / 'complete.json').read_text())['stage'] in ['baseline', 'train']
        for split in stats.SPLITS:
            rows = stats.read(ROOT / condition / f'eval_{split}.jsonl')
            assert list(map(stats.key, rows)) == list(map(stats.key, items[split]))
            for row, item in zip(rows, items[split]):
                for field, pred, gold in [('composed_ok', 'pred', 'Answer'),
                                         ('q1_ok', 'q1_pred', 'A1'), ('q2_ok', 'q2_pred', 'A2')]:
                    assert row[field] == is_correct(row[pred], item['e'][gold])
                    checked_labels += 1
            predictions[condition, split] = rows
        if condition == 'baseline':
            continue
        cfg = json.loads((ROOT / condition / 'run_config.json').read_text())
        assert cfg['n_unique'] == 864 and cfg['optimizer_steps'] == 216
        assert cfg['epochs'] == 2 and cfg['sample_exposures'] == 1728 and cfg['batch'] == 8
        source, seed = condition.split('_s')
        assert cfg['condition'] == source and cfg['train_seed'] == int(seed)
        assert not cfg['init_adapter_sha256'] and not cfg['smoke_test']
        assert cfg['targets_sha256'] == sha(ROOT / 'paired_targets' / f'{source}_targets.jsonl')
        actual = stats.read(ROOT / condition / 'train_examples.jsonl')
        targets = stats.read(ROOT / 'paired_targets' / f'{source}_targets.jsonl')
        assert list(map(stats.key, actual)) == list(map(stats.key, targets)) == list(map(stats.key, train))
        assert [r['target'] for r in actual] == [r['target'] for r in targets]
        losses = stats.read(ROOT / condition / 'loss.jsonl')
        assert [r['step'] for r in losses] == list(range(1, 217))
        assert all(math.isfinite(r['loss']) for r in losses)
        initial[source, seed] = json.loads((ROOT / condition / 'initialization.json').read_text())['lora_initial_sha256']
        configs[condition] = cfg
    for seed in ['0', '1', '2']:
        assert initial['context', seed] == initial['direct', seed]
    for group in report['groups']:
        split, field = group['split'], stats.FIELDS[group['metric']]
        b = [r[field] for r in predictions['baseline', split]]
        values = [[r[field] for r in predictions[f"{group['condition']}_s{k}", split]] for k in range(3)]
        same(stats.moments([100 * statistics.mean(v) for v in values]), group['accuracy_pct'])
        same(stats.moments([stats.paired(v, b)['gain_pp'] for v in values]), group['gain_pp'])
    primary, hops = [], []
    for record in report['per_run']:
        split, field = record['split'], stats.FIELDS[record['metric']]
        v = [r[field] for r in predictions[f"{record['condition']}_s{record['seed']}", split]]
        b = [r[field] for r in predictions['baseline', split]]
        recomputed = stats.paired(v, b)
        same(recomputed, {k: record[k] for k in recomputed})
        if split == 'D_reservedC':
            (primary if record['metric'] == 'composed' else hops).append((recomputed, record))
    for comp in report['context_minus_direct']:
        split, field = comp['split'], stats.FIELDS[comp['metric']]
        diffs, pairs = [], []
        for seed, saved in enumerate(comp['paired_seeds']):
            x = [r[field] for r in predictions[f'context_s{seed}', split]]
            y = [r[field] for r in predictions[f'direct_s{seed}', split]]
            rec = stats.paired(x, y)
            same(rec, {k: saved[k] for k in rec})
            pairs.append(rec)
            diffs.append([int(a) - int(b) for a, b in zip(x, y)])
            if split == 'D_reservedC':
                (primary if comp['metric'] == 'composed' else hops).append((rec, saved))
        same(stats.moments([r['gain_pp'] for r in pairs]), comp['gain_pp'])
        if split == 'D_reservedC':
            same(stats.cluster_interval(diffs, [r['person'] for r in items[split]]), comp['person_cluster'])
            same(stats.cluster_interval(diffs, [r['e']['A1'][0] for r in items[split]]), comp['bridge_country_cluster'])
    for family, n in [(primary, 9), (hops, 18)]:
        assert len(family) == n
        stats.holm([p[0] for p in family])
        for recalculated, saved in family:
            same(recalculated['mcnemar_p_holm'], saved['mcnemar_p_holm'])
    teacher = stats.read(ROOT / 'paired_targets/paired_diagnostics.jsonl')
    assert list(map(stats.key, teacher)) == list(map(stats.key, train))
    for row, item in zip(teacher, train):
        for c in ['context', 'direct']:
            assert row[f'{c}_ok'] == is_correct(row[f'{c}_target'], item['e']['Answer'])
    pair = stats.paired([r['context_ok'] for r in teacher], [r['direct_ok'] for r in teacher])
    same(pair, {k: report['teacher_advantage'][k] for k in pair})
    for field, saved in report['perplexity_diagnostics'].items():
        valid = [r[field] for r in teacher if r[field]['n_tokens']]
        nll = sum(v['sum_nll'] for v in valid) / sum(v['n_tokens'] for v in valid)
        same(nll, saved['token_weighted_nll'])
        same(math.exp(nll), saved['token_weighted_perplexity'])
    summary = {'completion_marker': complete, 'verified_sources': source_verified,
        'verified_prediction_labels': checked_labels, 'all_six_fits_and_budgets': True,
        'paired_initializations': True, 'target_hashes_and_training_examples': True,
        'sealed_D_identities_match': True, 'no_train_or_historical_C_identity_leakage': True,
        'all_reported_means_paired_tests_Holm_and_cluster_intervals_recomputed': True,
        'teacher_and_saved_NLL_aggregation_recomputed': True,
        'adapter_weights': 'Not downloaded; completion hash records exist but local binary rehash unavailable.',
        'raw_model_NLL': 'Not rerun on 14B; aggregates of saved token scores checked.',
        'Slurm_final_state': 'No sacct export in archive; computation completion marker and final successful log verified.'}
    stats.js(OUT / 'verification.json', summary)
    per_category = []
    for split in ['C_bothNew', 'D_reservedC']:
        for category in sorted({r['category'] for r in items[split]}):
            idx = [i for i, r in enumerate(items[split]) if r['category'] == category]
            b = [predictions['baseline', split][i]['composed_ok'] for i in idx]
            row = {'split': split, 'category': category, 'n': len(idx), 'baseline_pct': 100*statistics.mean(b)}
            for c in ['context', 'direct']:
                diffs = [stats.paired([predictions[f'{c}_s{k}', split][i]['composed_ok'] for i in idx], b)['gain_pp'] for k in range(3)]
                row[c + '_gain_pp_by_seed'] = diffs
                row[c + '_mean_gain_pp'] = statistics.mean(diffs)
            per_category.append(row)
    stats.js(OUT / 'category_effects.json', per_category)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('CATEGORY EFFECTS', json.dumps(per_category, ensure_ascii=False))


if __name__ == '__main__':
    main()
