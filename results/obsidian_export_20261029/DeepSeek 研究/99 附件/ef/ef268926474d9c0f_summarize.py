"""Original scorer only. No new scorer, semantic audit or mechanism inference."""
import argparse
import json
import math
import statistics
from pathlib import Path

from reuse import CONDITIONS, METRICS, key, read, sha, js
from scoring_v2_v1 import is_correct, old_is_correct

# Two-sided 95% Student t critical values, df 2 and 5. No SciPy dependency.
T95 = {3: 4.302652729749464, 6: 2.570581835636305}


def moments(values):
    mean, sd = statistics.mean(values), statistics.stdev(values)
    half = T95[len(values)] * sd / math.sqrt(len(values))
    return {'values': values, 'mean': mean, 'sample_sd': sd,
            'seed_t95': [mean - half, mean + half]}


def summarize(run, smoke=False):
    source = run / 'reused_700454'
    rows = read(source / 'plan/items_D_reservedC.jsonl')
    train = read(source / 'plan/train.jsonl')
    n = len(rows)
    if n != (2 if smoke else 500):
        raise ValueError('Incorrect D size')
    labels, per_seed = {}, []
    for seed in range(6):
        for c in CONDITIONS:
            folder = (source if seed < 3 else run) / f'{c}_s{seed}'
            cfg = json.loads((folder / 'run_config.json').read_text())
            complete = json.loads((folder / 'complete.json').read_text())
            ref = json.loads((source / f'{c}_s0/run_config.json').read_text())
            fixed = ['condition', 'round', 'epochs', 'batch', 'lr', 'rank', 'n_unique',
                     'sample_exposures', 'optimizer_steps', 'targets_sha256', 'init_adapter_sha256',
                     'optimizer_reset', 'scheduler_reset', 'smoke_test', 'scoring', 'plan_sha256',
                     'trained_target_tokens_including_eos', 'presented_prompt_tokens', 'max_sequence_tokens']
            if any(cfg[k] != ref[k] for k in fixed) or cfg['train_seed'] != seed:
                raise ValueError('Configuration drift: ' + str(folder))
            if complete['steps'] != cfg['optimizer_steps']:
                raise ValueError('Incomplete training: ' + str(folder))
            if seed >= 3:
                if sha(folder / 'adapter/adapter_model.safetensors') != complete['adapter_sha256']:
                    raise ValueError('Adapter hash mismatch')
                losses = read(folder / 'loss.jsonl')
                if len(losses) != cfg['optimizer_steps'] or any(
                        r['step'] != i + 1 or not math.isfinite(r['loss']) for i, r in enumerate(losses)):
                    raise ValueError('Loss/step mismatch')
                examples = read(folder / 'train_examples.jsonl')
                target = read(source / f'paired_targets/{c}_targets.jsonl')
                if list(map(key, examples)) != list(map(key, train)) or [r['target'] for r in examples] != [r['target'] for r in target]:
                    raise ValueError('Training example/target mismatch')
            labels[c, seed] = checked_predictions(folder, rows)
            per_seed.append({'seed': seed, 'condition': c, **{
                k: 100 * statistics.mean(v) for k, v in labels[c, seed].items()}})
        init = [json.loads(((source if seed < 3 else run) / f'{c}_s{seed}/initialization.json').read_text())
                for c in CONDITIONS]
        if init[0]['lora_initial_sha256'] != init[1]['lora_initial_sha256'] or any(
                r['train_seed'] != seed or r['continued_from'] is not None for r in init):
            raise ValueError('Pairing mismatch')
    baseline = checked_predictions(source / 'baseline', rows)
    base_acc = {m: 100 * statistics.mean(v) for m, v in baseline.items()}
    groups = {}
    for label, seeds in [('original_012', range(3)), ('added_345', range(3, 6)), ('all_012345', range(6))]:
        groups[label] = {}
        for metric in METRICS:
            acc = {c: [100 * statistics.mean(labels[c, s][metric]) for s in seeds] for c in CONDITIONS}
            groups[label][metric] = {
                'accuracy_pct': {c: moments(v) for c, v in acc.items()},
                'gain_over_base_pp': {c: moments([v - base_acc[metric] for v in vs]) for c, vs in acc.items()},
                'context_minus_direct_pp': moments([x-y for x,y in zip(acc['context'], acc['direct'])])}
    out = run / 'analysis_original_scorer'
    out.mkdir(exist_ok=False)
    report = {'scorer': 'frozen scoring_v2_v1.is_correct (signed floor)', 'primary_split': 'D_reservedC',
              'n': n, 'baseline_accuracy_pct': base_acc, 'per_seed': per_seed, 'groups': groups,
              'smoke_test': smoke, 'new_scorer_applied': False,
              'interpretation': 'Numbers only; wait for the new scorer. D is an observed stability replication set.'}
    js(out / 'report.json', report)
    lines = [f'COMPLETE: six added fits (3,4,5 x context/direct); D={n}',
             'Original frozen scorer only; baseline and targets reused; no regeneration.',
             'Seed intervals are across fits; not independent question replications.',
             f'Baseline: {base_acc}', 'Per-seed accuracy (%):']
    for row in per_seed:
        lines.append(f"seed={row['seed']} {row['condition']}: composed={row['composed_ok']:.4f}, hop1={row['q1_ok']:.4f}, hop2={row['q2_ok']:.4f}")
    for label, group in groups.items():
        for metric, values in group.items():
            c, d = [values['accuracy_pct'][k] for k in CONDITIONS]
            diff = values['context_minus_direct_pp']
            lines.append(f"{label} {metric}: context={c['mean']:.4f}% SD={c['sample_sd']:.4f}; direct={d['mean']:.4f}% SD={d['sample_sd']:.4f}; paired diff={diff['mean']:+.4f} pp, seed95={diff['seed_t95']}")
    (out / 'report.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


def checked_predictions(folder, rows):
    predictions = read(folder / 'eval_D_reservedC.jsonl')
    if list(map(key, predictions)) != list(map(key, rows)):
        raise ValueError('Prediction identity/order mismatch: ' + str(folder))
    labels = {}
    for metric, (field, answer) in METRICS.items():
        values = [is_correct(p[field], r['e'][answer]) for p,r in zip(predictions, rows)]
        if values != [p[metric] for p in predictions]:
            raise ValueError('Original scorer labels disagree: ' + str(folder))
        labels[metric] = values
    if any(old_is_correct(p['pred'], r['e']['Answer']) != p['legacy_composed_ok'] for p,r in zip(predictions, rows)):
        raise ValueError('Legacy scoring mismatch')
    return labels


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--smoke-test', action='store_true')
    args = ap.parse_args()
    summarize(args.run, args.smoke_test)
