"""Apply frozen v3 to 13 cached prediction sets without any model loading."""
import argparse
import collections
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
METRICS = {'composed': ('pred','Answer','composed_ok'), 'q1': ('q1_pred','A1','q1_ok'),
           'q2': ('q2_pred','A2','q2_ok')}
MODELS = ['baseline'] + [f'{c}_s{s}' for s in range(6) for c in ('context','direct')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def js(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def read(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def verify(root, local=False):
    # Frozen policy is checked before importing/evaluating the scorer.
    freeze = json.loads((HERE / 'SCORER_FREEZE.json').read_text())
    for name, digest in freeze['sha256'].items():
        if sha(HERE / name) != digest:
            raise ValueError('Frozen scoring rule changed: ' + name)
    manifest = json.loads((HERE / 'input_manifest.json').read_text())
    paths = json.loads((HERE / 'local_inputs.json').read_text()) if local else {
        k: str(root / v['remote']) for k,v in manifest.items()}
    if set(paths) != set(['plan'] + MODELS):
        raise ValueError('Input set differs from baseline + 12 fits + plan')
    for name, path in paths.items():
        if sha(path) != manifest[name]['sha256']:
            raise ValueError('Input checksum mismatch: ' + name)
    return {k:Path(v) for k,v in paths.items()}


def moments(values):
    n = len(values)
    critical = {3:4.302652729749464, 6:2.570581835636305}[n]
    mean, sd = statistics.mean(values), statistics.stdev(values)
    half = critical * sd / math.sqrt(n)
    return {'values':values,'mean':mean,'sd':sd,'seed_t95':[mean-half,mean+half]}


def run(root, out, local=False):
    paths = verify(root, local)
    if out.exists():
        raise FileExistsError('No overwrite: ' + str(out))
    from scorer_v3 import score
    from scoring_v2_v1 import is_correct as old_score
    plan = read(paths['plan'])
    keys = [(r['category'],r['person']) for r in plan]
    if len(keys) != 500 or len(set(keys)) != 500:
        raise ValueError('D must contain 500 unique items')
    start = time.time()
    out.mkdir(parents=True, exist_ok=False)
    js(out / 'scorer_freeze.json', json.loads((HERE / 'SCORER_FREEZE.json').read_text()))
    js(out / 'input_receipt.json', {k:{'path':str(v),'sha256':sha(v)} for k,v in paths.items()})
    js(out / 'code_hashes.json', {p.name:sha(p) for p in HERE.iterdir() if p.suffix in ('.py','.json','.txt','.sh')})
    with (out / 'items_D_reservedC.jsonl').open('x') as f:
        f.write(paths['plan'].read_text())
    models = {}
    queue_n = 0
    with (out / 'review_queue.jsonl').open('x') as queue:
        for model in MODELS:
            predictions = read(paths[model])
            if [(r['category'],r['person']) for r in predictions] != keys:
                raise ValueError('Item identity/order mismatch: ' + model)
            counts = {m:collections.Counter() for m in METRICS}
            reasons = {m:collections.Counter() for m in METRICS}
            with (out / f'{model}.jsonl').open('x') as f:
                for row, pred in zip(plan,predictions):
                    labels = {}
                    for metric,(field,gold,oldfield) in METRICS.items():
                        prior = bool(old_score(pred[field],row['e'][gold]))
                        if prior != bool(pred[oldfield]):
                            raise ValueError('Old score differs from cached label')
                        result = score(pred[field],row['e'][gold],row['category'],metric)
                        labels[metric] = {'old':prior,'v3':result['correct'],
                                          'changed':prior != result['correct'], **result}
                        counts[metric].update({'old_correct':int(prior),'new_correct':int(result['correct']),
                            'needs_review':int(result['needs_review']),
                            'old_true_new_false':int(prior and not result['correct']),
                            'old_false_new_true':int(not prior and result['correct'])})
                        reasons[metric][result['reason']] += 1
                        if labels[metric]['changed'] or result['needs_review']:
                            record = {'model':model,'category':row['category'],'person':row['person'],
                                'metric':metric,'prediction':pred[field],'reference':row['e'][gold],
                                'question':row['e'][{'composed':'Question','q1':'Q1','q2':'Q2'}[metric]],
                                **labels[metric]}
                            queue.write(json.dumps(record,ensure_ascii=False)+'\n')
                            queue_n += 1
                    f.write(json.dumps({'category':row['category'],'person':row['person'],
                        'original_prediction':pred,'scores':labels},ensure_ascii=False)+'\n')
            models[model] = {m:{**dict(v), 'old_accuracy_pct':v['old_correct']/5,
                               'new_accuracy_pct':v['new_correct']/5,'reasons':dict(reasons[m])}
                             for m,v in counts.items()}
            print(model, models[model]['composed'], flush=True)
    groups = {}
    for group, seeds in [('original_012',range(3)),('added_345',range(3,6)),('all_012345',range(6))]:
        groups[group] = {}
        for metric in METRICS:
            groups[group][metric] = {}
            for scoring in ('old','new'):
                acc = {c:[models[f'{c}_s{s}'][metric][scoring+'_accuracy_pct'] for s in seeds]
                       for c in ('context','direct')}
                base = models['baseline'][metric][scoring+'_accuracy_pct']
                groups[group][metric][scoring] = {
                    'accuracy_pct':{c:moments(v) for c,v in acc.items()},
                    'gain_over_base_pp':{c:moments([x-base for x in v]) for c,v in acc.items()},
                    'context_minus_direct_pp':moments([x-y for x,y in zip(acc['context'],acc['direct'])])}
    report = {'scorer_version':'v3','n_per_model':500,'models':models,'groups':groups,
              'review_queue_records':queue_n,'reference_policy':'unchanged_dataset_snapshot',
              'interpretation':'Post-hoc conservative whole-answer score; review flags are counted as unaccepted, not independently established factual errors.'}
    js(out / 'report.json', report)
    lines = ['EXP12 V3 — cached outputs only; unchanged D references; 500 items per model.',
             'Post-hoc conservative automated score. Unsupported forms are flagged, not dropped.',
             'Model | composed old% -> new% | composed needs_review/500']
    for name in MODELS:
        m = models[name]['composed']
        lines.append(f"{name}: {m['old_accuracy_pct']:.2f} -> {m['new_accuracy_pct']:.2f}; review={m['needs_review']}")
    for group, metrics in groups.items():
        for metric, scores in metrics.items():
            for name, values in scores.items():
                diff = values['context_minus_direct_pp']
                c,d = [values['accuracy_pct'][k]['mean'] for k in ('context','direct')]
                lines.append(f'{group} {metric} {name}: context={c:.4f}%, direct={d:.4f}%, paired difference={diff["mean"]:+.4f} pp, seed95={diff["seed_t95"]}')
    lines += ['No reference relabeling or human adjudication performed. No new training or prediction generation.']
    with (out / 'report.txt').open('x') as f:
        f.write('\n'.join(lines)+'\n')
    # Verify immutable inputs again before the completion marker.
    verify(root, local)
    js(out / 'complete.json', {'scorer_version':'v3','models':13,'items_per_model':500,
        'metric_labels':19500,'new_training_runs':0,'new_model_generations':0,
        'seconds':time.time()-start,'review_queue_records':queue_n,'references_relabelled':False})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path.home()/'inward')
    ap.add_argument('--out',type=Path)
    ap.add_argument('--local-inputs',action='store_true')
    ap.add_argument('--check-only',action='store_true')
    a = ap.parse_args()
    if a.check_only:
        verify(a.root,a.local_inputs)
        print('Frozen scorer and all 14 input files verified; no scoring performed.')
    else:
        if a.out is None:
            ap.error('--out required')
        run(a.root,a.out,a.local_inputs)
